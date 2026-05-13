"""Setup → ROI subtab.

Workflow: user loads the project folder (sidebar Browse), walks through
every image with Prev/Next, draws one or more closed-path / rect ROIs on
each, and tags them. The per-image ROI list lives in
``server_state.get_session(sid).extra['rois']`` keyed by absolute image
path — that's the single source of truth. The Dash side only carries:

  - ``roi-images-store``: {folder, images, idx} for navigation
  - ``roi-pending-shape``: the freshly-drawn shape awaiting a tag/save
  - ``roi-tick``: a counter; bump it whenever the ROI list for the current
    image changes, and the renderer + list updater read state and re-paint.

ROI entry format in state:
  {"tag": str, "type": "rect" | "path", "shape": <plotly shape dict>}
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tifffile
from dash import ALL, Input, Output, State, callback, ctx, dcc, html, no_update

from glia.prepare import prepared_dir
from glia.roi import save_project_rois, shape_anchor
from glia_dash import server_state
from glia_dash.components import empty_state


_IMAGE_CACHE: dict[str, tuple[float, np.ndarray]] = {}


def _load_image(path: str) -> np.ndarray:
    """mtime-aware cache: if the file changed (e.g. user re-prepared with a
    different channel), drop the stale entry and re-read."""
    try:
        mtime = Path(path).stat().st_mtime
    except OSError:
        return tifffile.imread(path)
    cached = _IMAGE_CACHE.get(path)
    if cached is None or cached[0] != mtime:
        _IMAGE_CACHE[path] = (mtime, tifffile.imread(path))
    return _IMAGE_CACHE[path][1]


def _list_images(folder: str, mode: str = "microglia") -> list[str]:
    """Return the prepared 8-bit TIFFs for the project, falling back to
    legacy top-level TIFFs if the Prepare step hasn't been run yet."""
    if not folder or not Path(folder).is_dir():
        return []
    prep = prepared_dir(folder, mode)
    if prep.is_dir():
        prepared = sorted(
            str(p) for p in prep.iterdir()
            if p.suffix.lower() in (".tif", ".tiff") and p.is_file()
        )
        if prepared:
            return prepared
    # Legacy fall-through: TIFFs sitting at the project root.
    return sorted(
        str(p) for p in Path(folder).iterdir()
        if p.suffix.lower() in (".tif", ".tiff") and p.is_file()
    )


def _rois_for(sid: str | None, image_path: str) -> list[dict]:
    state = server_state.get_session(sid)
    return list(state.extra.get("rois", {}).get(image_path, []))


def _set_rois_for(sid: str | None, image_path: str, rois: list[dict]):
    state = server_state.get_session(sid)
    state.extra.setdefault("rois", {})[image_path] = rois
    # Persist to disk under the project folder so the work survives a
    # restart. Best-effort — swallow errors so the UI doesn't break.
    try:
        save_project_rois(state.project_dir, state.extra["rois"])
    except Exception:
        pass


def _all_tags_in_project(sid: str | None) -> list[str]:
    """Every tag used in any image's ROI list, in first-seen order."""
    state = server_state.get_session(sid)
    seen: list[str] = []
    seen_set: set[str] = set()
    for rois in state.extra.get("rois", {}).values():
        for r in rois:
            t = r.get("tag")
            if t and t not in seen_set:
                seen.append(t)
                seen_set.add(t)
    return seen


def _normalize_tag(tag: str | None) -> str:
    """Make a tag safe for use as a filename segment.

    Filename encoding for single cells is ``<image_stem>__<roi_tag>__<N>.tif``
    where ``__`` (double underscore) separates the three groups. ROI tags
    therefore must not contain ``__``. Spaces collapse to hyphens; any
    other forbidden character is stripped.
    """
    if not tag:
        return ""
    t = tag.strip().replace(" ", "-")
    # Disallow the group-separator sequence inside a tag.
    while "__" in t:
        t = t.replace("__", "_")
    return t


# ── Layout ───────────────────────────────────────────────────────────


def roi_layout(sid: str | None) -> html.Div:
    state = server_state.get_session(sid)
    folder = state.project_dir
    mode = getattr(state, "mode", "microglia") or "microglia"
    images = _list_images(folder, mode)
    idx = int(state.extra.get("roi_image_idx", 0))
    if images and idx >= len(images):
        idx = 0
        state.extra["roi_image_idx"] = 0

    current_path = images[idx] if images else ""
    rois_here = _rois_for(sid, current_path) if current_path else []
    no_folder = not images

    nav_label = (f"Image {idx + 1} of {len(images)} · "
                 f"{os.path.basename(current_path)}") if images else ""

    banner = (empty_state(icon="📁", title="No project folder loaded",
                          text=("Pick a folder from the sidebar "
                                "(Browse folder…) containing the TIFFs you "
                                "want to analyze. ROI drawing happens here, "
                                "image by image, before thresholding."))
              if no_folder else None)

    return html.Div([
        html.Div(banner, style={"display": "block" if no_folder else "none"}),

        html.Div(style={"display": "none" if no_folder else "block"},
                 children=[
            html.Div([
                html.Span(nav_label, id="roi-nav-label",
                          style={"fontSize": "0.85rem",
                                 "color": "var(--ned-text)",
                                 "fontWeight": "500"}),
                html.Div(style={"flex": "1"}),
                html.Button("◀ Prev", id="roi-prev",
                            className="btn-ned-secondary",
                            disabled=(not images or idx == 0),
                            style={"marginRight": "6px"}),
                html.Button("Next ▶", id="roi-next",
                            className="btn-ned-secondary",
                            disabled=(not images or idx >= len(images) - 1)),
            ], style={"display": "flex", "alignItems": "center",
                      "marginBottom": "8px"}),

            html.Div([
                html.Span("Drawing tool:",
                          style={"fontSize": "0.78rem",
                                 "color": "var(--ned-text-muted)",
                                 "marginRight": "8px"}),
                dcc.RadioItems(
                    id="roi-draw-mode",
                    options=[
                        {"label": " Rectangle", "value": "drawrect"},
                        {"label": " Polygon", "value": "drawclosedpath"},
                    ],
                    value="drawclosedpath",
                    inline=True,
                    style={"fontSize": "0.82rem", "display": "inline-block"},
                ),
                html.Span(" · ", style={"opacity": "0.4",
                                        "marginLeft": "8px"}),
                html.Span("Use the modebar (top-right) to draw. After "
                          "drawing, give the ROI a tag below and click "
                          "Save ROI.",
                          style={"fontSize": "0.78rem",
                                 "color": "var(--ned-text-muted)",
                                 "marginLeft": "8px"}),
            ], style={"marginBottom": "8px"}),

            dcc.Graph(id="roi-graph",
                      figure=go.Figure(),
                      config={
                          "scrollZoom": True,
                          "displayModeBar": True,
                          "displaylogo": False,
                          "modeBarButtonsToAdd": [
                              "drawrect", "drawclosedpath", "eraseshape",
                          ],
                          "modeBarButtonsToRemove": [
                              "lasso2d", "select2d", "autoScale2d",
                          ],
                      },
                      style={"height": "600px"}),

            # Tag-reuse chips: every tag used anywhere in the project,
            # clickable to autofill the tag input. Reduces spelling drift.
            html.Div(id="roi-tag-chips",
                     children=_render_tag_chips(_all_tags_in_project(sid)),
                     style={"marginTop": "12px"}),

            html.Div([
                html.Span("New ROI tag:",
                          style={"fontSize": "0.78rem",
                                 "color": "var(--ned-text-muted)",
                                 "marginRight": "8px"}),
                dcc.Input(id="roi-new-tag", type="text",
                          placeholder="e.g. left-CA1",
                          style={"width": "180px"}),
                html.Button("Save ROI", id="roi-save",
                            className="btn-ned-primary",
                            style={"marginLeft": "12px"},
                            disabled=True),
                html.Span(id="roi-save-hint",
                          children=("Draw a shape on the image with the "
                                    "modebar tools, then this button "
                                    "becomes active."),
                          style={"fontSize": "0.78rem",
                                 "color": "var(--ned-text-muted)",
                                 "marginLeft": "12px"}),
            ], style={"marginTop": "8px",
                      "display": "flex", "alignItems": "center"}),

            html.Div(id="roi-list",
                     style={"marginTop": "12px"},
                     children=_render_roi_list(rois_here)),
        ]),

        dcc.Store(id="roi-images-store", data={"folder": folder,
                                                "images": images,
                                                "idx": idx}),
        dcc.Store(id="roi-pending-shape", data=None),
        dcc.Store(id="roi-tick", data=0),
    ])


def _render_tag_chips(tags: list[str]) -> list:
    """Clickable chip row of every tag used anywhere in the project."""
    if not tags:
        return [html.Span("No tags yet — type one below to start.",
                          style={"fontSize": "0.78rem",
                                 "color": "var(--ned-text-muted)",
                                 "fontStyle": "italic"})]
    return [
        html.Span("Reuse tag:",
                  style={"fontSize": "0.78rem",
                         "color": "var(--ned-text-muted)",
                         "marginRight": "8px",
                         "textTransform": "uppercase",
                         "letterSpacing": "0.5px"}),
        *[
            html.Button(
                t, id={"type": "roi-tag-chip", "tag": t},
                className="channel-chip",
                n_clicks=0,
                style={"marginRight": "4px"},
            )
            for t in tags
        ],
    ]


def _render_roi_list(rois: list[dict]) -> html.Div:
    if not rois:
        return html.Div("No ROIs yet for this image.",
                        style={"fontSize": "0.78rem",
                               "color": "var(--ned-text-muted)",
                               "fontStyle": "italic"})
    rows = []
    for i, r in enumerate(rois):
        rows.append(html.Div([
            html.Span(r["tag"],
                      style={"fontWeight": "600",
                             "color": "var(--ned-accent)",
                             "minWidth": "140px",
                             "display": "inline-block"}),
            html.Span(r["type"],
                      style={"fontSize": "0.78rem",
                             "color": "var(--ned-text-muted)",
                             "marginLeft": "12px",
                             "minWidth": "60px",
                             "display": "inline-block"}),
            html.Button("Delete", id={"type": "roi-delete", "index": i},
                        className="btn-ned-danger",
                        style={"padding": "2px 8px",
                               "fontSize": "0.75rem",
                               "marginLeft": "12px"}),
        ], style={"padding": "4px 0",
                  "display": "flex", "alignItems": "center"}))
    return html.Div([
        html.Div("ROIs on this image:",
                 style={"fontSize": "0.78rem",
                        "color": "var(--ned-text-muted)",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.5px",
                        "marginBottom": "6px"}),
        *rows,
    ])


# ── Figure helpers ────────────────────────────────────────────────────


def _image_figure(path: str, theme: str, rois: list[dict],
                  draw_mode: str = "drawclosedpath") -> go.Figure:
    img = _load_image(path)
    fig = px.imshow(img, color_continuous_scale="gray",
                    binary_string=True, aspect="equal")
    fig.update_layout(
        margin=dict(l=0, r=0, t=8, b=0),
        coloraxis_showscale=False,
        dragmode=draw_mode,
        newshape=dict(line_color="#58a6ff", line_width=2,
                      fillcolor="rgba(88,166,255,0.10)"),
        shapes=[s["shape"] for s in rois],
        annotations=[
            dict(
                x=shape_anchor(s["shape"])[0],
                y=shape_anchor(s["shape"])[1],
                text=s["tag"],
                showarrow=False,
                font=dict(color="#58a6ff", size=11),
                bgcolor="rgba(15,17,23,0.7)",
                bordercolor="#58a6ff",
                borderwidth=1,
                xref="x", yref="y",
            )
            for s in rois
        ],
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    bg, plot_bg, fg, grid = ("#ffffff", "#f6f8fa", "#1f2328", "#d0d7de") \
        if (theme or "light") == "light" \
        else ("#1c2128", "#0f1117", "#e6edf3", "#2d333b")
    fig.update_layout(paper_bgcolor=bg, plot_bgcolor=plot_bg,
                      font=dict(color=fg, family="IBM Plex Sans, sans-serif",
                                size=11))
    return fig


# Shape centroid helpers live in glia.roi (shared with setup_threshold).


def _current_image(store: dict) -> str:
    if not store:
        return ""
    images = store.get("images", [])
    idx = int(store.get("idx", 0))
    return images[idx] if (0 <= idx < len(images)) else ""


# ── Callbacks ─────────────────────────────────────────────────────────


@callback(
    Output("roi-graph", "figure"),
    Output("roi-list", "children"),
    Output("roi-tag-chips", "children"),
    Input("roi-images-store", "data"),
    Input("roi-tick", "data"),
    Input("roi-draw-mode", "value"),
    Input("theme-store", "data"),
    State("session-id", "data"),
    prevent_initial_call=False,
)
def render_roi_view(store, _tick, draw_mode, theme, sid):
    """Single renderer: image+shapes graph, ROI list, and tag chips. All
    three read from server-side state for the currently selected image so
    there's no double source of truth."""
    chips = _render_tag_chips(_all_tags_in_project(sid))
    if not store or not store.get("images"):
        return go.Figure(), _render_roi_list([]), chips
    img_path = _current_image(store)
    if not img_path:
        return go.Figure(), _render_roi_list([]), chips
    rois = _rois_for(sid, img_path)
    fig = _image_figure(img_path, theme or "light", rois,
                        draw_mode or "drawclosedpath")
    return fig, _render_roi_list(rois), chips


@callback(
    Output("roi-pending-shape", "data", allow_duplicate=True),
    Output("roi-save", "disabled"),
    Output("roi-save-hint", "children"),
    Input("roi-graph", "relayoutData"),
    prevent_initial_call=True,
)
def on_graph_relayout(relayout):
    """Stash the last drawn shape as the pending ROI."""
    if not relayout or "shapes" not in relayout:
        return no_update, no_update, no_update
    drawn = relayout["shapes"] or []
    if not drawn:
        return None, True, ("Draw a shape on the image with the modebar "
                            "tools, then this button becomes active.")
    return drawn[-1], False, "Type a tag and click Save ROI."


@callback(
    Output("roi-tick", "data", allow_duplicate=True),
    Output("roi-new-tag", "value"),
    Output("roi-pending-shape", "data", allow_duplicate=True),
    Input("roi-save", "n_clicks"),
    State("roi-pending-shape", "data"),
    State("roi-new-tag", "value"),
    State("roi-images-store", "data"),
    State("roi-tick", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_save_roi(n_clicks, pending, tag, store, tick, sid):
    if not n_clicks or pending is None:
        return no_update, no_update, no_update
    img_path = _current_image(store or {})
    if not img_path:
        return no_update, no_update, no_update
    rois = _rois_for(sid, img_path)
    tag = _normalize_tag(tag)
    if not tag:
        tag = f"roi-{len(rois) + 1}"
    shape_type = "rect" if pending.get("type") == "rect" else "path"
    rois.append({"tag": tag, "type": shape_type, "shape": pending})
    _set_rois_for(sid, img_path, rois)
    return (tick or 0) + 1, "", None


@callback(
    Output("roi-new-tag", "value", allow_duplicate=True),
    Input({"type": "roi-tag-chip", "tag": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def on_tag_chip_click(all_n_clicks):
    """Click a chip → autofill the new-tag input with that tag."""
    fired = any(n for n in (all_n_clicks or []))
    if not fired:
        return no_update
    trig = ctx.triggered_id
    if isinstance(trig, dict) and "tag" in trig:
        return trig["tag"]
    return no_update


@callback(
    Output("roi-tick", "data", allow_duplicate=True),
    Input({"type": "roi-delete", "index": ALL}, "n_clicks"),
    State("roi-images-store", "data"),
    State("roi-tick", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_delete_roi(all_n_clicks, store, tick, sid):
    trig = ctx.triggered_id
    if not isinstance(trig, dict) or "index" not in trig:
        return no_update
    # Only fire when the click count is non-zero (avoids ghost fires from
    # pattern-matching ID list churn).
    fired = any(n for n in (all_n_clicks or []))
    if not fired:
        return no_update
    img_path = _current_image(store or {})
    if not img_path:
        return no_update
    rois = _rois_for(sid, img_path)
    i = int(trig["index"])
    if not (0 <= i < len(rois)):
        return no_update
    rois.pop(i)
    _set_rois_for(sid, img_path, rois)
    return (tick or 0) + 1


@callback(
    Output("roi-images-store", "data", allow_duplicate=True),
    Output("roi-tick", "data", allow_duplicate=True),
    Input("roi-prev", "n_clicks"),
    Input("roi-next", "n_clicks"),
    State("roi-images-store", "data"),
    State("roi-tick", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_nav(prev_n, next_n, store, tick, sid):
    if not store:
        return no_update, no_update
    images = store.get("images", [])
    idx = int(store.get("idx", 0))
    if ctx.triggered_id == "roi-prev":
        idx = max(0, idx - 1)
    elif ctx.triggered_id == "roi-next":
        idx = min(len(images) - 1, idx + 1) if images else 0
    else:
        return no_update, no_update

    server_state.get_session(sid).extra["roi_image_idx"] = idx
    new_store = dict(store)
    new_store["idx"] = idx
    return new_store, (tick or 0) + 1


@callback(
    Output("roi-prev", "disabled"),
    Output("roi-next", "disabled"),
    Output("roi-nav-label", "children"),
    Input("roi-images-store", "data"),
    prevent_initial_call=False,
)
def update_nav_state(store):
    if not store or not store.get("images"):
        return True, True, ""
    images = store["images"]
    idx = int(store.get("idx", 0))
    label = (f"Image {idx + 1} of {len(images)} · "
             f"{os.path.basename(images[idx])}")
    return idx == 0, idx >= len(images) - 1, label
