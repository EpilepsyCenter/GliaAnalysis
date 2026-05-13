"""Setup → Threshold subtab — FIJI executable, thresholding params, preview."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tifffile
from dash import (
    ALL, Input, Output, State, callback, clientside_callback, ctx, dcc, html,
    no_update,
)
from skimage import measure
import dash_bootstrap_components as dbc

from glia.config import (
    DEFAULT_THRESHOLD_METHODS_GLOBAL,
    DEFAULT_THRESHOLD_METHODS_LOCAL,
)
from glia.fiji_runner import discover_fiji
from glia.preprocess import (
    MANUAL_METHOD,
    apply_threshold,
    component_area_histogram,
    compute_threshold_value,
    preprocess_fiji_style,
    unsupported_methods,
)
from glia.prepare import prepared_dir
from glia.roi import shape_anchor, union_mask
from glia_dash import server_state
from glia_dash.components import alert, browse_file, empty_state


# Tiny mtime-aware image cache so we don't re-read disk on every tweak,
# but DO see fresh data if the user re-prepared the image.
_IMAGE_CACHE: dict[str, tuple[float, np.ndarray]] = {}

# Last-rendered labeled mask for the current Setup preview. Used for the
# click-on-cell area lookup so we don't redo the threshold + label work.
_LABELS_CACHE: dict[str, np.ndarray] = {"labels": None}


def _markers_trace(selected: list[dict]) -> dict:
    """Build a scatter trace with red circle markers at selected centroids."""
    return {
        "type": "scatter",
        "x": [s["cx"] for s in selected],
        "y": [s["cy"] for s in selected],
        "mode": "markers",
        "marker": {
            "size": 18,
            "color": "rgba(248, 81, 73, 0.7)",
            "line": {"color": "white", "width": 2},
            "symbol": "circle",
        },
        "hoverinfo": "skip",
        "showlegend": False,
        "name": "selection",
    }


def _load_image(path: str) -> np.ndarray:
    try:
        mtime = Path(path).stat().st_mtime
    except OSError:
        return tifffile.imread(path)
    cached = _IMAGE_CACHE.get(path)
    if cached is None or cached[0] != mtime:
        _IMAGE_CACHE[path] = (mtime, tifffile.imread(path))
    return _IMAGE_CACHE[path][1]


def _list_project_images(folder: str) -> list[str]:
    """Return the prepared 8-bit TIFFs (Prepare step output) for the
    Threshold preview; legacy fall-through to top-level TIFFs."""
    if not folder or not Path(folder).is_dir():
        return []
    prep = prepared_dir(folder)
    if prep.is_dir():
        prepared = sorted(
            str(p) for p in prep.iterdir()
            if p.suffix.lower() in (".tif", ".tiff") and p.is_file()
        )
        if prepared:
            return prepared
    return sorted(
        str(p) for p in Path(folder).iterdir()
        if p.suffix.lower() in (".tif", ".tiff") and p.is_file()
    )


def _render_image_chips(folder: str, current_path: str) -> list:
    """Row of clickable filename chips for every TIFF in the project folder."""
    images = _list_project_images(folder)
    if not images:
        return [html.Span(
            "Load a project folder from the sidebar to scrub through images.",
            style={"fontSize": "0.82rem",
                   "color": "var(--ned-text-muted)"})]
    return [
        html.Span("Image:",
                  style={"fontSize": "0.72rem",
                         "color": "var(--ned-text-muted)",
                         "textTransform": "uppercase",
                         "letterSpacing": "0.5px",
                         "marginRight": "8px"}),
        *[
            html.Button(
                os.path.basename(p),
                id={"type": "setup-image-chip", "path": p},
                n_clicks=0,
                className=("channel-chip selected"
                           if p == current_path else "channel-chip"),
                style={"marginRight": "4px"},
            )
            for p in images
        ],
    ]


def threshold_layout(sid: str | None) -> html.Div:
    """Return the body of the Threshold subtab (no outer page header).

    The orchestrating Setup page (glia_dash.pages.setup) wraps this and the
    ROI subtab in a dbc.Tabs container.
    """
    state = server_state.get_session(sid)
    fiji_default = state.fiji_path or (discover_fiji() or "")

    methods_global = DEFAULT_THRESHOLD_METHODS_GLOBAL
    methods_local = DEFAULT_THRESHOLD_METHODS_LOCAL
    initial_methods = (methods_global if state.threshold_kind == "global"
                       else methods_local)

    preview_path = state.extra.get("preview_image_path", "")
    # If no preview picked yet but the project has images, default to the
    # first one — the chip row will start on a useful image.
    project_images = _list_project_images(state.project_dir)
    if not preview_path and project_images:
        preview_path = project_images[0]
        state.extra["preview_image_path"] = preview_path

    return html.Div([
        html.Div([
            html.Label("FIJI executable", className="form-label",
                       style={"fontSize": "0.82rem",
                              "color": "var(--ned-text-muted)"}),
            html.Div([
                dcc.Input(id="setup-fiji-path", type="text",
                          value=fiji_default,
                          placeholder="/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx",
                          style={"flex": "1"}),
                dbc.Button("Browse…", id="setup-fiji-browse",
                           className="btn-ned-secondary",
                           style={"marginLeft": "8px"}),
            ], style={"display": "flex", "alignItems": "center"}),
        ], style={"marginBottom": "24px"}),

        # Clickable chip row of every image in the project folder. The
        # currently-previewed image is highlighted with .selected.
        html.Div(id="setup-image-chip-row",
                 children=_render_image_chips(state.project_dir,
                                              preview_path),
                 style={"marginBottom": "16px",
                        "display": "flex",
                        "flexWrap": "wrap",
                        "alignItems": "center",
                        "gap": "0"}),

        # Threshold controls — single row across the full width
        html.Div([
            html.Div([
                html.Label("Kind",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dbc.RadioItems(
                    id="setup-threshold-kind",
                    options=[
                        {"label": "Global", "value": "global"},
                        {"label": "Local",  "value": "local"},
                    ],
                    value=state.threshold_kind, inline=True,
                ),
            ], style={"marginRight": "32px"}),

            html.Div([
                html.Label("Method",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dcc.Dropdown(
                    id="setup-threshold-method",
                    options=[{"label": m, "value": m} for m in initial_methods],
                    value=state.threshold_method,
                    clearable=False,
                    style={"width": "200px"},
                ),
            ], style={"marginRight": "32px"}),

            html.Div(id="setup-local-radius-wrap",
                     style={"display": "block" if state.threshold_kind
                            == "local" else "none",
                            "marginRight": "32px"},
                     children=[
                         html.Label("Local radius (px)",
                                    style={"fontSize": "0.72rem",
                                           "color":
                                               "var(--ned-text-muted)",
                                           "textTransform": "uppercase",
                                           "letterSpacing": "0.5px"}),
                         dcc.Input(id="setup-local-radius",
                                   type="number", min=1,
                                   value=state.local_radius,
                                   style={"width": "100px"}),
                     ]),

            html.Div([
                html.Label("Preprocess",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dbc.Switch(id="setup-preprocess",
                           label="FIJI chain",
                           value=state.preprocess,
                           style={"fontSize": "0.82rem"}),
            ], style={"marginRight": "32px"}),

            html.Div([
                html.Label("Area min (px²)",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dcc.Input(id="setup-area-min", type="number", min=0,
                          value=state.area_min, style={"width": "100px"}),
            ], style={"marginRight": "24px"}),

            html.Div([
                html.Label("Area max (px²)",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dcc.Input(id="setup-area-max", type="number", min=0,
                          value=state.area_max, style={"width": "100px"}),
            ]),
        ], style={"display": "flex", "alignItems": "flex-end",
                  "gap": "0", "marginBottom": "12px"}),

        # Threshold range slider (global only) — auto value from method,
        # draggable for fine override.
        html.Div(id="setup-threshold-slider-wrap",
                 style={"marginBottom": "20px",
                        "display": "block" if state.threshold_kind == "global"
                        else "none"},
                 children=[
                     html.Label("Threshold band (releasing the handle redraws)",
                                style={"fontSize": "0.72rem",
                                       "color": "var(--ned-text-muted)",
                                       "textTransform": "uppercase",
                                       "letterSpacing": "0.5px"}),
                     dcc.RangeSlider(
                         id="setup-threshold-slider",
                         min=0, max=255, step=1,
                         value=[state.threshold_lower, state.threshold_upper],
                         marks={0: "0", 64: "64", 128: "128",
                                192: "192", 255: "255"},
                         tooltip={"placement": "bottom",
                                  "always_visible": False},
                         updatemode="mouseup",
                     ),
                 ]),

        # Preview area — full width. The graphs always exist in the layout
        # (with empty figures initially) so Dash callbacks referencing them
        # have valid Input targets even before an image is picked. The
        # empty-state placeholder is shown/hidden by render_preview.
        html.Div(id="setup-preview-empty", children=[
            empty_state(
                icon="🔍",
                title="Pick a test image",
                text=("Choose one representative TIFF to preview the "
                      "threshold and the per-component size distribution. "
                      "Helps dial in area bounds before running FIJI on the "
                      "full set."),
            ),
        ]),
        html.Div(id="setup-preview-info",
                 style={"display": "none", "fontSize": "0.78rem",
                        "color": "var(--ned-text-muted)",
                        "marginBottom": "8px"}),
        html.Div(id="setup-preview-graphs-row",
                 style={"display": "none"},
                 children=[
                     dcc.Graph(id="setup-preview-orig", figure=go.Figure(),
                               config={
                                   "scrollZoom": True,
                                   "displayModeBar": True,
                                   "displaylogo": False,
                                   "modeBarButtonsToRemove":
                                       ["lasso2d", "select2d", "autoScale2d"],
                               },
                               style={"flex": "1", "height": "560px",
                                      "minWidth": "0"}),
                     dcc.Graph(id="setup-preview-binary", figure=go.Figure(),
                               config={
                                   "scrollZoom": True,
                                   "displayModeBar": True,
                                   "displaylogo": False,
                                   "modeBarButtonsToRemove":
                                       ["lasso2d", "select2d", "autoScale2d"],
                               },
                               style={"flex": "1", "height": "560px",
                                      "minWidth": "0", "marginLeft": "12px"}),
                 ]),
        dcc.Graph(id="setup-preview-histogram", figure=go.Figure(),
                  config={"displayModeBar": False},
                  style={"display": "none", "marginTop": "12px",
                         "height": "200px"}),

        # Click-on-cell area picker. Click ~5 representative cells (click
        # again to deselect), then Apply to set min/max bounds. The text
        # span is the only piece that updates dynamically — the buttons
        # need to be in the static layout so Dash callbacks referencing
        # them have valid Input targets.
        html.Div([
            html.Span(id="setup-selection-text",
                      children=("Click ~5 representative cells in either "
                                "preview to set area bounds.")),
            dbc.Button("Apply", id="setup-apply-bounds", size="sm",
                       className="btn-ned-primary", disabled=True),
            dbc.Button("Reset", id="setup-reset-bounds", size="sm",
                       className="btn-ned-secondary", disabled=True),
        ], id="setup-click-info",
            style={"marginTop": "8px", "minHeight": "32px",
                   "fontSize": "0.85rem",
                   "color": "var(--ned-text-muted)",
                   "display": "flex", "alignItems": "center",
                   "gap": "12px"}),

        dcc.Store(id="setup-preview-path-store", data=preview_path),
        # Selected cells: list of {"label": int, "area": int}
        dcc.Store(id="setup-selected-cells", data=[]),
        # Sink for the clientside zoom-sync callback.
        dcc.Store(id="setup-zoom-sync-sink"),
        # Dummy sink for the mirror_to_state callback below — Segment tab
        # reads threshold params from server-side state rather than from
        # Setup-tab Inputs (which don't exist when the user navigates away).
        dcc.Store(id="setup-state-sink"),
    ])


# ── Callbacks ─────────────────────────────────────────────────────────


# Sync zoom/pan between the two preview graphs. Clientside, so no server
# round-trip on every wheel event. The reentry guard prevents the mirror
# update from triggering another sync.
clientside_callback(
    """
    function(relayoutA, relayoutB) {
        const ctx = window.dash_clientside.callback_context;
        if (!ctx.triggered || !ctx.triggered.length) {
            return window.dash_clientside.no_update;
        }
        const trig = ctx.triggered[0];
        const data = trig.value;
        if (!data) return window.dash_clientside.no_update;

        const srcId = trig.prop_id.split('.')[0];
        const tgtId = (srcId === 'setup-preview-orig')
            ? 'setup-preview-binary' : 'setup-preview-orig';
        const tgtEl = document.getElementById(tgtId);
        if (!tgtEl) return window.dash_clientside.no_update;
        const tgtGraph = tgtEl.querySelector('.js-plotly-plot') || tgtEl;

        const update = {};
        if ('xaxis.range[0]' in data && 'xaxis.range[1]' in data) {
            update['xaxis.range'] = [data['xaxis.range[0]'],
                                     data['xaxis.range[1]']];
        }
        if ('yaxis.range[0]' in data && 'yaxis.range[1]' in data) {
            update['yaxis.range'] = [data['yaxis.range[0]'],
                                     data['yaxis.range[1]']];
        }
        if (data['xaxis.autorange'] === true) {
            update['xaxis.autorange'] = true;
        }
        if (data['yaxis.autorange'] === true) {
            update['yaxis.autorange'] = true;
        }
        if (Object.keys(update).length === 0) {
            return window.dash_clientside.no_update;
        }

        if (window._gliaZoomSyncing) {
            return window.dash_clientside.no_update;
        }
        window._gliaZoomSyncing = true;
        try {
            window.Plotly.relayout(tgtGraph, update);
        } catch (e) { /* graph may not be mounted yet */ }
        setTimeout(() => { window._gliaZoomSyncing = false; }, 50);
        return window.dash_clientside.no_update;
    }
    """,
    Output("setup-zoom-sync-sink", "data"),
    Input("setup-preview-orig", "relayoutData"),
    Input("setup-preview-binary", "relayoutData"),
    prevent_initial_call=True,
)


@callback(
    Output("setup-threshold-method", "options"),
    Output("setup-local-radius-wrap", "style"),
    Output("setup-threshold-slider-wrap", "style"),
    Input("setup-threshold-kind", "value"),
    prevent_initial_call=True,
)
def on_kind_change(kind):
    methods = (DEFAULT_THRESHOLD_METHODS_GLOBAL if kind == "global"
               else DEFAULT_THRESHOLD_METHODS_LOCAL)
    radius_style = {"display": "block" if kind == "local" else "none",
                    "marginRight": "32px"}
    slider_style = {"display": "block" if kind == "global" else "none",
                    "marginBottom": "20px"}
    return [{"label": m, "value": m} for m in methods], radius_style, slider_style


# When the method, image, or preprocessing flag changes, recompute the auto
# threshold and snap the slider to [auto, 255]. For Manual the slider stays
# wherever the user last left it.
@callback(
    Output("setup-threshold-slider", "value"),
    Input("setup-threshold-method", "value"),
    Input("setup-preview-path-store", "data"),
    Input("setup-preprocess", "value"),
    Input("setup-threshold-kind", "value"),
    State("setup-threshold-slider", "value"),
    prevent_initial_call=True,
)
def on_method_change(method, path, preprocess, kind, current):
    if kind != "global":
        return no_update
    if not path or not Path(path).is_file():
        return no_update
    if method == MANUAL_METHOD:
        return no_update
    try:
        img = _load_image(path)
        work = preprocess_fiji_style(img) if preprocess else img
        thresh, _ = compute_threshold_value(work, method or "Otsu")
    except Exception:
        return no_update
    lo = int(round(thresh))
    hi = int(current[1]) if current and len(current) == 2 else 255
    return [lo, hi]


@callback(
    Output("setup-preview-path-store", "data"),
    Input({"type": "setup-image-chip", "path": ALL}, "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_image_chip_click(all_n_clicks, sid):
    """Click a project file chip → set it as the preview image."""
    fired = any(n for n in (all_n_clicks or []))
    if not fired:
        return no_update
    trig = ctx.triggered_id
    if not isinstance(trig, dict) or "path" not in trig:
        return no_update
    path = trig["path"]
    server_state.get_session(sid).extra["preview_image_path"] = path
    return path


@callback(
    Output("setup-image-chip-row", "children"),
    Input("setup-preview-path-store", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def refresh_chip_active(path, sid):
    """Re-render the chip row so the active filename is highlighted."""
    folder = server_state.get_session(sid).project_dir
    return _render_image_chips(folder, path or "")


_PREVIEW_OUTPUTS = (
    Output("setup-preview-empty", "style"),
    Output("setup-preview-info", "style"),
    Output("setup-preview-graphs-row", "style"),
    Output("setup-preview-histogram", "style"),
    Output("setup-preview-info", "children"),
    Output("setup-preview-orig", "figure"),
    Output("setup-preview-binary", "figure"),
    Output("setup-preview-histogram", "figure"),
)


def _hide_preview(message: str | None = None):
    return (
        {"display": "block"},
        {"display": "none"},
        {"display": "none"},
        {"display": "none"},
        message or "",
        go.Figure(), go.Figure(), go.Figure(),
    )


@callback(
    *_PREVIEW_OUTPUTS,
    Input("setup-preview-path-store", "data"),
    Input("setup-threshold-method", "value"),
    Input("setup-threshold-kind", "value"),
    Input("setup-threshold-slider", "value"),
    Input("setup-local-radius", "value"),
    Input("setup-preprocess", "value"),
    Input("setup-area-min", "value"),
    Input("setup-area-max", "value"),
    Input("theme-store", "data"),
    Input("setup-selected-cells", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def render_preview(path, method, kind, slider, local_radius, preprocess,
                   area_min, area_max, theme, selected, sid):
    if not path or not Path(path).is_file():
        return _hide_preview()

    try:
        img = _load_image(path)
    except Exception:
        return _hide_preview()

    manual_lower = manual_upper = None
    if kind == "global" and slider and len(slider) == 2:
        manual_lower, manual_upper = float(slider[0]), float(slider[1])

    try:
        binary, info = apply_threshold(
            img,
            method=method or "Otsu",
            kind=kind or "global",
            local_radius=int(local_radius or 15),
            manual_lower=manual_lower,
            manual_upper=manual_upper,
            preprocess=bool(preprocess),
        )
    except Exception as e:
        return _hide_preview(f"Threshold failed: {e}")

    # If the user has drawn ROIs for this exact image in the ROI subtab,
    # mask everything outside the union. The histogram, click-on-cell
    # area picker, and visible thresholded image all then reflect what
    # the production Segment run will actually see.
    image_rois = server_state.get_session(sid).extra.get("rois", {}).get(path, [])
    info["roi_count"] = len(image_rois)
    if image_rois:
        roi_union = union_mask(image_rois, *binary.shape)
        binary = binary & roi_union
        # Recompute foreground fraction relative to the ROI area so it
        # reads as "fraction of analyzed region", not "fraction of frame".
        roi_pixels = int(roi_union.sum())
        if roi_pixels > 0:
            info["foreground_fraction"] = float(binary.sum()) / roi_pixels

    fig_orig = px.imshow(img, color_continuous_scale="gray",
                         binary_string=True, aspect="equal")
    fig_orig.update_layout(margin=dict(l=0, r=0, t=24, b=0),
                           coloraxis_showscale=False,
                           title=dict(text="Original", x=0.02,
                                      font=dict(size=12)))
    fig_orig.update_xaxes(visible=False)
    fig_orig.update_yaxes(visible=False)
    # Overlay each ROI shape + its tag on the Original so the user sees
    # what regions they've drawn against the raw signal.
    if image_rois:
        fig_orig.update_layout(
            shapes=[s["shape"] for s in image_rois],
            annotations=[
                dict(x=shape_anchor(s["shape"])[0],
                     y=shape_anchor(s["shape"])[1],
                     text=s["tag"],
                     showarrow=False,
                     font=dict(color="#58a6ff", size=11),
                     bgcolor="rgba(15,17,23,0.7)",
                     bordercolor="#58a6ff",
                     borderwidth=1,
                     xref="x", yref="y")
                for s in image_rois
            ],
        )

    fig_bin = px.imshow(binary.astype(np.uint8) * 255,
                        color_continuous_scale="gray",
                        binary_string=True, aspect="equal")
    fig_bin.update_layout(margin=dict(l=0, r=0, t=24, b=0),
                          coloraxis_showscale=False,
                          title=dict(text="Thresholded", x=0.02,
                                     font=dict(size=12)))
    fig_bin.update_xaxes(visible=False)
    fig_bin.update_yaxes(visible=False)

    # Stash the labeled mask + per-label area count so the click-on-cell
    # callback can look up connected components in O(1) without redoing the
    # threshold or labeling work.
    labels = measure.label(binary, connectivity=2)
    _LABELS_CACHE["labels"] = labels
    _LABELS_CACHE["areas"] = np.bincount(labels.ravel())

    # Histogram bins are computed in log10(area) space so bars render
    # uniformly across orders of magnitude. The x-axis is linear because
    # the values are already log-transformed; we hand-label the ticks
    # with the original-scale equivalents (10, 100, 1k, …) for readability.
    counts, log_edges = component_area_histogram(binary, n_bins=40, log=True)
    log_centers = 0.5 * (log_edges[:-1] + log_edges[1:])
    log_widths = np.diff(log_edges)
    hist = go.Figure(go.Bar(
        x=log_centers, y=counts, width=log_widths,
        marker_color="#0969da" if (theme or "light") == "light" else "#58a6ff",
        hovertemplate=("Area %{customdata[0]:,.0f}–%{customdata[1]:,.0f} px²"
                       "<br>Count %{y}<extra></extra>"),
        customdata=np.column_stack([10**log_edges[:-1], 10**log_edges[1:]]),
    ))
    tick_decades = list(range(int(np.floor(log_edges[0])),
                              int(np.ceil(log_edges[-1])) + 1))
    tick_labels = []
    for d in tick_decades:
        v = 10 ** d
        if v >= 1_000_000:
            tick_labels.append(f"{v/1e6:g}M")
        elif v >= 1_000:
            tick_labels.append(f"{v/1e3:g}k")
        else:
            tick_labels.append(f"{v:g}")
    hist.update_layout(
        margin=dict(l=40, r=20, t=24, b=40), height=180,
        xaxis_title="Component area (px²)",
        yaxis_title="Count",
        title=dict(text="Per-component size distribution",
                   x=0.02, font=dict(size=12)),
    )
    hist.update_xaxes(tickmode="array",
                      tickvals=tick_decades, ticktext=tick_labels)

    if area_min and area_min > 0:
        hist.add_vline(x=float(np.log10(area_min)),
                       line_color="#3fb950", line_dash="dash",
                       annotation_text="min", annotation_position="top")
    if area_max and area_max > 0:
        hist.add_vline(x=float(np.log10(area_max)),
                       line_color="#d29922", line_dash="dash",
                       annotation_text="max", annotation_position="top")

    # Apply dark/light Plotly theme manually since glia_dash.components helper
    # holds module-level state we can't be sure was set on this request path.
    bg, plot_bg, fg, grid = ("#ffffff", "#f6f8fa", "#1f2328", "#d0d7de") \
        if (theme or "light") == "light" \
        else ("#1c2128", "#0f1117", "#e6edf3", "#2d333b")
    for f in (fig_orig, fig_bin, hist):
        f.update_layout(paper_bgcolor=bg, plot_bgcolor=plot_bg,
                        font=dict(color=fg, family="IBM Plex Sans, sans-serif",
                                  size=11))
        f.update_xaxes(gridcolor=grid)
        f.update_yaxes(gridcolor=grid)

    if isinstance(info.get("threshold"), tuple):
        lo, hi = info["threshold"]
        thresh_str = f"band = [{lo:.0f}, {hi:.0f}]"
    elif "threshold" in info:
        thresh_str = f"threshold = {info['threshold']:.1f}"
    elif "block_size" in info:
        thresh_str = f"block = {info['block_size']} px"
    else:
        thresh_str = ""

    roi_str = (f" · masked by {info['roi_count']} ROI"
               f"{'s' if info['roi_count'] != 1 else ''}"
               if info.get("roi_count") else "")
    info_text = (
        f"Method: {info['requested']}"
        + (f" · fallback ({info['fallback']})" if info.get("fallback") else "")
        + (" · preprocessed" if info.get("preprocessed") else " · raw")
        + roi_str
        + f" · foreground {info['foreground_fraction']*100:.1f}%"
        + (f" · {thresh_str}" if thresh_str else "")
    )

    # Always attach a marker trace (possibly empty) at data[1] so the
    # lightweight Patch-based update_markers callback can update just its
    # x/y on click, without re-encoding the image trace at data[0].
    marker_trace = _markers_trace(selected or [])
    fig_orig.add_trace(marker_trace)
    fig_bin.add_trace(marker_trace)

    return (
        {"display": "none"},                                    # empty
        {"display": "block", "fontSize": "0.78rem",            # info
         "color": "var(--ned-text-muted)", "marginBottom": "8px"},
        {"display": "flex"},                                    # graphs row
        {"display": "block", "marginTop": "12px",              # histogram
         "height": "200px"},
        info_text,
        fig_orig, fig_bin, hist,
    )


# ── Click ~5 representative cells → suggest area bounds ─────────────


def _summary_text(selected: list[dict], note: str = "") -> str:
    if not selected:
        return ("Click ~5 representative cells in either preview to set "
                "area bounds.") + ((" " + note) if note else "")
    areas = [s["area"] for s in selected]
    lo, hi = min(areas), max(areas)
    base = (f"{len(selected)} cell{'s' if len(selected) != 1 else ''} "
            f"selected · area range {lo:,}–{hi:,} px² · click an already-"
            f"selected cell to remove it.")
    return base + ((" " + note) if note else "")


@callback(
    Output("setup-selected-cells", "data"),
    Output("setup-selection-text", "children"),
    Output("setup-apply-bounds", "disabled"),
    Output("setup-reset-bounds", "disabled"),
    Input("setup-preview-orig", "clickData"),
    Input("setup-preview-binary", "clickData"),
    Input("setup-reset-bounds", "n_clicks"),
    Input("setup-preview-path-store", "data"),
    State("setup-selected-cells", "data"),
    prevent_initial_call=True,
)
def on_selection_change(click_orig, click_bin, reset_n, path, selected):
    trigger = ctx.triggered_id
    selected = list(selected or [])

    # Reset on Reset button or when a new image is picked.
    if trigger in ("setup-reset-bounds", "setup-preview-path-store"):
        return [], _summary_text([]), True, True

    click = click_orig if trigger == "setup-preview-orig" else click_bin
    if not click or not click.get("points"):
        return no_update, no_update, no_update, no_update
    pt = click["points"][0]
    x = int(round(pt.get("x", 0)))
    y = int(round(pt.get("y", 0)))

    labels = _LABELS_CACHE.get("labels")
    areas = _LABELS_CACHE.get("areas")
    if labels is None or areas is None:
        return (no_update, "No preview yet — pick a test image first.",
                no_update, no_update)

    h, w = labels.shape
    if not (0 <= y < h and 0 <= x < w):
        return no_update, no_update, no_update, no_update
    lbl = int(labels[y, x])
    if lbl == 0:
        # Click landed on background — snap to the nearest cell within a
        # small radius. Display-pixel precision is limited so a click
        # 'just outside' a cell should still count as a hit on it.
        SNAP_RADIUS = 6
        y0 = max(0, y - SNAP_RADIUS); y1 = min(h, y + SNAP_RADIUS + 1)
        x0 = max(0, x - SNAP_RADIUS); x1 = min(w, x + SNAP_RADIUS + 1)
        sub = labels[y0:y1, x0:x1]
        if sub.any():
            ys_, xs_ = np.where(sub != 0)
            dists = (ys_ - (y - y0)) ** 2 + (xs_ - (x - x0)) ** 2
            i = int(np.argmin(dists))
            lbl = int(sub[ys_[i], xs_[i]])
        if lbl == 0:
            return (selected,
                    _summary_text(selected, "(clicked background)"),
                    len(selected) == 0, len(selected) == 0)

    # Toggle: remove if already selected, otherwise append. Store the
    # component's centroid (computed from the labeled mask) so the marker
    # overlay sits at the centre of the cell rather than wherever the user
    # happened to click.
    idx = next((i for i, s in enumerate(selected) if s["label"] == lbl), None)
    if idx is not None:
        selected.pop(idx)
    else:
        ys, xs = np.where(labels == lbl)
        cx = float(xs.mean()) if len(xs) else float(x)
        cy = float(ys.mean()) if len(ys) else float(y)
        selected.append({"label": lbl, "area": int(areas[lbl]),
                         "cx": cx, "cy": cy})

    has_selection = len(selected) > 0
    return (selected, _summary_text(selected),
            not has_selection, not has_selection)


@callback(
    Output("setup-area-min", "value", allow_duplicate=True),
    Output("setup-area-max", "value", allow_duplicate=True),
    Input("setup-apply-bounds", "n_clicks"),
    State("setup-selected-cells", "data"),
    prevent_initial_call=True,
)
def on_apply_bounds(n, selected):
    if not n or not selected:
        return no_update, no_update
    areas = [s["area"] for s in selected]
    return min(areas), max(areas)


# Note: marker overlay is included directly in render_preview's output
# (not via a separate Patch callback) to keep the update path single-source
# and avoid races with the initial empty go.Figure on first image pick.


@callback(
    Output("setup-state-sink", "data"),
    Input("setup-threshold-method", "value"),
    Input("setup-threshold-kind", "value"),
    Input("setup-threshold-slider", "value"),
    Input("setup-preprocess", "value"),
    Input("setup-area-min", "value"),
    Input("setup-area-max", "value"),
    Input("setup-local-radius", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def mirror_setup_to_state(method, kind, slider, preprocess,
                          area_min, area_max, local_radius, sid):
    """Push every Setup-tab control value into the SessionState so the
    Segment tab can build a SegmentParams without needing Setup to be
    mounted. Also snapshot the project settings so reopening the folder
    restores the chosen threshold + area bounds."""
    state = server_state.get_session(sid)
    if method:
        state.threshold_method = method
    if kind:
        state.threshold_kind = kind
    if slider and len(slider) == 2:
        state.threshold_lower = float(slider[0])
        state.threshold_upper = float(slider[1])
    state.preprocess = bool(preprocess)
    state.area_min = float(area_min or 0)
    state.area_max = float(area_max or 0)
    if local_radius:
        state.local_radius = int(local_radius)
    try:
        from glia.settings import save_project_settings
        save_project_settings(state.project_dir, state)
    except Exception:
        pass
    return no_update
