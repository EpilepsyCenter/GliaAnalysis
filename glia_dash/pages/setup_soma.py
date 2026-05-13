"""Setup → Soma subtab — radial soma + Sholl preview.

Picks one cell from the project's SingleCells folder, runs
``glia.radial.analyze_radial`` with the current ``gap_tol_deg``, and
shows the mask + soma polygon overlay + Sholl profile + radial scan.
The slider is the one tunable; its value persists into project
settings via the existing setup-state-sink callback in
setup_threshold.py — we mirror our own slot here for robustness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import tifffile
from dash import ALL, Input, Output, State, callback, ctx, dcc, html, no_update
import dash_bootstrap_components as dbc

from glia.radial import analyze_radial, dapi_centroid
from glia_dash import server_state
from glia_dash.components import empty_state


# Tiny mtime-aware mask cache so re-rendering the same cell with a new
# slider value doesn't re-read disk.
_MASK_CACHE: dict[str, tuple[float, np.ndarray]] = {}


def _singlecells_dir(project_dir: str) -> Path | None:
    if not project_dir:
        return None
    p = Path(project_dir) / "_gliaanalysis" / "SingleCells"
    return p if p.is_dir() else None


def _list_cells(project_dir: str) -> list[str]:
    folder = _singlecells_dir(project_dir)
    if folder is None:
        return []
    # Skip DAPI siblings — they're auxiliary crops, not cells.
    return sorted(str(p) for p in folder.glob("*.tif")
                  if not p.stem.endswith("__dapi"))


def _dapi_sibling(cell_path: str) -> Path | None:
    """Locate ``<cell_id>__dapi.tif`` next to the given cell mask."""
    p = Path(cell_path)
    sibling = p.with_name(p.stem + "__dapi.tif")
    return sibling if sibling.is_file() else None


def _load_mask(path: str) -> np.ndarray:
    try:
        mtime = Path(path).stat().st_mtime
    except OSError:
        return tifffile.imread(path) > 0
    cached = _MASK_CACHE.get(path)
    if cached is None or cached[0] != mtime:
        _MASK_CACHE[path] = (mtime, tifffile.imread(path) > 0)
    return _MASK_CACHE[path][1]


def _short_cell_label(cell_path: str) -> str:
    """One-line display name: <image_stem>  · #<label>.

    The list container is wide enough to fit typical filenames; very
    long stems are truncated with an ellipsis at render time via CSS
    (whiteSpace: nowrap + textOverflow: ellipsis). The full filename is
    on the button's title attribute for hover tooltips.
    """
    parts = Path(cell_path).stem.split("__")
    if len(parts) >= 3:
        return f"{parts[0]}  · #{parts[-1]}"
    return Path(cell_path).stem


def _render_cell_list(project_dir: str, current_path: str) -> list:
    """Vertical, scrollable list of clickable cells (one per line)."""
    cells = _list_cells(project_dir)
    if not cells:
        return [html.Div(
            "Run Segment to extract single cells before previewing "
            "soma detection.",
            style={"fontSize": "0.82rem",
                   "color": "var(--ned-text-muted)",
                   "padding": "8px"})]

    # No cap — scrollable container handles thousands of rows fine, and
    # users may want to find specific cells by stem.
    return [
        html.Div(
            f"{len(cells)} cell{'s' if len(cells) != 1 else ''}",
            style={"fontSize": "0.72rem",
                   "color": "var(--ned-text-muted)",
                   "textTransform": "uppercase",
                   "letterSpacing": "0.5px",
                   "padding": "4px 8px",
                   "borderBottom": "1px solid var(--ned-border)"},
        ),
        *[
            html.Button(
                _short_cell_label(p),
                id={"type": "soma-cell-chip", "path": p},
                n_clicks=0,
                className=("soma-cell-row selected"
                           if p == current_path else "soma-cell-row"),
                style={"display": "block",
                       "width": "100%",
                       "padding": "6px 10px",
                       "border": "none",
                       "borderBottom":
                           "1px solid rgba(125, 133, 144, 0.12)",
                       "textAlign": "left",
                       "background": ("rgba(56, 139, 253, 0.18)"
                                      if p == current_path
                                      else "transparent"),
                       "color": "var(--ned-text)",
                       "fontSize": "0.78rem",
                       "fontFamily":
                           "IBM Plex Mono, ui-monospace, monospace",
                       "cursor": "pointer",
                       "whiteSpace": "nowrap",
                       "overflow": "hidden",
                       "textOverflow": "ellipsis"},
                title=Path(p).name,
            )
            for p in cells
        ],
    ]


def soma_layout(sid: str | None) -> html.Div:
    state = server_state.get_session(sid)
    cells = _list_cells(state.project_dir)
    current = state.extra.get("soma_preview_cell", "")
    if not current and cells:
        current = cells[0]
        state.extra["soma_preview_cell"] = current

    gap_tol = float(getattr(state, "soma_gap_tol_deg", 20.0))
    use_dapi = bool(getattr(state, "use_dapi", False))

    return html.Div([
        html.Div(
            "Soma detection runs a single radial scan from each cell's "
            "deepest inscribed point (or its DAPI nucleus centroid, "
            "when enabled in Setup → Prepare). Rays that exit between "
            "processes trace the soma rim directly; rays that run down "
            "a process are interpolated from neighbours. The same scan "
            "yields the Sholl profile. Tune the gap tolerance below if "
            "the soma boundary creeps into thin processes or stops "
            "short.",
            style={"fontSize": "0.85rem",
                   "color": "var(--ned-text-muted)",
                   "marginBottom": "12px"},
        ),

        # DAPI colocalization banner — visible whenever DAPI is enabled.
        # The overlay only renders for cells that have a per-cell
        # ``__dapi.tif`` sibling, which is produced by the Segment
        # phase. After flipping DAPI on in Setup → Prepare and
        # re-running Prepare, the user must re-run Segment for the
        # overlay to appear here.
        html.Div(
            "DAPI colocalization (amber overlay inside the cell mask) "
            "will appear on each cell once Segment has been run with "
            "DAPI enabled. If you've toggled DAPI on after segmenting, "
            "re-run Setup → Prepare and then Segment to regenerate "
            "per-cell DAPI siblings.",
            style={"display": "block" if use_dapi else "none",
                   "fontSize": "0.78rem",
                   "color": "var(--ned-text-muted)",
                   "background": "rgba(251, 191, 36, 0.10)",
                   "border": "1px solid rgba(251, 191, 36, 0.35)",
                   "borderRadius": "6px",
                   "padding": "8px 12px",
                   "marginBottom": "16px"},
        ),

        # Gap-tolerance slider — drag-mode so the soma updates live.
        html.Div([
            html.Label("Gap tolerance (degrees)",
                       style={"fontSize": "0.72rem",
                              "color": "var(--ned-text-muted)",
                              "textTransform": "uppercase",
                              "letterSpacing": "0.5px"}),
            dcc.Slider(
                id="soma-gap-tol",
                min=5, max=45, step=1,
                value=gap_tol,
                marks={5: "5°", 15: "15°", 20: "20°",
                       30: "30°", 45: "45°"},
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode="drag",
            ),
            html.Div(
                "Smaller values commit to the soma boundary at the "
                "first sign of process notching; larger values let the "
                "soma absorb wider notches. 15–25° works for most "
                "microglia. Drag for live preview.",
                style={"fontSize": "0.75rem",
                       "color": "var(--ned-text-muted)",
                       "marginTop": "4px"},
            ),
        ], style={"marginBottom": "16px", "maxWidth": "640px"}),

        html.Div(id="soma-preview-info",
                 style={"display": "none", "fontSize": "0.82rem",
                        "color": "var(--ned-text-muted)",
                        "marginBottom": "8px"}),

        # Two-column body: cell list left, mask + Sholl stacked right.
        html.Div([
            # Left: scrollable cell list. Wide enough for the full
            # single-cell filenames (image stem + label index) without
            # truncation in the common case.
            html.Div(
                id="soma-cell-chip-row",
                children=_render_cell_list(state.project_dir, current),
                style={"flex": "0 0 440px",
                       "height": "820px",
                       "overflowY": "auto",
                       "border": "1px solid var(--ned-border)",
                       "borderRadius": "6px",
                       "marginRight": "12px",
                       "background": "var(--ned-panel)"},
            ),

            # Right: mask above, Sholl below.
            html.Div([
                html.Div(id="soma-preview-empty", children=[
                    empty_state(
                        icon="🔬",
                        title="No cell selected",
                        text=("Pick a cell on the left to preview "
                              "soma detection. If the list is empty, "
                              "run the Segment step first."),
                    ),
                ]),
                html.Div(id="soma-preview-graphs-col",
                         style={"display": "none"},
                         children=[
                             dcc.Graph(
                                 id="soma-preview-mask",
                                 figure=go.Figure(),
                                 config={
                                     "scrollZoom": True,
                                     "displayModeBar": True,
                                     "displaylogo": False,
                                     "modeBarButtonsToRemove":
                                         ["lasso2d", "select2d",
                                          "autoScale2d"]},
                                 style={"height": "520px",
                                        "marginBottom": "12px"}),
                             dcc.Graph(
                                 id="soma-preview-sholl",
                                 figure=go.Figure(),
                                 config={"displayModeBar": False},
                                 style={"height": "280px"}),
                         ]),
            ], style={"flex": "1", "minWidth": "0"}),
        ], style={"display": "flex", "alignItems": "flex-start"}),

        dcc.Store(id="soma-preview-cell-store", data=current),
        dcc.Store(id="soma-state-sink"),
    ])


# ── Callbacks ───────────────────────────────────────────────────────


@callback(
    Output("soma-preview-cell-store", "data"),
    Input({"type": "soma-cell-chip", "path": ALL}, "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_cell_chip_click(all_clicks, sid):
    if not any(n for n in (all_clicks or [])):
        return no_update
    trig = ctx.triggered_id
    if not isinstance(trig, dict) or "path" not in trig:
        return no_update
    server_state.get_session(sid).extra["soma_preview_cell"] = trig["path"]
    return trig["path"]


@callback(
    Output("soma-cell-chip-row", "children"),
    Input("soma-preview-cell-store", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def refresh_cell_chips(path, sid):
    folder = server_state.get_session(sid).project_dir
    return _render_cell_list(folder, path or "")


_PREVIEW_OUTPUTS = (
    Output("soma-preview-empty", "style"),
    Output("soma-preview-info", "style"),
    Output("soma-preview-graphs-col", "style"),
    Output("soma-preview-info", "children"),
    Output("soma-preview-mask", "figure"),
    Output("soma-preview-sholl", "figure"),
)


def _hide_preview(message: str | None = None):
    return (
        {"display": "block"},
        {"display": "none"},
        {"display": "none"},
        message or "",
        go.Figure(), go.Figure(),
    )


@callback(
    *_PREVIEW_OUTPUTS,
    Input("soma-preview-cell-store", "data"),
    Input("soma-gap-tol", "value"),
    Input("theme-store", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def render_preview(path, gap_tol, theme, sid):
    use_dapi = bool(getattr(server_state.get_session(sid),
                            "use_dapi", False))
    if not path or not Path(path).is_file():
        return _hide_preview()
    try:
        mask = _load_mask(path)
    except Exception as e:
        return _hide_preview(f"Failed to read mask: {e}")

    center_yx = None
    center_source = "EDT peak"
    dapi_img = None
    if use_dapi:
        dapi_path = _dapi_sibling(path)
        if dapi_path is not None:
            try:
                dapi_img = tifffile.imread(dapi_path)
                center_yx = dapi_centroid(dapi_img, cell_mask=mask)
                if center_yx is not None:
                    center_source = "DAPI nucleus"
            except Exception:
                dapi_img = None
                center_yx = None

    try:
        res = analyze_radial(mask, gap_tol_deg=float(gap_tol or 20.0),
                             center_yx=center_yx)
    except Exception as e:
        return _hide_preview(f"Radial analysis failed: {e}")

    bg, plot_bg, fg, grid = (
        ("#ffffff", "#f6f8fa", "#1f2328", "#d0d7de")
        if (theme or "light") == "light"
        else ("#1c2128", "#0f1117", "#e6edf3", "#2d333b")
    )

    # ── Mask + soma overlay figure ───────────────────────────────────
    # Stacking order (bottom → top):
    #   1. cell mask (grey)
    #   2. soma overlay (cyan fill)
    #   3. DAPI nucleus (amber fill + crisp outline) — drawn on top so
    #      the nucleus is always visible inside the soma.
    # We use distinctly different *hues* for soma and DAPI (cyan vs
    # amber) rather than two shades of blue, so the two regions are
    # immediately readable even at small sizes.
    h, w = mask.shape
    fig_mask = go.Figure()
    fig_mask.add_trace(go.Heatmap(
        z=mask.astype(np.uint8),
        colorscale=[[0, "#0d1117"], [1, "#9da7b3"]],
        showscale=False,
        hoverinfo="skip",
    ))
    # Soma overlay — cyan, semi-transparent.
    soma = res.soma_mask
    soma_z = np.where(soma, 1.0, np.nan)
    fig_mask.add_trace(go.Heatmap(
        z=soma_z,
        colorscale=[[0, "rgba(56,189,248,0.45)"],
                    [1, "rgba(56,189,248,0.45)"]],
        showscale=False,
        hoverinfo="skip",
    ))
    # DAPI nucleus overlay (top of stack).
    if dapi_img is not None:
        dapi_bin = (np.asarray(dapi_img) > 0) & mask
        if dapi_bin.any():
            dapi_z = np.where(dapi_bin, 1.0, np.nan)
            fig_mask.add_trace(go.Heatmap(
                z=dapi_z,
                colorscale=[[0, "rgba(251, 191, 36, 0.65)"],
                            [1, "rgba(251, 191, 36, 0.65)"]],
                showscale=False,
                hoverinfo="skip",
                name="DAPI nucleus",
            ))
            # Crisp amber outline of every DAPI component so the
            # nucleus shape reads clearly even where it overlaps the
            # soma fill.
            from skimage.measure import find_contours
            for ct in find_contours(dapi_bin.astype(float), 0.5):
                fig_mask.add_trace(go.Scatter(
                    x=ct[:, 1], y=ct[:, 0],
                    mode="lines",
                    line=dict(color="#fbbf24", width=1.8),
                    hoverinfo="skip",
                    showlegend=False,
                    name="DAPI outline",
                ))
    # Soma polygon outline.
    poly = res.soma_polygon
    poly_closed = np.vstack([poly, poly[:1]])
    fig_mask.add_trace(go.Scatter(
        x=poly_closed[:, 1], y=poly_closed[:, 0],
        mode="lines",
        line=dict(color="#ff7eb6", width=2.5),
        hoverinfo="skip",
        showlegend=False,
        name="soma boundary",
    ))
    # Critical radius ring (yellow dashed).
    cy, cx = res.center_yx
    theta_ring = np.linspace(0, 2 * np.pi, 200)
    fig_mask.add_trace(go.Scatter(
        x=cx + res.critical_radius * np.cos(theta_ring),
        y=cy + res.critical_radius * np.sin(theta_ring),
        mode="lines",
        line=dict(color="#f0e442", width=1.5, dash="dash"),
        hoverinfo="skip",
        showlegend=False,
        name="r₀",
    ))
    # r_out fan, color-coded by process membership.
    ang_rad = np.radians(res.angles_deg)
    end_y = cy + res.r_out * np.sin(ang_rad)
    end_x = cx + res.r_out * np.cos(ang_rad)
    step = max(1, len(ang_rad) // 90)
    rim_x, rim_y, proc_x, proc_y = [], [], [], []
    for i in range(0, len(ang_rad), step):
        if res.process_angle_mask[i]:
            proc_x.extend([cx, end_x[i], None])
            proc_y.extend([cy, end_y[i], None])
        else:
            rim_x.extend([cx, end_x[i], None])
            rim_y.extend([cy, end_y[i], None])
    if rim_x:
        fig_mask.add_trace(go.Scatter(
            x=rim_x, y=rim_y, mode="lines",
            line=dict(color="rgba(63,185,80,0.35)", width=0.8),
            hoverinfo="skip", showlegend=False, name="rim rays",
        ))
    if proc_x:
        fig_mask.add_trace(go.Scatter(
            x=proc_x, y=proc_y, mode="lines",
            line=dict(color="rgba(210,153,34,0.35)", width=0.8),
            hoverinfo="skip", showlegend=False, name="process rays",
        ))
    # Center marker.
    fig_mask.add_trace(go.Scatter(
        x=[cx], y=[cy], mode="markers",
        marker=dict(color="#f85149", size=10,
                    line=dict(color="white", width=1.5)),
        hoverinfo="skip", showlegend=False, name="center",
    ))

    fig_mask.update_layout(
        margin=dict(l=0, r=0, t=24, b=0),
        title=dict(text=f"{Path(path).stem}  ({w}×{h} px)",
                   x=0.02, font=dict(size=11)),
        paper_bgcolor=bg, plot_bgcolor=plot_bg,
        font=dict(color=fg, family="IBM Plex Sans, sans-serif", size=11),
    )
    fig_mask.update_xaxes(visible=False, range=[0, w])
    fig_mask.update_yaxes(visible=False, range=[h, 0], scaleanchor="x",
                          scaleratio=1)

    # ── Sholl + r_out(θ) figure ─────────────────────────────────────
    fig_sholl = go.Figure()
    fig_sholl.add_trace(go.Scatter(
        x=res.sholl_radii, y=res.sholl_intersections,
        mode="lines+markers",
        line=dict(color="#58a6ff", width=2),
        marker=dict(size=4),
        name="Sholl intersections",
        hovertemplate="r=%{x:.0f}px<br>#=%{y}<extra></extra>",
    ))
    fig_sholl.add_vline(x=res.critical_radius,
                        line_color="#f0e442", line_dash="dash",
                        annotation_text="r₀",
                        annotation_position="top")
    fig_sholl.update_layout(
        margin=dict(l=50, r=20, t=40, b=40),
        title=dict(text="Sholl profile — intersections vs r",
                   x=0.02, font=dict(size=11)),
        xaxis_title="r (px from soma center)",
        yaxis_title="# intersections",
        paper_bgcolor=bg, plot_bgcolor=plot_bg,
        font=dict(color=fg, family="IBM Plex Sans, sans-serif", size=11),
    )
    fig_sholl.update_xaxes(gridcolor=grid)
    fig_sholl.update_yaxes(gridcolor=grid, rangemode="tozero")

    # Surface the most common DAPI-misconfiguration: switch is on, but
    # no per-cell DAPI sibling has been produced yet. That happens when
    # the user enabled DAPI AFTER segmenting — the SingleCells/ folder
    # carries no ``<cell_id>__dapi.tif`` until Segment is re-run with
    # ``state.use_dapi`` on.
    dapi_hint = ""
    if use_dapi and _dapi_sibling(path) is None:
        dapi_hint = (" · ⚠ DAPI enabled but no sibling for this cell — "
                     "re-run Segment to regenerate single-cell crops "
                     "with DAPI nuclei")

    info_text = (
        f"centered on {center_source}"
        f" · r₀ = {res.critical_radius:.1f} px"
        f" · primary processes = {res.primary_process_count}"
        f" · max Sholl intersections = {res.max_intersections}"
        f" · ramification index = {res.ramification_index:.2f}"
        f" · soma area = {res.soma_area:.0f} px²"
        f" · soma:cell ratio = {res.soma_to_cell_area_ratio:.2f}"
        f" · soma circularity = {res.soma_circularity:.2f}"
        + dapi_hint
    )

    return (
        {"display": "none"},
        {"display": "block", "fontSize": "0.82rem",
         "color": "var(--ned-text-muted)", "marginBottom": "8px"},
        {"display": "block"},  # mask above + Sholl below (vertical)
        info_text,
        fig_mask, fig_sholl,
    )


@callback(
    Output("soma-state-sink", "data"),
    Input("soma-gap-tol", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def mirror_to_state(gap_tol, sid):
    """Persist the gap tolerance into SessionState + project settings.
    The DAPI toggle lives on the Prepare tab now."""
    state = server_state.get_session(sid)
    state.soma_gap_tol_deg = float(gap_tol or 20.0)
    try:
        from glia.settings import save_project_settings
        save_project_settings(state.project_dir, state)
    except Exception:
        pass
    return no_update
