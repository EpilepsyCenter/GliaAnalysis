"""Features page — compute the 27-feature per-cell table from Segment output."""

from __future__ import annotations

import time
from pathlib import Path

from dash import Input, Output, State, callback, dash_table, dcc, html, no_update
import dash_bootstrap_components as dbc

from glia.features import (
    extract_features_from_dir,
    load_skeleton_results,
    merge_geometric_and_skeleton,
)
from glia_dash import server_state
from glia_dash.components import alert, metric_card


_OUTPUT_SUBDIR = "_gliaanalysis"


def _output_paths(project_dir: str) -> tuple[Path, Path]:
    """Return (single_cells_dir, skeleton_dir) under the project's output."""
    out = Path(project_dir) / _OUTPUT_SUBDIR
    return out / "SingleCells", out / "SkeletonResults"


_DEFAULT_VISIBLE = [
    "ID", "roi_tag", "Area", "Perimeter", "Circularity",
    "# of branches", "# of junctions", "Maximum branch length",
]


# Themed cell / header styles for dash_table.DataTable so it doesn't fall
# back to Bootstrap-DARKLY's default dark-on-dark.
_TABLE_STYLE_CELL = {
    "fontFamily": "IBM Plex Sans, sans-serif",
    "fontSize": "12px",
    "padding": "6px 10px",
    "backgroundColor": "var(--ned-surface)",
    "color": "var(--ned-text)",
    "borderBottom": "1px solid var(--ned-border)",
    "borderLeft": "none",
    "borderRight": "none",
    "borderTop": "none",
    "textAlign": "left",
    "maxWidth": "320px",
    "overflow": "hidden",
    "textOverflow": "ellipsis",
}
_TABLE_STYLE_HEADER = {
    "backgroundColor": "var(--ned-bg)",
    "color": "var(--ned-text-muted)",
    "fontWeight": "600",
    "fontSize": "10.5px",
    "textTransform": "uppercase",
    "letterSpacing": "0.5px",
    "borderBottom": "2px solid var(--ned-border)",
    "padding": "8px 10px",
}
_TABLE_STYLE_TABLE = {
    "overflowX": "auto",
    "border": "1px solid var(--ned-border)",
    "borderRadius": "6px",
}
_TABLE_STYLE_DATA_CONDITIONAL = [
    {"if": {"row_index": "odd"},
     "backgroundColor": "var(--ned-surface-hover)"},
]


def layout(sid: str | None) -> html.Div:
    state = server_state.get_session(sid)
    project = state.project_dir

    if not project or not Path(project).is_dir():
        return html.Div([
            html.H4("Features", style={"marginBottom": "16px"}),
            alert("Pick a project folder from the sidebar first.",
                  variant="warning"),
        ])

    cells_dir, skel_dir = _output_paths(project)
    n_cells_on_disk = len(list(cells_dir.glob("*.tif"))) if cells_dir.exists() else 0
    n_csvs_on_disk = len(list(skel_dir.glob("*_results.csv"))) if skel_dir.exists() else 0
    df = state.features_df

    return html.Div([
        html.H4("Features", style={"marginBottom": "8px"}),
        html.Div(
            "Computes 18 FracLac-equivalent geometric features per cell and "
            "joins them with the 9 skeleton features → one row per cell, "
            "27 numeric columns + the cell ID.",
            style={"fontSize": "0.85rem",
                   "color": "var(--ned-text-muted)",
                   "marginBottom": "16px"},
        ),

        html.Div([
            metric_card("SingleCells on disk", str(n_cells_on_disk),
                        accent=(n_cells_on_disk > 0)),
            metric_card("Skeleton CSVs on disk", str(n_csvs_on_disk),
                        accent=(n_csvs_on_disk > 0)),
            metric_card("Features in memory",
                        str(len(df)) if df is not None else "—",
                        accent=(df is not None)),
        ], style={"display": "grid",
                  "gridTemplateColumns": "repeat(3, 1fr)",
                  "gap": "12px", "marginBottom": "16px"}),

        html.Div([
            dbc.Button("Extract features",
                       id="features-extract",
                       className="btn-ned-primary",
                       disabled=(n_cells_on_disk == 0)),
            html.Span(("Run the Segment tab first."
                       if n_cells_on_disk == 0
                       else f"Will process {n_cells_on_disk} cells."),
                      style={"marginLeft": "12px",
                             "fontSize": "0.78rem",
                             "color": "var(--ned-text-muted)"}),
        ], style={"display": "flex", "alignItems": "center"}),

        dcc.Loading(
            id="features-loading",
            type="default",
            children=html.Div(id="features-output",
                              style={"marginTop": "16px",
                                     "minHeight": "40px"},
                              children=(
                                  _render_preview(df)
                                  if df is not None else None
                              )),
        ),
    ])


def _render_preview(df):
    if df is None or len(df) == 0:
        return None

    # Add a derived `roi_tag` column for visual sanity-check; the Metadata
    # tab will do the canonical parse later.
    try:
        df = df.copy()
        df["roi_tag"] = df["ID"].str.split("__", n=2, expand=True)[1]
    except Exception:
        pass

    all_cols = list(df.columns)
    visible_default = [c for c in _DEFAULT_VISIBLE if c in all_cols]
    hidden_default = [c for c in all_cols if c not in visible_default]

    # Per-column number formatting: numeric columns get 3 sig figs; ID and
    # roi_tag stay as text.
    columns = []
    for c in all_cols:
        col = {"name": c, "id": c, "hideable": True}
        if df[c].dtype.kind in "fi":
            col["type"] = "numeric"
            col["format"] = {"specifier": ".3g"}
        columns.append(col)

    return html.Div([
        html.Div([
            html.Div([
                html.Label("Rows per page",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dcc.Dropdown(
                    id="features-page-size",
                    options=[{"label": str(n), "value": n}
                             for n in (10, 25, 50, 100, 250)],
                    value=25, clearable=False,
                    style={"width": "120px"},
                ),
            ], style={"marginRight": "24px"}),
            html.Div([
                html.Label("Visible columns",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dcc.Dropdown(
                    id="features-visible-cols",
                    options=[{"label": c, "value": c} for c in all_cols],
                    value=visible_default,
                    multi=True,
                    style={"minWidth": "420px"},
                ),
            ], style={"flex": "1"}),
        ], style={"display": "flex",
                  "alignItems": "flex-end",
                  "marginBottom": "12px"}),

        html.Div(
            f"{len(df):,} cells × {df.shape[1]} columns · "
            "click a column header to sort; rows are filterable.",
            style={"fontSize": "0.82rem",
                   "color": "var(--ned-text-muted)",
                   "marginBottom": "8px"},
        ),

        dash_table.DataTable(
            id="features-table",
            data=df.to_dict("records"),
            columns=columns,
            hidden_columns=hidden_default,
            page_size=25,
            page_action="native",
            sort_action="native",
            filter_action="native",
            style_cell=_TABLE_STYLE_CELL,
            style_header=_TABLE_STYLE_HEADER,
            style_table=_TABLE_STYLE_TABLE,
            style_data_conditional=_TABLE_STYLE_DATA_CONDITIONAL,
        ),
    ])


# ── Callbacks ────────────────────────────────────────────────────────


@callback(
    Output("features-output", "children"),
    Input("features-extract", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_extract(n_clicks, sid):
    if not n_clicks:
        return no_update
    state = server_state.get_session(sid)
    project = state.project_dir
    if not project or not Path(project).is_dir():
        return alert("No project folder loaded.", variant="danger")

    cells_dir, skel_dir = _output_paths(project)
    if not cells_dir.exists():
        return alert(f"No SingleCells folder at {cells_dir}. Run Segment "
                     "first.", variant="warning")

    t0 = time.time()
    try:
        geom = extract_features_from_dir(cells_dir)
    except Exception as e:
        return alert(f"Geometric feature extraction failed: {e}",
                     variant="danger")

    if skel_dir.exists():
        try:
            skel = load_skeleton_results(skel_dir)
            df = merge_geometric_and_skeleton(geom, skel)
        except Exception as e:
            return alert(f"Skeleton join failed: {e}", variant="danger")
    else:
        df = geom

    dt = time.time() - t0
    state.features_df = df
    try:
        from glia.features import save_features_df
        save_features_df(state.project_dir, df)
    except Exception:
        pass

    summary = alert(
        f"✓ Extracted features for {len(df)} cells "
        f"({df.shape[1]} columns) in {dt:.1f} s.",
        variant="success",
    )
    return html.Div([summary, _render_preview(df)])


# ── Page size + column-visibility controls wired to the DataTable ────


@callback(
    Output("features-table", "page_size"),
    Input("features-page-size", "value"),
    prevent_initial_call=True,
)
def update_page_size(size):
    return int(size or 25)


@callback(
    Output("features-table", "hidden_columns"),
    Input("features-visible-cols", "value"),
    State("features-table", "columns"),
    prevent_initial_call=True,
)
def update_visible_cols(visible, all_cols):
    if not all_cols:
        return no_update
    visible_set = set(visible or [])
    return [c["id"] for c in all_cols if c["id"] not in visible_set]
