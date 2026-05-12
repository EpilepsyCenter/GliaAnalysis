"""Export page — write the cell table, cluster mapping, and stats to disk.

Two ways to export:

1. **Download buttons** stream a CSV to the browser via dcc.Download. Use
   these to grab a single file quickly.
2. **Write all to project** saves everything to
   ``<project>/_gliaanalysis/exports/`` so the artifacts live next to the
   project they came from.

ColorByCluster.csv is the FIJI round-trip artifact (columns ``ID,Cluster``
matching the single-cell TIFF filenames) so the Ciernia macro can recolour
the original images.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from dash import Input, Output, State, callback, dcc, html, no_update

from glia_dash import server_state
from glia_dash.components import alert, metric_card


_EXPORT_SUBDIR = Path("_gliaanalysis") / "exports"


# ── Builders for each artifact (pure pandas; no Dash deps) ───────────


def build_color_by_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """The FIJI round-trip CSV: ID + Cluster + (label if present)."""
    if df is None or "Cluster" not in df.columns:
        return pd.DataFrame(columns=["ID", "Cluster"])
    cols = ["ID", "Cluster"]
    if "morphology_label" in df.columns:
        cols.append("morphology_label")
    return df[cols].copy()


def build_features_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Full cell-level table with metadata + features + cluster + PCs."""
    if df is None:
        return pd.DataFrame()
    return df.copy()


def build_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per cluster: n, percent of total, morphology label."""
    if df is None or "Cluster" not in df.columns:
        return pd.DataFrame()
    counts = df["Cluster"].value_counts().sort_index()
    out = pd.DataFrame({
        "Cluster": counts.index,
        "n": counts.values,
        "percent": (counts / counts.sum() * 100).round(2).values,
    })
    if "morphology_label" in df.columns:
        labels = (df.groupby("Cluster")["morphology_label"]
                    .first().reindex(out["Cluster"]).values)
        out["morphology_label"] = labels
    return out


# ── Layout ───────────────────────────────────────────────────────────


def layout(sid: str | None) -> html.Div:
    state = server_state.get_session(sid)
    df = state.features_df

    if df is None or len(df) == 0:
        return html.Div([
            html.H4("Export", style={"marginBottom": "16px"}),
            alert("No features in memory — run the Features tab first.",
                  variant="warning"),
        ])

    project = state.project_dir
    out_dir = (Path(project) / _EXPORT_SUBDIR) if project else None

    n_cells = len(df)
    n_clusters = (int(df["Cluster"].nunique()) if "Cluster" in df.columns
                  else 0)
    n_with_label = (df["morphology_label"].notna().sum()
                    if "morphology_label" in df.columns else 0)

    return html.Div([
        html.H4("Export", style={"marginBottom": "8px"}),
        html.Div(
            "Download the cell table and cluster mapping, or write "
            "everything to a folder inside the project. The "
            "ColorByCluster.csv format is what the Ciernia FIJI macro "
            "expects to recolour the original images.",
            style={"fontSize": "0.85rem",
                   "color": "var(--ned-text-muted)",
                   "marginBottom": "16px"},
        ),

        html.Div([
            metric_card("Cells", f"{n_cells:,}", accent=True),
            metric_card("Clusters", str(n_clusters),
                        accent=(n_clusters > 0)),
            metric_card("Labelled cells", f"{int(n_with_label):,}",
                        accent=(n_with_label > 0)),
        ], style={"display": "grid",
                  "gridTemplateColumns": "repeat(3, 1fr)",
                  "gap": "12px", "marginBottom": "20px"}),

        html.Div("Downloads",
                 style={"fontSize": "0.72rem",
                        "color": "var(--ned-text-muted)",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.5px",
                        "marginBottom": "6px"}),
        html.Div([
            html.Button("ColorByCluster.csv",
                        id="export-dl-color",
                        className="btn-ned-primary",
                        disabled=(n_clusters == 0),
                        style={"marginRight": "8px"}),
            html.Button("features.csv",
                        id="export-dl-features",
                        className="btn-ned-secondary",
                        style={"marginRight": "8px"}),
            html.Button("cluster_summary.csv",
                        id="export-dl-summary",
                        className="btn-ned-secondary",
                        disabled=(n_clusters == 0)),
        ], style={"display": "flex", "marginBottom": "20px"}),

        html.Div("Write to project",
                 style={"fontSize": "0.72rem",
                        "color": "var(--ned-text-muted)",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.5px",
                        "marginBottom": "6px"}),
        html.Div([
            html.Button("Write all to disk",
                        id="export-write-all",
                        className="btn-ned-primary"),
            html.Span(f"→ {out_dir}" if out_dir else "(no project folder)",
                      style={"marginLeft": "12px",
                             "fontSize": "0.78rem",
                             "color": "var(--ned-text-muted)"}),
        ], style={"display": "flex",
                  "alignItems": "center",
                  "marginBottom": "12px"}),

        html.Div(id="export-status",
                 style={"marginTop": "8px",
                        "fontSize": "0.85rem",
                        "color": "var(--ned-text-muted)"}),

        # dcc.Download takes care of the actual browser download.
        dcc.Download(id="export-download"),
    ])


# ── Callbacks ────────────────────────────────────────────────────────


@callback(
    Output("export-download", "data"),
    Input("export-dl-color", "n_clicks"),
    Input("export-dl-features", "n_clicks"),
    Input("export-dl-summary", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_download(n_color, n_features, n_summary, sid):
    """Stream the chosen CSV to the browser via dcc.Download."""
    from dash import ctx
    trig = ctx.triggered_id
    if not trig:
        return no_update
    state = server_state.get_session(sid)
    df = state.features_df
    if df is None:
        return no_update

    if trig == "export-dl-color":
        out = build_color_by_cluster(df)
        return dcc.send_data_frame(out.to_csv, "ColorByCluster.csv",
                                   index=False)
    if trig == "export-dl-features":
        out = build_features_csv(df)
        return dcc.send_data_frame(out.to_csv, "features.csv",
                                   index=False)
    if trig == "export-dl-summary":
        out = build_cluster_summary(df)
        return dcc.send_data_frame(out.to_csv, "cluster_summary.csv",
                                   index=False)
    return no_update


@callback(
    Output("export-status", "children"),
    Input("export-write-all", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_write_all(n_clicks, sid):
    if not n_clicks:
        return no_update
    state = server_state.get_session(sid)
    df = state.features_df
    if df is None:
        return alert("No features in memory.", variant="warning")
    project = state.project_dir
    if not project or not Path(project).is_dir():
        return alert("No project folder.", variant="warning")
    out_dir = Path(project) / _EXPORT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    try:
        if "Cluster" in df.columns:
            build_color_by_cluster(df).to_csv(
                out_dir / "ColorByCluster.csv", index=False)
            written.append("ColorByCluster.csv")
            build_cluster_summary(df).to_csv(
                out_dir / "cluster_summary.csv", index=False)
            written.append("cluster_summary.csv")
        build_features_csv(df).to_csv(
            out_dir / "features.csv", index=False)
        written.append("features.csv")
    except Exception as e:
        return alert(f"Write failed: {e}", variant="danger")

    return alert(
        f"✓ Wrote {len(written)} file(s) to {out_dir}: "
        + ", ".join(written),
        variant="success",
    )
