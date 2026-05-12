"""Export page — ColorByCluster CSV, features CSV, stats CSV, figures."""

from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

from glia_dash import server_state
from glia_dash.components import alert


def layout(sid: str | None) -> html.Div:
    state = server_state.get_session(sid)

    children = [
        html.H4("Export", style={"marginBottom": "8px"}),
        html.Div(
            "Round-trip artifacts back to FIJI and downstream tools: "
            "ColorByCluster.csv (ID, Cluster) for the FIJI macro, "
            "features.csv (full cell-level table with metadata), "
            "stats.csv (ANOVA + posthocs), and figures as PNG/SVG.",
            style={"fontSize": "0.85rem",
                   "color": "var(--ned-text-muted)",
                   "marginBottom": "16px"},
        ),
    ]

    if state.features_df is None:
        children.append(alert("Nothing to export yet.", "info"))
        return html.Div(children)

    children.append(dbc.Button("Download all", id="export-all",
                               className="btn-ned-primary", disabled=True))
    return html.Div(children)
