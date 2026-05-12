"""Setup page — composes ROI and Threshold subtabs.

The two subtab modules each register their own callbacks at import time;
this page only orchestrates their layouts inside a dbc.Tabs container.
"""

from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

# Import the subtab modules so their @callback decorators register at
# app start. The orchestrator only calls their layout functions.
from glia_dash.pages.setup_roi import roi_layout
from glia_dash.pages.setup_threshold import threshold_layout


def layout(sid: str | None) -> html.Div:
    return html.Div([
        html.H4("Setup", style={"marginBottom": "16px"}),
        dbc.Tabs(
            id="setup-subtabs",
            active_tab="setup-tab-roi",
            class_name="dbc-page-tabs",
            children=[
                dbc.Tab(roi_layout(sid),
                        label="1. ROIs",
                        tab_id="setup-tab-roi"),
                dbc.Tab(threshold_layout(sid),
                        label="2. Threshold",
                        tab_id="setup-tab-threshold"),
            ],
        ),
    ])
