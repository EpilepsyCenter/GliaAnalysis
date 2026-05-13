"""Setup page — composes ROI and Threshold subtabs.

The four subtab modules each register their own callbacks at import
time; this page only orchestrates their layouts inside a dbc.Tabs
container. The Soma subtab is microglia-only (its radial-scan output
isn't biologically meaningful on the GFAP syncytium), so we hide it
when the sidebar mode is "astrocyte".
"""

from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

# Import the subtab modules so their @callback decorators register at
# app start. The orchestrator only calls their layout functions.
from glia_dash import server_state
from glia_dash.pages.setup_prepare import prepare_layout
from glia_dash.pages.setup_roi import roi_layout
from glia_dash.pages.setup_soma import soma_layout
from glia_dash.pages.setup_threshold import threshold_layout


def layout(sid: str | None) -> html.Div:
    mode = (getattr(server_state.get_session(sid), "mode", "microglia")
            or "microglia")
    children = [
        dbc.Tab(prepare_layout(sid),
                label="1. Prepare",
                tab_id="setup-tab-prepare"),
        dbc.Tab(roi_layout(sid),
                label="2. ROIs",
                tab_id="setup-tab-roi"),
        dbc.Tab(threshold_layout(sid),
                label="3. Threshold",
                tab_id="setup-tab-threshold"),
    ]
    if mode != "astrocyte":
        children.append(
            dbc.Tab(soma_layout(sid),
                    label="4. Soma",
                    tab_id="setup-tab-soma")
        )
    return html.Div([
        html.H4("Setup", style={"marginBottom": "16px"}),
        dbc.Tabs(
            id="setup-subtabs",
            active_tab="setup-tab-prepare",
            class_name="dbc-page-tabs",
            children=children,
        ),
    ])
