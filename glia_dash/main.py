"""GliaAnalysis Dash application entry point.

Run with:  python -m glia_dash.main   (or `python app.py` from the project root)

Layout mirrors NED-Net: fixed 280px sidebar with brand + project info +
status, top tab bar across the main content area, and a single
``render_tab`` callback that hands off to per-page ``layout(sid)``
functions. Theme toggle (light/dark) is clientside via a data-theme
attribute on <html>.
"""

from __future__ import annotations

import os

from dash import (
    Dash, Input, Output, State, callback, ctx, dcc, html, no_update,
)
import dash_bootstrap_components as dbc

from glia_dash import server_state
from glia_dash.components import (
    alert,
    browse_folder,
    section_header,
    set_plotly_theme,
    sidebar_divider,
)
from glia_dash.pages import (
    cluster as cluster_page,
    explore as explore_page,
    export as export_page,
    features as features_page,
    metadata as metadata_page,
    segment as segment_page,
    setup as setup_page,
    stats as stats_page,
)


# ── App setup ─────────────────────────────────────────────────────────

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    assets_folder="assets",
    title="GliaAnalysis",
    update_title="GliaAnalysis | Loading…",
)
server = app.server


# ── Tab definitions ───────────────────────────────────────────────────

TAB_DEFS = [
    ("setup",    "Setup"),
    ("segment",  "Segment"),
    ("features", "Features"),
    ("metadata", "Metadata"),
    ("explore",  "Explore"),
    ("cluster",  "Cluster"),
    ("stats",    "Stats"),
    ("export",   "Export"),
]


# ── Sidebar ───────────────────────────────────────────────────────────


def _sidebar() -> html.Div:
    return html.Div(
        id="sidebar",
        children=[
            html.Div(
                id="sidebar-brand",
                children=[
                    html.H4("GliaAnalysis"),
                    html.Div("Microglia Morphology Pipeline",
                             className="subtitle"),
                ],
            ),
            html.Div(
                id="sidebar-content",
                children=[
                    section_header("MODE"),
                    dbc.RadioItems(
                        id="sidebar-mode",
                        options=[
                            {"label": "Microglia", "value": "microglia"},
                            {"label": "Astrocyte", "value": "astrocyte"},
                        ],
                        value="microglia",
                        inline=False,
                        style={"fontSize": "0.82rem"},
                    ),

                    sidebar_divider(),

                    section_header("PROJECT"),
                    html.Div(id="sidebar-project-info", children=[
                        html.Div("No project loaded", className="file-info",
                                 style={"opacity": "0.5"}),
                    ]),
                    dbc.Button("Browse folder…", id="sidebar-browse-project",
                               className="btn-ned-secondary",
                               size="sm",
                               style={"marginTop": "8px", "width": "100%"}),

                    sidebar_divider(),

                    section_header("FIJI"),
                    html.Div(id="sidebar-fiji-info", children=[
                        html.Div("Not configured", className="file-info",
                                 style={"opacity": "0.5"}),
                    ]),

                    sidebar_divider(),

                    section_header("STATUS"),
                    html.Div(id="sidebar-status", children=[
                        html.Div("⚪ No features yet",
                                 style={"fontSize": "0.82rem",
                                        "color": "var(--ned-text-muted)",
                                        "opacity": "0.6"}),
                    ]),
                ],
            ),
            html.Div(
                id="sidebar-footer",
                children=[
                    html.Div(
                        className="theme-toggle-container",
                        children=[
                            dbc.Switch(id="theme-toggle",
                                       label="Dark mode", value=False,
                                       style={"fontSize": "0.75rem"}),
                        ],
                    ),
                    html.Div([
                        html.Span("GliaAnalysis v0.1"),
                        html.Span(" · ", style={"opacity": "0.4"}),
                        html.Span("Microglia + Astrocyte"),
                    ]),
                ],
            ),
        ],
    )


# ── Tab bar ───────────────────────────────────────────────────────────


def _tab_bar() -> html.Div:
    nav_items = [
        dbc.NavLink(label, id=f"tab-{tid}",
                    active=(tid == "setup"), n_clicks=0)
        for tid, label in TAB_DEFS
    ]
    return html.Div(
        id="tab-bar",
        children=[
            dbc.Nav(nav_items, pills=False, className="nav-tabs",
                    style={"display": "flex", "flexWrap": "nowrap"}),
        ],
    )


# ── Main layout ───────────────────────────────────────────────────────

app.layout = html.Div(
    id="app-container",
    children=[
        dcc.Store(id="session-id", storage_type="session"),
        dcc.Store(id="active-tab", data="setup"),
        dcc.Store(id="tab-refresh", data=0),
        dcc.Store(id="theme-store", data="light"),
        _sidebar(),
        html.Div(
            id="main-content",
            children=[
                _tab_bar(),
                html.Div(id="tab-content"),
            ],
        ),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────


@callback(
    Output("session-id", "data"),
    Input("session-id", "data"),
)
def init_session(sid):
    sid = sid or server_state.create_session()
    # Rehydrate user-scoped prefs (FIJI path) on first hit of the session.
    try:
        from glia.settings import apply_user_settings, load_user_settings
        apply_user_settings(server_state.get_session(sid),
                            load_user_settings())
    except Exception:
        pass
    return sid


# Theme toggle — clientside flip <html data-theme="...">
app.clientside_callback(
    """
    function(isDark) {
        var theme = isDark ? "dark" : "light";
        document.documentElement.setAttribute("data-theme", theme);
        return theme;
    }
    """,
    Output("theme-store", "data"),
    Input("theme-toggle", "value"),
)


@callback(
    Output("tab-refresh", "data", allow_duplicate=True),
    Input("theme-store", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def on_theme_change(theme, refresh):
    set_plotly_theme(theme or "light")
    return (refresh or 0) + 1


@callback(
    Output("active-tab", "data"),
    *[Output(f"tab-{tid}", "active") for tid, _ in TAB_DEFS],
    *[Input(f"tab-{tid}", "n_clicks") for tid, _ in TAB_DEFS],
    State("active-tab", "data"),
    prevent_initial_call=True,
)
def switch_tab(*args):
    n = len(TAB_DEFS)
    current = args[n]
    triggered = ctx.triggered_id
    if triggered is None:
        active_flags = tuple(tid == current for tid, _ in TAB_DEFS)
        return (current,) + active_flags
    new_tab = triggered.replace("tab-", "")
    active_flags = tuple(tid == new_tab for tid, _ in TAB_DEFS)
    return (new_tab,) + active_flags


_PAGE_LAYOUTS = {
    "setup":    setup_page.layout,
    "segment":  segment_page.layout,
    "features": features_page.layout,
    "metadata": metadata_page.layout,
    "explore":  explore_page.layout,
    "cluster":  cluster_page.layout,
    "stats":    stats_page.layout,
    "export":   export_page.layout,
}


@callback(
    Output("tab-content", "children"),
    Input("active-tab", "data"),
    Input("tab-refresh", "data"),
    State("session-id", "data"),
)
def render_tab(active_tab, _refresh, sid):
    fn = _PAGE_LAYOUTS.get(active_tab)
    if fn is None:
        return html.Div("Unknown tab")
    return fn(sid)


# ── Sidebar: mode toggle ──────────────────────────────────────────────


@callback(
    Output("tab-refresh", "data", allow_duplicate=True),
    Input("sidebar-mode", "value"),
    State("session-id", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def on_mode_change(mode, sid, refresh):
    state = server_state.get_session(sid)
    state.mode = mode or "microglia"
    return (refresh or 0) + 1


# ── Sidebar: project dir folder picker ────────────────────────────────


@callback(
    Output("sidebar-project-info", "children"),
    Output("tab-refresh", "data", allow_duplicate=True),
    Input("sidebar-browse-project", "n_clicks"),
    State("session-id", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def on_browse_project(n_clicks, sid, refresh):
    if not n_clicks:
        return no_update, no_update
    folder = browse_folder("Select GliaAnalysis project folder")
    if not folder:
        return no_update, no_update
    state = server_state.get_session(sid)
    state.project_dir = folder
    # Rehydrate ROIs from <project>/.gliaanalysis_rois.json if present.
    try:
        from glia.roi import load_project_rois
        loaded = load_project_rois(folder)
        if loaded:
            state.extra["rois"] = loaded
    except Exception:
        pass
    # Restore the analysis params (threshold, area bounds, metadata fields,
    # cluster k, stats factors) from .gliaanalysis_settings.json so the
    # whole pipeline picks up where it left off.
    try:
        from glia.settings import apply_project_settings, load_project_settings
        apply_project_settings(state, load_project_settings(folder))
    except Exception:
        pass
    # Rehydrate the features dataframe (with parsed metadata + any prior
    # PCA / cluster assignments) so the user doesn't have to rerun the
    # Features and Metadata tabs after reopening.
    try:
        from glia.features import load_features_df
        df_loaded = load_features_df(folder)
        if df_loaded is not None:
            state.features_df = df_loaded
            if "Cluster" in df_loaded.columns:
                state.k = int(df_loaded["Cluster"].nunique())
    except Exception:
        pass
    info = _project_info_block(folder)
    return info, (refresh or 0) + 1


def _project_info_block(folder: str) -> html.Div:
    name = os.path.basename(folder) or folder
    return html.Div([
        html.Div(name, className="file-info",
                 style={"fontWeight": "600", "fontSize": "0.85rem",
                        "color": "var(--ned-text)", "wordBreak": "break-all"}),
        html.Div(folder, className="file-info",
                 style={"fontSize": "0.72rem",
                        "color": "var(--ned-text-muted)",
                        "opacity": "0.7", "wordBreak": "break-all"}),
    ])


# ── Sidebar: FIJI info reflects the Setup-tab input ───────────────────


@callback(
    Output("sidebar-fiji-info", "children"),
    Input("setup-fiji-path", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_fiji_path_change(path, sid):
    state = server_state.get_session(sid)
    state.fiji_path = path or ""
    try:
        from glia.settings import save_user_settings
        save_user_settings(state)
    except Exception:
        pass
    if not path:
        return html.Div("Not configured", className="file-info",
                        style={"opacity": "0.5"})
    return html.Div(os.path.basename(path), className="file-info",
                    style={"fontSize": "0.78rem",
                           "color": "var(--ned-text)"})


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=False, port=8050)
