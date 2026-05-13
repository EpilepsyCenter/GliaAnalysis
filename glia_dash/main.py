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
    ALL, Dash, Input, Output, State, callback, ctx, dcc, html, no_update,
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
    astrocyte_analysis as astrocyte_page,
    cluster as cluster_page,
    explore as explore_page,
    export as export_page,
    features as features_page,
    inflammation as inflammation_page,
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
#
# Every possible tab across all modes lives in TAB_DEFS so the global
# switch_tab callback can register Inputs for every tab.id. Only the
# subset returned by ``_visible_tabs(mode)`` is rendered into the tab
# bar at any given time — toggling sidebar mode reshapes the bar.
#
# microglia: setup → segment → features → explore → cluster →
#            inflammation → stats → export
# astrocyte: setup → astrocyte_analysis → explore → inflammation →
#            stats → export
# (Both modes share Setup, Explore, Inflammation, Stats, Export.)

TAB_DEFS = [
    ("setup",              "Setup"),
    ("segment",            "Segment"),
    ("features",           "Features"),
    ("astrocyte_analysis", "Astrocyte"),
    ("explore",            "Explore"),
    ("cluster",            "Cluster"),
    ("inflammation",       "Inflammation"),
    ("stats",              "Stats"),
    ("export",             "Export"),
]

_TABS_BY_MODE = {
    "microglia": ["setup", "segment", "features", "explore", "cluster",
                  "inflammation", "stats", "export"],
    "astrocyte": ["setup", "astrocyte_analysis", "explore",
                  "inflammation", "stats", "export"],
}


def _visible_tabs(mode: str) -> list[tuple[str, str]]:
    """Tab (id, label) pairs to render for the current mode."""
    ids = _TABS_BY_MODE.get(mode or "microglia",
                            _TABS_BY_MODE["microglia"])
    by_id = dict(TAB_DEFS)
    return [(tid, by_id[tid]) for tid in ids if tid in by_id]


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


def _nav_items_for(mode: str, active: str = "setup") -> list:
    """Build NavLinks only for the tabs visible in the given mode.

    IDs are pattern-matching dicts (``{"type": "tab", "id": tid}``)
    rather than static strings. This lets the global switch_tab
    callback target ALL such NavLinks without referencing IDs that
    don't currently exist in the DOM — important because mode-flip
    adds/removes NavLinks, and a static-id Input would crash with
    "nonexistent object" the moment its target isn't rendered.
    """
    return [
        dbc.NavLink(label, id={"type": "tab", "id": tid},
                    active=(tid == active), n_clicks=0)
        for tid, label in _visible_tabs(mode)
    ]


def _tab_bar() -> html.Div:
    return html.Div(
        id="tab-bar",
        children=[
            dbc.Nav(id="tab-nav",
                    children=_nav_items_for("microglia", active="setup"),
                    pills=False, className="nav-tabs",
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
    Output({"type": "tab", "id": ALL}, "active"),
    Input({"type": "tab", "id": ALL}, "n_clicks"),
    State("active-tab", "data"),
    State({"type": "tab", "id": ALL}, "id"),
    prevent_initial_call=True,
)
def switch_tab(n_clicks_list, current, ids):
    """Switch active tab in response to a NavLink click.

    Pattern-matching ``ALL`` targets any subset of tabs currently
    mounted, so the callback survives mode flips that
    add/remove NavLinks. Two gotchas this implementation handles:

    * **Phantom fires on remount.** When ``on_mode_change`` rebuilds
      ``tab-nav`` children, the new NavLinks each enter with
      ``n_clicks=0`` and Dash fires this callback with one of them as
      ``triggered_id``. We ignore that by requiring the triggered
      NavLink's actual ``n_clicks`` to be truthy.
    * **No ``no_update`` for pattern-matching outputs.** Always return
      a concrete list whose length matches the currently-rendered
      NavLinks; Dash rejects scalar ``no_update`` here.
    """
    triggered = ctx.triggered_id
    ids = ids or []
    n_clicks_list = n_clicks_list or []
    new_active = current or "setup"

    if isinstance(triggered, dict) and triggered.get("type") == "tab":
        for i, idict in enumerate(ids):
            if idict == triggered:
                clicked_count = (n_clicks_list[i]
                                 if i < len(n_clicks_list) else 0)
                if clicked_count:
                    new_active = triggered["id"]
                break

    active_flags = [idict.get("id") == new_active for idict in ids]
    return new_active, active_flags


_PAGE_LAYOUTS = {
    "setup":              setup_page.layout,
    "segment":            segment_page.layout,
    "features":           features_page.layout,
    "astrocyte_analysis": astrocyte_page.layout,
    "explore":            explore_page.layout,
    "cluster":            cluster_page.layout,
    "inflammation":       inflammation_page.layout,
    "stats":              stats_page.layout,
    "export":             export_page.layout,
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
    Output("tab-nav", "children"),
    Output("active-tab", "data", allow_duplicate=True),
    Input("sidebar-mode", "value"),
    State("active-tab", "data"),
    State("session-id", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def on_mode_change(mode, current_tab, sid, refresh):
    """Mode flip rebuilds the tab bar and snaps to a valid tab.

    Also reloads ``state.features_df`` from the right on-disk CSV
    (features.csv vs astrocyte_features.csv) so Explore / Stats /
    Inflammation see the correct dataframe immediately, without
    needing the user to rerun extraction.

    If the user was on a microglia-only tab (e.g. Cluster) and flips
    to astrocyte, that tab disappears from the bar; we redirect them
    to Setup so the page area doesn't try to render a hidden tab.
    """
    state = server_state.get_session(sid)
    mode = mode or "microglia"
    state.mode = mode

    # Swap the in-memory dataframe to match the new mode's on-disk CSV.
    if state.project_dir:
        try:
            if mode == "astrocyte":
                from glia.astrocyte import load_astrocyte_features_df
                df_loaded = load_astrocyte_features_df(state.project_dir)
            else:
                from glia.features import load_features_df
                df_loaded = load_features_df(state.project_dir)
            state.features_df = df_loaded
        except Exception:
            pass

    visible_ids = {tid for tid, _ in _visible_tabs(mode)}
    new_active = current_tab if current_tab in visible_ids else "setup"
    nav = _nav_items_for(mode, active=new_active)
    return (refresh or 0) + 1, nav, new_active


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
    # PCA / cluster assignments) so the user doesn't have to rerun
    # Features after reopening. In astrocyte mode the on-disk table is
    # ``astrocyte_features.csv`` (one row per (image, ROI)); in
    # microglia mode it's ``features.csv`` (one row per cell).
    try:
        if (state.mode or "microglia") == "astrocyte":
            from glia.astrocyte import load_astrocyte_features_df
            df_loaded = load_astrocyte_features_df(folder)
        else:
            from glia.features import load_features_df
            df_loaded = load_features_df(folder)
        if df_loaded is not None:
            state.features_df = df_loaded
            if "Cluster" in df_loaded.columns:
                state.k = int(df_loaded["Cluster"].nunique())
    except Exception:
        pass
    # Always re-broadcast the Prepare-tab metadata over the loaded
    # features so any edits the user made to Animal/Genotype/Treatment
    # after the last extraction take effect.
    try:
        from glia.metadata import ensure_metadata_joined
        ensure_metadata_joined(state)
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
