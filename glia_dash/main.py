"""GliaAnalysis Dash application entry point.

Run with:  python -m glia_dash.main   (or `python app.py` from the project root)

Desktop layout shared with NED-Net / ASTERISM / TRACER: a sidebar with the
logo, the cell-type switch and the page list; a toolbar with the open
project; a status bar along the bottom. A single ``render_tab`` callback
hands off to per-page ``layout(sid)`` functions. Light/dark is a
data-theme attribute on <html>; assets/app_shell.js follows the system
appearance, adds the keyboard shortcuts and the window.glia API the native
menu (desktop.py) calls.
"""

from __future__ import annotations

import os
from pathlib import Path

from dash import (
    ALL, Dash, Input, Output, State, callback, ctx, dcc, html, no_update,
)
import dash_bootstrap_components as dbc

from glia_dash import __version__, server_state
from glia_dash.components import browse_folder, logo, set_plotly_theme
from glia_dash.pages import (
    astrocyte_analysis as astrocyte_page,
    cluster as cluster_page,
    explore as explore_page,
    export as export_page,
    features as features_page,
    inflammation as inflammation_page,
    segment as segment_page,
    settings as settings_page,
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
    update_title=None,
)
server = app.server


# ── Navigation ────────────────────────────────────────────────────────
#
# Sidebar groups per mode: (group label, [(page id, label, icon)]).
# Icons are CSS masks in assets/style.css (.icon-<name>). The page id
# order across both modes is also the Cmd/Ctrl+1..9 order in
# assets/app_shell.js and the Go menu in desktop.py.
#
# microglia: setup → segment → features → cluster → explore →
#            inflammation → stats → export
# astrocyte: setup → astrocyte_analysis → explore → inflammation →
#            stats → export
# (Both modes share Setup, Explore, Inflammation, Stats, Export.)

_ANALYSIS_GROUP = ("Analysis", [
    ("explore", "Explore", "eye"),
    ("inflammation", "Inflammation", "flame"),
    ("stats", "Stats", "chart"),
])
_OUTPUT_GROUP = ("Output", [("export", "Export", "export")])
_PROJECT_GROUP = ("Project", [("setup", "Setup", "folder")])

NAV_GROUPS = {
    "microglia": [
        _PROJECT_GROUP,
        ("Microglia", [
            ("segment", "Segment", "segment"),
            ("features", "Features", "ruler"),
            ("cluster", "Cluster", "cluster"),
        ]),
        _ANALYSIS_GROUP,
        _OUTPUT_GROUP,
    ],
    "astrocyte": [
        _PROJECT_GROUP,
        ("Astrocyte", [
            ("astrocyte_analysis", "Astrocyte analysis", "astro"),
        ]),
        _ANALYSIS_GROUP,
        _OUTPUT_GROUP,
    ],
}


def _visible_ids(mode: str) -> list[str]:
    groups = NAV_GROUPS.get(mode or "microglia", NAV_GROUPS["microglia"])
    return [tid for _, items in groups for tid, _, _ in items]


def _nav_button(tid: str, label: str, icon: str, active: bool = False):
    return html.Button(
        [html.Span(className=f"nav-icon icon-{icon}"),
         html.Span(label, className="nav-label")],
        id={"type": "tab", "id": tid},
        n_clicks=0,
        className="nav-item active" if active else "nav-item",
    )


def _nav_children(mode: str, active: str = "setup") -> list:
    """Group labels and one button per page visible in ``mode``.

    IDs are pattern-matching dicts (``{"type": "tab", "id": tid}``) so the
    global switch_tab callback can target ALL of them without referencing
    IDs that a mode flip removes from the DOM.
    """
    out = []
    for group, items in NAV_GROUPS.get(mode or "microglia",
                                       NAV_GROUPS["microglia"]):
        out.append(html.Div(group, className="nav-group-label"))
        for tid, label, icon in items:
            out.append(_nav_button(tid, label, icon, active=(tid == active)))
    return out


# ── Sidebar ───────────────────────────────────────────────────────────


def _sidebar() -> html.Div:
    return html.Div(
        id="sidebar",
        children=[
            # Logo (same height as the toolbar beside it); goes to Setup
            html.Div(
                id="sidebar-brand",
                children=[html.Button(logo("wordmark", "brand-logo"),
                                      id="brand-home", className="brand-home",
                                      title="Setup")],
            ),
            # Cell type: reshapes the page list
            html.Div(
                className="sidebar-mode",
                children=[dbc.RadioItems(
                    id="sidebar-mode",
                    options=[
                        {"label": "Microglia", "value": "microglia"},
                        {"label": "Astrocyte", "value": "astrocyte"},
                    ],
                    value="microglia",
                    inline=True,
                    className="btn-group segmented",
                    inputClassName="btn-check",
                    labelClassName="btn btn-outline-secondary",
                    labelCheckedClassName="btn-ned-primary",
                )],
            ),
            html.Nav(_nav_children("microglia", active="setup"),
                     id="sidebar-nav", **{"aria-label": "Pages"}),
            html.Div(
                id="sidebar-footer",
                children=[
                    # Per-computer preferences (FIJI); Ctrl/⌘ ,
                    _nav_button("settings", "Settings", "gear"),
                    dbc.Switch(id="theme-toggle", label="Dark mode",
                               value=False, className="theme-switch"),
                    html.Div(
                        className="sidebar-version",
                        children=[html.Span(f"GliaAnalysis {__version__}")],
                    ),
                ],
            ),
        ],
    )


# ── Toolbar and status bar ────────────────────────────────────────────


def _title_block(title: str, subtitle: str, tooltip: str | None = None):
    """Toolbar title: project name over its folder."""
    return [
        html.Div(title, className="toolbar-title"),
        html.Div(subtitle, className="toolbar-subtitle", title=tooltip),
    ]


def _toolbar() -> html.Div:
    return html.Div(
        id="app-toolbar",
        children=[
            html.Div(id="toolbar-project", className="toolbar-title-block",
                     children=_title_block("No project open",
                                           "Open a folder of images to begin")),
            html.Div(className="toolbar-spacer"),
            dbc.Button("Open project…", id="toolbar-open-project",
                       className="btn-ned-secondary",
                       title="Open a project folder (Ctrl/⌘ O)"),
        ],
    )


def _status_bar() -> html.Div:
    return html.Div(
        id="status-bar",
        children=[html.Div(id="status-items", className="status-items")],
    )


# ── Main layout ───────────────────────────────────────────────────────

app.layout = html.Div(
    id="app-container",
    children=[
        dcc.Store(id="session-id", storage_type="session"),
        dcc.Store(id="active-tab", data="setup"),
        dcc.Store(id="tab-refresh", data=0),
        dcc.Store(id="status-refresh", data=0),
        dcc.Store(id="theme-store", data="light"),
        _sidebar(),
        html.Div(
            id="main-content",
            children=[
                _toolbar(),
                html.Div(id="tab-content"),
            ],
        ),
        _status_bar(),
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
        state = server_state.get_session(sid)
        apply_user_settings(state, load_user_settings())
        if not state.fiji_path:  # FIJI in its usual place: use it
            from glia.fiji_runner import discover_fiji
            state.fiji_path = discover_fiji() or ""
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
    Output({"type": "tab", "id": ALL}, "className"),
    Input({"type": "tab", "id": ALL}, "n_clicks"),
    State("active-tab", "data"),
    State({"type": "tab", "id": ALL}, "id"),
    prevent_initial_call=True,
)
def switch_tab(n_clicks_list, current, ids):
    """Make the clicked sidebar page active and highlight it.

    Pattern-matching ``ALL`` targets whichever page buttons are mounted, so
    the callback survives mode flips that add/remove them. Two gotchas:

    * **Phantom fires on remount.** When ``on_mode_change`` rebuilds the
      nav, the new buttons enter with ``n_clicks=0`` and Dash fires this
      callback with one of them as ``triggered_id``. We ignore that by
      requiring the triggered button's ``n_clicks`` to be truthy.
    * **No ``no_update`` for pattern-matching outputs.** Always return a
      concrete list whose length matches the mounted buttons.
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

    classes = ["nav-item active" if idict.get("id") == new_active
               else "nav-item" for idict in ids]
    return new_active, classes


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
    "settings":           settings_page.layout,
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
        return html.Div("Unknown page")
    return fn(sid)


# ── Cell type (sidebar switch) ──────────────────────────────────────────────


@callback(
    Output("tab-refresh", "data", allow_duplicate=True),
    Output("sidebar-nav", "children"),
    Output("active-tab", "data", allow_duplicate=True),
    Input("sidebar-mode", "value"),
    State("active-tab", "data"),
    State("session-id", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def on_mode_change(mode, current_tab, sid, refresh):
    """Mode flip rebuilds the page list and snaps to a valid page.

    Also reloads ``state.features_df`` from the right on-disk CSV
    (features.csv vs astrocyte_features.csv) so Explore / Stats /
    Inflammation see the correct dataframe immediately, without
    needing the user to rerun extraction.

    If the user was on a microglia-only page (e.g. Cluster) and flips
    to astrocyte, that page disappears from the list; we redirect them
    to Setup so the page area doesn't try to render a hidden page.
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

    keep = current_tab in _visible_ids(mode) or current_tab == "settings"
    new_active = current_tab if keep else "setup"
    nav = _nav_children(mode, active=new_active)
    return (refresh or 0) + 1, nav, new_active


# ── Open project (toolbar, File menu, start panel) ────────────────────────────────


@callback(
    Output("tab-refresh", "data", allow_duplicate=True),
    Input("toolbar-open-project", "n_clicks"),
    State("session-id", "data"),
    State("tab-refresh", "data"),
    prevent_initial_call=True,
)
def on_browse_project(n_clicks, sid, refresh):
    if not n_clicks:
        return no_update
    folder = browse_folder("Select GliaAnalysis project folder")
    if not folder:
        return no_update
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
    return (refresh or 0) + 1


# ── FIJI path (Settings page input) ──────────────────────────────────


@callback(
    Output("status-refresh", "data"),
    Input("settings-fiji-path", "value"),
    State("session-id", "data"),
    State("status-refresh", "data"),
    prevent_initial_call=True,
)
def on_fiji_path_change(path, sid, n):
    from glia.fiji_runner import resolve_fiji_path
    state = server_state.get_session(sid)
    state.fiji_path = resolve_fiji_path(path or "")
    try:
        from glia.settings import save_user_settings
        save_user_settings(state)
    except Exception:
        pass
    return (n or 0) + 1


# ── Toolbar title and status bar ──────────────────────────────────────


def _status(text: str, kind: str = "none", tooltip: str | None = None,
            go: str | None = None):
    """One status-bar item: coloured dot + short text.
    kind: ok (green), warn (amber), info (accent), none (grey).
    go: a page id; clicking the item opens that page (assets/app_shell.js)."""
    extra = {"data-go": go} if go else {}
    return html.Span(
        [html.Span(className=f"status-dot {kind}"), html.Span(text)],
        className="status-item link" if go else "status-item",
        title=tooltip, **extra,
    )


def _count_prepared(state) -> int:
    try:
        from glia.prepare import prepared_dir
        d = prepared_dir(state.project_dir, state.mode or "microglia")
        return len(list(d.glob("*.tif"))) if d.exists() else 0
    except Exception:
        return 0


@callback(
    Output("toolbar-project", "children"),
    Output("status-items", "children"),
    Input("tab-refresh", "data"),
    Input("active-tab", "data"),
    Input("status-refresh", "data"),
    Input("session-id", "data"),
)
def update_chrome(_refresh, _tab, _status_n, sid):
    """Project name in the toolbar; FIJI and pipeline progress in the
    status bar. Re-read whenever the page changes or re-renders."""
    state = server_state.get_session(sid)
    mode = state.mode or "microglia"
    folder = state.project_dir
    has_project = bool(folder) and Path(folder).is_dir()

    if has_project:
        title = _title_block(os.path.basename(folder.rstrip("/\\")) or folder,
                             folder, tooltip=folder)
    else:
        title = _title_block("No project open",
                             "Open a folder of images to begin")

    fiji = state.fiji_path
    items = [_status(f"FIJI: {os.path.basename(fiji.rstrip('/'))}", "ok", fiji,
                     go="settings")
             if fiji else _status("FIJI not set", "warn",
                                  "Set it in Settings", go="settings")]
    if has_project:
        n_prep = _count_prepared(state)
        items.append(_status(f"{n_prep} image{'s' if n_prep != 1 else ''} "
                             "prepared", "ok" if n_prep else "warn"))
        n_roi = sum(1 for r in (state.extra.get("rois") or {}).values() if r)
        items.append(_status(
            f"ROIs on {n_roi} image{'s' if n_roi != 1 else ''}" if n_roi
            else "No ROIs drawn", "ok" if n_roi else "none"))
        df = state.features_df
        if df is not None and len(df):
            unit = "ROI rows" if mode == "astrocyte" else "cells"
            items.append(_status(f"Features: {len(df):,} {unit}", "ok"))
            if mode == "microglia" and "Cluster" in df.columns:
                items.append(_status(
                    f"Clusters: k = {int(df['Cluster'].nunique())}", "info"))
        else:
            items.append(_status("No features yet"))
    items.append(html.Span(className="status-spacer"))
    items.append(html.Span("Astrocyte" if mode == "astrocyte" else "Microglia",
                           className="status-item"))
    return title, items


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=False, port=8050)
