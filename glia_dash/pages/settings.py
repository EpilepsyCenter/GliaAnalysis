"""Settings page — per-computer preferences (not part of any project).

Opened from the bottom of the sidebar, Ctrl/⌘ , or File › Settings…. Values
are saved for this computer in ~/.gliaanalysis/defaults.json (the FIJI path
is saved by ``main.on_fiji_path_change``, which also refreshes the status
bar).
"""

from __future__ import annotations

from dash import Input, Output, callback, ctx, dcc, html, no_update
import dash_bootstrap_components as dbc

from glia.fiji_runner import discover_fiji, resolve_fiji_path, validate_fiji_path
from glia_dash import server_state
from glia_dash.components import browse_file


def _fiji_status(path: str | None) -> html.Div:
    err = validate_fiji_path(path)
    if err is None:
        return html.Div("FIJI found.", className="settings-status ok")
    return html.Div(err, className="settings-status warn")


def layout(sid: str | None) -> html.Div:
    state = server_state.get_session(sid)
    fiji = state.fiji_path or (discover_fiji() or "")
    return html.Div(className="settings-page", children=[
        html.H4("Settings", style={"marginBottom": "4px"}),
        html.Div("Saved for this computer and used by every project.",
                 className="text-muted", style={"marginBottom": "18px"}),

        html.Section(className="settings-section", children=[
            html.H5("FIJI"),
            html.P(
                "Converts Olympus VSI files when images are prepared, and "
                "thresholds and skeletonizes cells in Segment and Astrocyte "
                "analysis. Point it at the FIJI program; picking the "
                "Fiji.app bundle or the Fiji folder also works.",
                className="settings-help",
            ),
            html.Label("FIJI executable", htmlFor="settings-fiji-path",
                       className="form-label"),
            html.Div(className="settings-row", children=[
                dcc.Input(id="settings-fiji-path", type="text", value=fiji,
                          debounce=True,
                          placeholder="/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx",
                          style={"flex": "1", "minWidth": "0"}),
                dbc.Button("Browse…", id="settings-fiji-browse",
                           className="btn-ned-secondary"),
            ]),
            html.Div(id="settings-fiji-status", children=_fiji_status(fiji)),
        ]),
    ])


@callback(
    Output("settings-fiji-path", "value"),
    Output("settings-fiji-status", "children"),
    Input("settings-fiji-browse", "n_clicks"),
    Input("settings-fiji-path", "value"),
    prevent_initial_call=True,
)
def on_fiji(n_clicks, typed):
    """Browse… or a typed path: a folder / Fiji.app becomes the program
    inside it; the line below the field says whether FIJI can be run."""
    path = typed or ""
    if ctx.triggered_id == "settings-fiji-browse":
        picked = browse_file("Select the FIJI program (or Fiji.app)",
                             filetypes=[("All files", "*")])
        if not picked:
            return no_update, no_update
        path = picked
    resolved = resolve_fiji_path(path)
    value = resolved if resolved != (typed or "") else no_update
    return value, _fiji_status(resolved)
