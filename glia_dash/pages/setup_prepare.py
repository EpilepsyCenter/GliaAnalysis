"""Setup → 1. Prepare subtab.

Converts whatever source format the user dropped into the project folder
(TIFF, OME-TIFF, CZI, ND2, LIF, …) into 2D 8-bit single-channel TIFFs in
``<project>/_gliaanalysis/Prepared/``. Also hosts the editable per-image
metadata table (Animal / Condition / etc.) so downstream stats can
group cells by any of those fields.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import dash_ag_grid as dag
from dash import (
    ALL, Input, Output, State, callback, ctx, dcc, html, no_update,
)
import dash_bootstrap_components as dbc

from glia.io import read_image_meta
from glia.prepare import (
    list_source_files, output_path_for, prepare_dapi_directory,
    prepare_directory, prepared_dapi_dir, prepared_dir,
)
from glia.settings import (
    get_image_metadata,
    set_image_metadata,
)
from glia.vsi_convert import (
    VSI_EXTENSIONS, ensure_vsi_converted, list_vsi_files, resolve_source,
)
from glia_dash import server_state
from glia_dash.components import alert, empty_state, metric_card
from glia_dash.pages.features import (
    _TABLE_STYLE_CELL, _TABLE_STYLE_DATA_CONDITIONAL,
    _TABLE_STYLE_HEADER, _TABLE_STYLE_TABLE,
)


_Z_OPTIONS = (
    {"label": "Max projection", "value": "max"},
    {"label": "Mean projection", "value": "mean"},
    {"label": "First slice", "value": "first"},
    {"label": "Center slice", "value": "center"},
)

# Built-in metadata columns the table always carries. User-defined
# columns sit alongside them and are persisted into the project JSON.
_FIXED_COLS = (
    "image", "channel", "channel_name",
    "dapi_channel", "dapi_channel_name",
    "z_projection", "exclude",
)
_FIXED_HEADERS = {
    "image": "Image",
    "channel": "Channel",
    "channel_name": "Channel",
    "dapi_channel": "DAPI",
    "dapi_channel_name": "DAPI",
    "z_projection": "Z mode",
    "exclude": "Include?",
}

# Sentinel used in the DAPI dropdown for "no DAPI on this image".
# Stored verbatim in metadata; the resolver maps it back to index -1.
_DAPI_NONE = "(none)"

# Columns we pre-create even on a fresh project so common analyses (one-
# way ANOVA on treatment, comparison by genotype) work without the user
# having to "Add column" first.
_DEFAULT_USER_COLS = ("Animal", "Genotype", "Treatment")

_INCLUDE_OPTIONS = (
    {"label": "Include", "value": "include"},
    {"label": "Exclude", "value": "exclude"},
)


def _default_animal_from_name(filename: str) -> str:
    """First underscore-separated token of the source filename, sans
    extension. Matches the typical lab nomenclature ``<animalID>_<rest>``.
    """
    stem = filename.rsplit(".", 1)[0]
    if "." in stem:  # handle .ome.tif etc.
        stem = stem.split(".", 1)[0]
    return stem.split("_", 1)[0]


# Cached metadata reads keyed by absolute path; reading channel info is
# cheap but still touches disk, and the table re-renders frequently.
_META_CACHE: dict[str, "tuple[int, list[str]]"] = {}


def _meta_for(src: Path, project_dir: str = "") -> tuple[int, list[str]]:
    """Return (n_channels, channel_names) for a source file, cached.

    For .vsi sources, the metadata is read from the cached OME-TIFF
    produced by glia.vsi_convert. If that cache doesn't exist yet (FIJI
    conversion hasn't run), return (1, []) so the table still renders
    rather than crashing.
    """
    key = str(src)
    if key in _META_CACHE:
        return _META_CACHE[key]
    read_from = resolve_source(project_dir, src) if project_dir else src
    if not Path(read_from).exists():
        _META_CACHE[key] = (1, [])
        return _META_CACHE[key]
    try:
        m = read_image_meta(str(read_from))
        out = (m.n_channels, list(m.channel_names))
    except Exception:
        out = (1, [])
    _META_CACHE[key] = out
    return out


def _channel_names_for(src: Path, project_dir: str = "") -> list[str]:
    """All channel names for a given source, padded with synthetic names
    (e.g. ``channel 1``) when bioio couldn't pull real ones."""
    n, names = _meta_for(src, project_dir)
    if names and len(names) >= n:
        return list(names)
    return [f"channel {i}" for i in range(n)]


def _resolve_channel_index(
    src: Path, channel_name: str, project_dir: str = "",
) -> int:
    """Find ``channel_name`` in this source's channels; -1 if absent.

    The DAPI sentinel ``"(none)"`` resolves to -1 the same way a missing
    channel does, which is the "no DAPI for this image" signal.
    """
    if channel_name == _DAPI_NONE or not channel_name:
        return -1
    names = _channel_names_for(src, project_dir)
    if channel_name in names:
        return names.index(channel_name)
    # Allow legacy numeric strings ("0", "1", ...) saved before the
    # name-based UX existed.
    try:
        idx = int(channel_name)
        if 0 <= idx < len(names):
            return idx
    except Exception:
        pass
    return -1


def _project_channel_pool(
    sources: list[Path], project_dir: str = "",
) -> list[str]:
    """Ordered union of every channel name across all source files."""
    pool: list[str] = []
    seen: set[str] = set()
    for src in sources:
        for n in _channel_names_for(src, project_dir):
            if n not in seen:
                seen.add(n)
                pool.append(n)
    return pool


# ── Metadata table helpers ──────────────────────────────────────────


def _ensure_rows_for_sources(
    project_dir: str, sources: list[Path],
    default_channel_name: str, default_z: str,
    default_dapi_name: str = _DAPI_NONE,
) -> list[dict]:
    """Build the metadata-table rows.

    Channel is stored as a *name* (e.g. ``Iba1``) so the table reads
    semantically; the numeric index is resolved per-file at prepare time.
    """
    by_name = {r.get("image"): r for r in get_image_metadata(project_dir)}
    out: list[dict] = []
    for src in sources:
        existing = by_name.get(src.name, {})
        names = _channel_names_for(src, project_dir)
        # Pick a channel name in this preference order:
        #   1) what the project JSON already had
        #   2) the batch default (if this file has that name)
        #   3) the first name available for the file
        chosen = (
            existing.get("channel_name")
            or (default_channel_name
                if default_channel_name in names else None)
            or (names[0] if names else "")
        )
        # Legacy field: if we previously stored a numeric index, resolve
        # it to a name so the table stays consistent.
        if not existing.get("channel_name") and "channel" in existing:
            try:
                ci = int(existing["channel"])
                if 0 <= ci < len(names):
                    chosen = names[ci]
            except Exception:
                pass
        # DAPI channel — same preference order as the primary channel.
        # ``_DAPI_NONE`` sentinel means "no DAPI on this image".
        dapi_chosen = existing.get("dapi_channel_name")
        if not dapi_chosen:
            dapi_chosen = (default_dapi_name
                           if default_dapi_name in names else _DAPI_NONE)
        # If the chosen DAPI name isn't actually in this file's channels
        # (other than the sentinel), demote to (none) so the dropdown
        # doesn't render a value AG Grid will reject.
        if dapi_chosen != _DAPI_NONE and dapi_chosen not in names:
            dapi_chosen = _DAPI_NONE

        row = {
            "image": src.name,
            "channel_name": chosen,
            "channel": _resolve_channel_index(src, chosen, project_dir),
            "dapi_channel_name": dapi_chosen,
            "dapi_channel": _resolve_channel_index(
                src, dapi_chosen, project_dir,
            ),
            "z_projection": str(existing.get("z_projection", default_z)),
            "exclude": existing.get("exclude") or "include",
            "Animal": existing.get("Animal")
                      or _default_animal_from_name(src.name),
            "Genotype": existing.get("Genotype", ""),
            "Treatment": existing.get("Treatment", ""),
        }
        # Carry over any other user-defined columns the user added before.
        for k, v in existing.items():
            if k not in row:
                row[k] = v
        out.append(row)
    return out


def _refresh_channel_indices(project_dir: str, rows: list[dict]) -> list[dict]:
    """Recompute the numeric channel indices from each row's channel_name
    and dapi_channel_name. The numeric columns are hidden from the UI
    but used by the Prepare step at run time."""
    for r in rows:
        src = Path(project_dir) / r.get("image", "")
        r["channel"] = _resolve_channel_index(
            src, r.get("channel_name", ""), project_dir,
        )
        r["dapi_channel"] = _resolve_channel_index(
            src, r.get("dapi_channel_name", _DAPI_NONE), project_dir,
        )
    return rows


def _user_columns(rows: list[dict]) -> list[str]:
    seen: set[str] = set(_FIXED_COLS)
    out: list[str] = []
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                out.append(k)
    return out


def _row_dicts_to_table_data(rows: list[dict]) -> list[dict]:
    """Stringify everything for dash_table consumption."""
    out = []
    for r in rows:
        out.append({k: ("" if v is None else v) for k, v in r.items()})
    return out


# ── Layout ──────────────────────────────────────────────────────────


def prepare_layout(sid: str | None) -> html.Div:
    state = server_state.get_session(sid)
    project = state.project_dir

    if not project or not Path(project).is_dir():
        return html.Div([
            empty_state(
                icon="📁",
                title="No project folder loaded",
                text=("Pick a folder from the sidebar (Browse folder…) "
                      "containing your microscopy images. The Prepare "
                      "step accepts TIFF, OME-TIFF, CZI, ND2, LIF, and "
                      "Olympus VSI (converted via FIJI Bio-Formats); "
                      "it writes 8-bit single-channel TIFFs to "
                      "_gliaanalysis/Prepared/ for the rest of the "
                      "pipeline."),
            ),
        ])

    sources = list_source_files(project)
    mode = getattr(state, "mode", "microglia") or "microglia"
    out_dir = prepared_dir(project, mode)
    n_prepared = (len(list(out_dir.glob("*.tif")))
                  if out_dir.exists() else 0)

    # VSI files need a one-time FIJI/Bio-Formats round-trip before bioio
    # can read them. ensure_vsi_converted is a no-op when the cache is
    # already fresh.
    vsi_present = bool(list_vsi_files(project))
    vsi_banners: list[html.Div] = []
    if vsi_present:
        rep = ensure_vsi_converted(project, getattr(state, "fiji_path", ""))
        if rep.warnings:
            vsi_banners.append(alert(
                "VSI layout issues:\n" + "\n".join(rep.warnings),
                variant="warning",
            ))
        if rep.errors:
            vsi_banners.append(alert(
                "VSI conversion problems:\n" + "\n".join(rep.errors),
                variant="warning",
            ))
        elif rep.n_converted:
            vsi_banners.append(alert(
                f"Converted {rep.n_converted} VSI file(s) via FIJI "
                f"Bio-Formats. Cached at _gliaanalysis/Converted/.",
                variant="info",
            ))
        # Bust the channel-meta cache for any VSI we just converted so
        # the dropdown picks up the real channel names instead of the
        # placeholder we returned while the cache was empty.
        for vsi in list_vsi_files(project):
            _META_CACHE.pop(str(vsi), None)

    channel_pool = _project_channel_pool(sources, project)
    default_channel_name = str(state.extra.get(
        "prepare_channel_name",
        (channel_pool[0] if channel_pool else "channel 0"),
    ))
    default_z = str(state.extra.get("prepare_z", "max"))
    default_dapi_name = str(state.extra.get(
        "prepare_dapi_channel_name", _DAPI_NONE,
    ))
    use_dapi = bool(getattr(state, "use_dapi", False))

    rows = _ensure_rows_for_sources(project, sources,
                                    default_channel_name, default_z,
                                    default_dapi_name=default_dapi_name)
    user_cols = _user_columns(rows)

    # Per-row channel-name option list (one source might have ['DAPI','Iba1'],
    # another only ['channel 0']). AG Grid's agSelectCellEditor reads the
    # per-row options from the rowData itself when given a callable
    # cellEditorParams, so we precompute them and stick them on each row
    # under a hidden key.
    for src, row in zip(sources, rows):
        names = _channel_names_for(src, project)
        row["_channel_options"] = names
        # DAPI dropdown adds an explicit "(none)" sentinel first.
        row["_dapi_channel_options"] = [_DAPI_NONE, *names]

    channel_options = [{"label": n, "value": n} for n in channel_pool]
    dapi_channel_options = [{"label": _DAPI_NONE, "value": _DAPI_NONE}] + [
        {"label": n, "value": n} for n in channel_pool
    ]
    editable_col_options = [
        {"label": c, "value": c}
        for c in ("channel_name", "dapi_channel_name",
                  "z_projection", "exclude", *user_cols)
    ]

    # AG Grid column definitions. Per-row valid channels are read from
    # each row's `_channel_options` array (set above) so the dropdown
    # only lists channels that actually exist in that file.
    z_values = [opt["value"] for opt in _Z_OPTIONS]
    include_values = [opt["value"] for opt in _INCLUDE_OPTIONS]
    aggrid_columns = [
        {"field": "image", "headerName": "Image", "editable": False,
         "checkboxSelection": True, "headerCheckboxSelection": True,
         "minWidth": 220},
        {"field": "channel_name",
         "headerName": _FIXED_HEADERS["channel_name"],
         "editable": True,
         "cellEditor": "agSelectCellEditor",
         "cellEditorParams": {
             "function": "params.data._channel_options"
         },
         "minWidth": 160},
        {"field": "dapi_channel_name",
         "headerName": _FIXED_HEADERS["dapi_channel_name"],
         "editable": True,
         "cellEditor": "agSelectCellEditor",
         "cellEditorParams": {
             "function": "params.data._dapi_channel_options"
         },
         "minWidth": 140,
         "hide": not use_dapi},
        {"field": "z_projection",
         "headerName": _FIXED_HEADERS["z_projection"],
         "editable": True,
         "cellEditor": "agSelectCellEditor",
         "cellEditorParams": {"values": z_values},
         "minWidth": 120},
        {"field": "exclude",
         "headerName": _FIXED_HEADERS["exclude"],
         "editable": True,
         "cellEditor": "agSelectCellEditor",
         "cellEditorParams": {"values": include_values},
         "minWidth": 110},
    ] + [
        {"field": c, "headerName": c, "editable": True, "minWidth": 120}
        for c in user_cols
    ]

    return html.Div([
        *vsi_banners,
        html.Div([
            metric_card("Source files", str(len(sources)), accent=True),
            metric_card("Prepared on disk", str(n_prepared),
                        accent=(n_prepared > 0)),
            metric_card("Metadata columns",
                        str(len(user_cols))),
        ], style={"display": "grid",
                  "gridTemplateColumns": "repeat(3, 1fr)",
                  "gap": "12px", "marginBottom": "16px"}),

        # ── DAPI opt-in ──────────────────────────────────────────────
        # Off by default. When on, the DAPI column appears in the table
        # below, a DAPI batch dropdown joins the defaults row, and the
        # Prepare runner produces Prepared_dapi/ in addition to the
        # primary prepared channel. Downstream (Segment + Features)
        # picks this up automatically; the Soma tab will use DAPI
        # nucleus centroids as the radial scan center whenever a real
        # nucleus is present, falling back to the EDT peak otherwise.
        html.Div([
            dbc.Switch(
                id="prepare-use-dapi",
                label="Use DAPI nucleus for soma centering",
                value=use_dapi,
            ),
            html.Div(
                "Optional. Pick a DAPI channel per image below; the "
                "soma boundary will be detected starting from the "
                "nucleus centroid instead of the cell's deepest "
                "inscribed point. Cells with no DAPI signal inside "
                "their mask automatically fall back to the EDT-peak "
                "center, so this is safe to leave on. Turn off if your "
                "DAPI quality is poor.",
                style={"fontSize": "0.78rem",
                       "color": "var(--ned-text-muted)",
                       "marginTop": "4px"},
            ),
        ], style={"marginBottom": "16px", "maxWidth": "720px"}),

        # Batch defaults row — pick a channel name (from the union across
        # all files) plus a Z mode, then "Apply to all rows" stamps every
        # row. The DAPI column is hidden unless the Soma tab's "Use DAPI"
        # switch is on; we still render the dropdown so the user can
        # pick a default before flipping the switch.
        html.Div([
            html.Div([
                html.Label("Default channel (batch)",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dcc.Dropdown(id="prepare-default-channel",
                             options=channel_options,
                             value=default_channel_name,
                             clearable=False,
                             style={"width": "240px"}),
            ], style={"marginRight": "24px"}),
            html.Div([
                html.Label("Default DAPI channel (batch)",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dcc.Dropdown(id="prepare-default-dapi-channel",
                             options=dapi_channel_options,
                             value=default_dapi_name,
                             clearable=False,
                             style={"width": "200px"}),
            ], id="prepare-default-dapi-wrap",
               style={"marginRight": "24px",
                      "display": "block" if use_dapi else "none"}),
            html.Div([
                html.Label("Default Z mode (batch)",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dcc.Dropdown(id="prepare-default-z",
                             options=list(_Z_OPTIONS),
                             value=default_z, clearable=False,
                             style={"width": "200px"}),
            ], style={"marginRight": "24px"}),
            dbc.Button("Apply defaults to all rows",
                       id="prepare-apply-defaults",
                       className="btn-ned-secondary",
                       size="sm"),
        ], style={"display": "flex",
                  "alignItems": "flex-end",
                  "marginBottom": "12px"}),
        # When DAPI is enabled but no row has it set, a small hint that
        # tells the user where to enable it.
        html.Div(
            id="prepare-dapi-hint",
            children=("DAPI is enabled — set the DAPI column to a "
                      "nucleus channel for each image you want "
                      "soma-centered on the nucleus. "
                      "Toggle off in Setup → Soma if you'd rather stay "
                      "on the EDT-peak center."),
            style={"display": "block" if use_dapi else "none",
                   "fontSize": "0.78rem",
                   "color": "var(--ned-text-muted)",
                   "background": "rgba(56,189,248,0.08)",
                   "border": "1px solid rgba(56,189,248,0.3)",
                   "borderRadius": "6px",
                   "padding": "6px 10px",
                   "marginBottom": "12px"},
        ),

        # Add-column row
        html.Div([
            html.Label("Metadata column (e.g. Sex, Cohort):",
                       style={"fontSize": "0.78rem",
                              "color": "var(--ned-text-muted)",
                              "marginRight": "8px"}),
            dcc.Input(id="prepare-new-col", type="text",
                      placeholder="Sex",
                      style={"width": "180px"}),
            dbc.Button("Add column", id="prepare-add-col",
                       className="btn-ned-secondary",
                       size="sm",
                       style={"marginLeft": "8px"}),
            dbc.Button("Remove empty columns",
                       id="prepare-drop-empty",
                       className="btn-ned-secondary",
                       size="sm",
                       style={"marginLeft": "8px"}),
        ], style={"display": "flex",
                  "alignItems": "center",
                  "marginBottom": "12px"}),

        # Selection-helper buttons. Dash DataTable's multi-select only
        # supports per-row clicks (no shift-click), so we offer Select
        # all + Clear as quick shortcuts; combine with the table's
        # native filter row to scope first.
        html.Div([
            html.Label("Select rows:",
                       style={"fontSize": "0.78rem",
                              "color": "var(--ned-text-muted)",
                              "marginRight": "8px"}),
            dbc.Button("All", id="prepare-select-all",
                       className="btn-ned-secondary", size="sm",
                       style={"marginRight": "6px"}),
            dbc.Button("Clear", id="prepare-select-clear",
                       className="btn-ned-secondary", size="sm"),
            html.Span("Tip: use the filter row above the table to "
                      "narrow first, then Select All.",
                      style={"fontSize": "0.72rem",
                             "color": "var(--ned-text-muted)",
                             "marginLeft": "12px",
                             "fontStyle": "italic"}),
        ], style={"display": "flex",
                  "alignItems": "center",
                  "marginBottom": "8px"}),

        # Batch edit row — apply a value to every selected row at once.
        html.Div([
            html.Label("Batch edit selected:",
                       style={"fontSize": "0.78rem",
                              "color": "var(--ned-text-muted)",
                              "marginRight": "8px"}),
            dcc.Dropdown(id="prepare-batch-col",
                         options=editable_col_options,
                         value=("Treatment" if "Treatment" in user_cols
                                else (editable_col_options[0]["value"]
                                      if editable_col_options else None)),
                         clearable=False,
                         style={"width": "180px"}),
            dcc.Input(id="prepare-batch-value", type="text",
                      placeholder="value (e.g. LPS)",
                      style={"width": "180px",
                             "marginLeft": "8px"}),
            dbc.Button("Apply to selected",
                       id="prepare-batch-apply",
                       className="btn-ned-primary",
                       size="sm",
                       style={"marginLeft": "8px"}),
            html.Span(id="prepare-batch-hint",
                      children="Tick the rows below first.",
                      style={"fontSize": "0.78rem",
                             "color": "var(--ned-text-muted)",
                             "marginLeft": "12px"}),
        ], style={"display": "flex",
                  "alignItems": "center",
                  "marginBottom": "12px"}),

        # The metadata table — row-selectable so the batch-edit row above
        # can target a chosen subset.
        dag.AgGrid(
            id="prepare-table",
            columnDefs=aggrid_columns,
            rowData=rows,
            defaultColDef={
                "sortable": True,
                "filter": True,
                "resizable": True,
                "flex": 1,
            },
            dashGridOptions={
                # Native multi-row selection + shift-click range select.
                "rowSelection": "multiple",
                "suppressRowClickSelection": True,  # checkbox only, not row
                "rowMultiSelectWithClick": False,
                # One click to open a dropdown editor — AG Grid defaults
                # to double-click, which feels broken next to other UI.
                "singleClickEdit": True,
                "stopEditingWhenCellsLoseFocus": True,
                "animateRows": True,
                "pagination": True,
                "paginationPageSize": 50,
                "rowHeight": 32,
            },
            className="ag-theme-alpine-dark",
            style={"height": "560px", "width": "100%"},
            getRowStyle={
                "styleConditions": [
                    {"condition": "params.data.exclude == 'exclude'",
                     "style": {"opacity": 0.4,
                               "textDecoration": "line-through"}},
                ],
            },
        ),

        html.Div([
            dbc.Button("Prepare all images",
                       id="prepare-run",
                       className="btn-ned-primary",
                       disabled=(len(sources) == 0)),
            html.Span(f"→ {out_dir}",
                      style={"marginLeft": "12px",
                             "fontSize": "0.78rem",
                             "color": "var(--ned-text-muted)"}),
        ], style={"display": "flex",
                  "alignItems": "center",
                  "marginTop": "12px"}),

        dcc.Loading(
            type="default",
            children=html.Div(id="prepare-status",
                              style={"marginTop": "12px",
                                     "minHeight": "20px"}),
        ),

        # Sink Store for the rowData → JSON persistence callback. We
        # write into this so the callback has somewhere to return to;
        # we never read it.
        dcc.Store(id="prepare-edit-sink"),
    ])


# ── Callbacks ───────────────────────────────────────────────────────


@callback(
    Output("prepare-table", "columnDefs", allow_duplicate=True),
    Output("prepare-dapi-hint", "style"),
    Input("prepare-use-dapi", "value"),
    State("prepare-table", "columnDefs"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_use_dapi_toggle(use_dapi, cols, sid):
    """Persist the project-wide DAPI toggle and live-update the table:
    show/hide the DAPI column and the hint banner without a page
    reload."""
    state = server_state.get_session(sid)
    state.use_dapi = bool(use_dapi)
    try:
        from glia.settings import save_project_settings
        save_project_settings(state.project_dir, state)
    except Exception:
        pass

    # Toggle the DAPI column's `hide` property in-place.
    new_cols = [dict(c) for c in (cols or [])]
    for c in new_cols:
        if c.get("field") == "dapi_channel_name":
            c["hide"] = not bool(use_dapi)

    hint_style = {"display": "block" if use_dapi else "none",
                  "fontSize": "0.78rem",
                  "color": "var(--ned-text-muted)",
                  "background": "rgba(56,189,248,0.08)",
                  "border": "1px solid rgba(56,189,248,0.3)",
                  "borderRadius": "6px",
                  "padding": "6px 10px",
                  "marginBottom": "12px"}
    return new_cols, hint_style


# Show / hide the batch-default DAPI dropdown's wrapper alongside the
# column itself when the user flips the project-wide DAPI switch.
@callback(
    Output("prepare-default-dapi-wrap", "style"),
    Input("prepare-use-dapi", "value"),
    prevent_initial_call=True,
)
def on_use_dapi_default_visibility(use_dapi):
    return {"marginRight": "24px",
            "display": "block" if use_dapi else "none"}


@callback(
    Output("prepare-table", "rowData", allow_duplicate=True),
    Input("prepare-apply-defaults", "n_clicks"),
    State("prepare-default-channel", "value"),
    State("prepare-default-dapi-channel", "value"),
    State("prepare-default-z", "value"),
    State("prepare-table", "virtualRowData"),
    State("prepare-table", "rowData"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_apply_defaults(n_clicks, channel_name, dapi_channel_name,
                      z_mode, virtual_rows, rows, sid):
    """Apply the batch default channel + DAPI channel + Z mode to every
    row, but skip files that don't have the chosen channel name —
    silently substituting channel 0 (typically DAPI) would corrupt the
    analysis. DAPI defaults to ``(none)`` for files that don't have it.
    Read virtualRowData so any pending cell edits aren't reverted.
    """
    if virtual_rows:
        rows = virtual_rows
    if not n_clicks or not rows:
        return no_update
    state = server_state.get_session(sid)
    state.extra["prepare_channel_name"] = str(channel_name or "")
    state.extra["prepare_dapi_channel_name"] = str(
        dapi_channel_name or _DAPI_NONE
    )
    state.extra["prepare_z"] = str(z_mode or "max")
    project = state.project_dir
    new_rows: list[dict] = []
    for r in rows:
        src = Path(project) / r.get("image", "")
        names = _channel_names_for(src, project)
        next_name = (channel_name
                     if channel_name and channel_name in names
                     else r.get("channel_name", ""))
        next_dapi = (dapi_channel_name
                     if (dapi_channel_name
                         and (dapi_channel_name == _DAPI_NONE
                              or dapi_channel_name in names))
                     else r.get("dapi_channel_name", _DAPI_NONE))
        new_rows.append({**r,
                         "channel_name": next_name,
                         "dapi_channel_name": next_dapi,
                         "z_projection": str(z_mode or "max")})
    new_rows = _refresh_channel_indices(project, new_rows)
    set_image_metadata(project, new_rows)
    return new_rows


@callback(
    Output("prepare-table", "rowData", allow_duplicate=True),
    Output("prepare-table", "columnDefs", allow_duplicate=True),
    Output("prepare-new-col", "value"),
    Input("prepare-add-col", "n_clicks"),
    State("prepare-new-col", "value"),
    State("prepare-table", "virtualRowData"),
    State("prepare-table", "rowData"),
    State("prepare-table", "columnDefs"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_add_col(n_clicks, name, virtual_rows, rows, cols, sid):
    if virtual_rows:
        rows = virtual_rows
    if not n_clicks:
        return no_update, no_update, no_update
    name = (name or "").strip()
    if not name or any(c.get("field") == name for c in cols):
        return no_update, no_update, no_update
    new_cols = list(cols) + [
        {"field": name, "headerName": name, "editable": True, "minWidth": 120}
    ]
    new_rows = [{**r, name: r.get(name, "")} for r in rows]
    # on_table_edit no longer listens on rowData, so persist here.
    state = server_state.get_session(sid)
    if state.project_dir:
        refreshed = _refresh_channel_indices(state.project_dir,
                                             [dict(r) for r in new_rows])
        set_image_metadata(state.project_dir, refreshed)
    return new_rows, new_cols, ""


@callback(
    Output("prepare-table", "rowData", allow_duplicate=True),
    Output("prepare-table", "columnDefs", allow_duplicate=True),
    Input("prepare-drop-empty", "n_clicks"),
    State("prepare-table", "virtualRowData"),
    State("prepare-table", "rowData"),
    State("prepare-table", "columnDefs"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_drop_empty(n_clicks, virtual_rows, rows, cols, sid):
    if virtual_rows:
        rows = virtual_rows
    if not n_clicks or not rows:
        return no_update, no_update
    fixed_ids = set(_FIXED_COLS)
    keep_ids: set[str] = set(fixed_ids) | {"_channel_options"}
    for r in rows:
        for k, v in r.items():
            if k in fixed_ids:
                continue
            if v not in (None, ""):
                keep_ids.add(k)
    new_cols = [c for c in cols if c.get("field") in keep_ids]
    new_rows = [{k: v for k, v in r.items() if k in keep_ids} for r in rows]
    # on_table_edit no longer listens on rowData, so persist here.
    state = server_state.get_session(sid)
    if state.project_dir:
        refreshed = _refresh_channel_indices(state.project_dir,
                                             [dict(r) for r in new_rows])
        set_image_metadata(state.project_dir, refreshed)
    return new_rows, new_cols


# Note: we use a dummy Store as the sink so this callback doesn't
# write back to the table (which would trigger an infinite loop). The
# real work is the side-effect: writing the project JSON.


@callback(
    Output("prepare-edit-sink", "data"),
    Input("prepare-table", "cellValueChanged"),
    State("prepare-table", "virtualRowData"),
    State("prepare-table", "rowData"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_table_edit(cell_event, virtual_rows, rows, sid):
    """Persist table state to the project JSON on every USER cell edit.

    Only ``cellValueChanged`` is an Input here. We deliberately do NOT
    listen on ``rowData`` because programmatic rebuilds (batch apply,
    apply defaults, add column) update ``rowData`` AND already call
    ``set_image_metadata`` themselves — if we also re-fired on the
    rowData change, we'd race with the table's own re-render: at that
    moment ``virtualRowData`` is still the pre-rebuild snapshot, so we'd
    write the old data back and overwrite the batch update.

    dash-ag-grid 35 doesn't reliably refresh ``virtualRowData`` between
    a text-cell commit and this callback firing, so we also apply the
    ``cellValueChanged`` event payload on top of whatever rowData
    snapshot we have. Each event carries the full updated row in
    ``ev['data']``; we match by ``image`` to merge.
    """
    state = server_state.get_session(sid)
    src_rows = virtual_rows if virtual_rows else rows
    if not state.project_dir or not src_rows:
        return no_update

    src_rows = [dict(r) for r in src_rows]

    events = []
    if isinstance(cell_event, list):
        events = [e for e in cell_event if isinstance(e, dict)]
    elif isinstance(cell_event, dict):
        events = [cell_event]
    for ev in events:
        data_row = ev.get("data")
        if isinstance(data_row, dict) and data_row.get("image"):
            for i, r in enumerate(src_rows):
                if r.get("image") == data_row["image"]:
                    src_rows[i] = {**r, **data_row}
                    break
            continue
        idx = ev.get("rowIndex")
        col = ev.get("colId")
        val = ev.get("value", ev.get("newValue"))
        if (isinstance(idx, int) and isinstance(col, str)
                and 0 <= idx < len(src_rows)):
            src_rows[idx][col] = val

    refreshed = _refresh_channel_indices(state.project_dir, src_rows)
    set_image_metadata(state.project_dir, refreshed)
    return len(refreshed)


@callback(
    Output("prepare-table", "rowData", allow_duplicate=True),
    Output("prepare-batch-value", "value"),
    Output("prepare-batch-hint", "children"),
    Input("prepare-batch-apply", "n_clicks"),
    State("prepare-batch-col", "value"),
    State("prepare-batch-value", "value"),
    State("prepare-table", "selectedRows"),
    State("prepare-table", "virtualRowData"),
    State("prepare-table", "rowData"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_batch_apply(n_clicks, col, value, selected,
                   virtual_rows, rows, sid):
    """``selected`` is the list of row-data dicts from AG Grid (not
    indices) — match each against the rowData by ``image`` field.
    Read from virtualRowData so any pending cell edits the user made
    before clicking Apply aren't reverted."""
    if virtual_rows:
        rows = virtual_rows
    if not n_clicks:
        return no_update, no_update, no_update
    if not col:
        return no_update, no_update, "Pick a column to edit."
    if not selected:
        return no_update, no_update, ("Select rows first (tick the "
                                       "checkboxes in the table).")
    coerced: object = value if value is not None else ""
    if col == "exclude":
        v = str(value or "").strip().lower()
        coerced = "exclude" if v in ("exclude", "x", "1", "yes", "true") \
            else "include"

    project = server_state.get_session(sid).project_dir
    selected_images = {s.get("image") for s in selected}
    new_rows = [dict(r) for r in rows]
    applied = 0
    skipped_no_channel: list[str] = []
    for r in new_rows:
        if r.get("image") not in selected_images:
            continue
        if col == "channel_name":
            src = Path(project) / r.get("image", "")
            if value and value not in _channel_names_for(src, project):
                skipped_no_channel.append(r.get("image", ""))
                continue
        r[col] = coerced
        applied += 1
    new_rows = _refresh_channel_indices(project, new_rows)
    set_image_metadata(project, new_rows)
    msg = (f"Applied {col} = {coerced!r} to {applied} row"
           f"{'s' if applied != 1 else ''}.")
    if skipped_no_channel:
        msg += (f" Skipped {len(skipped_no_channel)} row"
                f"{'s' if len(skipped_no_channel) != 1 else ''} "
                "where that channel doesn't exist in the file.")
    return new_rows, "", msg


# Selection helpers — AG Grid supports shift-click range selection
# natively. These buttons are convenience shortcuts for "Select all in
# current filter" and "Clear".


@callback(
    Output("prepare-table", "selectAll", allow_duplicate=True),
    Input("prepare-select-all", "n_clicks"),
    prevent_initial_call=True,
)
def on_select_all(n_clicks):
    if not n_clicks:
        return no_update
    # AG Grid filtered-selection: select every row that matches the
    # current filter (or every row if no filter is active).
    return {"filtered": True}


@callback(
    Output("prepare-table", "deselectAll", allow_duplicate=True),
    Input("prepare-select-clear", "n_clicks"),
    prevent_initial_call=True,
)
def on_select_clear(n_clicks):
    if not n_clicks:
        return no_update
    return True


@callback(
    Output("prepare-status", "children"),
    Output("roi-images-store", "data", allow_duplicate=True),
    Output("setup-image-chip-row", "children", allow_duplicate=True),
    Output("setup-preview-path-store", "data", allow_duplicate=True),
    Input("prepare-run", "n_clicks"),
    State("prepare-table", "virtualRowData"),
    State("prepare-table", "rowData"),
    State("prepare-default-channel", "value"),
    State("prepare-default-z", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_prepare(n_clicks, virtual_rows, rows,
               default_channel_name, default_z, sid):
    if virtual_rows:
        rows = virtual_rows
    if not n_clicks:
        return no_update, no_update, no_update, no_update
    state = server_state.get_session(sid)
    project = state.project_dir
    if not project or not Path(project).is_dir():
        return (alert("No project folder.", variant="warning"),
                no_update, no_update, no_update)

    # Persist whatever's currently in the table before running.
    if rows:
        rows = _refresh_channel_indices(project, [dict(r) for r in rows])
        set_image_metadata(project, rows)

    mode = getattr(state, "mode", "microglia") or "microglia"
    # Wipe Prepared/ so excluded files don't leave stale 8-bit TIFFs
    # behind from a previous run. Mode-aware so the *other* mode's
    # Prepared/ is untouched.
    out_dir = prepared_dir(project, mode)
    n_cleared = 0
    if out_dir.exists():
        for p in out_dir.iterdir():
            if p.is_file():
                try:
                    p.unlink()
                    n_cleared += 1
                except Exception:
                    pass

    # Build per-image overrides for the rows that are actually included.
    # Reject any row whose channel name doesn't exist in its file rather
    # than silently substituting channel 0 — that would corrupt the
    # analysis (e.g. preparing DAPI when the user picked Iba1).
    included = [r for r in (rows or []) if r.get("exclude") != "exclude"]
    per_image: dict = {}
    unresolved: list[str] = []
    for r in included:
        src = Path(project) / r["image"]
        ci = _resolve_channel_index(src, r.get("channel_name", ""), project)
        if ci < 0:
            unresolved.append(
                f"{r['image']} (no channel '{r.get('channel_name', '')}')"
            )
            continue
        per_image[r["image"]] = {
            "channel": ci,
            "z_projection": str(r.get("z_projection", default_z or "max")),
        }

    excluded_names = {r["image"] for r in (rows or [])
                      if r.get("exclude") == "exclude"}

    # Only include files whose channel actually resolved.
    included_names = set(per_image.keys())
    t0 = time.time()
    rep = prepare_directory(
        project, channel=0,
        z_projection=str(default_z or "max"),
        per_image=per_image,
        included_names=included_names,
        mode=mode,
    )

    # Optional DAPI pass — opt-in via the Soma tab toggle. Wipes
    # Prepared_dapi/ on each run so stale crops can't survive a config
    # change (e.g. user switched DAPI off for some images).
    dapi_rep = None
    dapi_dir_path = prepared_dapi_dir(project, mode)
    if dapi_dir_path.exists():
        for p in dapi_dir_path.iterdir():
            if p.is_file():
                try:
                    p.unlink()
                except Exception:
                    pass
    if getattr(state, "use_dapi", False):
        dapi_per_image: dict = {}
        for r in included:
            try:
                di = int(r.get("dapi_channel", -1))
            except (TypeError, ValueError):
                di = -1
            if di < 0:
                continue
            dapi_per_image[r["image"]] = {
                "channel": di,
                "z_projection": str(r.get("z_projection",
                                          default_z or "max")),
            }
        if dapi_per_image:
            dapi_rep = prepare_dapi_directory(
                project,
                dapi_per_image=dapi_per_image,
                z_projection=str(default_z or "max"),
                included_names=set(dapi_per_image.keys()),
                mode=mode,
            )

    dt = time.time() - t0
    if unresolved:
        rep.errors.extend(unresolved)
        rep.n_skipped += len(unresolved)

    body = [alert(
        f"✓ Prepared {rep.n_prepared} image"
        f"{'s' if rep.n_prepared != 1 else ''} in {dt:.1f} s.",
        variant="success",
    )]
    if dapi_rep is not None:
        body.append(alert(
            f"✓ Prepared {dapi_rep.n_prepared} DAPI channel"
            f"{'s' if dapi_rep.n_prepared != 1 else ''}.",
            variant="info",
        ))
        if dapi_rep.errors:
            body.append(alert(
                "DAPI prep failed for: "
                + "; ".join(dapi_rep.errors[:6])
                + (" …" if len(dapi_rep.errors) > 6 else ""),
                variant="warning",
            ))
    if rep.errors:
        body.append(alert(
            "Failed for: " + "; ".join(rep.errors[:6])
            + (" …" if len(rep.errors) > 6 else ""),
            variant="warning",
        ))
    body.append(html.Div(
        f"Output → {prepared_dir(project, mode)}"
        + (f"  +  {prepared_dapi_dir(project, mode)}"
           if dapi_rep is not None and dapi_rep.n_prepared > 0
           else ""),
        style={"fontSize": "0.78rem",
               "color": "var(--ned-text-muted)",
               "marginTop": "6px"},
    ))

    # Refresh the ROI and Threshold subtab image lists from the
    # freshly-prepared directory so the user doesn't have to reload
    # the project folder. dbc.Tabs renders all subtab layouts eagerly
    # at Setup-mount time, so the ROI store + Threshold chip row sit
    # in the DOM with the *pre-Prepare* image listing — these outputs
    # bring them up to date.
    from glia_dash.pages.setup_roi import _list_images as _roi_list_images
    from glia_dash.pages.setup_threshold import (
        _list_project_images as _threshold_list_images,
        _render_image_chips as _threshold_render_chips,
    )
    refreshed_images = _roi_list_images(project, mode)
    roi_store = {"folder": project, "images": refreshed_images, "idx": 0}
    threshold_images = _threshold_list_images(project, mode)
    threshold_current = (state.extra.get("preview_image_path", "")
                         or (threshold_images[0]
                             if threshold_images else ""))
    if threshold_current and threshold_current not in threshold_images:
        threshold_current = (threshold_images[0]
                             if threshold_images else "")
    state.extra["preview_image_path"] = threshold_current
    threshold_chips = _threshold_render_chips(
        project, threshold_current, mode,
    )

    return html.Div(body), roi_store, threshold_chips, threshold_current
