"""Metadata page — parse cell IDs into named columns.

The pipeline encodes per-cell metadata in the filename:
``<image_stem>__<roi_tag>__<N>``. ``image_stem`` is itself a list of
experimental fields joined by a user-chosen separator (default ``_``).
This tab lets the user name those fields and broadcasts them as
columns onto ``state.features_df`` so downstream tabs (Explore, Cluster,
Stats) can stratify on them.
"""

from __future__ import annotations

from dash import (
    Input, Output, State, callback, dash_table, dcc, html, no_update,
)
import dash_bootstrap_components as dbc
import pandas as pd

from glia_dash import server_state
from glia_dash.components import alert, metric_card
from glia_dash.pages.features import (
    _TABLE_STYLE_CELL, _TABLE_STYLE_DATA_CONDITIONAL,
    _TABLE_STYLE_HEADER, _TABLE_STYLE_TABLE,
)


_GROUP_SEP = "__"  # outer separator (image stem | roi tag | index)


# ── Parsing core ────────────────────────────────────────────────────


def parse_cell_id(cell_id: str, field_names: list[str], sep: str = "_"
                  ) -> dict:
    """Return {field_name: value, 'roi_tag': ..., 'cell_index': ...}.

    Missing trailing fields end up as empty strings rather than NaN so
    they group cleanly in groupbys.
    """
    out: dict = {name: "" for name in field_names}
    out["roi_tag"] = ""
    out["cell_index"] = ""
    if not cell_id:
        return out

    parts = cell_id.split(_GROUP_SEP)
    image_stem = parts[0] if parts else ""
    out["roi_tag"] = parts[1] if len(parts) > 1 else ""
    out["cell_index"] = parts[2] if len(parts) > 2 else ""

    fields = image_stem.split(sep) if image_stem else []
    for name, val in zip(field_names, fields):
        out[name] = val
    return out


def parse_features_df(df: pd.DataFrame, field_names: list[str],
                      sep: str = "_") -> pd.DataFrame:
    """Augment a features dataframe with parsed-metadata columns."""
    if df is None or len(df) == 0:
        return df
    parsed = df["ID"].apply(lambda cid: parse_cell_id(cid, field_names, sep))
    parsed_df = pd.DataFrame(list(parsed), index=df.index)
    # Don't overwrite columns that already exist (e.g. if Apply is run twice).
    new_cols = [c for c in parsed_df.columns if c not in df.columns
                or df[c].isna().all()]
    return pd.concat([df.drop(columns=[c for c in new_cols if c in df.columns]),
                      parsed_df[new_cols]], axis=1)


# ── Layout helpers ───────────────────────────────────────────────────


def _sample_id(state) -> str:
    df = state.features_df
    if df is None or len(df) == 0 or "ID" not in df.columns:
        return ""
    return str(df["ID"].iloc[0])


def _split_preview(sample: str, field_names: list[str], sep: str) -> html.Div:
    if not sample:
        return html.Span("(no cell IDs yet — run Features first)",
                         style={"fontSize": "0.82rem",
                                "color": "var(--ned-text-muted)",
                                "fontStyle": "italic"})
    parsed = parse_cell_id(sample, field_names, sep)
    chips = []
    for name in field_names:
        chips.append(html.Span([
            html.Span(name, style={"color": "var(--ned-text-muted)",
                                   "fontSize": "0.72rem",
                                   "textTransform": "uppercase",
                                   "letterSpacing": "0.5px",
                                   "marginRight": "4px"}),
            html.Span(parsed.get(name, "") or "—",
                      style={"color": "var(--ned-accent)",
                             "fontWeight": "500"}),
        ], style={"marginRight": "16px"}))
    chips.append(html.Span([
        html.Span("roi_tag", style={"color": "var(--ned-text-muted)",
                                    "fontSize": "0.72rem",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "0.5px",
                                    "marginRight": "4px"}),
        html.Span(parsed["roi_tag"] or "—",
                  style={"color": "var(--ned-success)", "fontWeight": "500"}),
    ], style={"marginRight": "16px"}))
    chips.append(html.Span([
        html.Span("idx", style={"color": "var(--ned-text-muted)",
                                "fontSize": "0.72rem",
                                "textTransform": "uppercase",
                                "letterSpacing": "0.5px",
                                "marginRight": "4px"}),
        html.Span(parsed["cell_index"] or "—",
                  style={"color": "var(--ned-text)"}),
    ]))
    return html.Div([
        html.Div(f"Sample ID: ", style={"display": "inline",
                                        "fontSize": "0.78rem",
                                        "color": "var(--ned-text-muted)"}),
        html.Code(sample, style={"fontSize": "0.85rem"}),
        html.Div(chips, style={"marginTop": "8px",
                               "display": "flex",
                               "flexWrap": "wrap"}),
    ])


def _render_preview(df: pd.DataFrame, field_names: list[str]):
    if df is None or len(df) == 0:
        return None
    all_cols = list(df.columns)
    visible_default = ["ID", *field_names, "roi_tag", "Area", "Circularity",
                       "# of branches"]
    visible_default = [c for c in visible_default if c in all_cols]
    hidden = [c for c in all_cols if c not in visible_default]
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
                dcc.Dropdown(id="metadata-page-size",
                             options=[{"label": str(n), "value": n}
                                      for n in (10, 25, 50, 100, 250)],
                             value=25, clearable=False,
                             style={"width": "120px"}),
            ], style={"marginRight": "24px"}),
            html.Div([
                html.Label("Visible columns",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dcc.Dropdown(id="metadata-visible-cols",
                             options=[{"label": c, "value": c}
                                      for c in all_cols],
                             value=visible_default, multi=True,
                             style={"minWidth": "420px"}),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "alignItems": "flex-end",
                  "marginBottom": "12px"}),

        html.Div(f"{len(df):,} cells × {df.shape[1]} columns",
                 style={"fontSize": "0.82rem",
                        "color": "var(--ned-text-muted)",
                        "marginBottom": "8px"}),

        dash_table.DataTable(
            id="metadata-table",
            data=df.to_dict("records"),
            columns=columns,
            hidden_columns=hidden,
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


# ── Layout ──────────────────────────────────────────────────────────


def layout(sid: str | None) -> html.Div:
    state = server_state.get_session(sid)
    df = state.features_df
    field_names = state.metadata_cols or ["Animal", "Condition"]
    sep = state.metadata_sep or "_"

    sample = _sample_id(state)

    return html.Div([
        html.H4("Metadata", style={"marginBottom": "8px"}),
        html.Div(
            "Cell IDs encode metadata as "
            "<image_stem>__<roi_tag>__<N>. Name the underscore-separated "
            "fields in the image stem and click Apply to add them as "
            "columns on the features table.",
            style={"fontSize": "0.85rem",
                   "color": "var(--ned-text-muted)",
                   "marginBottom": "16px"},
        ),

        html.Div([
            metric_card("Features in memory",
                        f"{len(df):,}" if df is not None else "—",
                        accent=(df is not None)),
            metric_card("Distinct ROI tags",
                        str(df["ID"].str.split("__", n=2, expand=True)[1]
                            .nunique()) if df is not None else "—",
                        accent=(df is not None)),
            metric_card("Distinct image stems",
                        str(df["ID"].str.split("__", n=1, expand=True)[0]
                            .nunique()) if df is not None else "—",
                        accent=(df is not None)),
        ], style={"display": "grid",
                  "gridTemplateColumns": "repeat(3, 1fr)",
                  "gap": "12px", "marginBottom": "16px"}),

        html.Div([
            html.Div([
                html.Label("Image-stem field names "
                           "(comma-separated, in order)",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dcc.Input(id="metadata-fields",
                          type="text",
                          value=",".join(field_names),
                          placeholder="Cohort,Animal,Condition,Sex",
                          style={"width": "100%", "maxWidth": "520px"}),
            ], style={"flex": "1"}),
            html.Div([
                html.Label("Separator",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dcc.Input(id="metadata-sep", type="text", value=sep,
                          maxLength=3, style={"width": "80px"}),
            ], style={"marginLeft": "16px"}),
        ], style={"display": "flex",
                  "alignItems": "flex-end",
                  "marginBottom": "12px"}),

        html.Div(id="metadata-split-preview",
                 children=_split_preview(sample, field_names, sep),
                 style={"padding": "10px 12px",
                        "border": "1px solid var(--ned-border)",
                        "borderRadius": "6px",
                        "background": "var(--ned-surface)",
                        "marginBottom": "16px"}),

        html.Div([
            dbc.Button("Apply metadata",
                       id="metadata-apply",
                       className="btn-ned-primary",
                       disabled=(df is None)),
            html.Span(("Run the Features tab first." if df is None
                       else f"Will tag {len(df):,} rows."),
                      style={"marginLeft": "12px",
                             "fontSize": "0.78rem",
                             "color": "var(--ned-text-muted)"}),
        ], style={"display": "flex", "alignItems": "center"}),

        html.Div(id="metadata-output",
                 style={"marginTop": "16px", "minHeight": "40px"},
                 children=_render_preview(df, field_names)
                          if df is not None else None),
    ])


# ── Callbacks ───────────────────────────────────────────────────────


@callback(
    Output("metadata-split-preview", "children"),
    Input("metadata-fields", "value"),
    Input("metadata-sep", "value"),
    State("session-id", "data"),
)
def live_split_preview(fields_csv, sep, sid):
    """Live-update the parsed-fields chip row as the user edits inputs."""
    field_names = [f.strip() for f in (fields_csv or "").split(",")
                   if f.strip()]
    state = server_state.get_session(sid)
    state.metadata_cols = field_names
    state.metadata_sep = sep or "_"
    return _split_preview(_sample_id(state), field_names, sep or "_")


@callback(
    Output("metadata-output", "children"),
    Input("metadata-apply", "n_clicks"),
    State("metadata-fields", "value"),
    State("metadata-sep", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_apply(n_clicks, fields_csv, sep, sid):
    if not n_clicks:
        return no_update
    state = server_state.get_session(sid)
    if state.features_df is None:
        return alert("No features in memory. Run the Features tab first.",
                     variant="warning")
    field_names = [f.strip() for f in (fields_csv or "").split(",")
                   if f.strip()]
    if not field_names:
        return alert("Provide at least one field name.", variant="warning")
    sep = sep or "_"
    try:
        df = parse_features_df(state.features_df, field_names, sep)
    except Exception as e:
        return alert(f"Parsing failed: {e}", variant="danger")
    state.features_df = df
    state.metadata_cols = field_names
    state.metadata_sep = sep
    return html.Div([
        alert(f"✓ Added {len(field_names) + 2} columns "
              f"({', '.join(field_names)}, roi_tag, cell_index).",
              variant="success"),
        _render_preview(df, field_names),
    ])


# ── Page size + visibility wiring (mirrors Features tab) ───────────


@callback(
    Output("metadata-table", "page_size"),
    Input("metadata-page-size", "value"),
    prevent_initial_call=True,
)
def update_page_size(size):
    return int(size or 25)


@callback(
    Output("metadata-table", "hidden_columns"),
    Input("metadata-visible-cols", "value"),
    State("metadata-table", "columns"),
    prevent_initial_call=True,
)
def update_visible_cols(visible, all_cols):
    if not all_cols:
        return no_update
    visible_set = set(visible or [])
    return [c["id"] for c in all_cols if c["id"] not in visible_set]
