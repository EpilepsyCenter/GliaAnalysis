"""Astrocyte Analysis page (astrocyte mode).

When the sidebar is set to mode = "astrocyte", this page replaces the
microglia Segment + Features tabs. The Setup tab (Prepare → ROIs →
Threshold) is unchanged — the user just picks the GFAP channel per
image in Prepare's metadata table. This page then reads the
thresholded GFAP masks, computes per-ROI network metrics, joins the
per-image metadata, and feeds the resulting dataframe into the same
Explore / Inflammation / Stats / Export tabs that the microglia path
uses.

Output: one row per (image, ROI) with columns:
  image_stem, roi_tag, plus the 9 ASTROCYTE_FEATURES, plus whatever
  per-image metadata columns the user set (Animal, Genotype,
  Treatment, …).
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update
import dash_bootstrap_components as dbc

from glia.astrocyte import (
    astrocyte_features_path,
    extract_astrocyte_features_from_project,
    load_astrocyte_features_df,
    save_astrocyte_features_df,
)
from glia.config import ASTROCYTE_FEATURES
from glia.settings import get_image_metadata
from glia_dash import server_state
from glia_dash.components import alert, empty_state, metric_card


def _thresholded_dir(project_dir: str, mode: str = "astrocyte") -> Path:
    from glia.prepare import glia_dir
    return glia_dir(project_dir, mode) / "ThresholdedImages"


def _join_metadata(df: pd.DataFrame, project_dir: str) -> pd.DataFrame:
    """Broadcast Prepare-tab per-image metadata onto each (image, ROI)
    row. The dataframe already carries ``image_stem`` and ``roi_tag``
    columns; we just look up by stem and copy every user-defined column.
    """
    if df is None or len(df) == 0:
        return df
    meta = get_image_metadata(project_dir)
    if not meta:
        return df

    _RESERVED = {"image", "channel", "channel_name",
                 "dapi_channel", "dapi_channel_name",
                 "z_projection", "exclude"}
    user_cols = sorted({
        k for row in meta for k, v in row.items()
        if k not in _RESERVED
        and not str(k).startswith("_")
        and isinstance(v, (str, int, float, bool, type(None)))
    })
    if not user_cols:
        return df

    lookup: dict[str, dict] = {}
    for row in meta:
        img = row.get("image", "")
        if not img:
            continue
        stem = img.rsplit(".", 1)[0]
        lookup[stem] = {c: row.get(c, "") for c in user_cols}

    out = df.copy()
    for c in user_cols:
        out[c] = out["image_stem"].map(
            lambda s: lookup.get(s, {}).get(c, "")
        )
    return out


# ── Layout ──────────────────────────────────────────────────────────


def layout(sid: str | None) -> html.Div:
    state = server_state.get_session(sid)
    project = state.project_dir

    if not project or not Path(project).is_dir():
        return html.Div([
            html.H4("Astrocyte Analysis", style={"marginBottom": "16px"}),
            alert("Pick a project folder from the sidebar first.",
                  variant="warning"),
        ])

    thresh_dir = _thresholded_dir(project)
    n_thresh = (len(list(thresh_dir.glob("*.tif")))
                if thresh_dir.is_dir() else 0)
    df_existing = load_astrocyte_features_df(project)
    n_rows = len(df_existing) if df_existing is not None else 0
    rois = state.extra.get("rois", {})
    n_roi_images = sum(1 for v in rois.values() if v)

    return html.Div([
        html.H4("Astrocyte Analysis", style={"marginBottom": "8px"}),
        html.Div(
            "Reads the thresholded GFAP images from Setup → Threshold "
            "and computes network metrics per (image, ROI): GFAP area "
            "fraction, total skeleton length, branches, junctions, "
            "branch density, mean intensity inside the mask, and a "
            "soma count (morphological opening with a small disk). "
            "Each row goes into Explore / Inflammation / Stats / "
            "Export, with the per-image metadata joined automatically. "
            "If you haven't drawn ROIs in Setup → ROIs, the whole "
            "image is the single 'all' ROI.",
            style={"fontSize": "0.85rem",
                   "color": "var(--ned-text-muted)",
                   "marginBottom": "16px"},
        ),

        html.Div([
            metric_card("Thresholded images", str(n_thresh),
                        accent=(n_thresh > 0)),
            metric_card("Images with ROIs", str(n_roi_images)),
            metric_card("Rows on disk", str(n_rows),
                        accent=(n_rows > 0)),
            metric_card("Mode", "astrocyte", accent=True),
        ], style={"display": "grid",
                  "gridTemplateColumns": "repeat(4, 1fr)",
                  "gap": "12px", "marginBottom": "16px"}),

        html.Div([
            html.Div([
                html.Label("Soma-disk radius (px)",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dcc.Input(id="astro-soma-radius", type="number",
                          min=1, max=20, step=1,
                          value=int(state.extra.get(
                              "astrocyte_soma_radius_px", 4)),
                          style={"width": "100px"}),
            ], style={"marginRight": "24px"}),
            dbc.Button("Run thresholding (FIJI)",
                       id="astro-threshold",
                       className="btn-ned-secondary",
                       style={"marginRight": "8px"}),
            dbc.Button("Run analysis", id="astro-run",
                       className="btn-ned-primary"),
            html.Span(
                ("Thresholding uses the Setup → Threshold params and "
                 "writes the GFAP binaries into the astrocyte-mode "
                 "ThresholdedImages folder. Run analysis then computes "
                 "the per-(image, ROI) metrics."),
                style={"fontSize": "0.78rem",
                       "color": "var(--ned-text-muted)",
                       "marginLeft": "12px"},
            ),
        ], style={"display": "flex",
                  "alignItems": "flex-end",
                  "flexWrap": "wrap",
                  "gap": "8px",
                  "marginBottom": "16px"}),

        dcc.Loading(
            type="default",
            children=html.Div(id="astro-output",
                              children=_render_summary(
                                  df_existing, project, fresh=False,
                              ) if df_existing is not None else
                              empty_state(
                                  icon="🌿",
                                  title="No astrocyte features yet",
                                  text=(f"{n_thresh} thresholded "
                                        f"image{'s' if n_thresh != 1 else ''} "
                                        "on disk. Click 'Run analysis' "
                                        "above to compute per-ROI "
                                        "metrics."),
                              ),
                              style={"marginTop": "12px",
                                     "minHeight": "40px"}),
        ),
    ])


def _render_summary(df: pd.DataFrame, project: str, *,
                    fresh: bool, dt: float | None = None) -> html.Div:
    if df is None or len(df) == 0:
        return alert("No rows produced. Are there thresholded images "
                     "in _gliaanalysis/ThresholdedImages/?",
                     variant="warning")

    n_images = df["image_stem"].nunique() if "image_stem" in df else 0
    n_rois = df.shape[0]

    timing = f" in {dt:.1f}s" if (fresh and dt is not None) else ""
    headline = alert(
        (f"✓ Computed {n_rois:,} (image, ROI) row"
         f"{'s' if n_rois != 1 else ''} across {n_images} "
         f"image{'s' if n_images != 1 else ''}{timing}. "
         "Explore / Inflammation / Stats now see these as their "
         "data source."),
        variant="success" if fresh else "info",
    )

    cols = [c for c in (["image_stem", "roi_tag"]
                        + list(ASTROCYTE_FEATURES))
            if c in df.columns]
    # Light truncation for the preview table.
    preview = df[cols].head(20).copy()
    preview = preview.round(3)

    table_rows = []
    for _, row in preview.iterrows():
        table_rows.append(html.Tr([
            html.Td(str(row.get(c, "")),
                    style={"fontFamily": "monospace",
                           "fontSize": "0.78rem",
                           "padding": "4px 8px",
                           "borderBottom":
                               "1px solid var(--ned-border)"})
            for c in cols
        ]))
    table = html.Table([
        html.Thead(html.Tr([
            html.Th(c, style={"textAlign": "left",
                              "fontSize": "0.72rem",
                              "color": "var(--ned-text-muted)",
                              "textTransform": "uppercase",
                              "letterSpacing": "0.5px",
                              "padding": "6px 8px",
                              "borderBottom":
                                  "2px solid var(--ned-border)"})
            for c in cols
        ])),
        html.Tbody(table_rows),
    ], style={"width": "100%",
              "borderCollapse": "collapse",
              "marginTop": "8px"})

    return html.Div([
        headline,
        html.Div(
            f"Showing first {len(preview)} of {len(df)} rows.",
            style={"fontSize": "0.78rem",
                   "color": "var(--ned-text-muted)",
                   "marginTop": "8px",
                   "marginBottom": "4px"},
        ),
        html.Div(table, style={"overflowX": "auto",
                               "marginBottom": "12px"}),
        html.Div(
            f"Saved to {astrocyte_features_path(project)}",
            style={"fontSize": "0.78rem",
                   "color": "var(--ned-text-muted)"},
        ),
    ])


# ── Callbacks ───────────────────────────────────────────────────────


@callback(
    Output("astro-output", "children", allow_duplicate=True),
    Input("astro-threshold", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_threshold(n_clicks, sid):
    """Run FIJI Phase 1 (threshold only) on the astrocyte-mode
    Prepared/ dir, writing the GFAP binaries into the astrocyte-mode
    ThresholdedImages/. Threshold params come from
    ``state.threshold_*`` set in Setup → Threshold."""
    if not n_clicks:
        return no_update
    state = server_state.get_session(sid)
    project = state.project_dir
    if not project or not Path(project).is_dir():
        return alert("No project folder loaded.", variant="warning")
    if not state.fiji_path or not Path(state.fiji_path).exists():
        return alert(
            f"FIJI not found at {state.fiji_path!r}. Configure it in "
            "Setup → Threshold.",
            variant="warning",
        )

    from glia.prepare import glia_dir, prepared_dir
    from glia.segment import SegmentParams, run_threshold_only

    mode = "astrocyte"
    prep_dir = prepared_dir(project, mode)
    if not prep_dir.is_dir() or not any(prep_dir.glob("*.tif")):
        return alert(
            "No prepared GFAP images yet. Go to Setup → Prepare and "
            "run preparation with the GFAP channel first.",
            variant="warning",
        )

    params = SegmentParams(
        input_dir=prep_dir,
        output_dir=glia_dir(project, mode),
        fiji_path=state.fiji_path,
        threshold_kind=state.threshold_kind,
        threshold_method=state.threshold_method,
        manual_lower=int(state.threshold_lower),
        manual_upper=int(state.threshold_upper),
        local_radius=int(state.local_radius),
        preprocess=bool(state.preprocess),
    )

    t0 = time.time()
    try:
        report = run_threshold_only(params)
    except Exception as e:
        return alert(f"Thresholding failed: {e}", variant="danger")
    dt = time.time() - t0

    msg = (f"✓ Thresholded {report.n_thresholded} of "
           f"{report.n_input_images} prepared images in {dt:.1f} s. "
           f"Binaries at {params.thresholded_dir}. "
           "Now click 'Run analysis' to compute metrics.")
    return alert(msg, variant="success")


@callback(
    Output("astro-output", "children"),
    Input("astro-run", "n_clicks"),
    State("astro-soma-radius", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_run(n_clicks, soma_radius, sid):
    if not n_clicks:
        return no_update
    state = server_state.get_session(sid)
    project = state.project_dir
    if not project or not Path(project).is_dir():
        return alert("No project folder loaded.", variant="warning")

    # Guard: if there are no thresholded images yet, nudge to threshold.
    if not any(_thresholded_dir(project).glob("*.tif")):
        return alert(
            "No thresholded GFAP images yet. Click 'Run thresholding "
            "(FIJI)' first (or run Setup → Prepare with the GFAP "
            "channel if you haven't prepared yet).",
            variant="warning",
        )

    soma_radius = int(soma_radius or 4)
    state.extra["astrocyte_soma_radius_px"] = soma_radius

    t0 = time.time()
    try:
        df = extract_astrocyte_features_from_project(
            project,
            rois=state.extra.get("rois", {}),
            soma_radius_px=soma_radius,
        )
    except Exception as e:
        return alert(f"Astrocyte analysis failed: {e}", variant="danger")

    # Join per-image metadata so Explore/Stats see Animal/Treatment/...
    df = _join_metadata(df, project)

    # Live + persisted output. features_df is the universal slot
    # downstream tabs read, so writing it here makes Explore /
    # Inflammation / Stats Just Work.
    state.features_df = df
    save_astrocyte_features_df(project, df)

    dt = time.time() - t0
    return _render_summary(df, project, fresh=True, dt=dt)
