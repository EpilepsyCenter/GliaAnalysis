"""Segment page — run the end-to-end segmentation pipeline on the project."""

from __future__ import annotations

import os
import time
from pathlib import Path

from dash import Input, Output, State, callback, dcc, html, no_update
import dash_bootstrap_components as dbc

from glia.segment import SegmentParams, run_pipeline
from glia_dash import server_state
from glia_dash.components import alert, metric_card


# Output directory layout: keep all generated artifacts inside the project
# folder under a hidden subfolder so the original images stay untouched and
# the user can wipe a run with a single rm.
_OUTPUT_SUBDIR = "_gliaanalysis"


def _project_tiffs(project_dir: str) -> list[str]:
    if not project_dir or not Path(project_dir).is_dir():
        return []
    return sorted(
        str(p) for p in Path(project_dir).iterdir()
        if p.suffix.lower() in (".tif", ".tiff") and p.is_file()
    )


def _summary_grid(state) -> html.Div:
    project = state.project_dir
    rois_map = state.extra.get("rois", {})
    images = _project_tiffs(project)
    n_rois_total = sum(len(rs) for rs in rois_map.values())
    n_images_with_rois = sum(1 for p in images if rois_map.get(p))

    method_label = f"{state.threshold_method} ({state.threshold_kind})"
    if state.threshold_kind == "global" and state.threshold_method != "Manual":
        method_label += f"  override [{int(state.threshold_lower)}, " \
                        f"{int(state.threshold_upper)}]"
    elif state.threshold_kind == "manual":
        method_label = f"Manual [{int(state.threshold_lower)}, " \
                       f"{int(state.threshold_upper)}]"

    return html.Div([
        metric_card("Project", os.path.basename(project) if project else "—"),
        metric_card("Images", str(len(images)), accent=True),
        metric_card("ROIs",
                    f"{n_rois_total} on {n_images_with_rois} img"
                    if n_rois_total else "none → 'all'"),
        metric_card("Method", method_label),
        metric_card("Area bounds",
                    f"{int(state.area_min)} – {int(state.area_max)} px²"),
        metric_card("Preprocess", "on" if state.preprocess else "off"),
    ], style={"display": "grid",
              "gridTemplateColumns": "repeat(3, 1fr)",
              "gap": "12px", "marginBottom": "20px"})


def layout(sid: str | None) -> html.Div:
    state = server_state.get_session(sid)
    project = state.project_dir
    fiji = state.fiji_path

    if not project or not Path(project).is_dir():
        return html.Div([
            html.H4("Segment", style={"marginBottom": "16px"}),
            alert("Pick a project folder from the sidebar first.",
                  variant="warning"),
        ])
    if not fiji or not Path(fiji).exists():
        return html.Div([
            html.H4("Segment", style={"marginBottom": "16px"}),
            alert("Set a valid FIJI executable in Setup → Threshold first.",
                  variant="warning"),
        ])

    output_dir = Path(project) / _OUTPUT_SUBDIR

    return html.Div([
        html.H4("Segment", style={"marginBottom": "16px"}),
        _summary_grid(state),

        html.Div([
            dbc.Button("Run pipeline", id="segment-run",
                       className="btn-ned-primary"),
            html.Span(f"Outputs → {output_dir}",
                      style={"marginLeft": "12px",
                             "fontSize": "0.78rem",
                             "color": "var(--ned-text-muted)"}),
        ], style={"display": "flex", "alignItems": "center"}),

        dcc.Loading(
            id="segment-loading",
            type="default",
            children=html.Div(id="segment-output",
                              style={"marginTop": "16px",
                                     "minHeight": "40px"}),
        ),
    ])


# ── Callbacks ────────────────────────────────────────────────────────


@callback(
    Output("segment-output", "children"),
    Input("segment-run", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_run(n_clicks, sid):
    if not n_clicks:
        return no_update
    state = server_state.get_session(sid)
    project = state.project_dir

    if not project or not Path(project).is_dir():
        return alert("No project folder loaded.", variant="danger")
    if not state.fiji_path or not Path(state.fiji_path).exists():
        return alert(f"FIJI not found at {state.fiji_path!r}.",
                     variant="danger")

    params = SegmentParams(
        input_dir=Path(project),
        output_dir=Path(project) / _OUTPUT_SUBDIR,
        fiji_path=state.fiji_path,
        threshold_kind=state.threshold_kind,
        threshold_method=state.threshold_method,
        manual_lower=int(state.threshold_lower),
        manual_upper=int(state.threshold_upper),
        local_radius=int(state.local_radius),
        area_min=float(state.area_min),
        area_max=float(state.area_max),
        preprocess=bool(state.preprocess),
        rois=state.extra.get("rois", {}),
    )

    t0 = time.time()
    try:
        report = run_pipeline(params)
    except Exception as e:
        return alert(f"Pipeline failed: {e}", variant="danger")
    dt = time.time() - t0

    cards = html.Div([
        metric_card("Input images",  str(report.n_input_images)),
        metric_card("Thresholded",   str(report.n_thresholded),
                    accent=True),
        metric_card("Single cells",  str(report.n_single_cells),
                    accent=True),
        metric_card("Skeleton CSVs", str(report.n_skeleton_csvs),
                    accent=True),
        metric_card("Prior cleared", str(report.n_cleared)),
        metric_card("Wall time",     f"{dt:.1f} s"),
    ], style={"display": "grid",
              "gridTemplateColumns": "repeat(6, 1fr)",
              "gap": "12px", "marginTop": "12px"})

    skipped_msg = (
        html.Div([
            html.Div(f"{len(report.skipped)} files skipped",
                     style={"fontSize": "0.85rem",
                            "color": "var(--ned-warning)",
                            "marginTop": "12px"}),
            html.Pre("\n".join(report.skipped[:20]),
                     style={"fontSize": "0.75rem",
                            "color": "var(--ned-text-muted)",
                            "maxHeight": "160px", "overflow": "auto",
                            "background": "var(--ned-surface)",
                            "padding": "8px",
                            "border": "1px solid var(--ned-border)",
                            "borderRadius": "6px"}),
        ])
        if report.skipped else None
    )

    return html.Div([
        alert(f"✓ Pipeline finished in {dt:.1f} s.", variant="success"),
        cards,
        skipped_msg,
        html.Div([
            html.Div("Output folder:",
                     style={"fontSize": "0.78rem",
                            "color": "var(--ned-text-muted)",
                            "marginTop": "16px"}),
            html.Code(str(params.output_dir),
                     style={"fontSize": "0.82rem"}),
        ]),
    ])
