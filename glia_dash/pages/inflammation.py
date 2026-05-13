"""Inflammation Index page.

Trains a supervised PCA score from two reference groups in the per-image
metadata (e.g. Saline vs LPS), then applies that score to every cell in
the project. Cells from groups that weren't used for training are still
scored — the trained axis quantifies how far along the activation
continuum any new condition sits.

Workflow:
  1. Pick the metadata column (defaults to ``state.factor_cols[0]`` /
     "Treatment" if present).
  2. Pick the **control** group (the "resting" anchor) and the
     **comparator** group (the "activated" anchor).
  3. Hit Train → forward-greedy feature selection finds the subset
     whose PC1 best separates the two anchor groups. Output: AUC,
     selected features, PC1 explained variance, training-cell n.
  4. Histogram of scores split by ALL metadata groups (including the
     untrained ones) + per-animal strip plot.
  5. Apply persists the score into ``features_df`` and
     ``features.csv`` so Explore / Stats can use it.

The trained model is persisted into the project settings under
``inflammation_model`` so reopening the folder re-applies it without
re-training.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update
import dash_bootstrap_components as dbc

from glia.config import ALL_FEATURES
from glia.features import save_features_df
from glia.inflammation_index import (
    DEFAULT_MAX_SUBSET_SIZE,
    DEFAULT_MIN_SUBSET_SIZE,
    InflammationModel,
    apply as ii_apply,
    per_animal_mean,
    train as ii_train,
)
from glia_dash import server_state
from glia_dash.components import alert, empty_state, metric_card


_ID_LIKE = {"ID", "roi_tag", "cell_index", "Cluster",
            "morphology_label", "center_source", "image"}


def _theme_palette(theme: str) -> dict:
    if (theme or "light") == "light":
        return dict(paper="#ffffff", plot="#f6f8fa", fg="#1f2328",
                    grid="#d0d7de",
                    colorway=["#0969da", "#1a7f37", "#9a6700", "#cf222e",
                              "#8250df", "#bf3989", "#0550ae", "#116329"])
    return dict(paper="#1c2128", plot="#0f1117", fg="#e6edf3",
                grid="#2d333b",
                colorway=["#58a6ff", "#3fb950", "#d29922", "#f85149",
                          "#bc8cff", "#f778ba", "#79c0ff", "#56d364"])


def _metadata_candidates(df: pd.DataFrame) -> list[str]:
    """Categorical-ish columns usable as the treatment column."""
    out: list[str] = []
    for c in df.columns:
        if c in _ID_LIKE:
            continue
        if c in ALL_FEATURES:
            continue
        if c.startswith("PC"):
            continue
        if c in ("inflammation_index",):
            continue
        if df[c].dtype.kind in "fi":
            continue
        out.append(c)
    return out


def _candidate_features(df: pd.DataFrame) -> list[str]:
    """The morphology features available for the inflammation axis."""
    return [c for c in ALL_FEATURES if c in df.columns
            and df[c].dtype.kind in "fi"]


# ── Layout ──────────────────────────────────────────────────────────


def layout(sid: str | None) -> html.Div:
    state = server_state.get_session(sid)
    from glia.metadata import ensure_metadata_joined
    ensure_metadata_joined(state)
    df = state.features_df

    if df is None or len(df) == 0:
        return html.Div([
            html.H4("Inflammation Index", style={"marginBottom": "16px"}),
            alert("No features in memory — run the Features tab first.",
                  variant="warning"),
        ])

    meta_cols = _metadata_candidates(df)
    feats = _candidate_features(df)
    if not meta_cols:
        return html.Div([
            html.H4("Inflammation Index", style={"marginBottom": "16px"}),
            alert("No metadata columns in the features table. Add at "
                  "least one (e.g. Treatment) in the Prepare tab.",
                  variant="warning"),
        ])

    default_treatment = (state.factor_cols[0]
                         if state.factor_cols
                         and state.factor_cols[0] in meta_cols
                         else meta_cols[0])

    # Recall any previously trained model, if any.
    model_dict = state.extra.get("inflammation_model")
    saved_model = None
    if isinstance(model_dict, dict):
        try:
            saved_model = InflammationModel.from_dict(model_dict)
        except Exception:
            saved_model = None

    # Pre-render the results panel from the saved model so the page
    # doesn't go blank after a tab switch. We re-score the in-memory
    # dataframe so the histograms / strip plot reflect the current
    # cell pool (in case features were re-extracted).
    initial_output: html.Div | None = None
    if saved_model is not None:
        try:
            scored = ii_apply(df, saved_model)
            # Don't write to features.csv here — that already happened
            # at training time. Just keep the live dataframe in sync.
            state.features_df = scored
            initial_output = _render_results(
                scored, saved_model, saved_model.treatment_col,
                dt=None, theme="light", state=state,
                extra_top=alert(
                    "Showing the previously trained model. "
                    "Retrain to update.",
                    variant="info",
                ),
            )
        except Exception:
            initial_output = None

    return html.Div([
        html.H4("Inflammation Index", style={"marginBottom": "8px"}),
        html.Div(
            "A supervised PCA-based morphology score. Pick two "
            "reference groups — typically a 'resting' control and an "
            "'activated' comparator. Forward-greedy feature selection "
            "finds the subset whose PC1 best separates them; PC1 is "
            "oriented so the comparator group scores higher. The "
            "trained axis is then applied to every cell in the "
            "project, including cells from groups the model never saw — "
            "their position on the axis tells you where they sit on "
            "the activation continuum. Animal-level means of this "
            "score are the recommended unit of statistical comparison.",
            style={"fontSize": "0.85rem",
                   "color": "var(--ned-text-muted)",
                   "marginBottom": "16px"},
        ),

        html.Div([
            metric_card("Cells", f"{len(df):,}", accent=True),
            metric_card("Candidate features", str(len(feats)),
                        accent=True),
            metric_card("Trained?", "yes" if saved_model else "no",
                        accent=bool(saved_model)),
            metric_card("Training AUC",
                        f"{saved_model.train_auc:.3f}" if saved_model
                        else "—", accent=bool(saved_model)),
        ], style={"display": "grid",
                  "gridTemplateColumns": "repeat(4, 1fr)",
                  "gap": "12px", "marginBottom": "16px"}),

        html.Div([
            html.Div([
                html.Label("Group by",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dcc.Dropdown(
                    id="ii-treatment-col",
                    options=[{"label": c, "value": c} for c in meta_cols],
                    value=default_treatment,
                    clearable=False,
                    style={"width": "220px"},
                ),
            ], style={"marginRight": "20px"}),
            html.Div([
                html.Label("Control group",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dcc.Dropdown(id="ii-control-group",
                             options=[], value=None, clearable=False,
                             style={"width": "200px"}),
            ], style={"marginRight": "20px"}),
            html.Div([
                html.Label("Comparator group",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dcc.Dropdown(id="ii-comparator-group",
                             options=[], value=None, clearable=False,
                             style={"width": "200px"}),
            ], style={"marginRight": "20px"}),
            html.Div([
                html.Label("Max features",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dcc.Input(id="ii-max-size", type="number",
                          min=DEFAULT_MIN_SUBSET_SIZE,
                          max=min(DEFAULT_MAX_SUBSET_SIZE, len(feats)),
                          step=1,
                          value=min(DEFAULT_MAX_SUBSET_SIZE, len(feats)),
                          style={"width": "100px"}),
            ], style={"marginRight": "20px"}),
            dbc.Button("Train + apply", id="ii-train",
                       className="btn-ned-primary"),
            html.Span(
                "Training also applies the score to every cell and "
                "writes it into features.csv, so Explore and Stats see "
                "'inflammation_index' immediately.",
                style={"fontSize": "0.75rem",
                       "color": "var(--ned-text-muted)",
                       "marginLeft": "12px"},
            ),
        ], style={"display": "flex", "alignItems": "flex-end",
                  "flexWrap": "wrap", "gap": "8px",
                  "marginBottom": "16px"}),

        dcc.Loading(
            type="default",
            children=html.Div(id="ii-output",
                              children=initial_output,
                              style={"marginTop": "12px",
                                     "minHeight": "40px"}),
        ),

        dcc.Store(id="ii-train-tick"),
    ])


# ── Callbacks ───────────────────────────────────────────────────────


@callback(
    Output("ii-control-group", "options"),
    Output("ii-comparator-group", "options"),
    Output("ii-control-group", "value"),
    Output("ii-comparator-group", "value"),
    Input("ii-treatment-col", "value"),
    State("session-id", "data"),
)
def populate_groups(treatment_col, sid):
    """Fill the control / comparator dropdowns from the chosen
    treatment column's unique values."""
    state = server_state.get_session(sid)
    df = state.features_df
    if df is None or not treatment_col or treatment_col not in df.columns:
        return [], [], None, None
    vals = sorted({str(v) for v in df[treatment_col].dropna().unique()
                   if str(v).strip()})
    opts = [{"label": v, "value": v} for v in vals]
    # If a model is already trained on this column, default to its
    # control/comparator.
    model_dict = state.extra.get("inflammation_model") or {}
    ctrl = model_dict.get("control_group") if (
        model_dict.get("treatment_col") == treatment_col
    ) else None
    comp = model_dict.get("comparator_group") if (
        model_dict.get("treatment_col") == treatment_col
    ) else None
    if ctrl not in vals:
        ctrl = vals[0] if vals else None
    if comp not in vals or comp == ctrl:
        comp = (vals[1] if len(vals) > 1
                else (vals[0] if vals else None))
    return opts, opts, ctrl, comp


@callback(
    Output("ii-output", "children"),
    Output("ii-train-tick", "data"),
    Input("ii-train", "n_clicks"),
    State("ii-treatment-col", "value"),
    State("ii-control-group", "value"),
    State("ii-comparator-group", "value"),
    State("ii-max-size", "value"),
    State("theme-store", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_train(n_clicks, treatment_col, ctrl, comp, max_size, theme, sid):
    if not n_clicks:
        return no_update, no_update
    state = server_state.get_session(sid)
    df = state.features_df
    if df is None:
        return alert("No features in memory.", variant="warning"), no_update
    if not treatment_col or not ctrl or not comp:
        return alert("Pick a treatment column and two groups first.",
                     variant="warning"), no_update
    if ctrl == comp:
        return alert("Control and comparator must differ.",
                     variant="warning"), no_update

    feats = _candidate_features(df)
    if not feats:
        return alert("No numeric morphology features in the dataframe.",
                     variant="danger"), no_update

    t0 = time.time()
    try:
        model = ii_train(
            df, treatment_col, ctrl, comp,
            candidate_features=feats,
            max_size=int(max_size or DEFAULT_MAX_SUBSET_SIZE),
        )
    except Exception as e:
        return alert(f"Training failed: {e}",
                     variant="danger"), no_update
    dt = time.time() - t0

    # Persist the model on state.extra (round-trips into project JSON
    # via save_project_settings → _PROJECT_EXTRA_FIELDS).
    state.extra["inflammation_model"] = model.to_dict()

    # Apply immediately so Explore / Stats see the new
    # ``inflammation_index`` column without an extra click. Persist
    # both the features dataframe and the project settings.
    try:
        scored = ii_apply(df, model)
        state.features_df = scored
        save_features_df(state.project_dir, scored)
    except Exception as e:
        scored = df.copy()
        scored["inflammation_index"] = np.nan
        from glia_dash.components import alert as _alert
        # Show the result panel with a warning so the user understands
        # why downstream tabs may not yet see the column.
        return _alert(
            f"Trained, but applying the score failed: {e}",
            variant="warning",
        ), no_update

    try:
        from glia.settings import save_project_settings
        save_project_settings(state.project_dir, state)
    except Exception:
        pass

    return _render_results(
        scored, model, treatment_col, dt, theme, state,
        extra_top=alert(
            f"✓ Trained and applied to {len(scored):,} cells "
            f"(AUC = {model.train_auc:.3f}). 'inflammation_index' is "
            f"now in Explore / Stats / features.csv.",
            variant="success",
        ),
    ), n_clicks


# ── Figure builders ─────────────────────────────────────────────────


def _build_hist(scored: pd.DataFrame, treatment_col: str,
                model: InflammationModel, theme: str) -> go.Figure:
    p = _theme_palette(theme)
    fig = go.Figure()
    # Iterate the groups in a stable order, with control / comparator
    # plotted first so they sit behind the held-out groups visually.
    groups = list(scored[treatment_col].dropna().unique())
    ordered = ([model.control_group, model.comparator_group]
               + [g for g in groups
                  if g not in (model.control_group,
                               model.comparator_group)])
    for i, g in enumerate(ordered):
        sub = scored[scored[treatment_col] == g]
        if sub.empty:
            continue
        color = p["colorway"][i % len(p["colorway"])]
        opacity = 0.75 if g in (model.control_group,
                                model.comparator_group) else 0.55
        fig.add_trace(go.Histogram(
            x=sub["inflammation_index"].dropna(),
            name=str(g) + (" (control)" if g == model.control_group
                           else " (comparator)"
                           if g == model.comparator_group else ""),
            opacity=opacity,
            marker_color=color,
            nbinsx=40,
        ))
    fig.update_layout(
        barmode="overlay",
        margin=dict(l=48, r=20, t=24, b=36), height=380,
        paper_bgcolor=p["paper"], plot_bgcolor=p["plot"],
        font=dict(color=p["fg"], family="IBM Plex Sans, sans-serif",
                  size=11),
        xaxis=dict(title="Inflammation Index", gridcolor=p["grid"]),
        yaxis=dict(title="Cell count", gridcolor=p["grid"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _build_animal_strip(scored: pd.DataFrame, treatment_col: str,
                        animal_col: str, theme: str) -> go.Figure | None:
    p = _theme_palette(theme)
    if animal_col not in scored.columns or not animal_col:
        return None
    animal_df = per_animal_mean(scored, animal_col)
    if treatment_col not in animal_df.columns:
        return None

    fig = go.Figure()
    groups = list(animal_df[treatment_col].dropna().unique())
    for i, g in enumerate(groups):
        sub = animal_df[animal_df[treatment_col] == g]
        color = p["colorway"][i % len(p["colorway"])]
        fig.add_trace(go.Box(
            y=sub["inflammation_index"],
            name=str(g),
            boxpoints="all", jitter=0.4, pointpos=0,
            marker=dict(color=color, size=8,
                        line=dict(color=p["paper"], width=1)),
            line=dict(color=color),
            fillcolor="rgba(0,0,0,0)",
        ))
    fig.update_layout(
        margin=dict(l=48, r=20, t=24, b=36), height=380,
        paper_bgcolor=p["paper"], plot_bgcolor=p["plot"],
        font=dict(color=p["fg"], family="IBM Plex Sans, sans-serif",
                  size=11),
        xaxis=dict(title=treatment_col, gridcolor=p["grid"]),
        yaxis=dict(title="Animal-level mean II", gridcolor=p["grid"]),
        showlegend=False,
    )
    return fig


def _build_feature_table(model: InflammationModel,
                         theme: str) -> html.Div:
    """Show the selected feature subset and PC1 loadings."""
    p = _theme_palette(theme)
    rows = []
    for feat, loading in zip(model.features, model.pc1_loadings):
        # Loadings live in the orientation-corrected space — flip so
        # the sign tells the user the direction of "more inflamed".
        signed = loading * model.score_orientation
        direction = "↑ inflamed" if signed > 0 else "↓ inflamed"
        rows.append(html.Tr([
            html.Td(feat),
            html.Td(f"{signed:+.3f}", style={"fontFamily": "monospace"}),
            html.Td(direction,
                    style={"color": ("#3fb950" if signed > 0
                                     else "#f85149"),
                           "fontSize": "0.82rem"}),
        ]))
    return html.Div([
        html.Table([
            html.Thead(html.Tr([
                html.Th("Feature"),
                html.Th("PC1 loading"),
                html.Th("Direction"),
            ])),
            html.Tbody(rows),
        ], style={"width": "100%",
                  "fontSize": "0.85rem",
                  "borderCollapse": "collapse"}),
    ])


def _render_results(scored: pd.DataFrame, model: InflammationModel,
                    treatment_col: str, dt: float | None,
                    theme: str, state,
                    extra_top: html.Div | None = None) -> html.Div:
    summary = []
    if extra_top is not None:
        summary.append(extra_top)
    timing = f" in {dt:.1f}s" if dt is not None else ""
    summary.append(alert(
        f"Trained on {model.n_train_cells:,} cells from "
        f"'{model.control_group}' (n={(scored[treatment_col] == model.control_group).sum():,}) "
        f"vs '{model.comparator_group}' (n={(scored[treatment_col] == model.comparator_group).sum():,})"
        f"{timing}. Forward-selected {len(model.features)} features; "
        f"PC1 captures {model.pc1_explained*100:.1f}% of variance; "
        f"training AUC = {model.train_auc:.3f}.",
        variant="info",
    ))

    fig_hist = _build_hist(scored, treatment_col, model, theme)
    fig_strip = _build_animal_strip(scored, treatment_col,
                                    state.animal_id_col, theme)

    blocks = [
        *summary,
        html.H6("Score distribution by group (cell-level)",
                style={"fontSize": "0.92rem",
                       "marginTop": "12px",
                       "marginBottom": "4px",
                       "color": "var(--ned-text)"}),
        dcc.Graph(figure=fig_hist,
                  config={"displayModeBar": False,
                          "toImageButtonOptions": {
                              "format": "png", "scale": 2,
                              "filename": "inflammation_index_cells"}}),
    ]
    if fig_strip is not None:
        blocks += [
            html.H6(f"Animal-level means (grouped by "
                    f"{state.animal_id_col})",
                    style={"fontSize": "0.92rem",
                           "marginTop": "16px",
                           "marginBottom": "4px",
                           "color": "var(--ned-text)"}),
            html.Div(
                "Animal-level mean is the unit of statistical "
                "comparison. Use Stats with 'inflammation_index' as "
                "the feature to run an ANOVA / posthoc test across "
                "any subset of groups.",
                style={"fontSize": "0.78rem",
                       "color": "var(--ned-text-muted)",
                       "marginBottom": "4px"},
            ),
            dcc.Graph(figure=fig_strip,
                      config={"displayModeBar": False,
                              "toImageButtonOptions": {
                                  "format": "png", "scale": 2,
                                  "filename":
                                      "inflammation_index_animals"}}),
        ]
    else:
        blocks.append(html.Div(
            "Set an Animal ID column in the Stats tab to also see "
            "animal-level means here.",
            style={"fontSize": "0.78rem",
                   "color": "var(--ned-text-muted)",
                   "marginTop": "8px"},
        ))

    blocks += [
        html.H6("Selected features (PC1 loadings)",
                style={"fontSize": "0.92rem",
                       "marginTop": "16px",
                       "marginBottom": "4px",
                       "color": "var(--ned-text)"}),
        html.Div(
            "Positive loading → larger value pushes the cell toward "
            "the comparator end of the axis. Forward selection added "
            "features greedily by AUC; later additions usually have "
            "smaller marginal contributions.",
            style={"fontSize": "0.78rem",
                   "color": "var(--ned-text-muted)",
                   "marginBottom": "6px"},
        ),
        _build_feature_table(model, theme),
    ]
    return html.Div(blocks)
