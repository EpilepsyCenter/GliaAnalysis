"""Stats page — animal-level ANOVA + posthocs for features and cluster %s."""

from __future__ import annotations

import numpy as np
import pandas as pd
from dash import (
    Input, Output, State, callback, dash_table, dcc, html, no_update,
)
import dash_bootstrap_components as dbc

from glia.config import ALL_FEATURES
from glia.stats import (
    aggregate_to_animal,
    cluster_percentages_per_animal,
    diagnostics as run_diagnostics,
    feature_test,
    group_sizes,
)
from glia_dash import server_state
from glia_dash.components import alert, metric_card
from glia_dash.pages.features import (
    _TABLE_STYLE_CELL, _TABLE_STYLE_DATA_CONDITIONAL,
    _TABLE_STYLE_HEADER, _TABLE_STYLE_TABLE,
)


_DEFAULT_FEATURES = [
    "Area", "Perimeter", "Circularity",
    "# of branches", "# of end point voxels", "Maximum branch length",
]
_PADJUST_OPTIONS = ("none", "bonf", "sidak", "holm", "fdr_bh")
_ID_LIKE_COLS = {"ID", "roi_tag", "cell_index", "Cluster", "morphology_label"}


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in ALL_FEATURES if c in df.columns]


def _metadata_candidates(df: pd.DataFrame) -> list[str]:
    """Columns that look like metadata (string-y, not numeric features)."""
    out = []
    for c in df.columns:
        if c in _ID_LIKE_COLS:
            continue
        if c in ALL_FEATURES:
            continue
        if c.startswith("PC"):
            continue
        if df[c].dtype.kind in "fi":
            continue
        out.append(c)
    return out


def _group_sizes_panel(sizes_df: pd.DataFrame, min_n: int) -> html.Div:
    """Per-group n with traffic-light styling at n<3 / n<5 / n≥5."""
    def color(n):
        if n < 3:
            return "var(--ned-danger)"
        if n < 5:
            return "var(--ned-warning)"
        return "var(--ned-success)"
    chips = []
    for _, row in sizes_df.iterrows():
        n = int(row["n"])
        label_parts = [str(row[c]) for c in sizes_df.columns if c != "n"]
        label = " · ".join(label_parts) if label_parts else "all"
        chips.append(html.Div([
            html.Div(label,
                     style={"fontSize": "0.72rem",
                            "color": "var(--ned-text-muted)",
                            "textTransform": "uppercase",
                            "letterSpacing": "0.5px"}),
            html.Div([
                html.Span(str(n),
                          style={"fontSize": "1.4rem",
                                 "fontWeight": "700",
                                 "color": color(n)}),
                html.Span(" n",
                          style={"fontSize": "0.78rem",
                                 "color": "var(--ned-text-muted)",
                                 "marginLeft": "4px"}),
            ]),
        ], style={"padding": "10px 14px",
                  "border": "1px solid var(--ned-border)",
                  "borderRadius": "6px",
                  "background": "var(--ned-surface)",
                  "minWidth": "100px"}))
    legend = html.Div([
        html.Span("● n ≥ 5", style={"color": "var(--ned-success)",
                                    "marginRight": "12px"}),
        html.Span("● 3 ≤ n < 5", style={"color": "var(--ned-warning)",
                                        "marginRight": "12px"}),
        html.Span("● n < 3", style={"color": "var(--ned-danger)"}),
    ], style={"fontSize": "0.72rem",
              "marginTop": "6px",
              "color": "var(--ned-text-muted)"})
    return html.Div([
        html.Div("Group sizes",
                 style={"fontSize": "0.72rem",
                        "color": "var(--ned-text-muted)",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.5px",
                        "marginBottom": "6px"}),
        html.Div(chips, style={"display": "flex",
                               "flexWrap": "wrap",
                               "gap": "8px"}),
        legend,
    ], style={"marginBottom": "12px"})


def _datatable(rows_df, table_id):
    if rows_df is None or rows_df.empty:
        return html.Div("(no rows)", style={"fontSize": "0.82rem",
                                            "color": "var(--ned-text-muted)",
                                            "fontStyle": "italic"})
    cols = []
    for c in rows_df.columns:
        col = {"name": str(c), "id": str(c)}
        if rows_df[c].dtype.kind in "fi":
            col["type"] = "numeric"
            col["format"] = {"specifier": ".3g"}
        cols.append(col)
    return dash_table.DataTable(
        id=table_id,
        data=rows_df.astype(object).where(pd.notnull(rows_df), None)
                  .to_dict("records"),
        columns=cols,
        page_size=20,
        page_action="native",
        sort_action="native",
        style_cell=_TABLE_STYLE_CELL,
        style_header=_TABLE_STYLE_HEADER,
        style_table=_TABLE_STYLE_TABLE,
        style_data_conditional=_TABLE_STYLE_DATA_CONDITIONAL,
    )


# ── Layout ──────────────────────────────────────────────────────────


def layout(sid: str | None) -> html.Div:
    state = server_state.get_session(sid)
    from glia.metadata import ensure_metadata_joined
    ensure_metadata_joined(state)
    df = state.features_df
    if df is None or len(df) == 0:
        return html.Div([
            html.H4("Stats", style={"marginBottom": "16px"}),
            alert("No features in memory — run the Features tab first.",
                  variant="warning"),
        ])

    meta_cols = _metadata_candidates(df)
    feats = _feature_columns(df)
    default_dv = [c for c in _DEFAULT_FEATURES if c in feats] or feats[:4]

    # Sensible defaults for animal + factors.
    animal_default = (state.animal_id_col
                      if state.animal_id_col in meta_cols
                      else ("Animal" if "Animal" in meta_cols
                            else (meta_cols[0] if meta_cols else "")))
    factor_default = [c for c in (state.factor_cols or [])
                      if c in meta_cols and c != animal_default]
    if not factor_default:
        if "Condition" in meta_cols and "Condition" != animal_default:
            factor_default = ["Condition"]

    n_animals = (df[animal_default].nunique() if animal_default
                 else 0)
    has_cluster = "Cluster" in df.columns

    return html.Div([
        html.H4("Stats", style={"marginBottom": "8px"}),
        html.Div(
            "Animal-level ANOVA with pairwise posthocs (pingouin). "
            "Aggregate cells → animals first to avoid pseudoreplication, "
            "then run a one- or two-way ANOVA per feature, plus a "
            "separate ANOVA on cluster proportions per animal "
            "(arcsine-sqrt transformed).",
            style={"fontSize": "0.85rem",
                   "color": "var(--ned-text-muted)",
                   "marginBottom": "16px"},
        ),

        html.Div([
            metric_card("Cells", f"{len(df):,}", accent=True),
            metric_card("Animals",
                        f"{n_animals}" if n_animals else "—",
                        accent=(n_animals > 0)),
            metric_card("Clusters",
                        str(int(df["Cluster"].nunique()))
                        if has_cluster else "—",
                        accent=has_cluster),
            metric_card("Available factors", str(len(meta_cols))),
        ], style={"display": "grid",
                  "gridTemplateColumns": "repeat(4, 1fr)",
                  "gap": "12px", "marginBottom": "16px"}),

        html.Div([
            html.Div([
                html.Label("Animal ID column",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dcc.Dropdown(id="stats-animal-id",
                             options=[{"label": c, "value": c}
                                      for c in meta_cols],
                             value=animal_default,
                             clearable=False,
                             style={"width": "200px"}),
            ], style={"marginRight": "24px"}),

            html.Div([
                html.Label("Experimental factors",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dcc.Dropdown(id="stats-factors",
                             options=[{"label": c, "value": c}
                                      for c in meta_cols],
                             value=factor_default, multi=True,
                             style={"minWidth": "320px"}),
            ], style={"marginRight": "24px"}),

            html.Div([
                html.Label("Method",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dcc.Dropdown(id="stats-method",
                             options=[
                                 {"label": "ANOVA (parametric)",
                                  "value": "anova"},
                                 {"label": "Welch ANOVA (unequal var)",
                                  "value": "welch"},
                                 {"label": "Kruskal-Wallis (non-parametric)",
                                  "value": "kruskal"},
                             ],
                             value="anova", clearable=False,
                             style={"width": "220px"}),
            ], style={"marginRight": "24px"}),

            html.Div([
                html.Label("Post-hoc correction",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dcc.Dropdown(id="stats-padjust",
                             options=[{"label": p, "value": p}
                                      for p in _PADJUST_OPTIONS],
                             value="holm", clearable=False,
                             style={"width": "140px"}),
            ]),
        ], style={"display": "flex",
                  "alignItems": "flex-end",
                  "marginBottom": "12px"}),

        html.Div([
            html.Div([
                html.Label("Features to test",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dcc.Dropdown(id="stats-features",
                             options=[{"label": c, "value": c}
                                      for c in feats],
                             value=default_dv, multi=True,
                             style={"minWidth": "420px"}),
            ], style={"flex": "1"}),
            html.Div([
                dbc.Switch(id="stats-aggregate",
                           label="Aggregate to animal level",
                           value=True,
                           style={"fontSize": "0.82rem"}),
            ], style={"marginLeft": "24px"}),
        ], style={"display": "flex",
                  "alignItems": "flex-end",
                  "marginBottom": "12px"}),

        html.Div([
            dbc.Button("Run stats", id="stats-run",
                       className="btn-ned-primary"),
            html.Span(id="stats-run-hint",
                      children="ANOVA + pairwise posthocs per feature.",
                      style={"marginLeft": "12px",
                             "fontSize": "0.78rem",
                             "color": "var(--ned-text-muted)"}),
        ], style={"display": "flex", "alignItems": "center"}),

        dcc.Loading(
            type="default",
            children=html.Div(id="stats-output",
                              style={"marginTop": "16px",
                                     "minHeight": "40px"}),
        ),

        _interpretation_panel(),
    ])


def _interpretation_panel() -> html.Div:
    section_h = {"fontSize": "0.82rem",
                 "fontWeight": "600",
                 "color": "var(--ned-text)",
                 "marginTop": "12px",
                 "marginBottom": "4px"}
    body = {"fontSize": "0.82rem",
            "color": "var(--ned-text-muted)",
            "lineHeight": "1.55"}
    return html.Div(style={"marginTop": "24px",
                           "padding": "16px 20px",
                           "border": "1px solid var(--ned-border)",
                           "borderRadius": "8px",
                           "background": "var(--ned-surface)"},
                    children=[
        html.Div("How to read these tests",
                 style={"fontSize": "0.92rem",
                        "fontWeight": "600",
                        "color": "var(--ned-text)"}),

        html.Div("Animal-level aggregation", style=section_h),
        html.Div("Each cell is not an independent sample — many cells "
                 "come from the same animal. Treating them as N would "
                 "inflate degrees of freedom and dramatically shrink "
                 "p-values (pseudoreplication). The default here "
                 "averages each feature within each animal before "
                 "testing, so the unit of analysis is the animal. "
                 "Cell-level mode is available but use only when you "
                 "really mean to ask a question about cells, not "
                 "treatments.",
                 style=body),

        html.Div("ANOVA table", style=section_h),
        html.Div([
            "One row per (feature, ANOVA term). Key columns: ",
            html.Code("Source", style={"color": "var(--ned-text)"}),
            " = factor or interaction; ",
            html.Code("F", style={"color": "var(--ned-text)"}),
            " = test statistic; ",
            html.Code("p-unc", style={"color": "var(--ned-text)"}),
            " = uncorrected p-value; ",
            html.Code("np2", style={"color": "var(--ned-text)"}),
            " = partial η² (effect size, 0–1). Use ",
            html.Code("p-unc", style={"color": "var(--ned-text)"}),
            " < 0.05 with an eye on effect size; small p but tiny η² "
            "often means a real but biologically minor effect.",
        ], style=body),

        html.Div("Posthoc table", style=section_h),
        html.Div([
            "When ANOVA finds an effect, posthocs tell you which group "
            "pairs differ. Columns: ",
            html.Code("A / B"), " (compared groups), ",
            html.Code("T"), " or ",
            html.Code("U-val"), " (test statistic), ",
            html.Code("p-corr"), " (multiplicity-corrected p), ",
            html.Code("hedges"), " (effect size). Correction method is ",
            "set by the dropdown above (Holm by default — balanced "
            "between Bonferroni's stringency and false discovery rate).",
        ], style=body),

        html.Div("Sample size", style=section_h),
        html.Div([
            "Minimum useful n per group is 3 (below that, variance can't "
            "be estimated and tests are undefined). The recommended floor "
            "for any meaningful inference is ",
            html.B("n ≥ 5 per group"),
            ", and pairwise posthocs really need ",
            html.B("n ≥ 6-8 per group"),
            " to detect typical effects after multiple-comparison "
            "correction. The Group-sizes panel shows each cell's n with "
            "traffic-light coloring — runs are blocked entirely if any "
            "cell has n < 3.",
        ], style=body),

        html.Div("Normality + equal variance", style=section_h),
        html.Div([
            "Classical ANOVA assumes (1) residuals are roughly Gaussian "
            "and (2) groups have similar variance. Shapiro-Wilk (per "
            "group, per feature) tests (1); Levene tests (2). If many "
            "Shapiro p-values are < 0.05 the assumption is violated — "
            "switch the Method dropdown to ", html.B("Kruskal-Wallis"),
            " (non-parametric, no normality assumption). If Levene fails "
            "but normality holds, use ", html.B("Welch's ANOVA"),
            ". Both alternatives are valid only for a single between "
            "factor — multi-factor designs need classical ANOVA.",
        ], style=body),

        html.Div("Cluster proportions", style=section_h),
        html.Div("Cluster percentages per animal are bounded in [0, 1] "
                 "so ANOVA on raw fractions is mis-specified. We "
                 "arcsine-sqrt transform (Φ-transform) to stabilize "
                 "variance, then run the same ANOVA + posthoc machinery. "
                 "Test rows are one cluster × one factor.",
                 style=body),
    ])


# ── Callbacks ───────────────────────────────────────────────────────


@callback(
    Output("stats-output", "children"),
    Input("stats-run", "n_clicks"),
    State("stats-animal-id", "value"),
    State("stats-factors", "value"),
    State("stats-features", "value"),
    State("stats-aggregate", "value"),
    State("stats-method", "value"),
    State("stats-padjust", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_run(n_clicks, animal, factors, features, aggregate, method, padjust,
           sid):
    if not n_clicks:
        return no_update
    state = server_state.get_session(sid)
    df = state.features_df
    if df is None or len(df) == 0:
        return alert("No features in memory.", variant="warning")

    factors = list(factors or [])
    features = list(features or [])
    padjust = padjust or "holm"

    if not factors:
        return alert("Pick at least one experimental factor.",
                     variant="warning")
    if not features:
        return alert("Pick at least one feature.", variant="warning")
    if not animal:
        return alert("Pick the column that identifies each animal.",
                     variant="warning")
    if animal in factors:
        return alert("Animal ID column cannot also be a factor.",
                     variant="warning")

    state.animal_id_col = animal
    state.factor_cols = factors
    try:
        from glia.settings import save_project_settings
        save_project_settings(state.project_dir, state)
    except Exception:
        pass

    feats = _feature_columns(df)
    # Clean inf so aggregation/mean doesn't propagate them.
    work = df.copy()
    work[feats] = work[feats].replace([np.inf, -np.inf], np.nan)

    if aggregate:
        try:
            work = aggregate_to_animal(work, animal, feats, factors)
        except Exception as e:
            return alert(f"Aggregation failed: {e}", variant="danger")
        unit_label = (f"{len(work)} animals "
                      f"({df[animal].nunique()} unique IDs)")
    else:
        unit_label = f"{len(df):,} cells (no aggregation)"

    warnings_div = (alert(
        "Cell-level mode — p-values may be inflated by "
        "pseudoreplication.", variant="warning",
    ) if not aggregate else None)

    method = method or "anova"
    if method in ("welch", "kruskal") and len(factors) > 1:
        return alert(f"{method.title()} only supports a single between "
                     "factor. Switch to ANOVA for multi-factor designs.",
                     variant="warning")

    # ── Group sizes + diagnostics ─────────────────────────────────────
    sizes = group_sizes(work, factors)
    min_n = int(sizes["n"].min()) if len(sizes) else 0
    sizes_panel = _group_sizes_panel(sizes, min_n)

    if min_n < 3:
        return html.Div([
            sizes_panel,
            alert(
                f"At least one group has n = {min_n} (< 3). Statistical "
                "tests need n ≥ 3 per group to compute variance and "
                "estimate p-values; with this dataset they would be "
                "meaningless. Add more animals to the under-powered "
                "group(s) or merge factor levels.",
                variant="danger",
            ),
        ])

    diag = run_diagnostics(work, features, factors)
    non_normal = (
        diag["normality"]
        .query("normal == False")
        if not diag["normality"].empty and "normal" in diag["normality"].columns
        else pd.DataFrame()
    )
    unequal_var = (
        diag["levene"]
        .query("equal_var == False")
        if not diag["levene"].empty and "equal_var" in diag["levene"].columns
        else pd.DataFrame()
    )

    diag_banner = None
    if len(non_normal) and method == "anova":
        diag_banner = alert(
            f"{len(non_normal)} (feature × group) cells fail Shapiro-Wilk "
            "normality (p < 0.05). Classical ANOVA assumes normal "
            "residuals — consider switching the Method dropdown to "
            "Kruskal-Wallis.",
            variant="warning",
        )
    elif len(unequal_var) and method == "anova":
        diag_banner = alert(
            f"{len(unequal_var)} feature(s) fail Levene's equal-variance "
            "test. Consider switching to Welch's ANOVA.",
            variant="warning",
        )

    try:
        anova_df, post_df = feature_test(
            work, features, factors, method=method,
            posthoc=True, padjust=padjust,
        )
    except Exception as e:
        return alert(f"{method.title()} test failed: {e}",
                     variant="danger")

    # Cluster proportions (if available)
    cluster_block = None
    if "Cluster" in df.columns:
        try:
            cp = cluster_percentages_per_animal(df, animal, factors,
                                                cluster_col="Cluster")
            # ANOVA on arcsine_sqrt by (Cluster × factors) — one ANOVA per
            # cluster id.
            rows = []
            posts = []
            for cid, sub in cp.groupby("Cluster"):
                a, p = feature_test(sub, ["arcsine_sqrt"],
                                    factors, method=method,
                                    posthoc=True, padjust=padjust)
                a["Cluster"] = int(cid)
                p["Cluster"] = int(cid)
                rows.append(a)
                posts.append(p)
            cluster_anova = (pd.concat(rows, ignore_index=True)
                             if rows else pd.DataFrame())
            cluster_post = (pd.concat(posts, ignore_index=True)
                            if posts else pd.DataFrame())
            cluster_block = html.Div([
                html.H6("Cluster proportions — ANOVA "
                        "(arcsine-sqrt transformed)",
                        style={"fontSize": "0.92rem",
                               "marginTop": "16px",
                               "marginBottom": "4px",
                               "color": "var(--ned-text)"}),
                _datatable(cluster_anova, "stats-cluster-anova"),
                html.H6("Cluster proportions — posthocs",
                        style={"fontSize": "0.92rem",
                               "marginTop": "12px",
                               "marginBottom": "4px",
                               "color": "var(--ned-text)"}),
                _datatable(cluster_post, "stats-cluster-post"),
            ])
        except Exception as e:
            cluster_block = alert(
                f"Cluster-proportion ANOVA failed: {e}", variant="warning")

    return html.Div([
        alert(f"✓ Stats run · unit: {unit_label} · "
              f"method: {method} · posthoc: {padjust}.",
              variant="success"),
        warnings_div if warnings_div else None,
        sizes_panel,
        diag_banner,

        html.H6(f"Feature test ({method})",
                style={"fontSize": "0.92rem",
                       "marginTop": "12px",
                       "marginBottom": "4px",
                       "color": "var(--ned-text)"}),
        _datatable(anova_df, "stats-feature-anova"),

        html.H6("Feature posthocs",
                style={"fontSize": "0.92rem",
                       "marginTop": "12px",
                       "marginBottom": "4px",
                       "color": "var(--ned-text)"}),
        _datatable(post_df, "stats-feature-post"),

        html.H6("Diagnostics — Shapiro-Wilk per group",
                style={"fontSize": "0.92rem",
                       "marginTop": "16px",
                       "marginBottom": "4px",
                       "color": "var(--ned-text)"}),
        _datatable(diag["normality"], "stats-normality"),

        html.H6("Diagnostics — Levene equal-variance test",
                style={"fontSize": "0.92rem",
                       "marginTop": "12px",
                       "marginBottom": "4px",
                       "color": "var(--ned-text)"}),
        _datatable(diag["levene"], "stats-levene"),

        cluster_block,
    ])
