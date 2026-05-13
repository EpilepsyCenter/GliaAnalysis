"""Stats page — animal-level ANOVA + posthocs for features and cluster %s."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import (
    Input, Output, State, callback, ctx, dash_table, dcc, html, no_update,
)
import dash_bootstrap_components as dbc

from glia.config import ALL_FEATURES, ASTROCYTE_FEATURES, DERIVED_FEATURES
from glia.stats import (
    aggregate_to_animal,
    cluster_percentages_per_animal,
    diagnostics as run_diagnostics,
    feature_test,
    group_sizes,
    mixed_effects_features,
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
# roi_tag is intentionally allowed through as a factor — when the user
# selects it, on_run dispatches to the mixed-effects (LMM) path which
# treats it as a within-subject factor with Animal as random intercept.
_ID_LIKE_COLS = {"ID", "cell_index", "Cluster", "morphology_label"}


def _feature_columns(df: pd.DataFrame, mode: str = "microglia") -> list[str]:
    base = (ASTROCYTE_FEATURES if (mode or "").lower() == "astrocyte"
            else ALL_FEATURES)
    pool = list(dict.fromkeys(base + DERIVED_FEATURES))
    return [c for c in pool if c in df.columns]


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


_PATSY_TERM = __import__("re").compile(r"^([^\[]+)\[T\.(.+)\]$")


def _humanize_effect(effect: str, references: dict[str, str]) -> str:
    """Translate Patsy 'Treatment[T.SV2A]:roi_tag[T.DG]' into plain English.

    'Treatment[T.SV2A]'                  -> 'SV2A vs Control'
    'roi_tag[T.DG]'                      -> 'DG vs CA1'
    'Treatment[T.SV2A]:roi_tag[T.DG]'    -> 'SV2A:DG interaction'
    """
    s = str(effect)
    if s.lower() == "intercept":
        return "(reference cell)"
    if s.startswith("("):
        return s
    parts = s.split(":")
    pieces = []
    main_terms = []  # "SV2A vs Control" pieces, used for "main effect"
    inter_terms = []  # "SV2A" / "DG" — used for "X:Y interaction"
    for part in parts:
        m = _PATSY_TERM.match(part.strip())
        if not m:
            pieces.append(part)
            continue
        col, lev = m.group(1), m.group(2)
        ref = references.get(col, "ref")
        main_terms.append(f"{lev} vs {ref}")
        inter_terms.append(lev)
    if len(parts) == 1:
        return main_terms[0] if main_terms else s
    return ":".join(inter_terms) + " interaction"


def _humanize_fixed_df(fixed_df: pd.DataFrame,
                       references: dict[str, str]) -> pd.DataFrame:
    """Return a copy of ``fixed_df`` with a human-readable effect column."""
    if fixed_df is None or fixed_df.empty:
        return fixed_df
    out = fixed_df.copy()
    out.insert(
        1, "effect (plain)",
        out["effect"].astype(str).map(
            lambda e: _humanize_effect(e, references)),
    )
    return out


def _round_for_display(df: pd.DataFrame, sig: int = 4) -> pd.DataFrame:
    """Pre-round numeric columns to ``sig`` significant figures.

    Dash's table format specifier is only honored when the column dtype
    is f/i. Object-dtype columns (which happen any time pingouin or
    statsmodels mixes None/NaN with floats) bypass the formatter and
    render with full repr precision. Pre-rounding sidesteps that.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if s.dtype.kind in "fi":
            out[c] = s.round(sig).astype(float)
            continue
        # Object column: coerce, round, convert back. Leave columns
        # that are mostly non-numeric (string labels) alone.
        coerced = pd.to_numeric(s, errors="coerce")
        non_null = coerced.notna().sum()
        if non_null and non_null >= s.notna().sum() * 0.5:
            rounded = coerced.round(sig)
            # Preserve any original strings that weren't numbers
            # (e.g. BF10 = '>1000' in pingouin output).
            keep_orig = coerced.isna() & s.notna()
            out[c] = rounded.where(~keep_orig, s)
    return out


_CORRECTED_P_COLS = ("p_adj", "p_corr", "p-corr", "p-adjust")
_RAW_P_COLS = ("p", "p_unc", "p-unc", "p_raw")


def _significance_styles(rows_df: pd.DataFrame) -> list[dict]:
    """Two-tier significance highlighting for a stats DataTable.

    Red row when the corrected p-value (p_adj / p-corr) < 0.05 — the
    'real' hit. Orange row when only the raw p-value is significant
    (corrected p ≥ 0.05) — a 'multiplicity-burned' candidate worth a
    look but not a headline result. Falls back to raw-p red highlight
    when no corrected column exists (e.g. user picked padjust=none).
    """
    if rows_df is None or rows_df.empty:
        return []
    cols = set(rows_df.columns.astype(str))
    corrected = next((c for c in _CORRECTED_P_COLS if c in cols), None)
    raw = next((c for c in _RAW_P_COLS if c in cols), None)
    rules: list[dict] = []
    red_bg = "rgba(207, 34, 46, 0.14)"
    orange_bg = "rgba(217, 119, 6, 0.13)"
    if corrected:
        rules.append({
            "if": {"filter_query": f"{{{corrected}}} < 0.05"},
            "backgroundColor": red_bg,
            "fontWeight": "600",
        })
        if raw and raw != corrected:
            # Orange only when corrected DIDN'T survive but raw is < .05.
            rules.append({
                "if": {"filter_query":
                       f"{{{corrected}}} >= 0.05 && {{{raw}}} < 0.05"},
                "backgroundColor": orange_bg,
            })
    elif raw:
        rules.append({
            "if": {"filter_query": f"{{{raw}}} < 0.05"},
            "backgroundColor": red_bg,
            "fontWeight": "600",
        })
    return rules


def _datatable(rows_df, table_id, highlight_significance: bool = True):
    if rows_df is None or rows_df.empty:
        return html.Div("(no rows)", style={"fontSize": "0.82rem",
                                            "color": "var(--ned-text-muted)",
                                            "fontStyle": "italic"})
    rows_df = _round_for_display(rows_df, sig=4)
    cols = []
    for c in rows_df.columns:
        col = {"name": str(c), "id": str(c)}
        if rows_df[c].dtype.kind in "fi":
            col["type"] = "numeric"
            col["format"] = {"specifier": ".4g"}
        cols.append(col)
    sig_rules = (_significance_styles(rows_df)
                 if highlight_significance else [])
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
        # Significance rules come AFTER the base zebra-striping so they
        # paint on top; otherwise alternating-row backgrounds win.
        style_data_conditional=(_TABLE_STYLE_DATA_CONDITIONAL
                                + sig_rules),
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
    feats = _feature_columns(df, getattr(state, "mode", "microglia"))

    # Honor the user's last selections (per-session, also persisted
    # into the project JSON so they survive a folder reopen). Fall
    # back to module defaults only on first use.
    saved_features = state.extra.get("stats_features")
    if isinstance(saved_features, list) and saved_features:
        default_dv = [c for c in saved_features if c in feats]
    else:
        default_dv = []
    if not default_dv:
        default_dv = [c for c in _DEFAULT_FEATURES if c in feats] or feats[:4]

    saved_method = state.extra.get("stats_method")
    method_default = (saved_method if saved_method in
                      ("anova", "welch", "kruskal") else "anova")
    saved_padjust = state.extra.get("stats_padjust")
    padjust_default = (saved_padjust if saved_padjust in _PADJUST_OPTIONS
                       else "holm")
    saved_aggregate = state.extra.get("stats_aggregate")
    aggregate_default = (bool(saved_aggregate)
                         if saved_aggregate is not None else True)

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

        # Quick-fill presets. Picks animal/factors/method/padjust in
        # one click. ROI-aware LMM auto-triggers the mixed-effects
        # dispatch by including roi_tag in factors.
        html.Div([
            dbc.DropdownMenu(
                label="Presets",
                color="secondary",
                size="sm",
                children=[
                    dbc.DropdownMenuItem(
                        "Animal-level ANOVA + Holm",
                        id="stats-preset-animal-anova",
                        n_clicks=0,
                    ),
                    dbc.DropdownMenuItem(
                        "ROI-aware LMM (Treatment × ROI)",
                        id="stats-preset-roi-lmm",
                        n_clicks=0,
                    ),
                ],
            ),
            html.Span(
                "Fills Animal / Factors / Method / Correction in one "
                "click. You still hit Run.",
                style={"marginLeft": "10px",
                       "fontSize": "0.78rem",
                       "color": "var(--ned-text-muted)"},
            ),
        ], style={"display": "flex", "alignItems": "center",
                  "marginBottom": "10px"}),

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
                             value=method_default, clearable=False,
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
                             value=padjust_default, clearable=False,
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
                           value=aggregate_default,
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


# ── LMM path (ROI-aware) ────────────────────────────────────────────


def _theme_palette(theme: str) -> dict:
    if (theme or "light") == "light":
        return dict(paper="#ffffff", plot="#f6f8fa", fg="#1f2328",
                    grid="#d0d7de", muted="#656d76",
                    pos="#cf222e", neg="#0969da", ns="#8c959f")
    return dict(paper="#1c2128", plot="#0f1117", fg="#e6edf3",
                grid="#2d333b", muted="#8b949e",
                pos="#f85149", neg="#58a6ff", ns="#6e7681")


def _lmm_table_for_plots(fixed_df: pd.DataFrame) -> pd.DataFrame:
    """Strip Intercept / failed / skipped rows and add derived columns.

    Forest and volcano both plot fixed-effect coefficients only — the
    Intercept row is mechanically the grand mean and isn't an effect.
    Failed/skipped rows have NaN estimates and would just clutter.
    """
    if fixed_df is None or fixed_df.empty:
        return fixed_df
    out = fixed_df.copy()
    out = out[out["effect"].astype(str).str.lower() != "intercept"]
    out = out[~out["effect"].astype(str).str.startswith("(")]
    out = out.dropna(subset=["estimate", "se", "p"])
    if out.empty:
        return out
    out["ci_lo"] = out["estimate"] - 1.96 * out["se"]
    out["ci_hi"] = out["estimate"] + 1.96 * out["se"]
    out["neglog10p"] = -np.log10(out["p"].clip(lower=1e-300))
    # Signed magnitude — sign of estimate, height = -log10(p).
    # Used as the forest plot color so the table reads as a heat map.
    out["signed_neglog10p"] = np.sign(out["estimate"]) * out["neglog10p"]
    out["is_interaction"] = out["effect"].astype(str).str.contains(":")
    return out


def _build_forest(fixed_df: pd.DataFrame, theme: str,
                  references: dict[str, str] | None = None) -> go.Figure:
    p = _theme_palette(theme)
    df = _lmm_table_for_plots(fixed_df)
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor=p["paper"], plot_bgcolor=p["plot"])
        return fig

    # Order rows so the largest (signed) effects float to the top of
    # each feature group — readers scan top-down for the strongest
    # signal. Group by measure to keep features visually together.
    df = df.assign(_abs=df["estimate"].abs())
    df = df.sort_values(["measure", "_abs"], ascending=[True, True])
    refs = references or {}
    labels = [f"{m} :: {_humanize_effect(e, refs)}"
              for m, e in zip(df["measure"], df["effect"])]

    sig_threshold = -np.log10(0.05)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["estimate"], y=labels,
        error_x=dict(type="data", symmetric=False,
                     array=df["ci_hi"] - df["estimate"],
                     arrayminus=df["estimate"] - df["ci_lo"],
                     thickness=1.4, width=4,
                     color=p["muted"]),
        mode="markers",
        marker=dict(
            size=8,
            color=df["signed_neglog10p"],
            colorscale=[[0.0, p["neg"]], [0.5, p["ns"]], [1.0, p["pos"]]],
            cmid=0,
            cmin=-max(3.0, float(df["neglog10p"].max())),
            cmax=max(3.0, float(df["neglog10p"].max())),
            colorbar=dict(
                title=dict(text="signed<br>−log₁₀ p", side="right",
                           font=dict(size=10)),
                thickness=10, len=0.65,
            ),
            line=dict(color=p["fg"], width=0.5),
        ),
        customdata=np.stack([
            df["p"].values, df["neglog10p"].values, df["se"].values,
        ], axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "estimate = %{x:.4g} ± %{customdata[2]:.3g} SE<br>"
            "p = %{customdata[0]:.3g} "
            "(−log₁₀ p = %{customdata[1]:.2f})<extra></extra>"),
        showlegend=False,
    ))
    fig.add_vline(x=0, line=dict(color=p["muted"], width=1, dash="dot"))

    # Height scales with row count so labels never overlap.
    height = max(220, 22 * len(df) + 80)
    fig.update_layout(
        margin=dict(l=240, r=40, t=18, b=40),
        height=height,
        paper_bgcolor=p["paper"], plot_bgcolor=p["plot"],
        font=dict(color=p["fg"], family="IBM Plex Sans, sans-serif",
                  size=10),
        xaxis=dict(title="coefficient (95% CI)",
                   gridcolor=p["grid"], zerolinecolor=p["grid"]),
        yaxis=dict(automargin=True, gridcolor=p["grid"]),
    )
    return fig


def _normalize_pairwise(post_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize pingouin's pairwise_tests output for plotting.

    pingouin's column set differs by method (T for parametric, U-val
    for Kruskal; p-corr only when padjust != 'none'). We pull out a
    consistent ``estimate`` (Hedges' g — the comparable effect size),
    a ``p`` and ``p_corr``, and a ``label`` like 'Contrast: A vs B'.
    """
    if post_df is None or post_df.empty:
        return pd.DataFrame()
    out = post_df.copy()
    cols = set(out.columns)
    # Label: prefer "Contrast: A vs B"; fall back to "A vs B" or Source.
    if {"A", "B"} <= cols:
        a = out["A"].astype(str)
        b = out["B"].astype(str)
        if "Contrast" in cols:
            out["label"] = (out["Contrast"].astype(str) + ": "
                            + a + " vs " + b)
            out["is_interaction"] = (out["Contrast"].astype(str)
                                     .str.contains(r"\*", regex=True))
        else:
            out["label"] = a + " vs " + b
            out["is_interaction"] = False
    elif "Source" in cols:
        out["label"] = out["Source"].astype(str)
        out["is_interaction"] = (out["Source"].astype(str)
                                 .str.contains(r"\*|:", regex=True))
    else:
        out["label"] = "(contrast)"
        out["is_interaction"] = False
    # Effect size: Hedges' g (parametric path) else fall back to T/U
    # converted to nothing — just leave at NaN if hedges absent.
    out["estimate"] = (out["hedges"] if "hedges" in cols
                       else (out["T"] if "T" in cols
                             else out.get("U-val", np.nan)))
    out["estimate"] = pd.to_numeric(out["estimate"], errors="coerce")
    raw_col = next((c for c in ("p_unc", "p-unc", "p")
                    if c in cols), None)
    corr_col = next((c for c in ("p_corr", "p-corr", "p_adjust")
                     if c in cols), None)
    raw = (out[raw_col] if raw_col
           else pd.Series(np.nan, index=out.index))
    corr = (out[corr_col] if corr_col else raw)
    out["p"] = pd.to_numeric(raw, errors="coerce")
    out["p_corr"] = pd.to_numeric(corr, errors="coerce")
    out["neglog10p"] = -np.log10(out["p_corr"].clip(lower=1e-300))
    out = out.dropna(subset=["estimate", "p_corr"])
    return out


def _build_pairwise_volcano(post_df: pd.DataFrame, theme: str) -> go.Figure:
    p = _theme_palette(theme)
    df = _normalize_pairwise(post_df)
    fig = go.Figure()
    if df.empty:
        fig.update_layout(paper_bgcolor=p["paper"], plot_bgcolor=p["plot"])
        return fig
    sig_threshold = -np.log10(0.05)
    for label, sub, color, symbol in [
        ("main effect", df[~df["is_interaction"]], p["neg"], "circle"),
        ("interaction", df[df["is_interaction"]], p["pos"], "diamond"),
    ]:
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["estimate"], y=sub["neglog10p"],
            mode="markers",
            marker=dict(size=9, color=color, symbol=symbol,
                        line=dict(color=p["fg"], width=0.6),
                        opacity=0.85),
            name=label,
            customdata=np.stack([
                sub["measure"].astype(str),
                sub["label"].astype(str),
                sub["p_corr"].values,
            ], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                "Hedges g = %{x:.3g}<br>"
                "p (corrected) = %{customdata[2]:.3g}"
                "<extra></extra>"),
        ))
    fig.add_hline(y=sig_threshold,
                  line=dict(color=p["muted"], width=1, dash="dot"),
                  annotation=dict(text="p_corr = 0.05",
                                  font=dict(size=9),
                                  xanchor="left", yanchor="bottom"),
                  annotation_position="top left")
    fig.add_vline(x=0, line=dict(color=p["muted"], width=1, dash="dot"))
    fig.update_layout(
        margin=dict(l=56, r=16, t=20, b=44), height=380,
        paper_bgcolor=p["paper"], plot_bgcolor=p["plot"],
        font=dict(color=p["fg"], family="IBM Plex Sans, sans-serif",
                  size=11),
        xaxis=dict(title="Hedges' g (effect size)",
                   gridcolor=p["grid"], zerolinecolor=p["grid"]),
        yaxis=dict(title="−log₁₀ p_corr",
                   gridcolor=p["grid"], zerolinecolor=p["grid"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=10)),
    )
    return fig


def _build_pairwise_forest(post_df: pd.DataFrame, theme: str) -> go.Figure:
    """Dot plot of Hedges' g per (feature, contrast), colored by p_corr.

    pingouin doesn't expose a usable SE for hedges in the pairwise
    output, so we don't draw 95% CI bars here — the color encodes
    significance instead. Rows sorted by |g| within each feature.
    """
    p = _theme_palette(theme)
    df = _normalize_pairwise(post_df)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor=p["paper"], plot_bgcolor=p["plot"])
        return fig
    df = df.assign(_abs=df["estimate"].abs())
    df = df.sort_values(["measure", "_abs"], ascending=[True, True])
    labels = [f"{m} :: {lbl}"
              for m, lbl in zip(df["measure"], df["label"])]
    df["signed_neglog10p"] = np.sign(df["estimate"]) * df["neglog10p"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["estimate"], y=labels,
        mode="markers",
        marker=dict(
            size=9,
            color=df["signed_neglog10p"],
            colorscale=[[0.0, p["neg"]], [0.5, p["ns"]], [1.0, p["pos"]]],
            cmid=0,
            cmin=-max(3.0, float(df["neglog10p"].max())),
            cmax=max(3.0, float(df["neglog10p"].max())),
            colorbar=dict(title=dict(text="signed<br>−log₁₀ p_corr",
                                     side="right",
                                     font=dict(size=10)),
                          thickness=10, len=0.65),
            line=dict(color=p["fg"], width=0.5),
        ),
        customdata=np.stack([df["p"].values,
                             df["p_corr"].values], axis=-1),
        hovertemplate=("<b>%{y}</b><br>"
                       "Hedges g = %{x:.3g}<br>"
                       "p = %{customdata[0]:.3g}, "
                       "p_corr = %{customdata[1]:.3g}"
                       "<extra></extra>"),
        showlegend=False,
    ))
    fig.add_vline(x=0, line=dict(color=p["muted"], width=1, dash="dot"))
    height = max(220, 22 * len(df) + 80)
    fig.update_layout(
        margin=dict(l=260, r=40, t=18, b=40),
        height=height,
        paper_bgcolor=p["paper"], plot_bgcolor=p["plot"],
        font=dict(color=p["fg"], family="IBM Plex Sans, sans-serif",
                  size=10),
        xaxis=dict(title="Hedges' g", gridcolor=p["grid"],
                   zerolinecolor=p["grid"]),
        yaxis=dict(automargin=True, gridcolor=p["grid"]),
    )
    return fig


def _build_volcano(fixed_df: pd.DataFrame, theme: str) -> go.Figure:
    p = _theme_palette(theme)
    df = _lmm_table_for_plots(fixed_df)
    fig = go.Figure()
    if df is None or df.empty:
        fig.update_layout(paper_bgcolor=p["paper"], plot_bgcolor=p["plot"])
        return fig

    sig_threshold = -np.log10(0.05)
    for label, sub, color, symbol in [
        ("main effect", df[~df["is_interaction"]], p["neg"], "circle"),
        ("interaction", df[df["is_interaction"]], p["pos"], "diamond"),
    ]:
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["estimate"], y=sub["neglog10p"],
            mode="markers",
            marker=dict(size=9, color=color, symbol=symbol,
                        line=dict(color=p["fg"], width=0.6),
                        opacity=0.85),
            name=label,
            customdata=np.stack([sub["measure"].astype(str),
                                 sub["effect"].astype(str),
                                 sub["p"].values], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                "estimate = %{x:.4g}<br>"
                "p = %{customdata[2]:.3g}<extra></extra>"),
        ))
    fig.add_hline(y=sig_threshold,
                  line=dict(color=p["muted"], width=1, dash="dot"),
                  annotation=dict(text="p = 0.05", font=dict(size=9),
                                  xanchor="left", yanchor="bottom"),
                  annotation_position="top left")
    fig.add_vline(x=0, line=dict(color=p["muted"], width=1, dash="dot"))

    fig.update_layout(
        margin=dict(l=56, r=16, t=20, b=44), height=380,
        paper_bgcolor=p["paper"], plot_bgcolor=p["plot"],
        font=dict(color=p["fg"], family="IBM Plex Sans, sans-serif",
                  size=11),
        xaxis=dict(title="coefficient (effect size)",
                   gridcolor=p["grid"], zerolinecolor=p["grid"]),
        yaxis=dict(title="−log₁₀ p (Wald)",
                   gridcolor=p["grid"], zerolinecolor=p["grid"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0,
                    font=dict(size=10)),
    )
    return fig


def _formula_panel(formula: str, n_obs: int, n_animals: int) -> html.Div:
    return html.Div([
        html.Div("Model", style={"fontSize": "0.72rem",
                                 "color": "var(--ned-text-muted)",
                                 "textTransform": "uppercase",
                                 "letterSpacing": "0.5px",
                                 "marginBottom": "4px"}),
        html.Pre(formula,
                 style={"fontSize": "0.82rem",
                        "fontFamily": "ui-monospace, SFMono-Regular, "
                                      "Menlo, monospace",
                        "padding": "8px 10px",
                        "background": "var(--ned-surface)",
                        "border": "1px solid var(--ned-border)",
                        "borderRadius": "6px",
                        "margin": "0 0 6px 0",
                        "color": "var(--ned-text)"}),
        html.Div(f"{n_obs} observations · {n_animals} animals · REML",
                 style={"fontSize": "0.74rem",
                        "color": "var(--ned-text-muted)"}),
    ], style={"marginBottom": "12px"})


def _run_lmm(work, features, factors, animal, padjust, theme="light"):
    """Mixed-effects branch — Treatment × ROI fixed, Animal random."""
    between = [c for c in factors if c != "roi_tag"]

    # Group sizes on the (animal × roi) frame so the chips reflect the
    # LMM's actual unit of analysis (animals per treatment × roi cell),
    # not cells.
    from glia.stats import aggregate_to_animal_roi
    agg = aggregate_to_animal_roi(work, animal, features, between,
                                  roi_col="roi_tag")
    sizes = group_sizes(agg, factors)
    min_n = int(sizes["n"].min()) if len(sizes) else 0
    sizes_panel = _group_sizes_panel(sizes, min_n)

    if min_n < 3:
        return html.Div([
            sizes_panel,
            alert(
                f"At least one (factor × ROI) cell has only n = {min_n} "
                "animals (< 3). LMM can fit but pairwise contrasts and "
                "Wald p-values are unreliable below that. Add more "
                "animals to the under-powered ROI(s) or drop the ROI "
                "from the factor picker to collapse across regions.",
                variant="danger",
            ),
        ])

    try:
        fixed_df, post_df, info = mixed_effects_features(
            work, features, animal, between,
            within_col="roi_tag", padjust=padjust,
        )
    except Exception as e:
        return alert(f"Mixed-effects fit failed: {e}",
                     variant="danger")

    re_df = info.get("random_effects", pd.DataFrame())
    contrasts_df = info.get("contrasts", pd.DataFrame())
    references = info.get("references", {})

    # Reference-level banner — these are the levels every Patsy
    # 'X[T.level]' coefficient is measured *against*. Without this,
    # users can't read the fixed-effect table at all.
    ref_chips = []
    for col, lev in references.items():
        ref_chips.append(html.Span([
            html.Span(f"{col} = ",
                      style={"color": "var(--ned-text-muted)"}),
            html.Span(str(lev),
                      style={"fontWeight": "600",
                             "color": "var(--ned-text)"}),
        ], style={"padding": "2px 8px",
                  "border": "1px solid var(--ned-border)",
                  "borderRadius": "10px",
                  "background": "var(--ned-surface)",
                  "marginRight": "6px",
                  "fontSize": "0.78rem"}))
    references_panel = html.Div([
        html.Div("Reference levels",
                 style={"fontSize": "0.72rem",
                        "color": "var(--ned-text-muted)",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.5px",
                        "marginBottom": "4px"}),
        html.Div(ref_chips, style={"display": "flex",
                                   "flexWrap": "wrap",
                                   "gap": "4px"}),
        html.Div(
            "Every coefficient below is measured *against* these "
            "levels. A row 'SV2A vs Control' is exactly that — the "
            "difference, evaluated in the reference ROI.",
            style={"fontSize": "0.74rem",
                   "color": "var(--ned-text-muted)",
                   "marginTop": "4px"}),
    ], style={"marginBottom": "12px"}) if ref_chips else None

    # Per-ROI contrasts panel — this is the "what does it mean" answer.
    within_col = info.get("within_col", "roi_tag")
    contrasts_panel = None
    if not contrasts_df.empty:
        contrasts_panel = html.Div([
            html.H6(f"Per-{within_col} Treatment contrasts (LMM Wald)",
                    style={"fontSize": "0.92rem",
                           "marginTop": "8px",
                           "marginBottom": "4px",
                           "color": "var(--ned-text)"}),
            html.Div(
                "Each row asks one direct question: 'In <this ROI>, "
                "does <treatment A> differ from <treatment B>?'. "
                "Estimates and standard errors come from the fitted "
                "LMM (so they benefit from the model's pooling across "
                "ROIs), not from independent t-tests. p_adj uses the "
                "post-hoc correction chosen in the dropdown above.",
                style={"fontSize": "0.78rem",
                       "color": "var(--ned-text-muted)",
                       "marginBottom": "6px"}),
            _datatable(contrasts_df, "stats-lmm-contrasts"),
        ])

    return html.Div([
        alert(f"✓ Stats run · unit: {info['n_obs']} (animal × ROI) "
              f"rows from {info['n_animals']} animals · "
              "model: linear mixed-effects (REML) · "
              f"posthoc: {padjust}.",
              variant="success"),
        _formula_panel(info["formula"],
                       info["n_obs"], info["n_animals"]),
        sizes_panel,
        alert(
            "ROI is treated as a within-subject factor with Animal as "
            "a random intercept. Animals missing some ROI levels are "
            "kept (LMM tolerates unbalanced designs); pingouin's "
            "mixed_anova would drop them. Shapiro-Wilk / Levene "
            "diagnostics are skipped here — LMM assumes normal "
            "residuals at the (animal × ROI) level, not raw cells.",
            variant="info",
        ),

        references_panel,
        contrasts_panel,

        html.Div([
            html.Div([
                html.H6("Forest plot — coefficients (95% CI)",
                        style={"fontSize": "0.92rem",
                               "marginTop": "12px",
                               "marginBottom": "2px",
                               "color": "var(--ned-text)"}),
                html.Div(
                    "Each row is one fixed-effect term per feature. "
                    "Bars are 95% Wald CIs; color is signed −log₁₀ p "
                    "(red = positive, blue = negative). Intercepts "
                    "and failed fits are dropped from the plot.",
                    style={"fontSize": "0.78rem",
                           "color": "var(--ned-text-muted)",
                           "marginBottom": "4px"},
                ),
                dcc.Graph(
                    figure=_build_forest(fixed_df, theme,
                                         references=references),
                    config={"displayModeBar": False,
                            "toImageButtonOptions": {
                                "format": "png", "scale": 2,
                                "filename": "lmm_forest"}},
                ),
            ], style={"flex": "1.4", "minWidth": "0"}),
            html.Div([
                html.H6("Volcano — effect size vs significance",
                        style={"fontSize": "0.92rem",
                               "marginTop": "12px",
                               "marginBottom": "2px",
                               "color": "var(--ned-text)"}),
                html.Div(
                    "Points above the dotted line clear p < 0.05 "
                    "(uncorrected). Circles = main effects, diamonds "
                    "= interactions.",
                    style={"fontSize": "0.78rem",
                           "color": "var(--ned-text-muted)",
                           "marginBottom": "4px"},
                ),
                dcc.Graph(
                    figure=_build_volcano(fixed_df, theme),
                    config={"displayModeBar": False,
                            "toImageButtonOptions": {
                                "format": "png", "scale": 2,
                                "filename": "lmm_volcano"}},
                ),
            ], style={"flex": "1", "minWidth": "0"}),
        ], style={"display": "flex", "gap": "20px",
                  "alignItems": "flex-start",
                  "marginTop": "8px"}),

        html.H6("Fixed effects (Wald)",
                style={"fontSize": "0.92rem",
                       "marginTop": "16px",
                       "marginBottom": "4px",
                       "color": "var(--ned-text)"}),
        html.Div(
            "'effect (plain)' translates the Patsy term in the next "
            "column. Each coefficient is a difference from the "
            "reference levels above.",
            style={"fontSize": "0.78rem",
                   "color": "var(--ned-text-muted)",
                   "marginBottom": "6px"}),
        _datatable(_humanize_fixed_df(fixed_df, references),
                   "stats-lmm-fixed"),

        html.H6("Random effect — Animal intercept SD",
                style={"fontSize": "0.92rem",
                       "marginTop": "16px",
                       "marginBottom": "4px",
                       "color": "var(--ned-text)"}),
        html.Div(
            "Larger SD = more between-animal variability the LMM is "
            "absorbing. A failed convergence flag means the Wald "
            "p-values for that feature should be discounted.",
            style={"fontSize": "0.78rem",
                   "color": "var(--ned-text-muted)",
                   "marginBottom": "6px"}),
        _datatable(re_df, "stats-lmm-re"),

        html.H6("Pairwise contrasts (on animal × ROI means)",
                style={"fontSize": "0.92rem",
                       "marginTop": "16px",
                       "marginBottom": "4px",
                       "color": "var(--ned-text)"}),
        _datatable(post_df, "stats-lmm-post"),
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
    State("theme-store", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_run(n_clicks, animal, factors, features, aggregate, method, padjust,
           theme, sid):
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
    # Remember the rest of the UI selections so the Stats tab opens
    # in the same configuration after a tab switch / folder reopen.
    state.extra["stats_features"] = list(features or [])
    state.extra["stats_method"] = str(method or "anova")
    state.extra["stats_padjust"] = str(padjust or "holm")
    state.extra["stats_aggregate"] = bool(aggregate)
    try:
        from glia.settings import save_project_settings
        save_project_settings(state.project_dir, state)
    except Exception:
        pass

    feats = _feature_columns(df, getattr(state, "mode", "microglia"))
    # Clean inf so aggregation/mean doesn't propagate them.
    work = df.copy()
    work[feats] = work[feats].replace([np.inf, -np.inf], np.nan)

    # ── ROI-aware LMM path ────────────────────────────────────────────
    # When the user picks roi_tag as a factor we switch to a linear
    # mixed-effects model with Animal as a random intercept. This
    # handles unbalanced ROI sets per animal natively — pingouin's
    # mixed_anova would drop any animal that's missing a ROI level.
    if "roi_tag" in factors and "roi_tag" in work.columns:
        return _run_lmm(work, features, factors, animal, padjust, theme)

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

        html.Div([
            html.Div([
                html.H6("Forest — pairwise effect sizes",
                        style={"fontSize": "0.92rem",
                               "marginTop": "12px",
                               "marginBottom": "2px",
                               "color": "var(--ned-text)"}),
                html.Div(
                    "One dot per (feature, contrast). x = Hedges' g, "
                    "color = signed −log₁₀ p_corr. Sorted by |g| "
                    "within each feature. No CI bars — pingouin "
                    "doesn't expose a usable SE for hedges here.",
                    style={"fontSize": "0.78rem",
                           "color": "var(--ned-text-muted)",
                           "marginBottom": "4px"}),
                dcc.Graph(
                    figure=_build_pairwise_forest(post_df, theme),
                    config={"displayModeBar": False,
                            "toImageButtonOptions": {
                                "format": "png", "scale": 2,
                                "filename":
                                    f"{method}_pairwise_forest"}},
                ),
            ], style={"flex": "1.4", "minWidth": "0"}),
            html.Div([
                html.H6("Volcano — effect vs significance",
                        style={"fontSize": "0.92rem",
                               "marginTop": "12px",
                               "marginBottom": "2px",
                               "color": "var(--ned-text)"}),
                html.Div(
                    "Points above the dotted line clear p_corr < "
                    "0.05. Circles = main-effect contrasts, "
                    "diamonds = interaction contrasts.",
                    style={"fontSize": "0.78rem",
                           "color": "var(--ned-text-muted)",
                           "marginBottom": "4px"}),
                dcc.Graph(
                    figure=_build_pairwise_volcano(post_df, theme),
                    config={"displayModeBar": False,
                            "toImageButtonOptions": {
                                "format": "png", "scale": 2,
                                "filename":
                                    f"{method}_pairwise_volcano"}},
                ),
            ], style={"flex": "1", "minWidth": "0"}),
        ], style={"display": "flex", "gap": "20px",
                  "alignItems": "flex-start",
                  "marginTop": "8px"}),

        html.H6("Feature posthocs",
                style={"fontSize": "0.92rem",
                       "marginTop": "16px",
                       "marginBottom": "4px",
                       "color": "var(--ned-text)"}),
        _datatable(post_df, "stats-feature-post"),

        html.H6("Diagnostics — Shapiro-Wilk per group",
                style={"fontSize": "0.92rem",
                       "marginTop": "16px",
                       "marginBottom": "4px",
                       "color": "var(--ned-text)"}),
        _datatable(diag["normality"], "stats-normality",
                   highlight_significance=False),

        html.H6("Diagnostics — Levene equal-variance test",
                style={"fontSize": "0.92rem",
                       "marginTop": "12px",
                       "marginBottom": "4px",
                       "color": "var(--ned-text)"}),
        _datatable(diag["levene"], "stats-levene",
                   highlight_significance=False),

        cluster_block,
    ])


@callback(
    Output("stats-animal-id", "value", allow_duplicate=True),
    Output("stats-factors", "value", allow_duplicate=True),
    Output("stats-method", "value", allow_duplicate=True),
    Output("stats-padjust", "value", allow_duplicate=True),
    Output("stats-aggregate", "value", allow_duplicate=True),
    Output("stats-features", "value", allow_duplicate=True),
    Input("stats-preset-animal-anova", "n_clicks"),
    Input("stats-preset-roi-lmm", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def apply_stats_preset(_n1, _n2, sid):
    """Fill all controls from a preset. Each preset is a sensible
    baseline; users still tweak from there and hit Run. Outputs stay
    at no_update when their target column isn't present in the
    current features table (e.g. roi_tag missing in a flat dataset).
    """
    state = server_state.get_session(sid)
    df = state.features_df
    if df is None or len(df) == 0:
        return (no_update,) * 6
    feats = _feature_columns(df, getattr(state, "mode", "microglia"))
    core = [c for c in _DEFAULT_FEATURES if c in feats] or feats[:4]

    # Pick the animal-ID column: respect a user-set state value if it
    # still exists in the frame, otherwise try the canonical "Animal".
    animal_val = (state.animal_id_col
                  if state.animal_id_col in df.columns
                  else ("Animal" if "Animal" in df.columns
                        else no_update))

    pid = ctx.triggered_id
    if pid == "stats-preset-animal-anova":
        factors = ["Treatment"] if "Treatment" in df.columns else []
        return animal_val, factors, "anova", "holm", True, core
    if pid == "stats-preset-roi-lmm":
        factors = []
        if "Treatment" in df.columns:
            factors.append("Treatment")
        if "roi_tag" in df.columns:
            factors.append("roi_tag")
        return animal_val, factors, "anova", "holm", True, core
    return (no_update,) * 6
