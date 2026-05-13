"""Explore page — feature distributions and pairwise correlations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

from glia.config import ALL_FEATURES, ASTROCYTE_FEATURES, DERIVED_FEATURES
from glia.transforms import apply_transform
from glia_dash import server_state
from glia_dash.components import alert, metric_card


_TRANSFORMS = ("none", "log", "zscore", "minmax")
_DEFAULT_DIST_FEATURES = [
    "Area", "Perimeter", "Circularity",
    "# of branches", "# of end point voxels", "Maximum branch length",
]
# Columns to hide from the grouping pickers. roi_tag stays available
# (Group by / Split by) so users can dissect Treatment × ROI effects.
_GROUPING_EXCLUDE = ("ID", "cell_index")


def _feature_columns(df: pd.DataFrame, mode: str = "microglia") -> list[str]:
    """Numeric columns to expose as features in the Explore tab.

    Microglia mode: the 36 per-cell morphology features.
    Astrocyte mode: the 9 per-(image, ROI) network metrics.

    Both modes also include ``DERIVED_FEATURES`` (e.g. the trained
    inflammation_index) when present.
    """
    base = (ASTROCYTE_FEATURES if (mode or "").lower() == "astrocyte"
            else ALL_FEATURES)
    # Use dict.fromkeys to dedupe while preserving order — some names
    # appear in both ALL_FEATURES and ASTROCYTE_FEATURES (e.g.
    # "# of branches"). Dedupe is cheap and keeps the picker tidy.
    pool = list(dict.fromkeys(base + DERIVED_FEATURES))
    return [c for c in pool if c in df.columns]


def _grouping_options(df: pd.DataFrame, mode: str = "microglia") -> list[dict]:
    """Possible categorical columns to group distributions by."""
    feats = set(_feature_columns(df, mode))
    drop = set(_GROUPING_EXCLUDE) | feats
    cats = [c for c in df.columns
            if c not in drop and df[c].dtype.kind not in "fi"]
    return [{"label": "(no grouping)", "value": ""}] + [
        {"label": c, "value": c} for c in cats
    ]


# ── Layout ───────────────────────────────────────────────────────────


def layout(sid: str | None) -> html.Div:
    state = server_state.get_session(sid)
    # Re-broadcast the Prepare-tab metadata in case the user edited it
    # after extracting features (or the persisted features.csv predates
    # the metadata).
    from glia.metadata import ensure_metadata_joined
    ensure_metadata_joined(state)
    df = state.features_df

    if df is None or len(df) == 0:
        return html.Div([
            html.H4("Explore", style={"marginBottom": "16px"}),
            alert("No features in memory — run the Features tab first.",
                  variant="warning"),
        ])

    mode = getattr(state, "mode", "microglia") or "microglia"
    feats = _feature_columns(df, mode)
    grouping_opts = _grouping_options(df, mode)
    valid_group_values = {opt["value"] for opt in grouping_opts}

    # Remember the user's last selections per session (and per project
    # if reopened from disk). Hardcoded defaults are only used the
    # first time the tab is touched on a fresh project.
    saved_group = state.extra.get("explore_group")
    if saved_group is None or saved_group not in valid_group_values:
        default_group = (grouping_opts[1]["value"]
                         if len(grouping_opts) > 1 else "")
    else:
        default_group = saved_group

    saved_split = state.extra.get("explore_split")
    if saved_split is None or saved_split not in valid_group_values:
        # Auto-suggest roi_tag as the secondary when it's available and
        # not already the primary. Users can clear it back to "(none)".
        default_split = ("roi_tag"
                         if "roi_tag" in valid_group_values
                         and "roi_tag" != default_group
                         else "")
    else:
        default_split = saved_split if saved_split != default_group else ""

    saved_features = state.extra.get("explore_features")
    if isinstance(saved_features, list):
        default_dist = [c for c in saved_features if c in feats]
    else:
        default_dist = []
    if not default_dist:
        default_dist = [c for c in _DEFAULT_DIST_FEATURES if c in feats]
    if not default_dist:
        default_dist = feats[:6]

    saved_show_points = state.extra.get("explore_show_points")
    show_points_default = (bool(saved_show_points)
                           if saved_show_points is not None else True)

    return html.Div([
        html.H4("Explore", style={"marginBottom": "8px"}),
        html.Div(
            "Inspect per-feature distributions (optionally split by an "
            "experimental factor) and the Spearman correlation matrix of "
            "every numeric feature extracted.",
            style={"fontSize": "0.85rem",
                   "color": "var(--ned-text-muted)",
                   "marginBottom": "16px"},
        ),

        html.Div([
            metric_card("Cells", f"{len(df):,}", accent=True),
            metric_card("Numeric features", str(len(feats)), accent=True),
            metric_card("Available groupings",
                        str(max(0, len(grouping_opts) - 1))),
        ], style={"display": "grid",
                  "gridTemplateColumns": "repeat(3, 1fr)",
                  "gap": "12px", "marginBottom": "16px"}),

        html.Div([
            html.Div([
                html.Label("Transform",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dbc.RadioItems(id="explore-transform",
                               options=[{"label": t, "value": t}
                                        for t in _TRANSFORMS],
                               value=state.transform or "none",
                               inline=True),
            ], style={"marginRight": "32px"}),

            html.Div([
                html.Label("Group by",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dcc.Dropdown(id="explore-group",
                             options=grouping_opts,
                             value=default_group, clearable=False,
                             style={"width": "180px"}),
            ], style={"marginRight": "16px"}),

            html.Div([
                html.Label("Split by",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dcc.Dropdown(id="explore-split",
                             options=grouping_opts,
                             value=default_split, clearable=True,
                             placeholder="(none)",
                             style={"width": "180px"}),
            ], style={"marginRight": "32px"}),

            html.Div([
                html.Label("Features to plot",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dcc.Dropdown(id="explore-features",
                             options=[{"label": c, "value": c}
                                      for c in feats],
                             value=default_dist, multi=True,
                             style={"minWidth": "420px"}),
            ], style={"flex": "1", "marginRight": "24px"}),

            html.Div([
                html.Label("Show points",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dbc.Switch(id="explore-show-points",
                           label="overlay individual cells",
                           value=show_points_default,
                           style={"fontSize": "0.82rem"}),
            ]),
        ], style={"display": "flex",
                  "alignItems": "flex-end",
                  "gap": "0", "marginBottom": "16px"}),

        dcc.Loading(
            children=[
                html.H6("Distributions",
                        style={"fontSize": "0.92rem",
                               "marginTop": "8px",
                               "marginBottom": "4px",
                               "color": "var(--ned-text)"}),
                dcc.Graph(id="explore-dist",
                          config={"displayModeBar": False},
                          style={"height": "440px"}),

                html.H6("Spearman correlations",
                        style={"fontSize": "0.92rem",
                               "marginTop": "16px",
                               "marginBottom": "4px",
                               "color": "var(--ned-text)"}),
                dcc.Graph(id="explore-corr",
                          config={"displayModeBar": False},
                          style={"height": "560px"}),
            ],
            type="default",
        ),
    ])


# ── Plotly figure builders ───────────────────────────────────────────


def _theme_palette(theme: str) -> dict:
    if (theme or "light") == "light":
        return dict(paper="#ffffff", plot="#f6f8fa", fg="#1f2328",
                    grid="#d0d7de", colorway=[
                        "#0969da", "#1a7f37", "#9a6700", "#cf222e",
                        "#8250df", "#bf3989", "#0550ae", "#116329",
                    ])
    return dict(paper="#1c2128", plot="#0f1117", fg="#e6edf3",
                grid="#2d333b", colorway=[
                    "#58a6ff", "#3fb950", "#d29922", "#f85149",
                    "#bc8cff", "#f778ba", "#79c0ff", "#56d364",
                ])


def _build_distributions(df: pd.DataFrame, features: list[str],
                         group: str, theme: str,
                         show_points: bool = True,
                         split: str = "",
                         animal_col: str = "") -> go.Figure:
    """Bar + SEM per (feature × group), publication-style.

    One subplot per feature. With one factor: one bar per primary group.
    With two factors: bars cluster by ``split`` (x-position) and color
    by ``group`` (legend shows primary levels once). Error bars are SEM.

    Point overlay:
      - single factor: per-cell jitter (legacy behavior).
      - two factors: aggregate to (animal × split × group) means first,
        so the cloud is per-animal — avoids pseudoreplicating cells.
        Falls back to per-cell if ``animal_col`` is empty.
    """
    if not features:
        return go.Figure()
    p = _theme_palette(theme)
    rows = max(1, (len(features) + 2) // 3)
    cols = min(3, len(features))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=features,
                        vertical_spacing=0.16, horizontal_spacing=0.07)

    has_primary = bool(group) and group in df.columns
    has_split = (bool(split) and split in df.columns
                 and split != group and split != animal_col)

    if has_primary:
        primaries = sorted(
            pd.Series(df[group]).fillna("(missing)").unique(),
            key=lambda x: str(x))
    else:
        primaries = [None]
    primary_names = ["all" if g is None else str(g) for g in primaries]
    colors = [p["colorway"][i % len(p["colorway"])]
              for i in range(len(primary_names))]

    rng = np.random.default_rng(0)

    if has_split:
        splits = sorted(
            pd.Series(df[split]).fillna("(missing)").unique(),
            key=lambda x: str(x))
        split_names = [str(s) for s in splits]
        split_positions = list(range(len(split_names)))
        n_primary = max(1, len(primaries))
        bar_w = 0.8 / n_primary
        primary_offsets = [
            (i - (n_primary - 1) / 2) * bar_w for i in range(n_primary)
        ]
    legend_seen: set[str] = set()

    # Per-animal aggregation key (deduped) — used only when has_split
    # so the SEM is animal-level, not cell-level.
    use_agg = (has_split and bool(animal_col)
               and animal_col in df.columns
               and animal_col not in (group, split))

    for i, feat in enumerate(features):
        r, c = i // cols + 1, i % cols + 1

        # ── Two-factor case: clustered bars (primary = color, split = x).
        if has_split:
            agg_df = None
            if use_agg:
                keys = [animal_col]
                if has_primary and group not in keys:
                    keys.append(group)
                if split not in keys:
                    keys.append(split)
                agg_df = (df[[*keys, feat]]
                          .dropna(subset=[feat])
                          .groupby(keys, dropna=False)[feat]
                          .mean()
                          .reset_index())

            for pi, prim in enumerate(primaries):
                means, sems, ns = [], [], []
                per_bar_values = []
                for sv in splits:
                    if agg_df is not None:
                        sub = agg_df
                        if has_primary:
                            sub = sub[sub[group].fillna("(missing)")
                                      == prim]
                        sub = sub[sub[split].fillna("(missing)") == sv]
                        v = sub[feat].to_numpy()
                    else:
                        sub = df
                        if has_primary:
                            sub = sub[sub[group].fillna("(missing)")
                                      == prim]
                        sub = sub[sub[split].fillna("(missing)") == sv]
                        v = sub[feat].dropna().to_numpy()
                    per_bar_values.append(v)
                    if len(v):
                        m = float(np.mean(v))
                        s = (float(np.std(v, ddof=1) / np.sqrt(len(v)))
                             if len(v) > 1 else 0.0)
                        means.append(m if np.isfinite(m) else 0.0)
                        sems.append(s if np.isfinite(s) else 0.0)
                        ns.append(int(len(v)))
                    else:
                        means.append(0.0)
                        sems.append(0.0)
                        ns.append(0)

                xs = [sp + primary_offsets[pi] for sp in split_positions]
                legend_key = primary_names[pi]
                show_in_legend = (has_primary
                                  and legend_key not in legend_seen)
                if show_in_legend:
                    legend_seen.add(legend_key)
                unit_label = ("animals"
                              if agg_df is not None else "cells")

                fig.add_trace(go.Bar(
                    x=xs, y=means,
                    error_y=dict(type="data", array=sems, visible=True,
                                 color=p["fg"], thickness=1.4, width=5),
                    marker=dict(color=colors[pi],
                                line=dict(color=p["fg"], width=0.5)),
                    name=legend_key,
                    legendgroup=legend_key,
                    showlegend=bool(show_in_legend),
                    customdata=[(legend_key, sn, n, unit_label)
                                for sn, n in zip(split_names, ns)],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b> · %{customdata[1]}"
                        "<br>mean = %{y:.3g}"
                        "<br>n = %{customdata[2]} %{customdata[3]}"
                        "<extra></extra>"),
                    width=bar_w * 0.92,
                ), row=r, col=c)

                if show_points:
                    for sj, vals in enumerate(per_bar_values):
                        if len(vals) == 0:
                            continue
                        # Drop non-finite values before scatter — plotly
                        # tolerates NaN but `inf` ruins the axis range.
                        vals = vals[np.isfinite(vals)]
                        if len(vals) == 0:
                            continue
                        jitter = rng.normal(0, bar_w * 0.18, len(vals))
                        fig.add_trace(go.Scatter(
                            x=[xs[sj] + j for j in jitter],
                            y=vals,
                            mode="markers",
                            marker=dict(color=p["fg"], size=3,
                                        opacity=0.32,
                                        line=dict(width=0)),
                            showlegend=False,
                            hoverinfo="skip",
                        ), row=r, col=c)

            fig.update_xaxes(tickmode="array",
                             tickvals=split_positions,
                             ticktext=split_names,
                             range=[-0.6, len(split_positions) - 0.4],
                             row=r, col=c)
            continue

        # ── Single-factor case: legacy layout — one bar per primary
        # level, multi-color via per-bar marker.color list, primary
        # names as x-axis ticks.
        positions = list(range(len(primaries)))
        means, sems, ns = [], [], []
        per_bar_values = []
        for prim in primaries:
            if has_primary:
                sub = df[df[group].fillna("(missing)") == prim]
            else:
                sub = df
            v = sub[feat].dropna().to_numpy()
            per_bar_values.append(v)
            if len(v):
                m = float(np.mean(v))
                s = (float(np.std(v, ddof=1) / np.sqrt(len(v)))
                     if len(v) > 1 else 0.0)
                means.append(m if np.isfinite(m) else 0.0)
                sems.append(s if np.isfinite(s) else 0.0)
                ns.append(int(len(v)))
            else:
                means.append(0.0)
                sems.append(0.0)
                ns.append(0)
        fig.add_trace(go.Bar(
            x=positions, y=means,
            error_y=dict(type="data", array=sems, visible=True,
                         color=p["fg"], thickness=1.4, width=5),
            marker=dict(color=colors,
                        line=dict(color=p["fg"], width=0.5)),
            showlegend=False,
            customdata=list(zip(primary_names, ns)),
            hovertemplate=("<b>%{customdata[0]}</b><br>"
                           "mean = %{y:.3g}<br>"
                           "n = %{customdata[1]}<extra></extra>"),
            width=0.65,
        ), row=r, col=c)

        if show_points:
            for gi, vals in enumerate(per_bar_values):
                if len(vals) == 0:
                    continue
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue
                jitter = rng.normal(0, 0.08, len(vals))
                fig.add_trace(go.Scatter(
                    x=[positions[gi] + j for j in jitter],
                    y=vals,
                    mode="markers",
                    marker=dict(color=p["fg"], size=3, opacity=0.28,
                                line=dict(width=0)),
                    showlegend=False,
                    hoverinfo="skip",
                ), row=r, col=c)

        fig.update_xaxes(tickmode="array", tickvals=positions,
                         ticktext=primary_names,
                         range=[-0.6, len(primary_names) - 0.4],
                         row=r, col=c)

    fig.update_layout(
        margin=dict(l=48, r=20, t=36, b=28),
        paper_bgcolor=p["paper"], plot_bgcolor=p["plot"],
        font=dict(color=p["fg"], family="IBM Plex Sans, sans-serif", size=11),
        bargap=0.18 if has_split else 0.30,
        bargroupgap=0.06,
        barmode="overlay",  # bars already positioned manually
        showlegend=bool(has_primary and has_split),
        legend=dict(orientation="h", yanchor="bottom", y=1.04,
                    xanchor="left", x=0,
                    font=dict(size=10)),
    )
    fig.update_xaxes(showline=True, linecolor=p["grid"],
                     gridcolor=p["grid"])
    fig.update_yaxes(showline=True, linecolor=p["grid"],
                     gridcolor=p["grid"], zerolinecolor=p["grid"])
    return fig


def _build_correlations(df: pd.DataFrame, theme: str,
                        mode: str = "microglia") -> go.Figure:
    p = _theme_palette(theme)
    feats = _feature_columns(df, mode)
    if len(feats) < 2:
        return go.Figure()
    sub = df[feats].copy()
    # Replace ±inf with NaN before .std() — otherwise the variance is
    # ill-defined and pandas emits a RuntimeWarning.
    sub = sub.replace([np.inf, -np.inf], np.nan)
    # Drop fully-NaN / zero-variance columns so Spearman doesn't return NaN
    # rows that would blank out the entire heatmap.
    std = sub.std(ddof=0)
    valid = std.fillna(0) > 0
    sub = sub.loc[:, valid].dropna()
    feats = list(sub.columns)
    if sub.shape[1] < 2 or len(sub) < 3:
        return go.Figure()
    rho, _ = spearmanr(sub.values)
    rho = np.asarray(rho)
    if rho.ndim == 0:
        return go.Figure()
    fig = go.Figure(go.Heatmap(
        z=rho, x=feats, y=feats,
        zmin=-1, zmax=1,
        colorscale="RdBu", reversescale=True,
        colorbar=dict(title="ρ", thickness=12),
        hovertemplate="%{y} vs %{x}<br>ρ = %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(l=140, r=20, t=20, b=140),
        paper_bgcolor=p["paper"], plot_bgcolor=p["plot"],
        font=dict(color=p["fg"], family="IBM Plex Sans, sans-serif",
                  size=10),
    )
    fig.update_xaxes(tickangle=-45, automargin=True, side="bottom",
                     gridcolor=p["grid"])
    fig.update_yaxes(automargin=True, gridcolor=p["grid"])
    return fig


# ── Callbacks ────────────────────────────────────────────────────────


@callback(
    Output("explore-dist", "figure"),
    Output("explore-corr", "figure"),
    Input("explore-transform", "value"),
    Input("explore-group", "value"),
    Input("explore-split", "value"),
    Input("explore-features", "value"),
    Input("explore-show-points", "value"),
    Input("theme-store", "data"),
    State("session-id", "data"),
    prevent_initial_call=False,
)
def update_plots(transform, group, split, features, show_points, theme, sid):
    state = server_state.get_session(sid)
    df_raw = state.features_df
    if df_raw is None or len(df_raw) == 0:
        return go.Figure(), go.Figure()
    state.transform = transform or "none"

    # Drop split if it duplicates the primary group — same factor on
    # both axes is meaningless and would draw N identical bars per
    # cluster.
    if split and split == group:
        split = ""

    # Mirror the user's current selections so re-rendering this page
    # (tab switch, refresh, folder reopen) restores them. Stored on
    # state.extra because they're UI preferences, not analysis params.
    state.extra["explore_features"] = list(features or [])
    state.extra["explore_group"] = group or ""
    state.extra["explore_split"] = split or ""
    state.extra["explore_show_points"] = bool(show_points)
    try:
        from glia.settings import save_project_settings
        save_project_settings(state.project_dir, state)
    except Exception:
        pass

    feats_all = _feature_columns(df_raw, getattr(state, "mode", "microglia"))
    # Replace ±inf with NaN once, up front. Several morphology features
    # (Circularity, Average branch length, etc.) can produce inf when a
    # denominator approaches zero, and any downstream .std() / .mean()
    # on those values both warns and corrupts the resulting figure.
    df_clean = df_raw.copy()
    df_clean[feats_all] = (df_clean[feats_all]
                           .replace([np.inf, -np.inf], np.nan))

    if transform and transform != "none":
        df_dist = apply_transform(df_clean, feats_all, transform)
    else:
        df_dist = df_clean
    # Spearman is rank-based, so transforms don't change the correlation
    # matrix. Always use the cleaned frame.
    try:
        dist_fig = _build_distributions(
            df_dist, list(features or []), group or "",
            theme, show_points=bool(show_points),
            split=split or "",
            animal_col=getattr(state, "animal_id_col", "") or "",
        )
    except Exception as e:
        dist_fig = _error_figure(theme,
                                 f"Distribution plot failed: {e}")
    try:
        corr_fig = _build_correlations(
            df_clean, theme,
            mode=getattr(state, "mode", "microglia"),
        )
    except Exception as e:
        corr_fig = _error_figure(theme,
                                 f"Correlation plot failed: {e}")
    return dist_fig, corr_fig


def _error_figure(theme: str, msg: str) -> go.Figure:
    """Return a non-crashing placeholder figure with the error inline."""
    p = _theme_palette(theme)
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False,
                       font=dict(color=p["fg"], size=12),
                       xref="paper", yref="paper")
    fig.update_layout(paper_bgcolor=p["paper"], plot_bgcolor=p["plot"],
                      xaxis=dict(visible=False),
                      yaxis=dict(visible=False),
                      margin=dict(l=20, r=20, t=20, b=20))
    return fig
