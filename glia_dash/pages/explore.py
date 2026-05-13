"""Explore page — feature distributions and pairwise correlations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

from glia.config import ALL_FEATURES
from glia.transforms import apply_transform
from glia_dash import server_state
from glia_dash.components import alert, metric_card


_TRANSFORMS = ("none", "log", "zscore", "minmax")
_DEFAULT_DIST_FEATURES = [
    "Area", "Perimeter", "Circularity",
    "# of branches", "# of end point voxels", "Maximum branch length",
]
_METADATA_COLS = ("ID", "roi_tag", "cell_index")


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """The 27 numeric feature columns present in the dataframe."""
    return [c for c in ALL_FEATURES if c in df.columns]


def _grouping_options(df: pd.DataFrame) -> list[dict]:
    """Possible categorical columns to group distributions by."""
    feats = set(_feature_columns(df))
    drop = set(_METADATA_COLS) | feats
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

    feats = _feature_columns(df)
    grouping_opts = _grouping_options(df)
    default_group = grouping_opts[1]["value"] if len(grouping_opts) > 1 else ""
    default_dist = [c for c in _DEFAULT_DIST_FEATURES if c in feats]
    if not default_dist:
        default_dist = feats[:6]

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
                             style={"width": "200px"}),
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
                           value=True,
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
                         show_points: bool = True) -> go.Figure:
    """Bar + SEM per (feature × group), publication-style.

    One subplot per feature, one bar per group, error bars from SEM.
    Optionally overlays individual cell values jittered around each bar.
    """
    if not features:
        return go.Figure()
    p = _theme_palette(theme)
    rows = max(1, (len(features) + 2) // 3)
    cols = min(3, len(features))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=features,
                        vertical_spacing=0.16, horizontal_spacing=0.07)

    if group and group in df.columns:
        groups = sorted(pd.Series(df[group]).fillna("(missing)").unique(),
                        key=lambda x: str(x))
    else:
        groups = [None]
    group_names = ["all" if g is None else str(g) for g in groups]
    colors = [p["colorway"][i % len(p["colorway"])]
              for i in range(len(group_names))]
    rng = np.random.default_rng(0)
    positions = list(range(len(group_names)))

    for i, feat in enumerate(features):
        r, c = i // cols + 1, i % cols + 1
        means: list[float] = []
        sems: list[float] = []
        ns: list[int] = []
        per_group_values: list[np.ndarray] = []
        for g in groups:
            sub_g = df if g is None else df[df[group].fillna("(missing)") == g]
            v = sub_g[feat].dropna().to_numpy()
            per_group_values.append(v)
            if len(v) > 0:
                means.append(float(v.mean()))
                sems.append(float(v.std(ddof=1) / np.sqrt(len(v)))
                            if len(v) > 1 else 0.0)
                ns.append(int(len(v)))
            else:
                means.append(0.0)
                sems.append(0.0)
                ns.append(0)

        # Bars on a numeric axis so the jittered scatter overlay aligns.
        fig.add_trace(go.Bar(
            x=positions, y=means,
            error_y=dict(type="data", array=sems, visible=True,
                         color=p["fg"], thickness=1.4, width=5),
            marker=dict(color=colors,
                        line=dict(color=p["fg"], width=0.5)),
            showlegend=False,
            customdata=list(zip(group_names, ns)),
            hovertemplate=("<b>%{customdata[0]}</b><br>"
                           "mean = %{y:.3g} ± %{error_y.array:.2g} SEM<br>"
                           "n = %{customdata[1]}<extra></extra>"),
            width=0.65,
        ), row=r, col=c)

        if show_points:
            for gi, vals in enumerate(per_group_values):
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
                         ticktext=group_names,
                         range=[-0.6, len(group_names) - 0.4],
                         row=r, col=c)

    fig.update_layout(
        margin=dict(l=48, r=20, t=36, b=28),
        paper_bgcolor=p["paper"], plot_bgcolor=p["plot"],
        font=dict(color=p["fg"], family="IBM Plex Sans, sans-serif", size=11),
        bargap=0.30,
        showlegend=False,
    )
    fig.update_xaxes(showline=True, linecolor=p["grid"],
                     gridcolor=p["grid"])
    fig.update_yaxes(showline=True, linecolor=p["grid"],
                     gridcolor=p["grid"], zerolinecolor=p["grid"])
    return fig


def _build_correlations(df: pd.DataFrame, theme: str) -> go.Figure:
    p = _theme_palette(theme)
    feats = _feature_columns(df)
    if len(feats) < 2:
        return go.Figure()
    sub = df[feats].copy()
    # Drop fully-NaN / zero-variance columns so Spearman doesn't return NaN
    # rows that would blank out the entire heatmap.
    valid = sub.std(ddof=0).fillna(0) > 0
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
    Input("explore-features", "value"),
    Input("explore-show-points", "value"),
    Input("theme-store", "data"),
    State("session-id", "data"),
    prevent_initial_call=False,
)
def update_plots(transform, group, features, show_points, theme, sid):
    state = server_state.get_session(sid)
    df_raw = state.features_df
    if df_raw is None or len(df_raw) == 0:
        return go.Figure(), go.Figure()
    state.transform = transform or "none"

    feats_all = _feature_columns(df_raw)
    if transform and transform != "none":
        df_dist = apply_transform(df_raw, feats_all, transform)
    else:
        df_dist = df_raw
    # Spearman is rank-based, so transforms don't change the correlation
    # matrix. Always use the raw frame — avoids zero-variance columns
    # blanking the heatmap after a log/z-score pass.
    return (
        _build_distributions(df_dist, list(features or []), group or "",
                             theme, show_points=bool(show_points)),
        _build_correlations(df_raw, theme),
    )
