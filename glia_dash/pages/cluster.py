"""Cluster page — PCA + KMeans + auto-labeling against morphology templates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import (
    ALL, Input, Output, State, callback, ctx, dash_table, dcc, html, no_update,
)
import dash_bootstrap_components as dbc

import os
from pathlib import Path

import tifffile
from skimage import measure

from glia.auto_label import auto_label
from glia.config import ALL_FEATURES, MORPHOLOGY_TEMPLATES
from glia.pca_cluster import (
    cluster_selection_scan,
    fit_pca,
    kmeans_cluster,
)
from glia.roi import per_roi_masks
from glia.segment import DEFAULT_ROI_TAG
from glia_dash import server_state
from glia_dash.components import alert, metric_card


_K_RANGE = range(2, 10)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in ALL_FEATURES if c in df.columns]


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


# ── Layout ──────────────────────────────────────────────────────────


def layout(sid: str | None) -> html.Div:
    state = server_state.get_session(sid)
    from glia.metadata import ensure_metadata_joined
    ensure_metadata_joined(state)
    df = state.features_df

    if df is None or len(df) == 0:
        return html.Div([
            html.H4("Cluster", style={"marginBottom": "16px"}),
            alert("No features in memory — run the Features tab first.",
                  variant="warning"),
        ])

    return html.Div([
        html.H4("Cluster", style={"marginBottom": "16px"}),
        dbc.Tabs(
            id="cluster-subtabs",
            active_tab="cluster-tab-main",
            class_name="dbc-page-tabs",
            children=[
                dbc.Tab(_clustering_subtab(state, df),
                        label="Clustering",
                        tab_id="cluster-tab-main"),
                dbc.Tab(_overlays_subtab(state, df),
                        label="Overlays",
                        tab_id="cluster-tab-overlays"),
            ],
        ),
    ])


def _clustering_subtab(state, df: pd.DataFrame) -> html.Div:
    feats = _feature_columns(df)
    has_cluster = "Cluster" in df.columns
    return html.Div([
        html.Div(
            "StandardScaler → PCA → KMeans. Pick k by inspecting the "
            "elbow + silhouette scan; the four canonical morphology "
            "templates (ameboid / hypertrophic / rod-like / ramified) "
            "are matched against each cluster's z-score profile via "
            "cosine similarity. You can override any label.",
            style={"fontSize": "0.85rem",
                   "color": "var(--ned-text-muted)",
                   "marginBottom": "16px"},
        ),

        html.Div([
            metric_card("Cells", f"{len(df):,}", accent=True),
            metric_card("Features", str(len(feats)), accent=True),
            metric_card("k (current)",
                        str(int(df["Cluster"].nunique())) if has_cluster
                        else "—", accent=has_cluster),
            metric_card("PCA components", str(state.pca_n_components)),
        ], style={"display": "grid",
                  "gridTemplateColumns": "repeat(4, 1fr)",
                  "gap": "12px", "marginBottom": "16px"}),

        html.Div([
            html.Div([
                html.Label("PCA components",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dcc.Input(id="cluster-pca-n", type="number",
                          min=2, max=20, step=1,
                          value=int(state.pca_n_components),
                          style={"width": "120px"}),
            ], style={"marginRight": "24px"}),
            html.Div([
                html.Label("Clusters (k)",
                           style={"fontSize": "0.72rem",
                                  "color": "var(--ned-text-muted)",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.5px"}),
                dcc.Input(id="cluster-k", type="number",
                          min=2, max=10, step=1, value=int(state.k),
                          style={"width": "120px"}),
            ], style={"marginRight": "24px"}),
            dbc.Button("Run PCA + cluster", id="cluster-run",
                       className="btn-ned-primary"),
            html.Span(id="cluster-run-hint",
                      children=(f"Will fit on {len(df):,} cells "
                                f"× {len(feats)} features."),
                      style={"marginLeft": "12px",
                             "fontSize": "0.78rem",
                             "color": "var(--ned-text-muted)"}),
        ], style={"display": "flex",
                  "alignItems": "flex-end",
                  "marginBottom": "16px"}),

        dcc.Loading(
            type="default",
            children=html.Div(id="cluster-output",
                              style={"marginTop": "12px",
                                     "minHeight": "40px"}),
        ),
    ])


def _overlays_subtab(state, df: pd.DataFrame) -> html.Div:
    """Image-by-image cluster overlay browser.

    Mirrors the Soma tab pattern: scrollable file list on the left,
    original image + cluster-colored overlay stacked on the right.
    A "Download overlay PNG" button beneath the overlay exports the
    composited image so it can be dropped into figures.

    We always build the full UI (file list + store + graphs + download)
    even when the features dataframe lacks a Cluster column, so that
    the callbacks have valid Input / Output targets and the subtab can
    re-populate live after the user runs PCA + cluster from the
    Clustering tab. The "run PCA first" banner is just shown / hidden
    by a separate callback that watches ``cluster-output``.
    """
    has_cluster = df is not None and "Cluster" in df.columns
    images = _project_image_paths(state.project_dir)
    current = state.extra.get("cluster_overlay_path", "")
    if (not current or current not in images) and images:
        current = images[0]
        state.extra["cluster_overlay_path"] = current

    return html.Div([
        # Banner shown until the features dataframe has a Cluster column.
        html.Div(
            "Run PCA + cluster in the Clustering tab first; once each "
            "cell has a cluster label, click an image on the left to "
            "see its overlay.",
            id="cluster-overlays-banner",
            style={"display": "none" if has_cluster else "block",
                   "fontSize": "0.82rem",
                   "color": "var(--ned-text-muted)",
                   "background": "rgba(56,189,248,0.08)",
                   "border": "1px solid rgba(56,189,248,0.3)",
                   "borderRadius": "6px",
                   "padding": "8px 12px",
                   "marginBottom": "12px"},
        ),
        html.Div(
            "Each segmented cell in the image is colored by its "
            "assigned cluster (palette matches the Clustering tab). "
            "Pick an image on the left; the original is shown on top, "
            "the overlay below. Use the download button to export the "
            "overlay as a PNG.",
            style={"fontSize": "0.85rem",
                   "color": "var(--ned-text-muted)",
                   "marginBottom": "16px"},
        ),

        html.Div([
            html.Div(id="cluster-overlay-list",
                     children=_render_overlay_file_list(images, current),
                     style={"flex": "0 0 440px",
                            "height": "880px",
                            "overflowY": "auto",
                            "border": "1px solid var(--ned-border)",
                            "borderRadius": "6px",
                            "marginRight": "12px",
                            "background": "var(--ned-panel)"}),

            html.Div([
                html.Div(id="cluster-overlay-info",
                         style={"display": "none",
                                "fontSize": "0.82rem",
                                "color": "var(--ned-text-muted)",
                                "marginBottom": "8px"}),
                html.Div(id="cluster-overlay-empty",
                         children=[
                             html.Div(
                                 "Pick an image on the left to render "
                                 "its cluster overlay.",
                                 style={"fontSize": "0.85rem",
                                        "color": "var(--ned-text-muted)",
                                        "padding": "40px"},
                             ),
                         ]),
                html.Div(id="cluster-overlay-graphs-col",
                         style={"display": "none"},
                         children=[
                             dcc.Graph(
                                 id="cluster-overlay-orig",
                                 figure=go.Figure(),
                                 config={"scrollZoom": True,
                                         "displayModeBar": True,
                                         "displaylogo": False,
                                         "toImageButtonOptions": {
                                             "format": "png",
                                             "filename": "original",
                                             "scale": 2,
                                         },
                                         "modeBarButtonsToRemove":
                                             ["lasso2d", "select2d",
                                              "autoScale2d"]},
                                 style={"height": "440px",
                                        "marginBottom": "12px"}),
                             dcc.Graph(
                                 id="cluster-overlay-img",
                                 figure=go.Figure(),
                                 config={"scrollZoom": True,
                                         "displayModeBar": True,
                                         "displaylogo": False,
                                         "toImageButtonOptions": {
                                             "format": "png",
                                             "filename": "cluster_overlay",
                                             "scale": 2,
                                         },
                                         "modeBarButtonsToRemove":
                                             ["lasso2d", "select2d",
                                              "autoScale2d"]},
                                 style={"height": "440px"}),
                             html.Div([
                                 dbc.Button(
                                     "Download overlay PNG",
                                     id="cluster-overlay-download-btn",
                                     className="btn-ned-secondary",
                                     size="sm",
                                 ),
                                 html.Span(
                                     "Saves a publication-ready PNG of "
                                     "the colored overlay at the "
                                     "image's native resolution. The "
                                     "Plotly camera button on each "
                                     "figure also works for quick "
                                     "screenshots.",
                                     style={"fontSize": "0.75rem",
                                            "color":
                                                "var(--ned-text-muted)",
                                            "marginLeft": "12px"},
                                 ),
                             ], style={"marginTop": "8px",
                                       "display": "flex",
                                       "alignItems": "center"}),
                             dcc.Download(id="cluster-overlay-download"),
                         ]),
            ], style={"flex": "1", "minWidth": "0"}),
        ], style={"display": "flex", "alignItems": "flex-start"}),

        dcc.Store(id="cluster-overlay-cell-store", data=current),
    ])


# ── Figure builders ─────────────────────────────────────────────────


def _build_scan(scan: pd.DataFrame, k_chosen: int, theme: str) -> go.Figure:
    p = _theme_palette(theme)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=scan["k"], y=scan["inertia"],
        mode="lines+markers", name="Inertia (elbow)",
        line=dict(color=p["colorway"][0], width=2),
        marker=dict(size=6),
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=scan["k"], y=scan["silhouette"],
        mode="lines+markers", name="Silhouette",
        line=dict(color=p["colorway"][1], width=2, dash="dot"),
        marker=dict(size=6),
        yaxis="y2",
    ))
    fig.add_vline(x=k_chosen, line_color=p["colorway"][3],
                  line_dash="dash",
                  annotation_text=f"k = {k_chosen}",
                  annotation_position="top")
    fig.update_layout(
        margin=dict(l=48, r=48, t=24, b=36), height=420,
        paper_bgcolor=p["paper"], plot_bgcolor=p["plot"],
        font=dict(color=p["fg"], family="IBM Plex Sans, sans-serif", size=11),
        xaxis=dict(title="k", gridcolor=p["grid"], dtick=1),
        yaxis=dict(title="Inertia", gridcolor=p["grid"]),
        yaxis2=dict(title="Silhouette", overlaying="y", side="right",
                    showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _build_pca_scatter(df: pd.DataFrame, theme: str) -> go.Figure:
    p = _theme_palette(theme)
    if "PC1" not in df.columns or "PC2" not in df.columns:
        return go.Figure()
    fig = go.Figure()
    clusters = sorted(df["Cluster"].unique())
    for i, cid in enumerate(clusters):
        sub = df[df["Cluster"] == cid]
        color = p["colorway"][int(cid) % len(p["colorway"])]
        label = df.loc[sub.index, "morphology_label"].iloc[0] \
            if "morphology_label" in df.columns else None
        name = (f"Cluster {int(cid)}"
                + (f" · {label}" if label else ""))
        fig.add_trace(go.Scattergl(
            x=sub["PC1"], y=sub["PC2"], mode="markers",
            name=name, marker=dict(color=color, size=5, opacity=0.7,
                                   line=dict(width=0)),
            hovertemplate=("Cluster %{text}<br>"
                           "PC1 = %{x:.2f}<br>"
                           "PC2 = %{y:.2f}<extra></extra>"),
            text=[str(int(cid))] * len(sub),
        ))
    fig.update_layout(
        margin=dict(l=48, r=20, t=20, b=36), height=420,
        paper_bgcolor=p["paper"], plot_bgcolor=p["plot"],
        font=dict(color=p["fg"], family="IBM Plex Sans, sans-serif", size=11),
        xaxis=dict(title="PC1", gridcolor=p["grid"],
                   zerolinecolor=p["grid"]),
        yaxis=dict(title="PC2", gridcolor=p["grid"],
                   zerolinecolor=p["grid"]),
        legend=dict(orientation="v"),
    )
    return fig


def _build_cluster_heatmap(df: pd.DataFrame, theme: str) -> go.Figure:
    p = _theme_palette(theme)
    feats = _feature_columns(df)
    means = df.groupby("Cluster")[feats].mean()
    # Row-scale to z-scores so the colors are comparable across very
    # different-magnitude features.
    z = (means - means.mean()) / means.std(ddof=0).replace(0, np.nan)
    z = z.fillna(0.0)
    fig = go.Figure(go.Heatmap(
        z=z.values, x=z.columns, y=[f"Cluster {int(c)}" for c in z.index],
        zmin=-2, zmax=2, colorscale="RdBu", reversescale=True,
        colorbar=dict(title="z", thickness=10),
        hovertemplate=("%{y}<br>%{x}<br>"
                       "z = %{z:.2f}<extra></extra>"),
    ))
    fig.update_layout(
        margin=dict(l=120, r=20, t=20, b=160), height=320,
        paper_bgcolor=p["paper"], plot_bgcolor=p["plot"],
        font=dict(color=p["fg"], family="IBM Plex Sans, sans-serif", size=10),
    )
    fig.update_xaxes(tickangle=-45, automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def _build_label_table(df: pd.DataFrame, scores: pd.DataFrame,
                       assignments: dict, overrides: dict,
                       theme: str) -> html.Div:
    p = _theme_palette(theme)
    templates = list(MORPHOLOGY_TEMPLATES)
    rows = []
    for cid in sorted(df["Cluster"].unique()):
        cid = int(cid)
        suggested = assignments.get(cid, "—")
        current = overrides.get(cid, suggested)
        n = int((df["Cluster"] == cid).sum())
        cluster_color = p["colorway"][cid % len(p["colorway"])]
        score_row = scores.loc[cid] if cid in scores.index else None
        scores_str = ""
        if score_row is not None:
            scores_str = " · ".join(
                f"{t} {score_row[t]:+.2f}"
                for t in sorted(score_row.index,
                                key=lambda t: -score_row[t])
            )
        rows.append(html.Tr([
            html.Td([
                html.Span("●", style={"color": cluster_color,
                                      "marginRight": "8px"}),
                html.Span(f"Cluster {cid}",
                          style={"fontWeight": "600",
                                 "color": "var(--ned-text)"}),
            ]),
            html.Td(f"{n:,}",
                    style={"color": "var(--ned-text-muted)"}),
            html.Td(suggested,
                    style={"color": "var(--ned-text-muted)"}),
            html.Td(dcc.Dropdown(
                id={"type": "cluster-label-override", "cid": cid},
                options=[{"label": t, "value": t} for t in templates]
                        + [{"label": "(custom — leave blank to clear)",
                            "value": ""}],
                value=current if current in templates else None,
                clearable=True,
                style={"width": "240px"},
            )),
            html.Td(scores_str,
                    style={"color": "var(--ned-text-muted)",
                           "fontSize": "0.78rem"}),
        ]))
    return html.Div([
        html.Table([
            html.Thead(html.Tr([
                html.Th("Cluster"), html.Th("n"),
                html.Th("Auto-label"), html.Th("Override"),
                html.Th("Template scores (cosine sim, sorted)"),
            ])),
            html.Tbody(rows),
        ], style={"width": "100%",
                  "fontSize": "0.85rem",
                  "borderCollapse": "collapse"}),
    ])


def _render_results(df, scan, scores, assignments, overrides,
                    theme) -> html.Div:
    return html.Div([
        html.Div([
            html.Div([
                html.H6("k selection (elbow + silhouette)",
                        style={"fontSize": "0.92rem",
                               "marginBottom": "4px",
                               "color": "var(--ned-text)"}),
                dcc.Graph(figure=_build_scan(scan,
                                             int(df["Cluster"].nunique()),
                                             theme),
                          config={"displayModeBar": False}),
            ], style={"flex": "1"}),
            html.Div([
                html.H6("PCA scatter (PC1 vs PC2)",
                        style={"fontSize": "0.92rem",
                               "marginBottom": "4px",
                               "color": "var(--ned-text)"}),
                dcc.Graph(figure=_build_pca_scatter(df, theme),
                          config={"displayModeBar": False}),
            ], style={"flex": "1", "marginLeft": "16px"}),
        ], style={"display": "flex", "marginTop": "8px"}),

        html.H6("Cluster mean feature heatmap (z-scored across clusters)",
                style={"fontSize": "0.92rem",
                       "marginTop": "16px",
                       "marginBottom": "4px",
                       "color": "var(--ned-text)"}),
        dcc.Graph(figure=_build_cluster_heatmap(df, theme),
                  config={"displayModeBar": False}),

        html.H6("Auto-labels (override below)",
                style={"fontSize": "0.92rem",
                       "marginTop": "16px",
                       "marginBottom": "8px",
                       "color": "var(--ned-text)"}),
        _build_label_table(df, scores, assignments, overrides, theme),

        _interpretation_panel(),
    ])


# ── Cluster overlay on original image ────────────────────────────────


_OUTPUT_SUBDIR = "_gliaanalysis"


def _project_image_paths(project_dir: str) -> list[str]:
    """Prepared TIFFs by default; legacy fall-through to top-level."""
    if not project_dir or not Path(project_dir).is_dir():
        return []
    from glia.prepare import prepared_dir
    prep = prepared_dir(project_dir)
    if prep.is_dir():
        prepared = sorted(
            str(p) for p in prep.iterdir()
            if p.suffix.lower() in (".tif", ".tiff") and p.is_file()
        )
        if prepared:
            return prepared
    return sorted(str(p) for p in Path(project_dir).iterdir()
                  if p.suffix.lower() in (".tif", ".tiff") and p.is_file())


def _thresholded_path(project_dir: str, image_path: str) -> Path:
    """Find the matching <stem>_thresholded.tif under the project's output."""
    stem = Path(image_path).name  # threshold.ijm saves <filename>_thresholded.tif
    return (Path(project_dir) / _OUTPUT_SUBDIR /
            "ThresholdedImages" / f"{stem}_thresholded.tif")


def _build_cluster_overlay(
    image_path: str, project_dir: str,
    rois_by_path: dict, df: pd.DataFrame, theme: str,
) -> tuple[go.Figure, go.Figure, dict]:
    """Return (fig_original, fig_overlay, info)."""
    p = _theme_palette(theme)
    info: dict = {"matched": 0, "unmatched": 0}

    img = tifffile.imread(image_path)
    fig_orig = px.imshow(img, color_continuous_scale="gray",
                         binary_string=True, aspect="equal")
    fig_orig.update_layout(
        margin=dict(l=0, r=0, t=24, b=0),
        coloraxis_showscale=False,
        paper_bgcolor=p["paper"], plot_bgcolor=p["plot"],
        font=dict(color=p["fg"]),
        title=dict(text="Original", x=0.02, font=dict(size=12)),
    )
    fig_orig.update_xaxes(visible=False)
    fig_orig.update_yaxes(visible=False)

    th_path = _thresholded_path(project_dir, image_path)
    if not th_path.exists():
        empty = go.Figure()
        empty.update_layout(
            paper_bgcolor=p["paper"], plot_bgcolor=p["plot"],
            font=dict(color=p["fg"]),
            annotations=[dict(text=f"No thresholded image at {th_path}",
                              xref="paper", yref="paper",
                              x=0.5, y=0.5, showarrow=False,
                              font=dict(color=p["fg"]))],
        )
        info["error"] = f"Thresholded image not found at {th_path}"
        return fig_orig, empty, info

    binary = tifffile.imread(th_path) > 0
    h, w = binary.shape

    image_stem = Path(image_path).stem
    image_rois = rois_by_path.get(image_path, [])
    if image_rois:
        passes = per_roi_masks(image_rois, h, w)
    else:
        passes = [(DEFAULT_ROI_TAG, np.ones((h, w), dtype=bool))]

    # Build an RGBA overlay: same shape as the original image, blank
    # background, cells filled with their cluster colour.
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    df_image = df[df["ID"].str.startswith(image_stem + "__")]
    id_to_cluster = dict(zip(df_image["ID"], df_image["Cluster"]))

    clusters_present = sorted({int(c) for c in df_image["Cluster"].unique()})

    def hex_to_rgb(h_):
        h_ = h_.lstrip("#")
        return (int(h_[0:2], 16), int(h_[2:4], 16), int(h_[4:6], 16))

    for tag, roi_arr in passes:
        roi_binary = binary & roi_arr
        if not roi_binary.any():
            continue
        labels = measure.label(roi_binary, connectivity=1)
        for rp in measure.regionprops(labels):
            cell_id = f"{image_stem}__{tag}__{rp.label}"
            cluster = id_to_cluster.get(cell_id)
            if cluster is None:
                info["unmatched"] += 1
                continue
            info["matched"] += 1
            r, g, b = hex_to_rgb(
                p["colorway"][int(cluster) % len(p["colorway"])]
            )
            pixel_mask = (labels == rp.label)
            overlay[pixel_mask, 0] = r
            overlay[pixel_mask, 1] = g
            overlay[pixel_mask, 2] = b
            overlay[pixel_mask, 3] = 255

    # Compose: original grayscale (dimmed) underneath, RGBA overlay on top.
    base = np.repeat(img[..., None], 3, axis=-1)
    base = (base.astype(np.float32) * 0.6).clip(0, 255).astype(np.uint8)
    rgb = base.copy()
    alpha = overlay[..., 3:4].astype(np.float32) / 255.0
    rgb = (rgb * (1 - alpha) + overlay[..., :3] * alpha).astype(np.uint8)

    fig_overlay = px.imshow(rgb, aspect="equal")
    fig_overlay.update_layout(
        margin=dict(l=0, r=0, t=24, b=0),
        coloraxis_showscale=False,
        paper_bgcolor=p["paper"], plot_bgcolor=p["plot"],
        font=dict(color=p["fg"]),
        title=dict(text="Cells coloured by cluster", x=0.02,
                   font=dict(size=12)),
    )
    fig_overlay.update_xaxes(visible=False)
    fig_overlay.update_yaxes(visible=False)
    info["clusters_present"] = clusters_present
    return fig_orig, fig_overlay, info


def _render_overlay_file_list(images: list[str], current: str) -> list:
    """Vertical, scrollable list of project images (one per line)."""
    if not images:
        return [html.Div(
            "No project images on disk. Prepare images first.",
            style={"fontSize": "0.82rem",
                   "color": "var(--ned-text-muted)",
                   "padding": "8px"})]
    return [
        html.Div(
            f"{len(images)} image{'s' if len(images) != 1 else ''}",
            style={"fontSize": "0.72rem",
                   "color": "var(--ned-text-muted)",
                   "textTransform": "uppercase",
                   "letterSpacing": "0.5px",
                   "padding": "4px 8px",
                   "borderBottom": "1px solid var(--ned-border)"},
        ),
        *[
            html.Button(
                Path(p).stem,
                id={"type": "cluster-overlay-row", "path": p},
                n_clicks=0,
                className=("soma-cell-row selected"
                           if p == current else "soma-cell-row"),
                style={"display": "block",
                       "width": "100%",
                       "padding": "6px 10px",
                       "border": "none",
                       "borderBottom":
                           "1px solid rgba(125, 133, 144, 0.12)",
                       "textAlign": "left",
                       "background": ("rgba(56, 139, 253, 0.18)"
                                      if p == current
                                      else "transparent"),
                       "color": "var(--ned-text)",
                       "fontSize": "0.78rem",
                       "fontFamily":
                           "IBM Plex Mono, ui-monospace, monospace",
                       "cursor": "pointer",
                       "whiteSpace": "nowrap",
                       "overflow": "hidden",
                       "textOverflow": "ellipsis"},
                title=Path(p).name,
            )
            for p in images
        ],
    ]


def _interpretation_panel() -> html.Div:
    """How-to-read panel for the four plots above. Plain language."""
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
        html.Div("How to read these plots",
                 style={"fontSize": "0.92rem",
                        "fontWeight": "600",
                        "color": "var(--ned-text)",
                        "letterSpacing": "0.3px"}),

        html.Div("PCA in one paragraph", style=section_h),
        html.Div(
            "Each cell is a point in 27-dimensional feature space (one "
            "dimension per morphology measurement). PCA finds the axes "
            "along which cells differ from each other most. PC1 is the "
            "direction of greatest variance; PC2 is the next greatest, "
            "uncorrelated with PC1. Their meaning depends on which "
            "features load onto them — in microglia data PC1 typically "
            "captures size / ramification (small ameboid ↔ large branched), "
            "PC2 captures elongation (round ↔ rod). Inspect the heatmap "
            "below to see which features drive each cluster.",
            style=body,
        ),

        html.Div("k-selection plot (left)", style=section_h),
        html.Div([
            html.Span("Inertia (blue, solid) ", style={"fontWeight": "600",
                                                       "color": "var(--ned-text)"}),
            html.Span("= within-cluster sum of squares; falls as k grows "
                     "because more clusters fit the data more tightly. "
                     "Look for the 'elbow' where the curve flattens — "
                     "adding clusters past that point yields diminishing "
                     "returns. "),
            html.Br(),
            html.Span("Silhouette (green, dotted) ", style={"fontWeight": "600",
                                                            "color": "var(--ned-text)"}),
            html.Span("ranges from -1 to +1; higher = better-separated "
                     "clusters. Pick a k where silhouette peaks AND the "
                     "elbow has flattened — usually k = 3 or 4 for "
                     "microglia (matching the four canonical phenotypes)."),
        ], style=body),

        html.Div("PCA scatter (right)", style=section_h),
        html.Div("Each dot is one cell, positioned by its PC1 and PC2 "
                 "scores; color = its KMeans cluster. Visually well-"
                 "separated colored blobs = clusters that capture real "
                 "structure; heavily overlapping blobs = clusters that "
                 "may be artificial. Outliers far from the main mass are "
                 "often debris or merged cells worth double-checking on "
                 "the original image.",
                 style=body),

        html.Div("Cluster mean feature heatmap", style=section_h),
        html.Div("Each row is one cluster, each column is one of the 27 "
                 "features. Cells are z-scored across clusters: ",
                 style=body),
        html.Ul(style={**body, "paddingLeft": "20px", "marginTop": "0"},
                children=[
            html.Li([html.Span("red ", style={"color": "#cf222e",
                                              "fontWeight": "600"}),
                     "= this cluster is well above the cross-cluster "
                     "mean for that feature"]),
            html.Li([html.Span("blue ", style={"color": "#0969da",
                                               "fontWeight": "600"}),
                     "= well below the mean"]),
            html.Li("near-white = average. The colored pattern across the "
                    "row is the cluster's morphology signature."),
        ]),

        html.Div("Auto-label table", style=section_h),
        html.Div("The four canonical microglia phenotypes (ameboid, "
                 "hypertrophic, rod-like, ramified) each have a "
                 "characteristic z-score profile over six diagnostic "
                 "features (circularity, area, # branches, # endpoints, "
                 "span ratio, max branch length — see "
                 "glia.config.MORPHOLOGY_TEMPLATES). Each cluster is "
                 "scored against each template via cosine similarity; the "
                 "labels are then assigned greedily without reuse. A high "
                 "score (closer to +1) means the cluster's signature "
                 "matches that template; near 0 or negative means it "
                 "doesn't. Use the override dropdown if the biology of "
                 "your dataset suggests a different label.",
                 style=body),
    ])


# ── Callbacks ───────────────────────────────────────────────────────


@callback(
    Output("cluster-output", "children"),
    Input("cluster-run", "n_clicks"),
    State("cluster-pca-n", "value"),
    State("cluster-k", "value"),
    State("theme-store", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_run(n_clicks, n_components, k, theme, sid):
    if not n_clicks:
        return no_update
    state = server_state.get_session(sid)
    df = state.features_df
    if df is None or len(df) == 0:
        return alert("No features in memory.", variant="warning")
    feats = _feature_columns(df)
    if len(feats) < 2:
        return alert("Need at least 2 numeric feature columns to cluster.",
                     variant="warning")

    n_components = int(n_components or 5)
    k = int(k or 4)
    state.k = k
    state.pca_n_components = n_components

    # Strip any columns left over from a prior Cluster run so fit_pca
    # doesn't concat duplicate PC columns and the merge stays clean.
    derived = [c for c in df.columns
               if c == "Cluster" or c == "morphology_label"
               or (c.startswith("PC") and c[2:].isdigit())]
    if derived:
        df = df.drop(columns=derived)

    # Some geometric features can be inf (e.g. radii_ratio when a shape is
    # degenerate). sklearn rejects inf, so clean before fitting; replace
    # inf with NaN, then impute NaN with the per-column median. Keeps the
    # cell row count constant so the Cluster series aligns by index.
    df_clean = df.copy()
    df_clean[feats] = (df_clean[feats]
                       .replace([np.inf, -np.inf], np.nan)
                       .fillna(df_clean[feats].replace([np.inf, -np.inf], np.nan)
                                .median()))

    try:
        _, df_pca = fit_pca(df_clean, feats, n_components=n_components)
        scan = cluster_selection_scan(df_clean, feats, k_range=_K_RANGE)
        clusters = kmeans_cluster(df_clean, feats, k=k)
    except Exception as e:
        return alert(f"Clustering failed: {e}", variant="danger")

    df_out = df_pca.copy()
    # Drop any prior columns named the same to avoid duplicates on re-run.
    for c in ("Cluster",):
        if c in df_out.columns:
            df_out = df_out.drop(columns=[c])
    df_out["Cluster"] = clusters.values

    assignments, scores = auto_label(df_out, cluster_col="Cluster")
    df_out["morphology_label"] = df_out["Cluster"].map(assignments)
    state.features_df = df_out
    state.cluster_labels = dict(assignments)
    state.extra["cluster_scores"] = scores
    try:
        from glia.settings import save_project_settings
        save_project_settings(state.project_dir, state)
        from glia.features import save_features_df
        save_features_df(state.project_dir, df_out)
    except Exception:
        pass

    return _render_results(df_out, scan, scores, assignments,
                           overrides=state.cluster_labels, theme=theme)


@callback(
    Output("cluster-output", "children", allow_duplicate=True),
    Input({"type": "cluster-label-override", "cid": ALL}, "value"),
    State("theme-store", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_label_override(values, theme, sid):
    """Write override label into state and re-render so morphology_label
    updates everywhere."""
    state = server_state.get_session(sid)
    df = state.features_df
    if df is None or "Cluster" not in df.columns:
        return no_update
    trig = ctx.triggered_id
    if not isinstance(trig, dict) or "cid" not in trig:
        return no_update
    new_label = next(
        (v for v in values
         if v is not None and ctx.inputs_list and True), None
    )
    # Simpler: read the triggered value directly via ctx.triggered
    new_value = ctx.triggered[0]["value"] if ctx.triggered else None
    cid = int(trig["cid"])
    overrides = dict(state.cluster_labels or {})
    if new_value:
        overrides[cid] = new_value
    else:
        overrides.pop(cid, None)
    state.cluster_labels = overrides
    df["morphology_label"] = df["Cluster"].map(overrides)
    state.features_df = df

    scan = cluster_selection_scan(df, _feature_columns(df), k_range=_K_RANGE)
    scores = state.extra.get("cluster_scores")
    if scores is None:
        # Recompute if not cached
        _, scores = auto_label(df, cluster_col="Cluster")
        state.extra["cluster_scores"] = scores
    return _render_results(df, scan, scores,
                           assignments=state.cluster_labels,
                           overrides=overrides, theme=theme)


# ── Cluster-overlay callbacks ────────────────────────────────────────


@callback(
    Output("cluster-overlays-banner", "style"),
    Input("cluster-output", "children"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def hide_overlays_banner(_children, sid):
    """When PCA + cluster has just run, the features dataframe now has
    a Cluster column. Hide the warning banner so the Overlays subtab
    becomes usable without reloading the page."""
    state = server_state.get_session(sid)
    df = state.features_df
    has_cluster = df is not None and "Cluster" in df.columns
    return {"display": "none" if has_cluster else "block",
            "fontSize": "0.82rem",
            "color": "var(--ned-text-muted)",
            "background": "rgba(56,189,248,0.08)",
            "border": "1px solid rgba(56,189,248,0.3)",
            "borderRadius": "6px",
            "padding": "8px 12px",
            "marginBottom": "12px"}


@callback(
    Output("cluster-overlay-cell-store", "data"),
    Output("cluster-overlay-list", "children"),
    Input({"type": "cluster-overlay-row", "path": ALL}, "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_overlay_row_click(_clicks, sid):
    """Clicking a file row updates the persistent store and re-renders
    the list so the selected entry highlights."""
    if not any(n for n in (_clicks or [])):
        return no_update, no_update
    trig = ctx.triggered_id
    if not isinstance(trig, dict) or "path" not in trig:
        return no_update, no_update
    path = trig["path"]
    state = server_state.get_session(sid)
    state.extra["cluster_overlay_path"] = path
    images = _project_image_paths(state.project_dir)
    return path, _render_overlay_file_list(images, path)


_OVERLAY_OUTPUTS = (
    Output("cluster-overlay-empty", "style"),
    Output("cluster-overlay-info", "style"),
    Output("cluster-overlay-graphs-col", "style"),
    Output("cluster-overlay-info", "children"),
    Output("cluster-overlay-orig", "figure"),
    Output("cluster-overlay-img", "figure"),
)


def _hide_overlay(message: str | None = None):
    return (
        {"display": "block"},
        {"display": "none"},
        {"display": "none"},
        message or "",
        go.Figure(), go.Figure(),
    )


@callback(
    *_OVERLAY_OUTPUTS,
    Input("cluster-overlay-cell-store", "data"),
    Input("theme-store", "data"),
    Input("cluster-output", "children"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def render_overlay(path, theme, _cluster_output, sid):
    if not path:
        return _hide_overlay()
    state = server_state.get_session(sid)
    df = state.features_df
    if df is None or "Cluster" not in df.columns:
        return _hide_overlay("Run PCA + cluster in the Clustering "
                             "tab first.")
    rois = state.extra.get("rois", {})
    try:
        fig_orig, fig_overlay, info = _build_cluster_overlay(
            path, state.project_dir, rois, df, theme,
        )
    except Exception as e:
        return _hide_overlay(f"Overlay rendering failed: {e}")
    if "error" in info:
        return _hide_overlay(info["error"])

    info_text = (
        f"{Path(path).name} · matched {info.get('matched', 0):,} cells"
        f" ({info.get('unmatched', 0):,} unmatched)"
        f" · clusters present: "
        f"{', '.join(map(str, info.get('clusters_present', [])))}"
    )

    return (
        {"display": "none"},
        {"display": "block", "fontSize": "0.82rem",
         "color": "var(--ned-text-muted)", "marginBottom": "8px"},
        {"display": "block"},
        info_text,
        fig_orig, fig_overlay,
    )


@callback(
    Output("cluster-overlay-download", "data"),
    Input("cluster-overlay-download-btn", "n_clicks"),
    State("cluster-overlay-cell-store", "data"),
    State("theme-store", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def on_overlay_download(n_clicks, path, theme, sid):
    """Composite the cluster overlay into a PNG and stream it back.

    We re-build the overlay at native image resolution (no Plotly
    chrome) so the saved file is publication-ready. The figure already
    holds the same RGB array via px.imshow, but going straight from the
    composite numpy array → PIL → PNG bytes is simpler and produces a
    crisper output than re-rendering the figure.
    """
    if not n_clicks or not path:
        return no_update
    state = server_state.get_session(sid)
    df = state.features_df
    if df is None or "Cluster" not in df.columns:
        return no_update
    rgb = _composite_overlay_rgb(
        path, state.project_dir, state.extra.get("rois", {}), df,
    )
    if rgb is None:
        return no_update
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    buf.seek(0)
    filename = Path(path).stem + "__cluster_overlay.png"
    return dcc.send_bytes(buf.getvalue(), filename)


def _composite_overlay_rgb(
    image_path: str, project_dir: str, rois_by_path: dict,
    df: pd.DataFrame,
) -> np.ndarray | None:
    """Re-build the RGB composite used by _build_cluster_overlay,
    without the Plotly figure wrapping. Returns ``None`` if the
    thresholded image isn't available.
    """
    p = _theme_palette("light")  # palette is fixed regardless of theme
    img = tifffile.imread(image_path)
    th_path = _thresholded_path(project_dir, image_path)
    if not th_path.exists():
        return None
    binary = tifffile.imread(th_path) > 0
    h, w = binary.shape

    image_stem = Path(image_path).stem
    image_rois = rois_by_path.get(image_path, [])
    if image_rois:
        passes = per_roi_masks(image_rois, h, w)
    else:
        passes = [(DEFAULT_ROI_TAG, np.ones((h, w), dtype=bool))]

    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    df_image = df[df["ID"].str.startswith(image_stem + "__")]
    id_to_cluster = dict(zip(df_image["ID"], df_image["Cluster"]))

    def hex_to_rgb(h_):
        h_ = h_.lstrip("#")
        return (int(h_[0:2], 16), int(h_[2:4], 16), int(h_[4:6], 16))

    for tag, roi_arr in passes:
        roi_binary = binary & roi_arr
        if not roi_binary.any():
            continue
        labels = measure.label(roi_binary, connectivity=1)
        for rp in measure.regionprops(labels):
            cell_id = f"{image_stem}__{tag}__{rp.label}"
            cluster = id_to_cluster.get(cell_id)
            if cluster is None:
                continue
            r, g, b = hex_to_rgb(
                p["colorway"][int(cluster) % len(p["colorway"])]
            )
            pixel_mask = (labels == rp.label)
            overlay[pixel_mask, 0] = r
            overlay[pixel_mask, 1] = g
            overlay[pixel_mask, 2] = b
            overlay[pixel_mask, 3] = 255

    # Normalize the base to 0-255 if needed (e.g. uint16 source).
    if img.dtype != np.uint8:
        base = img.astype(np.float32)
        base = (base - base.min()) / max(base.ptp(), 1)
        base = (base * 255).clip(0, 255).astype(np.uint8)
    else:
        base = img
    base_rgb = np.repeat(base[..., None], 3, axis=-1)
    base_rgb = (base_rgb.astype(np.float32) * 0.6).clip(0, 255).astype(np.uint8)
    alpha = overlay[..., 3:4].astype(np.float32) / 255.0
    rgb = (base_rgb * (1 - alpha) + overlay[..., :3] * alpha).astype(np.uint8)
    return rgb
