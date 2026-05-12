"""Plotly helpers shared across tabs."""
from __future__ import annotations

import pandas as pd
import plotly.express as px


def correlation_heatmap(corr: pd.DataFrame, title: str = ""):
    fig = px.imshow(
        corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        aspect="auto", title=title,
    )
    return fig


def pca_scatter(df: pd.DataFrame, x: str, y: str, color: str | None = None):
    return px.scatter(df, x=x, y=y, color=color, opacity=0.6)


def feature_distribution(df: pd.DataFrame, feature: str, color: str | None = None):
    return px.histogram(df, x=feature, color=color, marginal="box", nbins=40)


def cluster_heatmap(cluster_means: pd.DataFrame, title: str = ""):
    return px.imshow(
        cluster_means.T, aspect="auto", color_continuous_scale="RdBu_r", title=title,
    )
