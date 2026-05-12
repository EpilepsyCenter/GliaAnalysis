"""Rule-based morphology labels for k-means clusters.

For each cluster we compute the z-score of its mean feature vector across
all clusters on a 6-feature diagnostic panel. We then score the cluster
against four canonical templates (ameboid, hypertrophic, rod-like, ramified)
using cosine similarity. Assignment is greedy without reuse — each label is
applied at most once unless the user explicitly overrides.

The scoring exposed to the UI lets users see *why* a label was suggested
and override it if the cluster looks atypical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from glia.config import DIAGNOSTIC_FEATURES, MORPHOLOGY_TEMPLATES


def cluster_zscore_profiles(
    df: pd.DataFrame, cluster_col: str = "Cluster"
) -> pd.DataFrame:
    """Mean feature vector per cluster, then z-scored across clusters."""
    means = df.groupby(cluster_col)[DIAGNOSTIC_FEATURES].mean()
    z = (means - means.mean()) / means.std(ddof=0).replace(0, np.nan)
    return z.fillna(0.0)


def score_templates(z_profiles: pd.DataFrame) -> pd.DataFrame:
    """Cosine similarity between each cluster's z-profile and each template."""
    template_df = pd.DataFrame(MORPHOLOGY_TEMPLATES).T[DIAGNOSTIC_FEATURES].fillna(0)
    # row-normalize both sides
    def _unit(M):
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return M / norms

    Z = _unit(z_profiles.to_numpy())
    T = _unit(template_df.to_numpy())
    sim = Z @ T.T
    return pd.DataFrame(sim, index=z_profiles.index, columns=template_df.index)


def greedy_label_assignment(scores: pd.DataFrame) -> dict[int, str]:
    """Pick the highest-scoring (cluster, template) pair iteratively without reuse."""
    s = scores.copy()
    assignments: dict[int, str] = {}
    while not s.empty and s.shape[1] > 0:
        cluster_idx, template = s.stack().idxmax()
        assignments[int(cluster_idx)] = template
        s = s.drop(index=cluster_idx, columns=template, errors="ignore")
    return assignments


def auto_label(df: pd.DataFrame, cluster_col: str = "Cluster") -> tuple[dict, pd.DataFrame]:
    """Convenience wrapper. Returns (assignments, full score matrix)."""
    z = cluster_zscore_profiles(df, cluster_col=cluster_col)
    scores = score_templates(z)
    return greedy_label_assignment(scores), scores
