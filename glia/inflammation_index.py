"""Inflammation Index — supervised PCA-based morphology axis.

Trains on two reference groups (e.g. "saline" vs "LPS") to find a
combination of features whose PC1 best separates those two groups.
That trained axis can then be applied to every cell in the project —
including cells from groups that weren't used for training — to put
every cell on a single, interpretable activation scale.

The method follows BrainEnergyLab's R/ImageJ Inflammation-Index package
(Heindl-style supervised PCA), generalized to the morphology feature
set this pipeline produces. See ``InflammationModel`` for what gets
trained and persisted.

Why two-group training:
    A single morphology axis is well-posed only when you anchor it
    with two endpoints. Multi-class discrimination (LDA, multinomial)
    yields N-1 axes and is no longer the "Inflammation Index" — it's
    a different statistical object. With 3+ experimental groups, the
    cleanest pattern is to train on the two reference groups and let
    the scoring step quantify where the other groups sit on that
    axis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# Hyperparameters with sensible defaults. ``corr_cutoff`` mirrors the
# BrainEnergyLab default; subset-size bounds match the R package.
DEFAULT_MIN_SUBSET_SIZE = 5
DEFAULT_MAX_SUBSET_SIZE = 15
DEFAULT_CORR_CUTOFF = 0.9


@dataclass
class InflammationModel:
    """Trained Inflammation Index.

    Carries everything needed to score a fresh dataframe via
    :func:`apply`. Serializable to project settings.
    """
    treatment_col: str
    control_group: str
    comparator_group: str
    features: list[str]                 # feature columns used in PC1
    feature_means: list[float]          # StandardScaler.mean_
    feature_scales: list[float]         # StandardScaler.scale_
    pc1_loadings: list[float]           # PCA.components_[0]
    pc1_explained: float                # variance ratio captured by PC1
    score_orientation: int              # +1 or -1; flips PC1 so comparator > control
    train_score_mean: float             # for centering (mean of training cells' scores)
    train_score_std: float              # for scaling (std of training cells' scores)
    train_auc: float                    # AUC at training time, control vs comparator
    n_train_cells: int
    forward_path: list[dict] = field(default_factory=list)
    # ^ one entry per forward-selection step:
    #   {"step": k, "feature": name, "auc": float, "size": k}

    def to_dict(self) -> dict:
        return {
            "treatment_col": self.treatment_col,
            "control_group": self.control_group,
            "comparator_group": self.comparator_group,
            "features": list(self.features),
            "feature_means": list(self.feature_means),
            "feature_scales": list(self.feature_scales),
            "pc1_loadings": list(self.pc1_loadings),
            "pc1_explained": float(self.pc1_explained),
            "score_orientation": int(self.score_orientation),
            "train_score_mean": float(self.train_score_mean),
            "train_score_std": float(self.train_score_std),
            "train_auc": float(self.train_auc),
            "n_train_cells": int(self.n_train_cells),
            "forward_path": list(self.forward_path),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InflammationModel":
        return cls(
            treatment_col=d["treatment_col"],
            control_group=d["control_group"],
            comparator_group=d["comparator_group"],
            features=list(d["features"]),
            feature_means=list(d["feature_means"]),
            feature_scales=list(d["feature_scales"]),
            pc1_loadings=list(d["pc1_loadings"]),
            pc1_explained=float(d["pc1_explained"]),
            score_orientation=int(d.get("score_orientation", 1)),
            train_score_mean=float(d.get("train_score_mean", 0.0)),
            train_score_std=float(d.get("train_score_std", 1.0)),
            train_auc=float(d.get("train_auc", 0.5)),
            n_train_cells=int(d.get("n_train_cells", 0)),
            forward_path=list(d.get("forward_path", [])),
        )


def _prune_correlated(
    df_train: pd.DataFrame, candidates: list[str], cutoff: float,
) -> list[str]:
    """Drop one of each pair with |Pearson| > cutoff.

    Order-stable: we keep the first feature in ``candidates`` order and
    drop the later one of each correlated pair. That makes the result
    deterministic given the input ordering.
    """
    if len(candidates) < 2:
        return list(candidates)
    sub = df_train[candidates].select_dtypes(include="number")
    # Some features may be constant within the training subset — drop them.
    nonconst = [c for c in sub.columns if sub[c].std(ddof=0) > 1e-12]
    sub = sub[nonconst]
    if sub.shape[1] < 2:
        return list(sub.columns)
    corr = sub.corr().abs()
    keep: list[str] = []
    for c in sub.columns:
        clash = False
        for k in keep:
            if corr.loc[c, k] > cutoff:
                clash = True
                break
        if not clash:
            keep.append(c)
    return keep


def _pc1_auc(
    df_train: pd.DataFrame, feature_subset: list[str], y: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Fit StandardScaler+PCA on ``feature_subset``, return (AUC, scores).

    The AUC is invariant to PC1's sign, so we report the better of
    AUC(score) and AUC(-score). The caller decides orientation later.
    """
    X = df_train[feature_subset].to_numpy(dtype=float)
    # Guard against constant columns in the supplied subset.
    if not np.all(X.std(axis=0, ddof=0) > 1e-12):
        return 0.5, np.zeros(len(X))
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    pca = PCA(n_components=1)
    score = pca.fit_transform(Xz)[:, 0]
    try:
        auc = float(roc_auc_score(y, score))
    except ValueError:
        return 0.5, score
    return max(auc, 1.0 - auc), score


def _forward_select(
    df_train: pd.DataFrame, pool: list[str], y: np.ndarray,
    min_size: int, max_size: int,
) -> tuple[list[str], float, list[dict]]:
    """Greedy forward selection of feature subset by PC1 AUC.

    At each step, try adding each remaining pool feature; keep the one
    that gives the highest PC1 AUC. Continue until either
    ``max_size`` is reached or AUC stops improving for two consecutive
    additions (we still try up to ``max_size`` so the best size is
    captured, then prune back to that best size).
    """
    selected: list[str] = []
    remaining = list(pool)
    path: list[dict] = []
    best_auc_so_far = 0.5
    best_size = 0

    while remaining and len(selected) < max_size:
        candidate_results: list[tuple[float, str]] = []
        for f in remaining:
            trial = selected + [f]
            auc, _ = _pc1_auc(df_train, trial, y)
            candidate_results.append((auc, f))
        candidate_results.sort(key=lambda t: -t[0])
        best_auc, best_feat = candidate_results[0]
        selected.append(best_feat)
        remaining.remove(best_feat)
        path.append({
            "step": len(selected),
            "feature": best_feat,
            "auc": best_auc,
            "size": len(selected),
        })
        if best_auc > best_auc_so_far + 1e-9:
            best_auc_so_far = best_auc
            best_size = len(selected)

    # Prune back to the size that achieved the best AUC, respecting
    # min_size. If best_size < min_size, keep min_size.
    final_size = max(best_size, min_size)
    final_size = min(final_size, len(selected))
    final = selected[:final_size]
    return final, best_auc_so_far, path


def train(
    df: pd.DataFrame,
    treatment_col: str,
    control_group: str,
    comparator_group: str,
    candidate_features: list[str],
    *,
    min_size: int = DEFAULT_MIN_SUBSET_SIZE,
    max_size: int = DEFAULT_MAX_SUBSET_SIZE,
    corr_cutoff: float = DEFAULT_CORR_CUTOFF,
) -> InflammationModel:
    """Fit an Inflammation Index from two reference groups.

    Args:
        df: cell-level dataframe. Must carry ``treatment_col`` and all
            ``candidate_features`` as columns.
        treatment_col: the metadata column used to split groups (e.g.
            ``"Treatment"``).
        control_group: label of the "resting" reference group.
        comparator_group: label of the "activated" reference group.
            PC1 is oriented so this group's scores are higher.
        candidate_features: starting feature pool. The 36-feature set
            from ``glia.config.ALL_FEATURES`` is the natural input.
        min_size, max_size: forward-selection size bounds (BrainEnergyLab
            defaults).
        corr_cutoff: drop one of each pair with |Pearson| > cutoff
            among the training cells, before forward selection.

    Returns:
        :class:`InflammationModel`.

    Raises:
        ValueError if either reference group has zero cells, or if no
        usable features remain after pruning.
    """
    if treatment_col not in df.columns:
        raise ValueError(f"Treatment column '{treatment_col}' not in df.")
    mask = df[treatment_col].isin([control_group, comparator_group])
    sub = df.loc[mask].copy()
    if (sub[treatment_col] == control_group).sum() == 0:
        raise ValueError(
            f"No training cells for control group '{control_group}'.")
    if (sub[treatment_col] == comparator_group).sum() == 0:
        raise ValueError(
            f"No training cells for comparator group '{comparator_group}'.")

    pool = [c for c in candidate_features if c in sub.columns]
    if not pool:
        raise ValueError("None of the candidate features are in df.")

    # Drop any rows with NaN in the pool — PCA can't handle them.
    sub = sub.dropna(subset=pool).reset_index(drop=True)
    if len(sub) < 2:
        raise ValueError(
            "Fewer than 2 training cells survive NaN filtering on "
            "the candidate feature set.")

    pruned = _prune_correlated(sub, pool, corr_cutoff)
    if len(pruned) < min_size:
        # Don't fail — just train on whatever's left.
        if not pruned:
            raise ValueError(
                "No features survive correlation pruning at "
                f"|r|>{corr_cutoff}.")
        min_size = len(pruned)

    y = (sub[treatment_col] == comparator_group).to_numpy(dtype=int)
    selected, train_auc, path = _forward_select(
        sub, pruned, y,
        min_size=min_size, max_size=min(max_size, len(pruned)),
    )

    # Fit the final model on the chosen subset.
    X = sub[selected].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    pca = PCA(n_components=1)
    score = pca.fit_transform(Xz)[:, 0]

    # Orient PC1 so comparator scores are higher than control's.
    mean_ctrl = float(score[y == 0].mean())
    mean_comp = float(score[y == 1].mean())
    orientation = 1 if mean_comp >= mean_ctrl else -1
    score = score * orientation

    return InflammationModel(
        treatment_col=treatment_col,
        control_group=control_group,
        comparator_group=comparator_group,
        features=selected,
        feature_means=scaler.mean_.tolist(),
        feature_scales=scaler.scale_.tolist(),
        pc1_loadings=pca.components_[0].tolist(),
        pc1_explained=float(pca.explained_variance_ratio_[0]),
        score_orientation=orientation,
        train_score_mean=float(score.mean()),
        train_score_std=float(score.std(ddof=0)) or 1.0,
        train_auc=float(train_auc),
        n_train_cells=int(len(sub)),
        forward_path=path,
    )


def apply(
    df: pd.DataFrame, model: InflammationModel, *,
    column_name: str = "inflammation_index",
) -> pd.DataFrame:
    """Score every row of ``df`` on the trained Inflammation Index.

    The output column is ``inflammation_index`` (z-scored against the
    training cells' score distribution, so it's unitless and ≈ 0 at
    the training population mean).

    Rows missing any of the model's features get NaN scores rather
    than silently breaking. The returned dataframe is a copy with the
    new column appended.
    """
    out = df.copy()
    feats = model.features
    missing = [f for f in feats if f not in out.columns]
    if missing:
        raise ValueError(
            f"Features missing from df for scoring: {missing}")

    X = out[feats].to_numpy(dtype=float)
    means = np.asarray(model.feature_means, dtype=float)
    scales = np.asarray(model.feature_scales, dtype=float)
    loadings = np.asarray(model.pc1_loadings, dtype=float)

    # Per-row NaN masking — we don't want one NaN feature to NaN out
    # an entire dataset. Rows with any NaN in the model's features get
    # NaN scores (the user can decide whether to drop them).
    row_has_nan = np.isnan(X).any(axis=1)
    Xz = (X - means) / scales
    raw = Xz @ loadings
    raw = raw * model.score_orientation
    z = (raw - model.train_score_mean) / max(model.train_score_std, 1e-12)
    z[row_has_nan] = np.nan
    out[column_name] = z
    return out


def per_animal_mean(
    df: pd.DataFrame, animal_col: str, score_col: str = "inflammation_index",
) -> pd.DataFrame:
    """Collapse per-cell scores to per-animal means.

    The animal-level mean is the recommended unit of statistical
    analysis — cells from the same animal are not independent samples,
    so direct cell-level t-tests inflate n and give false significance.

    Returns columns ``[animal_col, score_col, "n_cells"]`` plus
    whatever other metadata columns are constant within each animal
    (e.g. Treatment, Genotype).
    """
    if score_col not in df.columns:
        raise ValueError(f"'{score_col}' not in dataframe.")
    if animal_col not in df.columns:
        raise ValueError(f"'{animal_col}' not in dataframe.")
    grp = df.groupby(animal_col, as_index=False)
    out = grp.agg(**{
        score_col: (score_col, "mean"),
        "n_cells": (score_col, "count"),
    })
    # Carry over metadata that's constant within each animal (one value).
    meta_candidates = [c for c in df.columns
                       if c not in (animal_col, score_col, "n_cells")
                       and df[c].dtype.kind not in "f"]
    for c in meta_candidates:
        per_animal_unique = df.groupby(animal_col)[c].nunique()
        if int(per_animal_unique.max()) == 1:
            # Constant within each animal — safe to carry over.
            first_val = df.groupby(animal_col, as_index=False)[c].first()
            out = out.merge(first_val, on=animal_col, how="left")
    return out
