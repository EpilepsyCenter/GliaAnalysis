"""Stats wrappers — pingouin-based, animal-level aggregation by default.

Rationale: the original R package used cell-level mixed models with a random
effect for animal ID. Python lacks a clean lme4 equivalent. The standard
workaround in glia research is to aggregate cells to animal-level means
before running fixed-effects stats — slight power cost, no R dependency,
much friendlier UI. Cell-level mode is available but flagged for
pseudoreplication.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pingouin as pg


def aggregate_to_animal(
    df: pd.DataFrame, animal_col: str, feature_cols: list[str], factor_cols: list[str]
) -> pd.DataFrame:
    """One row per animal: mean of each feature; experimental factors carried over."""
    agg = df.groupby([animal_col, *factor_cols])[feature_cols].mean().reset_index()
    return agg


def cluster_percentages_per_animal(
    df: pd.DataFrame, animal_col: str, factor_cols: list[str],
    cluster_col: str = "Cluster",
) -> pd.DataFrame:
    """% of each cluster per animal. Arcsine-sqrt transforms the proportion."""
    counts = (
        df.groupby([animal_col, *factor_cols, cluster_col])
        .size()
        .reset_index(name="n")
    )
    totals = counts.groupby([animal_col, *factor_cols])["n"].transform("sum")
    counts["proportion"] = counts["n"] / totals
    counts["arcsine_sqrt"] = np.arcsin(np.sqrt(counts["proportion"]))
    return counts


def feature_anova(
    df: pd.DataFrame, feature_cols: list[str], factors: list[str],
    posthoc: bool = True, padjust: str = "holm",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two-way (or more) ANOVA per feature, plus pairwise posthocs.

    Returns (anova_table, posthoc_table). Both have a 'measure' column to
    indicate which feature the row applies to.
    """
    anova_rows = []
    posthoc_rows = []
    for f in feature_cols:
        a = pg.anova(data=df, dv=f, between=factors, detailed=True)
        a["measure"] = f
        anova_rows.append(a)
        if posthoc:
            ph = pg.pairwise_tests(
                data=df, dv=f, between=factors, padjust=padjust,
            )
            ph["measure"] = f
            posthoc_rows.append(ph)
    anova = pd.concat(anova_rows, ignore_index=True)
    post = pd.concat(posthoc_rows, ignore_index=True) if posthoc_rows else pd.DataFrame()
    return anova, post


def normality_check(
    df: pd.DataFrame, feature_cols: list[str], group_col: str | None = None,
) -> pd.DataFrame:
    """Shapiro-Wilk per feature (per group if group_col given)."""
    if group_col is None:
        return pg.normality(df[feature_cols])
    return pg.normality(data=df, dv=feature_cols, group=group_col)


def group_sizes(df: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """Row count per combination of factor levels."""
    if not factors:
        return pd.DataFrame({"n": [len(df)]})
    return (df.groupby(factors, dropna=False).size()
              .reset_index(name="n"))


def diagnostics(
    df: pd.DataFrame, feature_cols: list[str], factors: list[str],
) -> dict:
    """Per-feature normality (Shapiro per group) + Levene homogeneity.

    With 2+ factors we cross them into a single label so the diagnostics
    apply to the actual ANOVA cells.
    """
    if not factors or not feature_cols:
        return {"normality": pd.DataFrame(), "levene": pd.DataFrame()}
    group_label = df[factors].astype(str).agg("|".join, axis=1)
    work = df.assign(_grp=group_label)
    norm_rows = []
    levene_rows = []
    for f in feature_cols:
        try:
            n = pg.normality(data=work, dv=f, group="_grp")
            n = n.reset_index().rename(columns={"index": "group"})
            n["measure"] = f
            norm_rows.append(n)
        except Exception:
            pass
        try:
            lv = pg.homoscedasticity(data=work, dv=f, group="_grp",
                                     method="levene")
            lv["measure"] = f
            levene_rows.append(lv)
        except Exception:
            pass
    return {
        "normality": (pd.concat(norm_rows, ignore_index=True)
                      if norm_rows else pd.DataFrame()),
        "levene":    (pd.concat(levene_rows, ignore_index=True)
                      if levene_rows else pd.DataFrame()),
    }


def feature_test(
    df: pd.DataFrame, feature_cols: list[str], factors: list[str],
    method: str = "anova", posthoc: bool = True, padjust: str = "holm",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the chosen omnibus test + pairwise posthocs per feature.

    method:
      - "anova":   classical ANOVA (assumes normality + equal variances)
      - "welch":   Welch's ANOVA (unequal variances; one factor only)
      - "kruskal": Kruskal-Wallis (non-parametric; one factor only)
    """
    omnibus_rows = []
    posthoc_rows = []
    parametric = method != "kruskal"
    for f in feature_cols:
        try:
            if method == "anova":
                a = pg.anova(data=df, dv=f, between=factors, detailed=True)
            elif method == "welch":
                a = pg.welch_anova(data=df, dv=f, between=factors[0])
            elif method == "kruskal":
                a = pg.kruskal(data=df, dv=f, between=factors[0])
            else:
                raise ValueError(f"Unknown method: {method}")
            a["measure"] = f
            omnibus_rows.append(a)
        except Exception as e:
            omnibus_rows.append(pd.DataFrame([{
                "Source": "(failed)", "p-unc": None,
                "measure": f, "error": str(e),
            }]))
        if posthoc:
            try:
                ph = pg.pairwise_tests(
                    data=df, dv=f, between=factors,
                    padjust=padjust, parametric=parametric,
                )
                ph["measure"] = f
                posthoc_rows.append(ph)
            except Exception:
                pass
    omnibus = pd.concat(omnibus_rows, ignore_index=True)
    post = (pd.concat(posthoc_rows, ignore_index=True)
            if posthoc_rows else pd.DataFrame())
    return omnibus, post
