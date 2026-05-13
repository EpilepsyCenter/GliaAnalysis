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


def aggregate_to_animal_roi(
    df: pd.DataFrame, animal_col: str, feature_cols: list[str],
    factor_cols: list[str], roi_col: str = "roi_tag",
) -> pd.DataFrame:
    """One row per (animal, roi_tag): mean of each feature.

    Factors are carried through. Used as the unit-of-analysis frame for
    the mixed-effects path so each animal contributes one observation
    per ROI and pseudoreplication across cells/images collapses to a
    proper within-subject design.
    """
    keys = [animal_col, *[c for c in factor_cols if c != roi_col], roi_col]
    return (df.groupby(keys, dropna=False)[feature_cols]
              .mean()
              .reset_index())



def mixed_effects_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    animal_col: str,
    between_cols: list[str],
    within_col: str = "roi_tag",
    padjust: str = "holm",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Linear mixed-effects model per feature, Animal as random intercept.

    Fits ``feature ~ between * within + (1 | animal)`` via
    ``statsmodels.formula.api.mixedlm`` after aggregating cells/images
    to (animal × within) means. Handles unbalanced ROI sets per
    animal natively (REML) — no animal is dropped for missing a level,
    unlike ``pingouin.mixed_anova``.

    Returns
    -------
    fixed_df : DataFrame
        One row per (feature, fixed-effect term) with estimate, SE,
        Wald z, and Wald p-value. The Intercept row is kept (useful
        sanity check) but flagged.
    pairwise_df : DataFrame
        Pingouin pairwise t-tests on the aggregated frame across the
        full between × within cross, with the chosen p-adjustment.
    info : dict
        ``formula`` (str), ``n_obs``, ``n_animals``, and a per-feature
        ``random_effects`` DataFrame with Animal-level SD + convergence
        flag, so users can see how much variance the random effect
        absorbed.
    """
    import statsmodels.formula.api as smf

    factors = [c for c in between_cols if c != within_col]
    work = aggregate_to_animal_roi(df, animal_col, feature_cols,
                                   factors, roi_col=within_col)

    rhs_terms = [*factors, within_col]
    rhs = " * ".join(rhs_terms) if len(rhs_terms) > 1 else rhs_terms[0]
    formula_template = "Q('{f}') ~ " + rhs

    # Reference levels (alphabetically first — Patsy's default for
    # treatment coding). We need these to translate parameter names
    # like 'Treatment[T.SV2A]' into 'SV2A vs Control' and to compute
    # per-ROI Treatment contrasts.
    references: dict[str, str] = {}
    levels: dict[str, list[str]] = {}
    for col in rhs_terms:
        uniq = sorted(str(v) for v in work[col].dropna().unique())
        levels[col] = uniq
        if uniq:
            references[col] = uniq[0]

    fx_rows: list[pd.DataFrame] = []
    pw_rows: list[pd.DataFrame] = []
    ct_rows: list[pd.DataFrame] = []
    re_rows: list[dict] = []

    for f in feature_cols:
        sub = work[[animal_col, *rhs_terms, f]].dropna(subset=[f])
        # mixedlm needs ≥2 distinct groups and some variance.
        if sub[f].nunique() < 2 or sub[animal_col].nunique() < 2:
            fx_rows.append(pd.DataFrame([{
                "measure": f, "effect": "(skipped)",
                "estimate": np.nan, "se": np.nan,
                "z": np.nan, "p": np.nan,
                "note": "insufficient variance or only one animal",
            }]))
            continue
        try:
            md = smf.mixedlm(formula_template.format(f=f),
                             data=sub, groups=sub[animal_col])
            mdf = md.fit(reml=True, method="lbfgs")
            # cov_re is the random-effects covariance (Animal Var here).
            try:
                animal_var = float(mdf.cov_re.iloc[0, 0])
            except Exception:
                animal_var = float("nan")
            re_rows.append({
                "measure": f,
                "animal_sd": (float(np.sqrt(animal_var))
                              if animal_var == animal_var and animal_var >= 0
                              else None),
                "converged": bool(getattr(mdf, "converged", False)),
            })
            params = mdf.params.copy()
            # Drop the random-effects variance row from the fixed-effect
            # table — it's reported separately in re_rows.
            for drop in ("Group Var", "Group x animal Cov"):
                params = params.drop(labels=[drop], errors="ignore")
            tidy = pd.DataFrame({
                "measure": f,
                "effect": params.index,
                "estimate": params.values,
                "se": mdf.bse.reindex(params.index).values,
                "z": mdf.tvalues.reindex(params.index).values,
                "p": mdf.pvalues.reindex(params.index).values,
            })
            fx_rows.append(tidy)
        except Exception as e:
            fx_rows.append(pd.DataFrame([{
                "measure": f, "effect": "(failed)",
                "estimate": np.nan, "se": np.nan,
                "z": np.nan, "p": np.nan,
                "note": str(e),
            }]))
            continue

        # Per-(within) Treatment contrasts from the LMM itself. These
        # answer the question users actually have ("Is SV2A different
        # from Control in DG?") instead of the raw coefficient table,
        # which only gives differences-from-reference and interactions.
        # Computed by combining fixed-effect parameters with their
        # covariance — i.e. proper Wald tests on linear combinations.
        try:
            ct = _per_within_contrasts(
                mdf, factors, within_col, levels, measure=f,
            )
            if not ct.empty:
                ct_rows.append(ct)
        except Exception:
            pass

        # Pairwise contrasts on the aggregated frame (pingouin) — kept
        # as a secondary view. Independent of the LMM's pooling so
        # numbers may differ slightly from the contrasts above.
        try:
            ph = pg.pairwise_tests(
                data=sub, dv=f,
                between=rhs_terms,
                padjust=padjust, parametric=True,
            )
            ph["measure"] = f
            pw_rows.append(ph)
        except Exception:
            pass

    fixed_df = (pd.concat(fx_rows, ignore_index=True)
                if fx_rows else pd.DataFrame())
    pairwise_df = (pd.concat(pw_rows, ignore_index=True)
                   if pw_rows else pd.DataFrame())
    contrasts_df = (pd.concat(ct_rows, ignore_index=True)
                    if ct_rows else pd.DataFrame())

    # Apply the same multiple-comparison correction the user picked for
    # the pingouin pairwise table, so the two views are comparable.
    if not contrasts_df.empty and "p" in contrasts_df.columns:
        try:
            from statsmodels.stats.multitest import multipletests
            valid = contrasts_df["p"].notna()
            if valid.any() and padjust and padjust != "none":
                method = {"bonf": "bonferroni", "fdr_bh": "fdr_bh",
                          "sidak": "sidak", "holm": "holm"}.get(
                              padjust, "holm")
                pvals = contrasts_df.loc[valid, "p"].to_numpy()
                _, p_adj, _, _ = multipletests(pvals, method=method)
                contrasts_df.loc[valid, "p_adj"] = p_adj
            else:
                contrasts_df["p_adj"] = contrasts_df["p"]
        except Exception:
            pass

    info = {
        "formula": f"feature ~ {rhs} + (1 | {animal_col})",
        "n_obs": int(len(work)),
        "n_animals": int(work[animal_col].nunique()),
        "random_effects": (pd.DataFrame(re_rows)
                           if re_rows else pd.DataFrame()),
        "contrasts": contrasts_df,
        "references": references,
        "within_col": within_col,
        "between_cols": list(factors),
        "padjust": padjust,
    }
    return fixed_df, pairwise_df, info


def _per_within_contrasts(
    mdf, between_cols: list[str], within_col: str,
    levels: dict[str, list[str]], measure: str,
) -> pd.DataFrame:
    """All pairwise contrasts of the first between factor, within each
    level of ``within_col``, computed from the fitted LMM.

    For an unbalanced 3×3 design (Treatment ∈ {Ctrl, A, B}, ROI ∈
    {CA1, CA3, DG}) this returns 9 rows per feature: 3 pairwise
    Treatment contrasts (A−Ctrl, B−Ctrl, B−A) inside each ROI.

    The contrast is a Wald test on a linear combination of the LMM's
    fixed-effect parameters — so the standard error uses the model's
    parameter covariance and the test inherits the LMM's pooling
    (better than independent t-tests when sample sizes are small).
    """
    if not between_cols:
        return pd.DataFrame()
    treat_col = between_cols[0]
    treat_levels = levels.get(treat_col, [])
    within_levels = levels.get(within_col, [])
    if len(treat_levels) < 2 or not within_levels:
        return pd.DataFrame()
    treat_ref = treat_levels[0]
    within_ref = within_levels[0]

    # t_test operates on fixed effects only — params also contains
    # 'Group Var', which would shift the contrast vector and silently
    # produce wrong contrasts.
    params = mdf.fe_params
    param_index = {n: i for i, n in enumerate(params.index)}

    def add(coef: dict[str, float], name: str, c: float):
        coef[name] = coef.get(name, 0.0) + c

    rows: list[dict] = []
    for win_level in within_levels:
        for i, a in enumerate(treat_levels):
            for b in treat_levels[i + 1:]:
                # E[a in r] - E[b in r] in parameter space.
                # See module docstring above for the algebra.
                coef: dict[str, float] = {}
                if a != treat_ref:
                    add(coef, f"{treat_col}[T.{a}]", +1.0)
                if b != treat_ref:
                    add(coef, f"{treat_col}[T.{b}]", -1.0)
                if win_level != within_ref:
                    if a != treat_ref:
                        add(coef, f"{treat_col}[T.{a}]:"
                                  f"{within_col}[T.{win_level}]", +1.0)
                    if b != treat_ref:
                        add(coef, f"{treat_col}[T.{b}]:"
                                  f"{within_col}[T.{win_level}]", -1.0)

                # Build the contrast vector L in parameter order. Skip
                # the contrast entirely if any term refers to a
                # parameter that wasn't fit (e.g. cell collapsed by
                # missing data) — silently dropping a term would
                # produce a misleading "significant" result.
                L = np.zeros(len(params))
                missing = False
                for name, c in coef.items():
                    if name not in param_index:
                        missing = True
                        break
                    L[param_index[name]] = c
                if missing or not np.any(L):
                    continue
                try:
                    # mixedlm.t_test requires a 2D contrast matrix
                    # (n_contrasts × n_fixed_params), not a 1D vector.
                    # Return objects can be 1D arrays, 2D matrices, or
                    # 0-d scalars depending on the statsmodels version,
                    # so flatten before scalar conversion.
                    tt = mdf.t_test(L.reshape(1, -1))
                    est = float(np.asarray(tt.effect).ravel()[0])
                    se = float(np.asarray(tt.sd).ravel()[0])
                    z = float(np.asarray(tt.tvalue).ravel()[0])
                    p = float(np.asarray(tt.pvalue).ravel()[0])
                except Exception:
                    continue
                rows.append({
                    "measure": measure,
                    within_col: str(win_level),
                    "contrast": f"{a} − {b}",
                    "estimate": est,
                    "se": se,
                    "z": z,
                    "p": p,
                })
    return pd.DataFrame(rows)


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
