# GliaAnalysis architecture

## Goal

User-friendly GUI replacement for the Ciernia Lab pipeline:
- FIJI **MicrogliaMorphology** (.ijm macros) — kept as the segmentation engine,
  but called headlessly and stripped of its manual dialogs.
- **MicrogliaMorphologyR** (R package) — replaced entirely by Python.
- **FracLac** (FIJI plugin, GUI-only) — replaced by scikit-image.
- **BioVoxxel ThresholdCheck** (FIJI plugin, GUI-only) — replaced by an in-app
  threshold preview.

## Stack

`dash + dash-bootstrap-components + pandas + scikit-image + scikit-learn +
pingouin + plotly` plus a disk-installed FIJI for the validated
thresholding/skeleton steps. Shares the design system (CSS tokens, sidebar
layout, native folder/file pickers) with the sibling NED-Net Dash app —
see `glia_dash/assets/style.css`.

## Pipeline

```
TIFFs (one Iba1 channel, optional ROI)
       │
       ▼
[Setup tab]   threshold method + area bounds (live preview)
       │
       ▼
[Segment tab] FIJI headless: threshold → AnalyzeParticles → Skeletonize +
              AnalyzeSkeleton → writes ThresholdedImages/, SingleCells/,
              SkeletonResults/, Areas.csv
       │
       ▼
[Features tab] scikit-image: 18 FracLac-equivalent geometric features per cell;
               merge with the 9 AnalyzeSkeleton features → 27-feature df
       │
       ▼
[Metadata tab] parse "Cohort_Animal_Condition_Sex_Region" filename or join CSV
       │
       ▼
[Explore tab] feature distributions, correlation heatmap, transform toggle
       │
       ▼
[Cluster tab] StandardScaler → PCA → KMeans; elbow + silhouette guidance;
              auto-labeling against ameboid/hypertrophic/rod/ramified templates
              via cosine similarity on a 6-feature z-score profile;
              override-able
       │
       ▼
[Stats tab]  aggregate cells → animals; pingouin ANOVA per feature + per
             cluster % (arcsine-sqrt transformed); Tukey/Holm posthocs
       │
       ▼
[Export tab] ColorByCluster.csv (round-trip to FIJI), features.csv,
             stats.csv, figures
```

## Key design decisions

1. **Keep FIJI for segmentation.** AnalyzeSkeleton is well-validated; rebuilding
   it in skimage would invite a new validation burden. We strip out the manual
   dialogs and call via subprocess.
2. **Replace FracLac with scikit-image.** FracLac's "Hull and Circle Results"
   are geometric, not fractal — all 18 features are computable from a binary
   mask with `regionprops` + `ConvexHull` + minimum-enclosing-circle.
   **Risk:** numerical drift vs FracLac. Validation: per-feature Spearman
   correlation ≥ 0.95 on a real test set before declaring done.
3. **pingouin + animal-level aggregation > lme4-style mixed models.** Slight
   power cost vs cell-level mixed models, but no R dependency and a much
   cleaner UI. Cell-level mode available but flagged for pseudoreplication.
4. **Auto cluster labeling with override.** Score cluster z-profiles against
   four canonical morphology templates via cosine similarity; assign greedily;
   show the score matrix so users can override anything that looks off.
5. **Astrocyte mode scaffolded but deferred to v0.2.** Simpler pipeline:
   threshold → area fraction + mean intensity. No clustering needed.

## Known unknowns

- FracLac equivalence: needs real-data side-by-side check (waiting on images).
- Threshold method default: 'Otsu' is a safe placeholder but may not be best
  for fluorescent Iba1.
- Auto-label template weights in `glia.config.MORPHOLOGY_TEMPLATES` are
  educated guesses; will need tuning against labeled examples.
- FIJI executable path discovery: macOS-leaning, will need cross-platform
  hardening if anyone outside the lab uses this.
