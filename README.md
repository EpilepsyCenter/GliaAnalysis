# GliaAnalysis

A streamlined GUI pipeline for microglia morphology and astrocyte (GFAP) analysis.

Built as a user-friendly alternative to the Ciernia Lab `MicrogliaMorphology` (FIJI)
+ `MicrogliaMorphologyR` (R) stack. Keeps the validated FIJI thresholding/skeleton
analysis as the segmentation engine, replaces FracLac with scikit-image, and replaces
the R post-analysis with a Dash app using pingouin for stats. Shares its design
system (theme, sidebar layout, dialog helpers) with the sibling NED-Net app.

## Status

Scaffold only. Targets v0.1 once test images are available.

## Run

```
pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:8050/>. Requires a FIJI/ImageJ installation; path set
on the Setup tab.

## Layout

```
app.py                       Dash entry (thin wrapper around glia_dash.main)
glia_dash/                   UI layer
  main.py                    Dash() app, sidebar + tab bar, callbacks
  components.py              Layout helpers + native folder/file pickers
  server_state.py            UUID-keyed session state
  assets/style.css           Shared design tokens (same as NED-Net)
  pages/                     Per-tab layout(sid) modules
glia/                        Library code (UI-agnostic)
  config.py                  Constants, feature lists
  fiji_runner.py             Headless FIJI subprocess wrapper
  features.py                scikit-image FracLac-equivalent + skeleton parsing
  ingest.py                  CSV reading, filename metadata parsing
  transforms.py              log / z-score / min-max
  pca_cluster.py             PCA + k-means
  auto_label.py              Rule-based morphology labeling
  stats.py                   pingouin wrappers, animal-level aggregation
  plots.py                   Plotly helpers
fiji_macros/                 Stripped-down headless .ijm macros
tests/                       Synthetic-binary tests for feature equivalence
docs/architecture.md         Design notes
```
