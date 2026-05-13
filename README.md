# GliaAnalysis

A streamlined GUI pipeline for **microglia morphology** and **astrocyte (GFAP) network**
analysis. Built as a user-friendly alternative to the Ciernia Lab
`MicrogliaMorphology` (FIJI) + `MicrogliaMorphologyR` stack and the BrainEnergyLab
`Inflammation-Index` package, with both microglia and astrocyte workflows in one app.

The segmentation engine is FIJI/ImageJ (so thresholding stays validated against the
literature), but the morphology extraction, clustering, statistics, and inflammation
scoring all run in Python — no R required.

## What it does

**Per-cell microglia morphology** (mode = "microglia"):

- 36 features per cell: 18 FracLac-equivalent geometric (area, perimeter, hull,
  bounding circle, span ratio), 9 AnalyzeSkeleton-equivalent (branches, junctions,
  branch lengths), 5 radial-scan soma metrics (area, circularity, soma:cell ratio,
  primary process count), 4 Sholl-derived metrics (critical radius, ramification
  index, max extent, max intersections).
- Optional **DAPI nucleus-seeded soma centering** when a DAPI channel is available.
- StandardScaler → PCA → KMeans clustering with auto-labeling against the four
  canonical morphologies (ameboid / hypertrophic / rod-like / ramified).
- Per-image cluster overlays with PNG export.

**Per-(image, ROI) astrocyte network** (mode = "astrocyte"):

- 9 GFAP-network features per ROI: area fraction, total skeleton length,
  branches / junctions, mean branch length, branch density per 1000 px², mean
  intensity inside the mask, soma count.
- Same Prepare → Threshold pipeline as microglia, but with the GFAP channel.

**Both modes share:**

- **Inflammation Index** — supervised PCA score trained on two reference groups
  (e.g. Control vs LPS); applied to every cell or ROI so held-out groups land on the
  same activation axis. Generalizes the BrainEnergyLab R package.
- **Explore** distributions + Spearman correlations.
- **Stats** — animal-level ANOVA / Welch / Kruskal-Wallis with pingouin post-hocs;
  cluster-proportion ANOVA (microglia mode only).
- **Export** — features.csv / astrocyte_features.csv, group plots, project state.
- Per-project persistence: ROIs, threshold settings, metadata, trained
  Inflammation Index, UI selections — all round-trip in `.gliaanalysis_settings.json`.

## Installation

### Prerequisites — all platforms

1. **Python 3.10 or newer** (3.11 / 3.12 / 3.13 work).
2. **FIJI / ImageJ** with Bio-Formats. Download from
   <https://imagej.net/software/fiji/downloads>. Bio-Formats ships bundled.

### Mac / Linux

```bash
git clone https://github.com/EpilepsyCenter/GliaAnalysis.git
cd GliaAnalysis

# Optional but recommended: isolated venv
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# Start the app
python app.py
```

Then open <http://127.0.0.1:8050/> in your browser.

### Windows

```powershell
git clone https://github.com/EpilepsyCenter/GliaAnalysis.git
cd GliaAnalysis

py -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
# or
.\.venv\Scripts\activate.bat   # cmd.exe

pip install -r requirements.txt

python app.py
```

Then open <http://127.0.0.1:8050/>.

On Windows, when configuring FIJI in the Setup → Threshold tab, point at the
`ImageJ-win64.exe` inside the FIJI folder (typically `C:\Fiji.app\ImageJ-win64.exe`).
On macOS the path is `/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx`.
On Linux it's the `ImageJ-linux64` binary inside the Fiji folder.

### Optional: Olympus `.vsi` files

`.vsi` requires a one-time Bio-Formats conversion via FIJI. The app does this
automatically on first folder load — make sure FIJI is configured before opening
a project that contains `.vsi` files.

## Quick start

### 1. Make a project folder

Any folder containing your raw microscopy images works:

- TIFF / OME-TIFF, CZI, ND2, LIF, VSI.
- Multi-channel files are fine — the app lets you pick which channel is the
  microglia / astrocyte / DAPI signal per image.
- Z-stacks are projected (max / mean / first / centre) at Prepare time.

### 2. Configure mode

In the left sidebar, set **Mode** to either **Microglia** or **Astrocyte**. The
tab bar reshapes for the active mode:

- Microglia: Setup → Segment → Features → Explore → Cluster → Inflammation → Stats → Export.
- Astrocyte: Setup → Astrocyte → Explore → Inflammation → Stats → Export.

Each mode keeps its own `Prepared/` and `ThresholdedImages/` folders under
`_gliaanalysis/` so switching modes doesn't overwrite the other mode's outputs.

### 3. Setup → Prepare

Set FIJI executable path (one-time, per machine). For each image, pick:

- The **primary channel** (Iba1 for microglia, GFAP for astrocyte).
- Optionally the **DAPI channel** (microglia mode) for nucleus-seeded soma centering.
- The **Z-projection** mode (max is the typical default).
- Metadata (Animal, Genotype, Treatment, …) — used by Stats and Inflammation.

Click **Prepare all images**.

### 4. Setup → ROIs (optional)

Draw rectangles / ellipses / polygons over regions you want to analyze separately
(e.g. CA1, DG). Tag each ROI. Cells / metrics are reported per ROI tag.

### 5. Setup → Threshold

Tune the threshold (global Otsu by default), set area bounds, preview on a
representative image. The same params drive the FIJI pipeline in step 6.

### 6a. Microglia: Segment → Features

- **Segment** runs FIJI thresholding + Python single-cell extraction + skeleton
  analysis. Produces `_gliaanalysis/SingleCells/` and `_gliaanalysis/SkeletonResults/`.
- **Features** computes the 36 morphology features per cell, joins metadata, writes
  `features.csv`.

### 6b. Astrocyte: Astrocyte Analysis

- **Run thresholding (FIJI)** writes GFAP binaries into the astrocyte-mode
  `ThresholdedImages/`.
- **Run analysis** computes per-(image, ROI) network metrics, joins metadata,
  writes `astrocyte_features.csv`.

### 7. Cluster (microglia mode only)

PCA + KMeans + auto-labeling against the four canonical morphologies. Override any
label. The **Overlays** subtab shows each image with cells colored by cluster; a
"Download overlay PNG" button exports publication-ready figures.

### 8. Inflammation Index

Pick a control and a comparator group from the metadata; click **Train + apply**.
Forward-greedy feature selection finds the subset whose PC1 best separates the two
anchors, the model scores every cell or ROI (including held-out groups), and
`inflammation_index` becomes a column on Explore / Stats.

### 9. Explore + Stats

Inspect feature distributions and pairwise correlations in Explore. Run
animal-level ANOVA / Kruskal-Wallis with pingouin post-hocs in Stats.

### 10. Export

Save features tables, group plots, and project artifacts to
`_gliaanalysis/exports/`.

## Project layout

```
app.py                       Dash entry point
glia_dash/                   UI layer
  main.py                    Dash app, sidebar, mode-aware tab bar
  components.py              Shared layout helpers + native folder/file pickers
  server_state.py            UUID-keyed session state
  pages/                     One module per tab
    setup_*.py               Setup subtabs (Prepare / ROIs / Threshold / Soma)
    segment.py               Microglia segmentation pipeline runner
    features.py              Per-cell feature extraction
    cluster.py               PCA + KMeans + Overlays subtab
    inflammation.py          Inflammation Index training + applying
    explore.py               Distributions + correlations
    stats.py                 Animal-level ANOVA + posthocs
    astrocyte_analysis.py    Astrocyte (GFAP) per-ROI metrics
    export.py                CSV / plot export
glia/                        Library (UI-agnostic, importable from notebooks)
  config.py                  Feature lists (microglia + astrocyte)
  prepare.py                 Channel projection, mode-aware Prepared/ dirs
  segment.py                 FIJI threshold + Python single-cell extract
  skeleton.py                Pure-Python skeleton analysis
  features.py                FracLac-equivalent geometric features
  radial.py                  Unified soma + Sholl radial scan
  astrocyte.py               GFAP network metrics per (image, ROI)
  pca_cluster.py             PCA + KMeans + selection scan
  auto_label.py              Cluster → morphology template matching
  inflammation_index.py      Supervised PCA Inflammation Index
  stats.py                   pingouin wrappers, animal-level aggregation
  metadata.py                Per-image metadata join
  settings.py                Project-settings JSON read/write (file-locked)
  io.py                      Multi-format image loading (bioio)
  vsi_convert.py             Headless FIJI Bio-Formats VSI → OME-TIFF
  roi.py                     Per-ROI mask generation
fiji_macros/                 Headless .ijm / .bsh macros
docs/                        Design notes
tests/                       Pytest synthetic-binary feature tests
```

## Per-project artifacts

Each project folder gets a `_gliaanalysis/` (microglia) and/or
`_gliaanalysis/astrocyte/` (astrocyte) subdirectory:

```
<project>/
  ImageA.nd2
  ImageB.nd2
  …
  .gliaanalysis_settings.json    threshold, metadata, UI selections, trained models
  .gliaanalysis_rois.json        drawn ROIs per image
  _gliaanalysis/
    Converted/                   cached VSI → OME-TIFF (if any)
    Prepared/                    8-bit microglia-channel projections
    Prepared_dapi/               8-bit DAPI projections (if used)
    ThresholdedImages/           FIJI binary output (microglia)
    SingleCells/                 per-cell crops
    SkeletonResults/             per-cell skeleton CSVs
    SkeletonImages/              per-cell skeleton overlays
    features.csv                 microglia feature table
    exports/                     user-driven exports
    astrocyte/
      Prepared/                  8-bit GFAP-channel projections
      ThresholdedImages/         FIJI binary output (astrocyte)
      astrocyte_features.csv     astrocyte feature table
```

These files round-trip on folder reopen; nothing is lost between sessions.

## Status

In active development. The pipelines are functional end-to-end; expect occasional
sharp edges. Report issues at
<https://github.com/EpilepsyCenter/GliaAnalysis/issues>.

## License

MIT.
