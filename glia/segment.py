"""Orchestrate the three-stage segmentation pipeline.

Phase 1 (FIJI headless): threshold every input TIFF
Phase 2 (Python):        label components, filter by area, crop per-cell TIFFs
Phase 3 (Python):        skeletonize + per-skeleton metrics (see glia.skeleton)

Phases 2 and 3 are Python because FIJI ``--headless`` has two blocking
issues: RoiManager throws ``HeadlessException`` (kills Phase 2) and
``Analyze Skeleton (2D/3D)`` silently fails to populate its Results table
(kills Phase 3). The metrics, naming, and CSV format are kept compatible
with what the original Ciernia macro produces, so the downstream loader
(:func:`glia.features.load_skeleton_results`) is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
from skimage import measure

from glia.fiji_runner import run_headless
from glia.roi import per_roi_masks
from glia.skeleton import analyze_directory as analyze_skeleton_directory


# Default ROI tag for images that have none drawn. Cells get
# "<image_stem>__all__<N>.tif" so downstream parsing still has three groups.
DEFAULT_ROI_TAG = "all"


_THRESHOLD_MACRO = Path(__file__).parent.parent / "fiji_macros" / "threshold.ijm"


@dataclass
class SegmentParams:
    input_dir: Path
    output_dir: Path
    fiji_path: str
    threshold_kind: str = "global"        # global | local | manual
    threshold_method: str = "Otsu"
    manual_lower: int = 0
    manual_upper: int = 255
    local_radius: int = 100
    area_min: float = 100.0
    area_max: float = 10_000.0
    preprocess: bool = True
    # rois maps absolute image path -> list of ROI dicts (each with tag,
    # type, shape). Pass {} (default) to process whole-image with tag "all".
    rois: dict = field(default_factory=dict)
    # Optional parallel directory of DAPI / nucleus-channel grayscale TIFFs.
    # Filenames must match the prepared cell-channel TIFFs (same stems).
    # When set, single-cell extraction also crops the DAPI image at each
    # cell's bbox and writes a sibling ``<cell_id>__dapi.tif`` so the
    # downstream radial analysis can seed its center on the nucleus.
    dapi_dir: Path | None = None

    def fiji_args(self) -> dict:
        return {
            "input_dir": str(self.input_dir),
            "output_dir": str(self.thresholded_dir),
            "threshold_kind": self.threshold_kind,
            "threshold_method": self.threshold_method,
            "manual_lower": int(self.manual_lower),
            "manual_upper": int(self.manual_upper),
            "local_radius": int(self.local_radius),
            "preprocess": 1 if self.preprocess else 0,
        }

    @property
    def thresholded_dir(self) -> Path:
        return self.output_dir / "ThresholdedImages"

    @property
    def cells_dir(self) -> Path:
        return self.output_dir / "SingleCells"

    @property
    def skeleton_dir(self) -> Path:
        return self.output_dir / "SkeletonResults"

    @property
    def skeleton_img_dir(self) -> Path:
        return self.output_dir / "SkeletonImages"


@dataclass
class SegmentReport:
    n_input_images: int = 0
    n_thresholded: int = 0
    n_single_cells: int = 0
    n_skeleton_csvs: int = 0
    n_cleared: int = 0          # prior-run files wiped before this run
    phase1_stdout: str = ""
    skipped: list[str] = field(default_factory=list)


def _source_stem(thresholded_tif: Path) -> str:
    """Recover the original image stem from a thresholded TIFF filename.

    The threshold.ijm macro saves as ``<original>.tif_thresholded.tif``,
    so the stem still has the source extension baked in. Strip both the
    ``_thresholded`` suffix and the trailing ``.tif``/``.tiff`` so the
    stem reads as the original image base (e.g. ``449380_Control``).
    """
    stem = thresholded_tif.stem
    if stem.endswith("_thresholded"):
        stem = stem[: -len("_thresholded")]
    if stem.endswith(".tif"):
        stem = stem[:-4]
    elif stem.endswith(".tiff"):
        stem = stem[:-5]
    return stem


def _find_source_path(input_dir: Path, stem: str) -> str:
    for ext in (".tif", ".tiff", ".TIF", ".TIFF"):
        p = input_dir / f"{stem}{ext}"
        if p.exists():
            return str(p)
    return ""


def extract_single_cells(
    thresholded_dir: Path,
    cells_dir: Path,
    area_min: float,
    area_max: float,
    input_dir: Path | None = None,
    rois: dict | None = None,
    dapi_dir: Path | None = None,
) -> tuple[int, list[str]]:
    """Read every thresholded TIFF, label foreground components, and save
    each cell whose pixel area falls in (area_min, area_max) as a cropped
    TIFF.

    Filename: ``<image_stem>__<roi_tag>__<N>.tif``. `__` (double
    underscore) separates the three groups; single underscores stay free
    for image-level metadata and intra-tag separators.

    When ``rois`` is provided and the source image has at least one ROI,
    we run the labeling/extraction once per ROI with that ROI's mask
    AND-ed into the binary, and tag each cell with the ROI's name.
    Images with no ROIs fall through to a single pass tagged
    :data:`DEFAULT_ROI_TAG` (=\"all\").
    """
    cells_dir.mkdir(parents=True, exist_ok=True)
    rois = rois or {}
    n_total = 0
    skipped: list[str] = []

    for tif in sorted(thresholded_dir.glob("*.tif")):
        try:
            img = tifffile.imread(tif)
        except Exception as e:
            skipped.append(f"{tif.name}: {e}")
            continue
        binary = img > 0
        h, w = binary.shape

        stem = _source_stem(tif)
        source_path = (_find_source_path(input_dir, stem)
                       if input_dir is not None else "")
        image_rois = rois.get(source_path, []) if source_path else []

        # Match a parallel DAPI image by stem if one was supplied. We
        # apply a single global Otsu threshold to the full DAPI image
        # here — not per-crop later — so faint cells don't get noise-
        # thresholded into spurious centroids. The sibling file written
        # below is therefore a binary mask, and downstream
        # ``dapi_centroid`` just takes the centroid of the largest
        # surviving connected component.
        #
        # Shape mismatch (channel registration off) silently disables
        # DAPI for this image — better than producing wrong centers.
        dapi_binary = None
        if dapi_dir is not None:
            for ext in (".tif", ".tiff"):
                candidate = Path(dapi_dir) / f"{stem}{ext}"
                if candidate.is_file():
                    try:
                        di = tifffile.imread(candidate)
                        if di.shape[:2] == (h, w):
                            from skimage.filters import threshold_otsu
                            try:
                                t = float(threshold_otsu(di))
                                dapi_binary = (di > t).astype(np.uint8) * 255
                            except Exception:
                                dapi_binary = None
                    except Exception:
                        pass
                    break

        if image_rois:
            passes = per_roi_masks(image_rois, h, w)
        else:
            passes = [(DEFAULT_ROI_TAG, np.ones((h, w), dtype=bool))]

        for tag, roi_arr in passes:
            roi_binary = binary & roi_arr
            if not roi_binary.any():
                continue
            # 4-connectivity matches FIJI's Analyze Particles default;
            # 8-connectivity merges distinct cells touching only at corners.
            labels = measure.label(roi_binary, connectivity=1)
            for rp in measure.regionprops(labels):
                area = rp.area
                if not (area_min < area < area_max):
                    continue
                r0, c0, r1, c1 = rp.bbox
                mask = (labels[r0:r1, c0:c1] == rp.label)
                crop = (mask.astype(np.uint8)) * 255
                cell_id = f"{stem}__{tag}__{rp.label}"
                tifffile.imwrite(cells_dir / f"{cell_id}.tif", crop)
                if dapi_binary is not None:
                    dapi_crop = dapi_binary[r0:r1, c0:c1]
                    tifffile.imwrite(
                        cells_dir / f"{cell_id}__dapi.tif", dapi_crop,
                    )
                n_total += 1
    return n_total, skipped


def run_pipeline(params: SegmentParams) -> SegmentReport:
    """End-to-end: threshold → extract single cells → skeletonize.

    Returns a SegmentReport with counts and FIJI stdout for diagnostics.
    """
    if not Path(params.fiji_path).exists():
        raise FileNotFoundError(f"FIJI executable not found: {params.fiji_path}")
    if not params.input_dir.is_dir():
        raise NotADirectoryError(f"input_dir is not a directory: {params.input_dir}")
    params.output_dir.mkdir(parents=True, exist_ok=True)

    report = SegmentReport()
    inputs = sorted([p for p in params.input_dir.glob("*.tif")] +
                    [p for p in params.input_dir.glob("*.tiff")])
    report.n_input_images = len(inputs)

    # ── Wipe prior output ────────────────────────────────────────────
    # Default behaviour is overwrite — if you re-run with a different
    # threshold, the new component labels and indices won't line up with
    # the old per-cell files, so leftover files create ID mismatches
    # downstream. Wipe the four sub-folders fully.
    cleared = 0
    for d in (params.thresholded_dir, params.cells_dir,
              params.skeleton_dir, params.skeleton_img_dir):
        if d.exists():
            for p in d.iterdir():
                if p.is_file():
                    try:
                        p.unlink()
                        cleared += 1
                    except Exception:
                        pass
    # Also wipe Areas.csv leftover from threshold.ijm.
    legacy_areas = params.thresholded_dir / "Areas.csv"
    if legacy_areas.exists():
        try:
            legacy_areas.unlink()
            cleared += 1
        except Exception:
            pass
    report.n_cleared = cleared

    # ── Phase 1: FIJI thresholding ───────────────────────────────────
    params.thresholded_dir.mkdir(parents=True, exist_ok=True)
    proc1 = run_headless(params.fiji_path, _THRESHOLD_MACRO, params.fiji_args())
    report.phase1_stdout = proc1.stdout + ("\n" + proc1.stderr if proc1.stderr else "")
    report.n_thresholded = len(list(params.thresholded_dir.glob("*.tif")))

    # ── Phase 2: Python single-cell extraction (ROI-aware) ───────────
    n_cells, skipped = extract_single_cells(
        params.thresholded_dir, params.cells_dir,
        params.area_min, params.area_max,
        input_dir=params.input_dir,
        rois=params.rois,
        dapi_dir=params.dapi_dir,
    )
    report.n_single_cells = n_cells
    report.skipped.extend(skipped)

    # ── Phase 3: Python skeleton analysis ────────────────────────────
    if n_cells == 0:
        return report
    n_csvs, skel_skipped = analyze_skeleton_directory(
        params.cells_dir, params.skeleton_dir, params.skeleton_img_dir,
    )
    report.n_skeleton_csvs = n_csvs
    report.skipped.extend(skel_skipped)
    return report
