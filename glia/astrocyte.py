"""Astrocyte / GFAP field-level metrics.

GFAP astrocytes don't segment cleanly into individual cells the way Iba1
microglia do — astrocyte processes blend across the syncytium, so per-
cell morphology clustering isn't well-posed. Instead, we compute network
metrics on the thresholded GFAP image, one row per (image, ROI). The
output dataframe has the same metadata-join shape as the microglia
features.csv and feeds the same Explore / Stats / Inflammation Index
machinery.

The Setup pipeline (Prepare → ROIs → Threshold) is reused as-is: the
user picks the GFAP channel in the Prepare metadata table, and the
threshold step writes the binary GFAP mask to
``_gliaanalysis/ThresholdedImages/``. This module reads those binaries
plus the prepared grayscale images and emits per-ROI metrics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import binary_opening
from skimage import measure
from skimage.morphology import disk, skeletonize

from glia.config import ASTROCYTE_FEATURES
from glia.roi import per_roi_masks
from glia.segment import DEFAULT_ROI_TAG


ASTROCYTE_FEATURES_FILENAME = "astrocyte_features.csv"


@dataclass
class AstrocyteMetrics:
    gfap_area_px: float
    gfap_area_fraction: float
    skeleton_length_total_px: float
    n_branches: int
    n_junctions: int
    mean_branch_length_px: float
    branch_density_per_1000px2: float
    mean_intensity_in_mask: float
    soma_count: int


# Mapping from dataclass field → canonical column name in ASTROCYTE_FEATURES.
_FIELD_MAP = {
    "gfap_area_px": "GFAP area (px²)",
    "gfap_area_fraction": "GFAP area fraction",
    "skeleton_length_total_px": "Skeleton length total (px)",
    "n_branches": "# of branches",
    "n_junctions": "# of junctions",
    "mean_branch_length_px": "Mean branch length (px)",
    "branch_density_per_1000px2": "Branch density (per 1000 px²)",
    "mean_intensity_in_mask": "Mean intensity in mask",
    "soma_count": "Soma count",
}


def compute_astrocyte_metrics(
    binary_mask: np.ndarray,
    intensity_image: np.ndarray | None = None,
    roi_mask: np.ndarray | None = None,
    *,
    soma_radius_px: int = 4,
) -> AstrocyteMetrics:
    """Compute GFAP network metrics on one (image, ROI) combination.

    Args:
        binary_mask: 2D bool, foreground = GFAP-positive pixels.
        intensity_image: optional 2D grayscale of the same shape as
            ``binary_mask``. Used to compute mean intensity inside the
            mask. If None, the corresponding metric is 0.
        roi_mask: optional 2D bool, the ROI window. Metrics are
            computed on ``binary_mask & roi_mask`` and the area
            fraction is normalized to the ROI area. If None, the full
            image is the ROI.
        soma_radius_px: opening-disk radius used to count compact
            soma-like components. 4 px is a reasonable default at 20×;
            very high-magnification images may want larger.

    Returns:
        :class:`AstrocyteMetrics`.

    Raises:
        ValueError if ``binary_mask`` isn't 2D.
    """
    bm = np.asarray(binary_mask).astype(bool)
    if bm.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape {bm.shape}")

    if roi_mask is None:
        roi = np.ones_like(bm, dtype=bool)
    else:
        roi = np.asarray(roi_mask).astype(bool)
        if roi.shape != bm.shape:
            raise ValueError("roi_mask shape mismatch with binary_mask.")

    fg = bm & roi
    roi_area = float(roi.sum())
    fg_area = float(fg.sum())
    area_fraction = fg_area / roi_area if roi_area > 0 else 0.0

    # ── Skeleton-derived metrics ────────────────────────────────────
    if fg.any():
        skel = skeletonize(fg)
        skel_total = float(skel.sum())
        # Branches / junctions / endpoints from the 3x3 neighbour count
        # on the skeleton: junctions have >2 neighbours, endpoints have
        # 1, slabs have 2. # of branches ≈ (# endpoints + 2*# junctions)
        # / 2 (Euler-style; same as MicrogliaMorphology's reporting).
        neigh = _skeleton_neighbours(skel)
        n_endpoints = int(((skel) & (neigh == 1)).sum())
        n_junctions = int(((skel) & (neigh >= 3)).sum())
        n_branches = (n_endpoints + 2 * n_junctions) // 2
        mean_branch_len = (skel_total / n_branches
                           if n_branches > 0 else 0.0)
    else:
        skel_total = 0.0
        n_endpoints = 0
        n_junctions = 0
        n_branches = 0
        mean_branch_len = 0.0

    # ── Branch density per 1000 px² of ROI ──────────────────────────
    branch_density = (1000.0 * n_branches / roi_area
                      if roi_area > 0 else 0.0)

    # ── Mean intensity inside the mask ──────────────────────────────
    if intensity_image is not None and fg.any():
        img = np.asarray(intensity_image)
        if img.shape == fg.shape:
            mean_intensity = float(img[fg].astype(float).mean())
        else:
            mean_intensity = 0.0
    else:
        mean_intensity = 0.0

    # ── Soma count via morphological opening with a disk ────────────
    if fg.any():
        opened = binary_opening(fg, structure=disk(soma_radius_px))
        labels = measure.label(opened, connectivity=2)
        soma_count = int(labels.max())
    else:
        soma_count = 0

    return AstrocyteMetrics(
        gfap_area_px=fg_area,
        gfap_area_fraction=area_fraction,
        skeleton_length_total_px=skel_total,
        n_branches=int(n_branches),
        n_junctions=int(n_junctions),
        mean_branch_length_px=float(mean_branch_len),
        branch_density_per_1000px2=float(branch_density),
        mean_intensity_in_mask=float(mean_intensity),
        soma_count=int(soma_count),
    )


def metrics_to_row(metrics: AstrocyteMetrics) -> dict:
    """Map dataclass fields to the canonical ASTROCYTE_FEATURES columns."""
    d = asdict(metrics)
    return {_FIELD_MAP[k]: v for k, v in d.items()}


def _skeleton_neighbours(skel: np.ndarray) -> np.ndarray:
    """Per-pixel 8-neighbour count on a binary skeleton."""
    s = skel.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    # 8-neighbour count via convolution, minus the centre itself.
    from scipy.signal import convolve2d
    return convolve2d(s, kernel, mode="same", boundary="fill",
                      fillvalue=0) - s


def _source_stem(thresholded_tif: Path) -> str:
    """Recover the original image stem from a thresholded TIFF.

    The threshold.ijm macro saves as ``<original>.tif_thresholded.tif``;
    strip the trailing ``_thresholded`` and the inner extension.
    """
    stem = thresholded_tif.stem
    if stem.endswith("_thresholded"):
        stem = stem[: -len("_thresholded")]
    if stem.endswith(".tif"):
        stem = stem[:-4]
    elif stem.endswith(".tiff"):
        stem = stem[:-5]
    return stem


def extract_astrocyte_features_from_project(
    project_dir: str,
    *,
    rois: dict | None = None,
    soma_radius_px: int = 4,
    mode: str = "astrocyte",
) -> pd.DataFrame:
    """Walk every thresholded GFAP image in the project, compute
    metrics per ROI, return a DataFrame with one row per (image, ROI).

    ``rois`` maps absolute prepared-image path → list of ROI dicts (the
    same shape ``SegmentParams.rois`` uses). Images with no ROIs get a
    single row tagged ``DEFAULT_ROI_TAG`` ("all").

    Joins per-image metadata at the caller's discretion — the metadata
    join is handled separately so this function stays format-agnostic.

    Columns: ``image_stem``, ``roi_tag``, plus the 9
    :data:`ASTROCYTE_FEATURES`.
    """
    from glia.prepare import glia_dir, prepared_dir
    thresh_dir = glia_dir(project_dir, mode) / "ThresholdedImages"
    prep_dir = prepared_dir(project_dir, mode)
    if not thresh_dir.is_dir():
        return pd.DataFrame(
            columns=["image_stem", "roi_tag"] + ASTROCYTE_FEATURES,
        )

    rois = rois or {}
    rows: list[dict] = []
    for tif in sorted(thresh_dir.glob("*.tif")):
        try:
            binary = tifffile.imread(tif) > 0
        except Exception:
            continue
        h, w = binary.shape
        stem = _source_stem(tif)

        # Intensity from the prepared (8-bit) image, if available.
        prep_path = prep_dir / f"{stem}.tif"
        intensity = None
        if prep_path.is_file():
            try:
                intensity = tifffile.imread(prep_path)
            except Exception:
                intensity = None

        # Find the absolute prepared-image path for ROI matching. The
        # ``rois`` dict is keyed on the absolute path of the *source*
        # the user drew on — that's the prepared image.
        rois_key = str(prep_path) if prep_path.is_file() else ""
        image_rois = rois.get(rois_key, []) if rois_key else []

        if image_rois:
            passes = per_roi_masks(image_rois, h, w)
        else:
            passes = [(DEFAULT_ROI_TAG, np.ones((h, w), dtype=bool))]

        for tag, roi_arr in passes:
            metrics = compute_astrocyte_metrics(
                binary, intensity_image=intensity, roi_mask=roi_arr,
                soma_radius_px=soma_radius_px,
            )
            row = {"image_stem": stem, "roi_tag": tag,
                   **metrics_to_row(metrics)}
            rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=["image_stem", "roi_tag"] + ASTROCYTE_FEATURES,
        )
    return pd.DataFrame(rows)


# ── Persistence ─────────────────────────────────────────────────────


def astrocyte_features_path(project_dir: str,
                            mode: str = "astrocyte") -> Path | None:
    """Where astrocyte_features.csv lives. Mode is included so the
    file lands inside the astrocyte subfolder (alongside the
    matching Prepared / ThresholdedImages dirs)."""
    if not project_dir or not Path(project_dir).is_dir():
        return None
    from glia.prepare import glia_dir
    return glia_dir(project_dir, mode) / ASTROCYTE_FEATURES_FILENAME


def save_astrocyte_features_df(
    project_dir: str, df: pd.DataFrame,
) -> str | None:
    path = astrocyte_features_path(project_dir)
    if path is None or df is None:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except Exception:
        return None
    return str(path)


def load_astrocyte_features_df(project_dir: str) -> pd.DataFrame | None:
    path = astrocyte_features_path(project_dir)
    if path is None or not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None
