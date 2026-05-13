"""Per-cell skeleton analysis in pure Python.

Replaces FIJI's ``Analyze Skeleton (2D/3D)`` because that plugin doesn't
write its Results table when ImageJ runs in ``--headless`` mode (verified on
Fiji 2024). We re-derive the same 9 metrics from ``skimage.morphology.skeletonize``
+ ``skan.Skeleton`` and write per-cell CSVs whose columns match the FIJI
output so :func:`glia.features.load_skeleton_results` works unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import convolve
from skan import Skeleton
from skimage.morphology import skeletonize


_DEGREE_KERNEL = np.ones((3, 3), dtype=np.uint8)
_DEGREE_KERNEL[1, 1] = 0


_EMPTY_ROW = {
    "# Branches": 0,
    "# Junctions": 0,
    "# End-point voxels": 0,
    "# Junction voxels": 0,
    "# Slab voxels": 0,
    "Average Branch Length": 0.0,
    "# Triple points": 0,
    "# Quadruple points": 0,
    "Maximum Branch Length": 0.0,
}


def analyze_skeleton(binary_mask: np.ndarray) -> dict:
    """Compute the 9 AnalyzeSkeleton-equivalent metrics for one cell mask.

    Returns a dict whose keys match the column names FIJI's Summarize would
    have produced (so the same downstream loader keeps working). For masks
    that produce a sub-pixel skeleton (empty / single-pixel), every count
    is 0 and lengths are 0.0.
    """
    mask = np.asarray(binary_mask) > 0
    if not mask.any():
        return dict(_EMPTY_ROW)

    sk = skeletonize(mask)
    if sk.sum() < 2:
        return dict(_EMPTY_ROW)

    # Per-pixel skeleton degree via an 8-neighbour count.
    neighbours = convolve(sk.astype(np.uint8), _DEGREE_KERNEL,
                          mode="constant", cval=0)
    deg = np.where(sk, neighbours, 0)

    endpoint_voxels = int((deg == 1).sum())
    slab_voxels     = int((deg == 2).sum())
    triple_points   = int((deg == 3).sum())
    quad_points     = int((deg == 4).sum())
    junction_voxels = int((deg >= 3).sum())
    junctions       = junction_voxels  # FIJI labels these the same in 2D

    # Branch count + lengths from skan. Wrap because skan can complain about
    # very small skeletons (e.g. a single 2-pixel segment).
    try:
        skel_obj = Skeleton(sk.astype(np.uint8))
        branches = int(skel_obj.n_paths)
        distances = np.asarray(skel_obj.path_lengths())
        if branches == 0 or distances.size == 0:
            avg_len, max_len = 0.0, 0.0
        else:
            avg_len = float(distances.mean())
            max_len = float(distances.max())
    except Exception:
        branches = 1 if sk.sum() >= 2 else 0
        avg_len = float(sk.sum())
        max_len = avg_len

    return {
        "# Branches": int(branches),
        "# Junctions": int(junctions),
        "# End-point voxels": endpoint_voxels,
        "# Junction voxels": junction_voxels,
        "# Slab voxels": slab_voxels,
        "Average Branch Length": avg_len,
        "# Triple points": triple_points,
        "# Quadruple points": quad_points,
        "Maximum Branch Length": max_len,
    }


def analyze_directory(
    single_cells_dir: Path,
    skeleton_out: Path,
    skeleton_img_out: Path | None = None,
) -> tuple[int, list[str]]:
    """Run skeleton analysis on every TIFF in ``single_cells_dir``.

    Writes ``<cell_id>.tif_results.csv`` to ``skeleton_out`` for each input
    (matching the FIJI macro's naming). If ``skeleton_img_out`` is given,
    also writes the skeletonized image as ``<cell_id>.tif_taggedskeleton.tif``.
    Returns (n_csvs_written, skipped_filenames).
    """
    skeleton_out.mkdir(parents=True, exist_ok=True)
    if skeleton_img_out is not None:
        skeleton_img_out.mkdir(parents=True, exist_ok=True)

    skipped: list[str] = []
    written = 0
    for tif in sorted(single_cells_dir.glob("*.tif")):
        # Skip auxiliary sibling crops (e.g. ``<cell_id>__dapi.tif``).
        # Those are grayscale nucleus channels, not cell masks.
        if tif.stem.endswith("__dapi"):
            continue
        try:
            img = tifffile.imread(tif)
        except Exception as e:
            skipped.append(f"{tif.name}: read failed ({e})")
            continue
        try:
            row = analyze_skeleton(img > 0)
        except Exception as e:
            skipped.append(f"{tif.name}: skeleton failed ({e})")
            continue
        pd.DataFrame([row]).to_csv(skeleton_out / f"{tif.name}_results.csv",
                                   index=False)
        if skeleton_img_out is not None:
            sk = skeletonize(img > 0).astype(np.uint8) * 255
            tifffile.imwrite(
                skeleton_img_out / f"{tif.name}_taggedskeleton.tif", sk,
            )
        written += 1
    return written, skipped
