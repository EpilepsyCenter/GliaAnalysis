"""Feature extraction from single-cell binary masks.

Replaces FracLac's "Hull and Circle Results" output with scikit-image +
scipy + a minimum-enclosing-circle implementation, so the full pipeline
can run headlessly from Python without a GUI plugin.

Validation strategy: run side-by-side with FracLac on a test set of real
images and confirm per-feature Spearman correlation >= 0.95 before declaring
the substitution acceptable. See tests/test_features.py for synthetic
sanity-checks; the real-data check belongs in a notebook once images arrive.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import binary_fill_holes
from scipy.spatial import ConvexHull
from skimage import measure
from skimage.segmentation import find_boundaries

from glia.config import FRACLAC_FEATURES, SKELETON_FEATURES


@dataclass
class GeometricFeatures:
    foreground_pixels: int
    density_in_hull: float
    span_ratio_hull: float
    max_span_hull: float
    area: float
    perimeter: float
    circularity: float
    bbox_width: float
    bbox_height: float
    max_radius_hull_com: float
    radii_ratio_hull_com: float
    radii_cv_hull_com: float
    mean_radius_hull_com: float
    diameter_bounding_circle: float
    max_radius_circle_com: float
    radii_ratio_circle_com: float
    radii_cv_circle_com: float
    mean_radius_circle_com: float


def compute_geometric_features(binary_mask: np.ndarray) -> GeometricFeatures:
    """Compute 18 FracLac-equivalent geometric features from a 2D binary mask.

    Expects a 2D boolean / uint8 array where the cell is foreground (True/255)
    on a black background. Multiple disconnected components are merged.

    Implementation notes per feature group:
      - Area, perimeter, circularity, bbox: skimage.measure.regionprops on the
        single combined component (after filling holes).
      - Convex hull metrics: scipy.spatial.ConvexHull on the foreground points;
        max span = max pairwise distance among hull vertices.
      - Radii from hull centroid: distances from hull centroid to each hull
        vertex.
      - Minimum enclosing circle: Welzl's algorithm on the hull vertices.
      - Radii from circle centroid: distances from circle center to each
        foreground edge pixel.
    """
    mask = np.asarray(binary_mask).astype(bool)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
    if not mask.any():
        raise ValueError("Empty mask — no foreground pixels.")

    # Fill internal holes, then treat all foreground as one merged region.
    filled = binary_fill_holes(mask)
    foreground_pixels = int(filled.sum())

    # regionprops with label=1 over the entire foreground (merges components).
    labels = filled.astype(np.uint8)
    rp = measure.regionprops(labels)[0]
    area = float(rp.area)
    perimeter = float(rp.perimeter) if rp.perimeter > 0 else 0.0
    circularity = (
        4.0 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0
    )
    minr, minc, maxr, maxc = rp.bbox
    bbox_height = float(maxr - minr)
    bbox_width = float(maxc - minc)

    # Convex hull of foreground pixel coordinates (row, col).
    coords = np.column_stack(np.nonzero(filled))      # (N, 2) as (row, col)
    pts = coords[:, [1, 0]].astype(float)              # (x=col, y=row)
    if len(pts) < 3 or np.linalg.matrix_rank(pts - pts.mean(0)) < 2:
        # Degenerate (single pixel or collinear). Build a tiny artificial hull.
        hull_vertices = pts
        hull_area = max(area, 1.0)
    else:
        hull = ConvexHull(pts)
        hull_vertices = pts[hull.vertices]
        hull_area = float(hull.volume)  # 2D: .volume is the polygon area

    density_in_hull = foreground_pixels / hull_area if hull_area > 0 else 0.0

    # Max span across hull: max pairwise distance among hull vertices.
    if len(hull_vertices) >= 2:
        diff = hull_vertices[:, None, :] - hull_vertices[None, :, :]
        max_span_hull = float(np.sqrt((diff ** 2).sum(-1)).max())
    else:
        max_span_hull = 0.0

    # Hull span ratio (major/minor axis). PCA on hull vertices: ratio of stddevs.
    span_ratio_hull = _principal_axis_ratio(hull_vertices)

    # Radii from hull's centroid to each hull vertex.
    hull_com = hull_vertices.mean(axis=0)
    hull_radii = np.linalg.norm(hull_vertices - hull_com, axis=1)
    (mean_radius_hull_com, max_radius_hull_com,
     radii_ratio_hull_com, radii_cv_hull_com) = _radii_stats(hull_radii)

    # Minimum enclosing circle (Welzl) on hull vertices.
    cx, cy, circle_r = _min_enclosing_circle(hull_vertices)
    diameter_bounding_circle = 2.0 * circle_r

    # Radii from the circle's centroid to each foreground edge pixel.
    edge_mask = find_boundaries(filled, mode="outer")
    if not edge_mask.any():
        edge_mask = filled  # fallback for tiny masks
    edge_rc = np.column_stack(np.nonzero(edge_mask))
    edge_xy = edge_rc[:, [1, 0]].astype(float)
    circle_radii = np.linalg.norm(edge_xy - np.array([cx, cy]), axis=1)
    (mean_radius_circle_com, max_radius_circle_com,
     radii_ratio_circle_com, radii_cv_circle_com) = _radii_stats(circle_radii)

    return GeometricFeatures(
        foreground_pixels=foreground_pixels,
        density_in_hull=density_in_hull,
        span_ratio_hull=span_ratio_hull,
        max_span_hull=max_span_hull,
        area=area,
        perimeter=perimeter,
        circularity=circularity,
        bbox_width=bbox_width,
        bbox_height=bbox_height,
        max_radius_hull_com=max_radius_hull_com,
        radii_ratio_hull_com=radii_ratio_hull_com,
        radii_cv_hull_com=radii_cv_hull_com,
        mean_radius_hull_com=mean_radius_hull_com,
        diameter_bounding_circle=diameter_bounding_circle,
        max_radius_circle_com=max_radius_circle_com,
        radii_ratio_circle_com=radii_ratio_circle_com,
        radii_cv_circle_com=radii_cv_circle_com,
        mean_radius_circle_com=mean_radius_circle_com,
    )


def _principal_axis_ratio(points: np.ndarray) -> float:
    """Ratio of major to minor stddev along PCA axes of `points`."""
    if len(points) < 2:
        return 1.0
    centered = points - points.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    if np.isscalar(cov) or cov.shape == ():
        return 1.0
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0, None)
    major, minor = float(eigvals.max()), float(eigvals.min())
    if minor <= 1e-12:
        return float("inf") if major > 0 else 1.0
    return float(np.sqrt(major / minor))


def _radii_stats(radii: np.ndarray) -> tuple[float, float, float, float]:
    """Return (mean, max, max/min ratio, CV) for an array of radii."""
    if len(radii) == 0:
        return 0.0, 0.0, 1.0, 0.0
    mean_r = float(radii.mean())
    max_r = float(radii.max())
    min_r = float(radii.min())
    ratio = max_r / min_r if min_r > 1e-12 else float("inf")
    cv = float(radii.std() / mean_r) if mean_r > 1e-12 else 0.0
    return mean_r, max_r, ratio, cv


# ── Minimum enclosing circle (Welzl) ──────────────────────────────────


def _min_enclosing_circle(points: np.ndarray) -> tuple[float, float, float]:
    """Return (cx, cy, r) of the smallest circle enclosing all points.

    Welzl's randomized algorithm; expected O(n).
    """
    pts = [tuple(p) for p in points]
    rng = random.Random(0)
    rng.shuffle(pts)
    return _welzl(pts, [], len(pts))


def _welzl(P: list, R: list, n: int) -> tuple[float, float, float]:
    if n == 0 or len(R) == 3:
        return _trivial_circle(R)
    p = P[n - 1]
    D = _welzl(P, R, n - 1)
    if _in_circle(p, D):
        return D
    return _welzl(P, R + [p], n - 1)


def _in_circle(p, circle) -> bool:
    cx, cy, r = circle
    return (p[0] - cx) ** 2 + (p[1] - cy) ** 2 <= r ** 2 + 1e-9


def _trivial_circle(R: list) -> tuple[float, float, float]:
    if not R:
        return (0.0, 0.0, 0.0)
    if len(R) == 1:
        return (R[0][0], R[0][1], 0.0)
    if len(R) == 2:
        (x1, y1), (x2, y2) = R
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        r = np.hypot(x1 - x2, y1 - y2) / 2
        return (float(cx), float(cy), float(r))
    return _circle_from_3(R[0], R[1], R[2])


def _circle_from_3(a, b, c) -> tuple[float, float, float]:
    ax, ay = a; bx, by = b; cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        # Collinear — fall back to the circle through the two farthest points.
        pairs = [(a, b), (a, c), (b, c)]
        dists = [np.hypot(p[0] - q[0], p[1] - q[1]) for p, q in pairs]
        p, q = pairs[int(np.argmax(dists))]
        return _trivial_circle([p, q])
    ux = ((ax ** 2 + ay ** 2) * (by - cy)
          + (bx ** 2 + by ** 2) * (cy - ay)
          + (cx ** 2 + cy ** 2) * (ay - by)) / d
    uy = ((ax ** 2 + ay ** 2) * (cx - bx)
          + (bx ** 2 + by ** 2) * (ax - cx)
          + (cx ** 2 + cy ** 2) * (bx - ax)) / d
    r = np.hypot(ax - ux, ay - uy)
    return (float(ux), float(uy), float(r))


def features_to_row(feats: GeometricFeatures, cell_id: str) -> dict:
    """Map dataclass fields to the 18 named columns in FRACLAC_FEATURES."""
    field_map = dict(zip(
        [
            "foreground_pixels", "density_in_hull", "span_ratio_hull",
            "max_span_hull", "area", "perimeter", "circularity",
            "bbox_width", "bbox_height", "max_radius_hull_com",
            "radii_ratio_hull_com", "radii_cv_hull_com",
            "mean_radius_hull_com", "diameter_bounding_circle",
            "max_radius_circle_com", "radii_ratio_circle_com",
            "radii_cv_circle_com", "mean_radius_circle_com",
        ],
        FRACLAC_FEATURES,
    ))
    d = asdict(feats)
    return {"ID": cell_id, **{field_map[k]: v for k, v in d.items()}}


def extract_features_from_dir(single_cells_dir: Path) -> pd.DataFrame:
    """Iterate every single-cell .tif in a directory, return one row per cell."""
    rows = []
    for tif in sorted(Path(single_cells_dir).glob("*.tif")):
        mask = tifffile.imread(tif) > 0
        feats = compute_geometric_features(mask)
        rows.append(features_to_row(feats, cell_id=tif.stem))
    return pd.DataFrame(rows)


# Map AnalyzeSkeleton's column names → the canonical names in SKELETON_FEATURES.
_FIJI_TO_CANONICAL = {
    "# Branches":              "# of branches",
    "# Junctions":             "# of junctions",
    "# End-point voxels":      "# of end point voxels",
    "# Junction voxels":       "# of junction voxels",
    "# Slab voxels":           "# of slab voxels",
    "Average Branch Length":   "Average branch length",
    "# Triple points":         "# of triple points",
    "# Quadruple points":      "# of quadruple points",
    "Maximum Branch Length":   "Maximum branch length",
}


def _normalize_column(name: str) -> str:
    """Lowercased, punctuation-stripped form for tolerant column matching."""
    n = name.lower().replace("#", " ").replace("-", " ").replace("_", " ")
    n = n.replace(" of ", " ")
    return " ".join(n.split())


def _row_from_fiji_skeleton(s: pd.Series) -> dict:
    """Map one AnalyzeSkeleton result row to the 9 canonical feature columns."""
    by_norm = {_normalize_column(c): c for c in s.index}
    out: dict = {}
    for fiji_name, canonical in _FIJI_TO_CANONICAL.items():
        if fiji_name in s.index:
            out[canonical] = s[fiji_name]
        else:
            out[canonical] = s[by_norm.get(_normalize_column(fiji_name), "")] \
                if _normalize_column(fiji_name) in by_norm else float("nan")
    return out


def load_skeleton_results(skeleton_dir: Path) -> pd.DataFrame:
    """Concatenate FIJI AnalyzeSkeleton per-cell CSVs into a single dataframe.

    Mirrors MicrogliaMorphologyR::skeleton_tidying. Files with more than one
    row are skipped — those indicate the cell fragmented during skeletonization
    and the original pipeline excludes them. Returns a dataframe with columns
    ``ID`` plus the 9 names in ``SKELETON_FEATURES``.
    """
    rows: list[dict] = []
    for csv_path in sorted(Path(skeleton_dir).glob("*_results.csv")):
        df = pd.read_csv(csv_path)
        if len(df) != 1:
            continue
        cell_id = csv_path.stem
        if cell_id.endswith("_results"):
            cell_id = cell_id[: -len("_results")]
        # Both the FIJI reference macro and our Python skeleton step name
        # files "<cell_id>.tif_results.csv", leaving a trailing ".tif" after
        # the suffix strip. Drop it so the ID matches what
        # extract_features_from_dir derives from the single-cell TIFF.
        if cell_id.endswith(".tif"):
            cell_id = cell_id[:-4]
        row = _row_from_fiji_skeleton(df.iloc[0])
        row["ID"] = cell_id
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["ID"] + SKELETON_FEATURES)
    out = pd.DataFrame(rows)
    return out[["ID"] + SKELETON_FEATURES]


def merge_geometric_and_skeleton(
    geom: pd.DataFrame, skel: pd.DataFrame
) -> pd.DataFrame:
    """Inner-join on cell ID, return the full 27-feature cell-level table."""
    return geom.merge(skel, on="ID", how="inner")


# ── Per-project features-table persistence ──────────────────────────


FEATURES_PERSIST_FILENAME = "features.csv"


def features_persist_path(project_dir: str) -> Path | None:
    if not project_dir or not Path(project_dir).is_dir():
        return None
    return Path(project_dir) / "_gliaanalysis" / FEATURES_PERSIST_FILENAME


def save_features_df(project_dir: str, df: pd.DataFrame) -> str | None:
    """Write the current features dataframe to <project>/_gliaanalysis/features.csv.

    Includes whatever derived columns are present (PC1..PCN, Cluster,
    morphology_label, parsed metadata). Best-effort — errors swallowed.
    """
    path = features_persist_path(project_dir)
    if path is None or df is None:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except Exception:
        return None
    return str(path)


def load_features_df(project_dir: str) -> pd.DataFrame | None:
    """Reload the persisted features table, or None if absent / unreadable."""
    path = features_persist_path(project_dir)
    if path is None or not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None
