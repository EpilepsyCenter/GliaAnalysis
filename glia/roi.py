"""Rasterize Plotly ROI shapes into binary masks.

The Setup → ROI subtab stores ROIs as raw Plotly layout-shape dicts. To
use them downstream (threshold preview, segmentation) we need them as
``H x W`` boolean numpy arrays in image-pixel coordinates. This module
handles both the rect and closed-path varieties Plotly emits.
"""

from __future__ import annotations

import re

import numpy as np
from skimage.draw import polygon as draw_polygon


_PATH_COORD_RE = re.compile(
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)[,\s]+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


# ── Centroid helpers (used by both Setup subtabs for label placement) ─


def path_centroid(path: str) -> tuple[float, float]:
    """Centroid of all coordinate pairs in an SVG path string.

    Plotly's draw modebar emits paths with no required whitespace between
    command and coords (e.g. ``M100,50L200,60Z``) — pull every (x,y) pair
    out via regex and average them.
    """
    if not path:
        return (0.0, 0.0)
    pairs = _PATH_COORD_RE.findall(path)
    if not pairs:
        return (0.0, 0.0)
    xs = [float(x) for x, _ in pairs]
    ys = [float(y) for _, y in pairs]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def shape_anchor(shape: dict) -> tuple[float, float]:
    """A reasonable (x, y) to place a label inside the shape."""
    if shape.get("type") == "rect":
        cx = 0.5 * (shape.get("x0", 0) + shape.get("x1", 0))
        cy = 0.5 * (shape.get("y0", 0) + shape.get("y1", 0))
        return (cx, cy)
    return path_centroid(shape.get("path", ""))


def roi_mask(shape: dict, height: int, width: int) -> np.ndarray:
    """Rasterize one Plotly shape dict to a HxW boolean mask."""
    mask = np.zeros((height, width), dtype=bool)
    if not shape:
        return mask

    if shape.get("type") == "rect":
        x0 = float(shape.get("x0", 0.0))
        x1 = float(shape.get("x1", 0.0))
        y0 = float(shape.get("y0", 0.0))
        y1 = float(shape.get("y1", 0.0))
        lo_x, hi_x = sorted((x0, x1))
        lo_y, hi_y = sorted((y0, y1))
        cmin = max(0, int(round(lo_x)))
        cmax = min(width,  int(round(hi_x)))
        rmin = max(0, int(round(lo_y)))
        rmax = min(height, int(round(hi_y)))
        if cmin < cmax and rmin < rmax:
            mask[rmin:rmax, cmin:cmax] = True
        return mask

    # Treat anything else as a polygonal path.
    path = shape.get("path", "")
    pairs = _PATH_COORD_RE.findall(path)
    if len(pairs) < 3:
        return mask
    xs = np.array([float(x) for x, _ in pairs])
    ys = np.array([float(y) for _, y in pairs])
    rr, cc = draw_polygon(ys, xs, shape=(height, width))
    mask[rr, cc] = True
    return mask


def union_mask(
    rois: list[dict], height: int, width: int,
) -> np.ndarray:
    """Boolean union of all ROIs' masks (the 'inside ROIs' region)."""
    out = np.zeros((height, width), dtype=bool)
    for r in rois:
        out |= roi_mask(r.get("shape", {}), height, width)
    return out


def per_roi_masks(
    rois: list[dict], height: int, width: int,
) -> list[tuple[str, np.ndarray]]:
    """List of (tag, mask) for each ROI, preserving the order from the editor."""
    return [(r.get("tag", ""), roi_mask(r.get("shape", {}), height, width))
            for r in rois]
