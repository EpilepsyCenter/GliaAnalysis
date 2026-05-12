"""Threshold preview implemented in Python (skimage), independent of FIJI.

The Setup tab uses this for a fast live preview; the production Segment tab
still calls FIJI headless for the real run. Method names track the FIJI
auto-threshold catalog where skimage has an equivalent; methods without a
skimage match are reported via ``unsupported_methods()`` and fall back to
Otsu.

Also exposes the FIJI-style preprocessing chain (Enhance Contrast 0.35 →
Unsharp Mask r=3 m=0.6 → Despeckle) so the preview matches what the real
FIJI Segment run will produce.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from skimage import exposure, filters
from skimage.filters import threshold_local
from skimage.util import img_as_ubyte


# ── Global threshold method dispatch ─────────────────────────────────

_GLOBAL_METHODS: dict[str, callable] = {
    "Otsu":      filters.threshold_otsu,
    "Li":        filters.threshold_li,
    "Yen":       filters.threshold_yen,
    "IsoData":   filters.threshold_isodata,
    "Triangle":  filters.threshold_triangle,
    "Minimum":   filters.threshold_minimum,
    "Mean":      filters.threshold_mean,
}

_LOCAL_METHODS = {
    "Mean":     ("gaussian", None),
    "Median":   ("median", None),
    "Niblack":  ("niblack", None),
    "Sauvola":  ("sauvola", None),
    "Otsu":     ("otsu", None),
}

MANUAL_METHOD = "Manual"


def supported_global_methods() -> list[str]:
    return sorted(_GLOBAL_METHODS) + [MANUAL_METHOD]


def supported_local_methods() -> list[str]:
    return sorted(_LOCAL_METHODS)


def unsupported_methods(kind: str) -> set[str]:
    from glia.config import (
        DEFAULT_THRESHOLD_METHODS_GLOBAL,
        DEFAULT_THRESHOLD_METHODS_LOCAL,
    )
    if kind == "global":
        return set(DEFAULT_THRESHOLD_METHODS_GLOBAL) - set(_GLOBAL_METHODS)
    return set(DEFAULT_THRESHOLD_METHODS_LOCAL) - set(_LOCAL_METHODS)


# ── FIJI-style preprocessing chain ──────────────────────────────────


def preprocess_fiji_style(image: np.ndarray) -> np.ndarray:
    """Apply the Ciernia MicrogliaMorphology preprocessing chain.

    Steps from MicrogliaMorphology_Program.ijm:
      1. Enhance Contrast 0.35  — saturate 0.35% of the brightest/darkest
         pixels then linearly stretch to [0, 255]. We use the symmetric
         skimage equivalent (rescale_intensity with percentile clipping).
      2. Unsharp Mask r=3, m=0.6 — increases edge contrast.
      3. Despeckle — 3x3 median filter to remove salt-and-pepper noise.
    """
    if image.dtype != np.uint8:
        # Normalize to 8-bit first so contrast stretching is meaningful.
        image = img_as_ubyte(exposure.rescale_intensity(image))

    p_lo, p_hi = np.percentile(image, (0.175, 100 - 0.175))
    stretched = exposure.rescale_intensity(image, in_range=(p_lo, p_hi))

    sharpened = filters.unsharp_mask(stretched, radius=3, amount=0.6,
                                     preserve_range=True)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    despeckled = median_filter(sharpened, size=3)
    return despeckled


# ── Threshold ─────────────────────────────────────────────────────────


def compute_threshold_value(
    image: np.ndarray,
    method: str = "Otsu",
) -> tuple[float, dict]:
    """Return the auto-threshold value for ``image`` under the given method.

    Returns (threshold, info). ``info`` records any fallback. Method
    ``Manual`` returns the middle of the image's dynamic range so the UI
    slider has a sensible starting point.
    """
    info: dict = {"requested": method}
    if method == MANUAL_METHOD:
        info["manual"] = True
        return float((float(image.min()) + float(image.max())) / 2), info
    fn = _GLOBAL_METHODS.get(method)
    if fn is None:
        fn = _GLOBAL_METHODS["Otsu"]
        info["fallback"] = f"{method!r} not in skimage; using Otsu"
    return float(fn(image)), info


def apply_threshold(
    image: np.ndarray,
    method: str = "Otsu",
    kind: str = "global",
    local_radius: int = 15,
    manual_lower: float | None = None,
    manual_upper: float | None = None,
    preprocess: bool = True,
) -> tuple[np.ndarray, dict]:
    """Return (binary_mask, info).

    For global thresholding, when ``manual_lower`` / ``manual_upper`` are
    provided (typically driven by the Setup-tab range slider), the band
    ``[manual_lower, manual_upper]`` is used directly — the ``method``
    argument is only consulted to compute the *initial* threshold value
    via :func:`compute_threshold_value`, which the caller uses to populate
    the slider.

    For local thresholding, the user-selected method drives the per-pixel
    threshold from :func:`skimage.filters.threshold_local`.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")

    info: dict = {"requested": method, "kind": kind, "preprocessed": preprocess}
    work = preprocess_fiji_style(image) if preprocess else image

    if kind == "global":
        if manual_lower is None and manual_upper is None:
            thresh, sub = compute_threshold_value(work, method)
            info.update(sub)
            binary = work > thresh
            info["threshold"] = thresh
        else:
            lo = float(manual_lower) if manual_lower is not None else 0.0
            hi = float(manual_upper) if manual_upper is not None else 255.0
            binary = (work >= lo) & (work <= hi)
            info["threshold"] = (lo, hi)
            if method == MANUAL_METHOD:
                info["manual"] = True
    elif kind == "local":
        spec = _LOCAL_METHODS.get(method)
        if spec is None:
            spec = _LOCAL_METHODS["Mean"]
            info["fallback"] = f"{method!r} not in skimage; using local Mean"
        block_size = int(local_radius) * 2 + 1
        sk_method, _ = spec
        thresh = threshold_local(work, block_size=block_size, method=sk_method)
        binary = work > thresh
        info["block_size"] = block_size
    else:
        raise ValueError(f"kind must be 'global' or 'local', got {kind!r}")

    info["foreground_fraction"] = float(binary.mean())
    return binary.astype(bool), info


def component_area_histogram(
    binary: np.ndarray, n_bins: int = 40, log: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Return (counts, edges). When ``log=True`` (the default), edges are in
    log10(area) space so bars render uniformly on a log-scale display."""
    from skimage import measure
    labels = measure.label(binary, connectivity=2)
    if labels.max() == 0:
        return np.array([0]), np.array([0.0, 1.0])
    areas = np.bincount(labels.ravel())[1:]
    areas = areas[areas > 0]
    if len(areas) == 0:
        return np.array([0]), np.array([0.0, 1.0])
    if log:
        counts, edges = np.histogram(np.log10(areas), bins=n_bins)
    else:
        counts, edges = np.histogram(areas, bins=n_bins)
    return counts, edges
