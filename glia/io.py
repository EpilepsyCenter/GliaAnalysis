"""Multi-format image I/O via bioio.

The Prepare step uses these helpers to read whatever source format the
user dropped into the project folder (TIFF, OME-TIFF, CZI, ND2, LIF, etc.)
and emit a 2D ``uint8`` array suitable for the rest of the pipeline.

Only bioio's pure-Python readers are needed — no Java / Bio-Formats jar.
"""

from __future__ import annotations

import contextlib
import io as _io
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# bioio prints reader-attempt errors to stderr on every load (which is
# expected, because it tries readers in priority order until one matches).
# Suppress that chatter — we already report any real load failure.
logging.getLogger("bioio").setLevel(logging.ERROR)
logging.getLogger("bioio_ome_tiff").setLevel(logging.ERROR)


@dataclass
class ImageMeta:
    path: str
    suffix: str
    dtype: str
    n_t: int
    n_channels: int
    n_z: int
    height: int
    width: int
    channel_names: list[str]

    @property
    def is_multichannel(self) -> bool:
        return self.n_channels > 1

    @property
    def is_3d(self) -> bool:
        return self.n_z > 1


def _bioimage(path: str):
    """Open with bioio while suppressing its noisy reader-fallback prints."""
    from bioio import BioImage
    # bioio writes the "Attempted file...failed" line via plain print(); the
    # logger-level mute above doesn't catch it. Redirect stderr just for
    # this call so the UI logs stay clean.
    devnull = open(os.devnull, "w")
    try:
        with contextlib.redirect_stderr(devnull):
            return BioImage(path)
    finally:
        devnull.close()


def read_image_meta(path: str) -> ImageMeta:
    """Cheap metadata-only read."""
    img = _bioimage(path)
    try:
        names = [str(n) for n in (img.channel_names or [])]
    except Exception:
        names = []
    # bioio defaults to "Channel:0:0" for plain TIFFs — too noisy. If the
    # names look like the default placeholder, suppress them so the UI
    # falls back to "channel N".
    if names and all(n.startswith("Channel:") for n in names):
        names = []
    # Sidecar overlay: VSI conversions write channel names to
    # <stem>.tif.channels.json because the FileSaver TIFF format can't
    # carry OME channel names. Prefer those over anything bioio invents.
    try:
        from glia.vsi_convert import load_sidecar_channel_names
        sidecar_names = load_sidecar_channel_names(Path(path))
    except Exception:
        sidecar_names = []
    if sidecar_names:
        names = sidecar_names

    return ImageMeta(
        path=str(path),
        suffix=Path(path).suffix.lower(),
        dtype=str(img.dtype),
        n_t=int(img.dims.T),
        n_channels=int(img.dims.C),
        n_z=int(img.dims.Z),
        height=int(img.dims.Y),
        width=int(img.dims.X),
        channel_names=names,
    )


def _project_z(arr3d: np.ndarray, mode: str) -> np.ndarray:
    """Collapse a (Z, Y, X) volume to (Y, X) via the chosen projection."""
    if arr3d.ndim == 2:
        return arr3d
    if mode == "mean":
        return arr3d.mean(axis=0)
    if mode in ("first", "center"):
        idx = 0 if mode == "first" else arr3d.shape[0] // 2
        return arr3d[idx]
    # Default: max projection — most common for microglia / astrocytes.
    return arr3d.max(axis=0)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Min-max stretch to 0–255 uint8 (matches what FIJI's ``run('8-bit')`` does
    on a non-8-bit image). No-op when input is already uint8."""
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    lo = float(np.nanmin(a))
    hi = float(np.nanmax(a))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    a = (a - lo) * (255.0 / (hi - lo))
    return np.clip(a, 0, 255).astype(np.uint8)


def load_image_2d_uint8(
    path: str, channel: int = 0, z_projection: str = "max",
) -> tuple[np.ndarray, dict]:
    """Read ``path``, pick ``channel``, project ``Z`` if needed, return a
    2D uint8 image plus an info dict describing what happened.

    Handles four cases:
      * Plain 2D image → straight through.
      * Z-stack → projected with the chosen mode.
      * Multi-channel → picks the requested channel index.
      * RGB / brightfield with samples-per-pixel (S > 1) → averages the
        samples to luminance so the saved TIFF is single-channel rather
        than a 3-channel RGB the rest of the pipeline can't handle.
    """
    img = _bioimage(path)
    c = max(0, min(int(channel), int(img.dims.C) - 1))
    has_s = False
    try:
        has_s = int(img.dims.S) > 1
    except Exception:
        has_s = False

    if has_s:
        arr = img.get_image_data("ZYXS", T=0, C=c)
        arr = arr.mean(axis=-1)  # collapse samples → grayscale
    else:
        arr = img.get_image_data("ZYX", T=0, C=c)

    projected = _project_z(arr, z_projection)

    # Defensive: if anything still leaks an extra axis (e.g. an exotic
    # reader returned a singleton sample axis), squeeze + drop until 2D.
    while projected.ndim > 2:
        projected = projected[..., 0] if projected.shape[-1] <= 4 \
            else projected[0]

    arr8 = _to_uint8(projected)
    return arr8, {
        "dtype_in": str(img.dtype),
        "channel": c,
        "n_channels": int(img.dims.C),
        "z_projection": z_projection if int(img.dims.Z) > 1 else "n/a",
        "n_z": int(img.dims.Z),
        "height": int(img.dims.Y),
        "width": int(img.dims.X),
        "samples_per_pixel": int(img.dims.S) if has_s else 1,
    }
