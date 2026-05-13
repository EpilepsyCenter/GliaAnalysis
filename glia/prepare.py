"""Project Prepare step: convert raw images → 8-bit single-channel TIFFs.

Inputs: any format bioio can read (TIFF, OME-TIFF, CZI, ND2, LIF, ...).
Output: ``<project>/_gliaanalysis/Prepared/<stem>.tif`` — 2D ``uint8``, the
single contract the rest of the pipeline operates on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tifffile

from glia.io import load_image_2d_uint8, read_image_meta
from glia.vsi_convert import VSI_EXTENSIONS, resolve_source


_PREPARED_NAME = "Prepared"
_PREPARED_DAPI_NAME = "Prepared_dapi"
_GLIA_BASE = "_gliaanalysis"
_ASTROCYTE_SUBDIR = "astrocyte"


def glia_dir(project_dir: str, mode: str = "microglia") -> Path:
    """Mode-aware base output dir under the project.

    Microglia keeps the historical ``_gliaanalysis/`` layout. Astrocyte
    nests under ``_gliaanalysis/astrocyte/`` so its Prepared,
    ThresholdedImages, etc. don't overwrite the microglia outputs when
    the user switches modes and re-runs Prepare with a different
    channel.
    """
    base = Path(project_dir) / _GLIA_BASE
    if (mode or "microglia") == "astrocyte":
        return base / _ASTROCYTE_SUBDIR
    return base

# Source extensions the Prepare step is willing to attempt. bioio's plugin
# selection covers everything in this list except .vsi, which is routed
# through FIJI Bio-Formats first (glia.vsi_convert) and then read as the
# cached OME-TIFF.
SOURCE_EXTENSIONS = (
    ".tif", ".tiff",
    ".ome.tif", ".ome.tiff",
    ".czi",
    ".nd2",
    ".lif",
    ".vsi",
)


def prepared_dir(project_dir: str, mode: str = "microglia") -> Path:
    return glia_dir(project_dir, mode) / _PREPARED_NAME


def prepared_dapi_dir(project_dir: str, mode: str = "microglia") -> Path:
    """Parallel directory of prepared DAPI / nucleus-channel TIFFs.

    Mirrors the layout of ``prepared_dir`` so the segment step can match
    DAPI to cell-channel images by stem. Only written when the project
    opts into DAPI-seeded soma centering.
    """
    return glia_dir(project_dir, mode) / _PREPARED_DAPI_NAME


def list_source_files(project_dir: str) -> list[Path]:
    """Top-level files in the project folder with a supported extension."""
    if not project_dir or not Path(project_dir).is_dir():
        return []
    out: list[Path] = []
    for p in sorted(Path(project_dir).iterdir()):
        if not p.is_file():
            continue
        name = p.name.lower()
        if any(name.endswith(ext) for ext in SOURCE_EXTENSIONS):
            out.append(p)
    return out


def output_path_for(project_dir: str, source: Path,
                    mode: str = "microglia") -> Path:
    """Where the prepared TIFF lands for a given source file."""
    return prepared_dir(project_dir, mode) / (source.stem + ".tif")


def output_path_for_dapi(project_dir: str, source: Path,
                         mode: str = "microglia") -> Path:
    """Where the prepared DAPI TIFF lands for a given source file."""
    return prepared_dapi_dir(project_dir, mode) / (source.stem + ".tif")


@dataclass
class PrepareReport:
    n_prepared: int = 0
    n_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    written: list[str] = field(default_factory=list)


def prepare_one(
    source: Path, output: Path, channel: int = 0,
    z_projection: str = "max", project_dir: str | None = None,
) -> dict:
    """Read, project, normalize, write. Returns the info dict from io.

    ``project_dir`` is only needed when the source is a .vsi; in that case
    the reader is redirected to the cached OME-TIFF produced by
    glia.vsi_convert.ensure_vsi_converted.
    """
    read_from = resolve_source(project_dir or "", source) if project_dir \
        else source
    if not read_from.exists():
        raise FileNotFoundError(
            f"{source.name}: expected converted file at {read_from}. "
            "Run VSI conversion first (FIJI required)."
        )
    arr, info = load_image_2d_uint8(str(read_from), channel=channel,
                                    z_projection=z_projection)
    output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output, arr)
    info["source"] = str(source)
    info["read_from"] = str(read_from)
    info["output"] = str(output)
    return info


def prepare_dapi_directory(
    project_dir: str,
    dapi_per_image: dict,
    z_projection: str = "max",
    included_names: set[str] | None = None,
    mode: str = "microglia",
) -> PrepareReport:
    """Prepare DAPI / nucleus channel into ``Prepared_dapi/``.

    ``dapi_per_image`` maps source basename → ``{"channel": int,
    "z_projection": str}``. Rows missing from the mapping or with a
    negative channel index are silently skipped — that's the per-image
    "no DAPI for this file" signal.

    Same dtype contract as the primary prepare step: 2D uint8.
    """
    rep = PrepareReport()
    sources = list_source_files(project_dir)
    out_dir = prepared_dapi_dir(project_dir, mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in sources:
        if included_names is not None and src.name not in included_names:
            continue
        cfg = dapi_per_image.get(src.name)
        if not cfg:
            continue
        try:
            ch = int(cfg.get("channel", -1))
        except (TypeError, ValueError):
            continue
        if ch < 0:
            continue
        z = str(cfg.get("z_projection", z_projection))
        try:
            prepare_one(src,
                        output_path_for_dapi(project_dir, src, mode),
                        channel=ch, z_projection=z,
                        project_dir=project_dir)
            rep.n_prepared += 1
            rep.written.append(src.name)
        except Exception as e:
            rep.n_skipped += 1
            rep.errors.append(f"{src.name}: {e}")
    return rep


def prepare_directory(
    project_dir: str,
    channel: int = 0,
    z_projection: str = "max",
    per_image: dict | None = None,
    included_names: set[str] | None = None,
    mode: str = "microglia",
) -> PrepareReport:
    """Prepare source files in ``project_dir``.

    ``per_image`` lets callers override channel / z_projection on a
    per-file basis; it maps the source filename (basename) to a dict
    that may include ``channel`` and/or ``z_projection``. Missing keys
    fall back to the batch defaults.

    ``included_names`` (if given) is the allow-list of basenames to
    process; anything else found in the project is skipped silently.
    """
    rep = PrepareReport()
    per_image = per_image or {}
    sources = list_source_files(project_dir)
    out_dir = prepared_dir(project_dir, mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in sources:
        if included_names is not None and src.name not in included_names:
            continue
        override = per_image.get(src.name, {})
        c = int(override.get("channel", channel))
        z = str(override.get("z_projection", z_projection))
        try:
            prepare_one(src, output_path_for(project_dir, src, mode),
                        channel=c, z_projection=z,
                        project_dir=project_dir)
            rep.n_prepared += 1
            rep.written.append(src.name)
        except Exception as e:
            rep.n_skipped += 1
            rep.errors.append(f"{src.name}: {e}")
    return rep
