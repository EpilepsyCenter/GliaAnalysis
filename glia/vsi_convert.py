"""Olympus VSI → OME-TIFF conversion via headless FIJI Bio-Formats.

VSI is a proprietary multi-file Olympus format with no native Python reader,
so we round-trip through FIJI/Bio-Formats once and cache the result. The
cache lives at ``<project>/_gliaanalysis/Converted/<stem>.ome.tif``; the rest
of the pipeline reads from that cache via bioio just like any other source.

Conversion is by-mtime cached: re-running is a no-op when the OME-TIFF on
disk is newer than its source .vsi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from glia.fiji_runner import run_headless


_CONVERTED_SUBDIR = Path("_gliaanalysis") / "Converted"
_MACRO_PATH = (
    Path(__file__).resolve().parent.parent / "fiji_macros" / "vsi_convert.bsh"
)

VSI_EXTENSIONS = (".vsi",)


def converted_dir(project_dir: str) -> Path:
    return Path(project_dir) / _CONVERTED_SUBDIR


def converted_path_for(project_dir: str, vsi_path: Path) -> Path:
    return converted_dir(project_dir) / (vsi_path.stem + ".tif")


def channels_sidecar_for(converted_tif: Path) -> Path:
    """Sibling JSON next to the converted TIFF, holding channel names
    extracted from the VSI's OME metadata at conversion time."""
    return converted_tif.with_suffix(converted_tif.suffix + ".channels.json")


def load_sidecar_channel_names(converted_tif: Path) -> list[str]:
    sidecar = channels_sidecar_for(converted_tif)
    if not sidecar.exists():
        return []
    import json
    try:
        data = json.loads(sidecar.read_text())
    except Exception:
        return []
    names = data.get("channel_names") if isinstance(data, dict) else None
    if not isinstance(names, list):
        return []
    return [str(n) for n in names]


def list_vsi_files(project_dir: str) -> list[Path]:
    if not project_dir or not Path(project_dir).is_dir():
        return []
    return sorted(
        p for p in Path(project_dir).iterdir()
        if p.is_file() and p.name.lower().endswith(VSI_EXTENSIONS)
    )


def companion_dir(vsi_path: Path) -> Path:
    """Olympus pairs ``Foo.vsi`` with a sibling folder ``_Foo_``."""
    return vsi_path.parent / f"_{vsi_path.stem}_"


def has_companion(vsi_path: Path) -> bool:
    d = companion_dir(vsi_path)
    return d.is_dir() and any(d.iterdir())


def audit_vsi_layout(project_dir: str) -> tuple[list[Path], list[Path]]:
    """Sanity-check VSI ↔ companion-folder pairing.

    Returns ``(missing_companion, orphan_companions)``:
      * ``missing_companion`` — .vsi files with no ``_<stem>_`` folder
        next to them (Bio-Formats will fail on these).
      * ``orphan_companions`` — ``_<stem>_`` folders whose .vsi sibling
        is absent (often a typo in either name; user-fixable).
    """
    root = Path(project_dir)
    if not root.is_dir():
        return [], []
    vsi_files = list_vsi_files(project_dir)
    vsi_stems = {p.stem for p in vsi_files}
    missing = [p for p in vsi_files if not has_companion(p)]
    orphans: list[Path] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if len(name) >= 3 and name.startswith("_") and name.endswith("_"):
            stem = name[1:-1]
            if stem and stem not in vsi_stems:
                orphans.append(d)
    return missing, orphans


def needs_conversion(project_dir: str, vsi_path: Path) -> bool:
    """True if no fresh cached OME-TIFF exists for ``vsi_path``."""
    out = converted_path_for(project_dir, vsi_path)
    if not out.exists():
        return True
    try:
        return out.stat().st_mtime < vsi_path.stat().st_mtime
    except OSError:
        return True


def resolve_source(project_dir: str, source: Path) -> Path:
    """Map a source path to the file the rest of the pipeline should read.

    For .vsi sources this returns the cached OME-TIFF. For everything else
    it returns the input path unchanged. Callers don't need to know whether
    a given source went through FIJI conversion.
    """
    if source.suffix.lower() in VSI_EXTENSIONS:
        return converted_path_for(project_dir, source)
    return source


@dataclass
class VsiConvertReport:
    n_converted: int = 0
    n_cached: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    converted: list[str] = field(default_factory=list)
    fiji_stdout: str = ""
    fiji_stderr: str = ""


def ensure_vsi_converted(
    project_dir: str, fiji_path: str | None,
    timeout: int = 1800,
) -> VsiConvertReport:
    """Convert every .vsi in ``project_dir`` whose cache is missing or stale.

    No-op (but still scans) when there are no VSI files. Raises nothing —
    failures are recorded on the report so the UI can surface them.
    """
    rep = VsiConvertReport()
    vsi_files = list_vsi_files(project_dir)
    if not vsi_files:
        return rep

    missing, orphans = audit_vsi_layout(project_dir)
    if missing:
        rep.warnings.append(
            "These .vsi files have no '_<stem>_' companion folder and "
            "will be skipped (Bio-Formats needs both): "
            + ", ".join(p.name for p in missing)
        )
    if orphans:
        rep.warnings.append(
            "Found companion folders with no matching .vsi file "
            "(likely renamed or typo): "
            + ", ".join(d.name for d in orphans)
        )

    convertible = [p for p in vsi_files if has_companion(p)]
    pending = [p for p in convertible if needs_conversion(project_dir, p)]
    rep.n_cached = len(convertible) - len(pending)
    if not pending:
        return rep

    if not fiji_path or not Path(fiji_path).exists():
        rep.errors.append(
            "FIJI is not configured — set the FIJI path in Setup before "
            "loading VSI files."
        )
        return rep
    if not _MACRO_PATH.exists():
        rep.errors.append(f"VSI macro missing at {_MACRO_PATH}.")
        return rep

    out_dir = converted_dir(project_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Macro iterates the whole input_dir; cached files are skipped inside
    # the macro by mtime, matching needs_conversion() above.
    try:
        proc = run_headless(
            fiji_path, _MACRO_PATH,
            {"input_dir": project_dir, "output_dir": str(out_dir)},
            timeout=timeout,
        )
    except Exception as e:
        rep.errors.append(f"FIJI subprocess failed: {e}")
        return rep
    rep.fiji_stdout = proc.stdout or ""
    rep.fiji_stderr = proc.stderr or ""

    for src in pending:
        out = converted_path_for(project_dir, src)
        if out.exists() and not needs_conversion(project_dir, src):
            rep.n_converted += 1
            rep.converted.append(src.name)
        else:
            rep.errors.append(f"{src.name}: conversion produced no output")
    return rep
