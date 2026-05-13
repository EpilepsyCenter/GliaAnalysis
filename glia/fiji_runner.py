"""Headless FIJI subprocess wrapper.

We can run FIJI without a GUI for the parts of the original pipeline that
don't depend on the BioVoxxel or FracLac GUI plugins. That covers:
    - Thresholding (any of the 16 global / 9 local auto methods)
    - Analyze Particles for single-cell extraction
    - Skeletonize + AnalyzeSkeleton

BioVoxxel ThresholdCheck is replaced by an in-app preview in the Setup tab.
FracLac is replaced entirely by glia.features.compute_geometric_features.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def run_headless(
    fiji_path: str | Path,
    macro_path: str | Path,
    args: dict,
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    """Invoke FIJI in headless mode with a script + pipe-delimited args.

    Supports two script flavours:
      * ``.ijm`` — IJ1 macro, args read via ``getArgument()``.
      * ``.bsh`` — Beanshell, args read via ``bsh.args[0]``. We use this
        for tasks where the IJ1 macro layer trips over headless GUI
        prompts (e.g. Bio-Formats Importer's prompter dialog).

    A pipe delimiter is used rather than commas because filenames
    sometimes contain commas.
    """
    arg_string = "|".join(f"{k}={v}" for k, v in args.items())
    macro_path = str(macro_path)
    if macro_path.lower().endswith(".bsh"):
        # FIJI auto-detects script language from extension when a script
        # path is passed positionally. The trailing argument lands in
        # bsh.args[0].
        cmd = [
            str(fiji_path),
            "--headless",
            "--console",
            macro_path,
            arg_string,
        ]
    else:
        cmd = [
            str(fiji_path),
            "--headless",
            "--console",
            "-macro",
            macro_path,
            arg_string,
        ]
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )


def discover_fiji() -> str | None:
    """Best-effort lookup of FIJI on disk. Returns None if not found."""
    candidates = [
        "/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx",
        "/Applications/Fiji/Contents/MacOS/ImageJ-macosx",
        "/usr/local/bin/fiji",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None
