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
import sys
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
    # On Windows, ImageJ-win64.exe is a GUI-subsystem binary with no stdout.
    # Passing --console makes the launcher relaunch itself to attach a console
    # ("GUI program launched from unknown parent process"), and that relaunch
    # mangles the trailing "-macro <path>" arguments — the macro path is
    # dropped and "-macro" itself is taken as the script name ("Macro or
    # script not found: …/macros/-macro"). We detect results by reading the
    # output files, not stdout, so just omit --console on Windows. On
    # macOS/Linux it's harmless and gives us the FIJI log for diagnostics.
    console = [] if sys.platform == "win32" else ["--console"]
    if macro_path.lower().endswith(".bsh"):
        # FIJI auto-detects script language from extension when a script
        # path is passed positionally. The trailing argument lands in
        # bsh.args[0].
        cmd = [
            str(fiji_path),
            "--headless",
            *console,
            macro_path,
            arg_string,
        ]
    else:
        cmd = [
            str(fiji_path),
            "--headless",
            *console,
            "-macro",
            macro_path,
            arg_string,
        ]
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )


def validate_fiji_path(fiji_path: str | Path | None) -> str | None:
    """Return a human-readable error if ``fiji_path`` can't be launched.

    Returns ``None`` when the path looks runnable. The common Windows
    mistake is pointing at the ``Fiji.app`` *folder* rather than the
    executable inside it: ``Path.exists()`` is True for a directory, so
    naive checks pass and ``subprocess.run`` then dies with
    ``WinError 5: Access is denied``. We catch that here with a clear hint.
    """
    if not fiji_path:
        return ("No FIJI path configured. Set it to the FIJI executable in "
                "Settings.")
    p = Path(fiji_path)
    if not p.exists():
        return f"FIJI executable not found: {fiji_path}"
    if p.is_dir():
        return (
            f"FIJI path points to a folder, not the executable: {fiji_path}\n"
            "Point it at the program file inside the Fiji folder — on Windows "
            r"that's ...\Fiji.app\ImageJ-win64.exe (or fiji-windows-x64.exe on "
            "newer builds); on macOS it's "
            ".../Fiji.app/Contents/MacOS/ImageJ-macosx."
        )
    return None


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


# Program names inside a Fiji installation, newest launchers first.
_FIJI_EXECUTABLES = (
    "Contents/MacOS/fiji-macos-arm64", "Contents/MacOS/fiji-macos-x64",
    "Contents/MacOS/fiji-macos", "Contents/MacOS/ImageJ-macosx",
    "fiji-windows-x64.exe", "ImageJ-win64.exe", "ImageJ-win32.exe",
    "fiji-linux-x64", "ImageJ-linux64", "ImageJ-linux32",
)


def resolve_fiji_path(path: str | Path | None) -> str:
    """The FIJI program for a picked path: a folder or ``Fiji.app`` bundle
    becomes the executable inside it (the first known launcher name that
    exists); anything else is returned unchanged."""
    if not path:
        return ""
    p = Path(path)
    if p.is_dir():
        for name in _FIJI_EXECUTABLES:
            exe = p / name
            if exe.is_file():
                return str(exe)
    return str(path)
