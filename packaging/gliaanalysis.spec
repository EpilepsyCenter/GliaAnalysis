# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller recipe for the GliaAnalysis desktop app (after NED-Net's).

Build on each platform (PyInstaller does not cross-compile), from a clean
environment with the app's dependencies installed:

    python -m venv .venv-build && source .venv-build/bin/activate   # Windows: .venv-build\\Scripts\\activate
    pip install -r requirements-lock.txt -r requirements-desktop.txt pyinstaller
    pyinstaller packaging/gliaanalysis.spec --noconfirm

Output in dist/:
    macOS    GliaAnalysis.app
    Windows  GliaAnalysis\\GliaAnalysis.exe (with its folder)
    Linux    GliaAnalysis/GliaAnalysis (with its folder)

A folder build ("onedir"), not a single file: a single-file app unpacks itself
on every start, which is slow and makes virus scanners check it again each
time. FIJI is not bundled; the app runs the user's own FIJI (Settings page).
"""
import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).parent
APP = ROOT / "glia_dash"
VERSION = re.search(r'__version__\s*=\s*"([^"]+)"',
                    (APP / "__init__.py").read_text()).group(1)

# bioio finds its image readers (CZI, LIF, ND2, OME-TIFF, TIFF) through
# package entry points, i.e. the packages' metadata: bundle it with the code.
BIOIO = ["bioio", "bioio-base", "bioio-czi", "bioio-lif", "bioio-nd2",
         "bioio-ome-tiff", "bioio-tifffile"]

datas = [
    (str(APP / "assets"), "glia_dash/assets"),
    # glia/segment.py and glia/vsi_convert.py look for ../fiji_macros
    (str(ROOT / "fiji_macros"), "fiji_macros"),
]
datas += collect_data_files("dash_ag_grid")
datas += collect_data_files("pingouin")
datas += collect_data_files("ome_types")
for dist in BIOIO + ["ome-types", "xsdata"]:
    datas += copy_metadata(dist)

hiddenimports = (
    collect_submodules("glia")
    + collect_submodules("glia_dash")
    + [m for dist in BIOIO for m in collect_submodules(dist.replace("-", "_"))]
    # ome-types (OME-XML metadata) has xsdata load its pydantic plugin through
    # an entry point; the plugin module ships inside the ome-types distribution.
    + collect_submodules("ome_types")
    + collect_submodules("xsdata_pydantic_basemodel")
    # imagecodecs imports each codec (LZW, zstd, JPEG …) by name only when a
    # compressed TIFF needs it
    + collect_submodules("imagecodecs")
    # Windows file dialogs run tkinter from a string (file_dialogs._tk_frozen)
    + ["tkinter", "tkinter.filedialog", "waitress"]
)

excludes = ["IPython", "jupyter_client", "jupyter_core", "notebook", "pytest"]
if not sys.platform.startswith("linux"):
    # pywebview uses the system web engine on macOS and Windows; Qt only on Linux
    excludes += ["PyQt5", "PyQt6", "PySide2", "PySide6", "qtpy"]

if sys.platform == "darwin":
    ICON = APP / "assets" / "glia.icns"
elif sys.platform.startswith("win"):
    ICON = APP / "assets" / "glia_icon.ico"
else:
    ICON = APP / "assets" / "glia_icon.png"

a = Analysis(
    [str(ROOT / "packaging" / "glia_entry.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="GliaAnalysis",
    console=False,          # a desktop app: no terminal window
    icon=str(ICON),
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="GliaAnalysis", upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="GliaAnalysis.app",
        icon=str(ICON),
        bundle_identifier="se.lu.med.gliaanalysis",
        version=VERSION,
        info_plist={
            "CFBundleName": "GliaAnalysis",
            "CFBundleDisplayName": "GliaAnalysis",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,   # follow Dark Mode
            "LSMinimumSystemVersion": "11.0",
        },
    )
