# Running GliaAnalysis on Windows 11

This guide gets the app running on a fresh Windows 11 (x64) machine and
gives a step-by-step checklist to verify it. No code changes are required
— only the pinned `requirements-lock.txt` and the steps below.

> **Why this should "just work":** unlike Intel Macs (where `numba`/
> `llvmlite` have no wheel), **every** dependency in `requirements-lock.txt`
> ships a Windows (`win_amd64`) or pure-Python wheel for Python 3.13, so
> `pip install` downloads binaries only — **no compiler / "Visual C++
> Build Tools" prompt**. If you *do* see pip trying to *build* a package,
> stop and tell the maintainer which one (see Troubleshooting).

---

## 0. Get the code

**Option A — Git (recommended; lets you `git pull` updates).**
Install *Git for Windows* from <https://git-scm.com/download/win> (accept
the defaults), then in **PowerShell**:

```powershell
git clone https://github.com/EpilepsyCenter/GliaAnalysis.git
cd GliaAnalysis
```

Later, to update to the newest version: `git pull`.

**Option B — download a ZIP (no Git needed).**
On the GitHub page (<https://github.com/EpilepsyCenter/GliaAnalysis>) click
**Code ▸ Download ZIP**, extract it (right-click ▸ *Extract All…*), then
open the extracted `GliaAnalysis` folder in a terminal. To update later you
re-download the ZIP (there's no `git pull`).

All steps below assume you're **inside the project folder** — the one that
contains `app.py` and `requirements-lock.txt`.

---

## 1. Prerequisites

1. **Python 3.13 (64-bit) from python.org** — *not* the Microsoft Store
   build and *not* a minimal conda env.
   - Reason: the **"Browse folder…"** picker relies on `tkinter`, which the
     python.org installer includes but the Store build often omits. With
     python.org Python this works out of the box.
   - During install, tick **"Add python.exe to PATH"**.
   - Verify in a new terminal:
     ```
     py -3.13 --version        # -> Python 3.13.x
     py -3.13 -c "import tkinter; print('tk ok')"
     ```

2. **Fiji (ImageJ) for Windows** — download the Windows x64 build from
   https://fiji.sc and unzip it (e.g. to `C:\Fiji.app`). Note the
   executable path — one of:
   - `C:\Fiji.app\ImageJ-win64.exe`
   - `C:\Fiji.app\fiji-windows-x64.exe` (newer builds)

   FIJI is Java-based and runs natively on Windows; the app already passes
   it POSIX-style macro paths and omits `--console` on Windows, so no
   path-mangling workarounds are needed.

---

## 2. Install

From the project folder (the one containing `app.py`), in **PowerShell**:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1          # cmd.exe: .\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
```

> Do **not** copy a `.venv` from a Mac — it's architecture-specific and
> won't run on Windows. Always create the venv on the Windows machine.

The install should print `Downloading …whl` / `Using cached …whl` lines
and finish without compiling anything.

---

## 3. Run + configure

```powershell
python app.py
```

Open **http://127.0.0.1:8050** in a browser. Then:

1. **Sidebar → Mode**: pick *Microglia* or *Astrocyte* (re-select after each
   restart — mode isn't persisted yet).
2. **Sidebar → Browse folder…**: select the project folder (the one with
   your images / `.gliaanalysis_*` files).
3. **Setup → Threshold → FIJI path**: paste the full path to the FIJI
   executable from step 1.2 (the field is a text box — paste it directly;
   FIJI is not auto-discovered on Windows).

---

## 4. Verification checklist (run on the Windows 11 box)

Tick each; if one fails, copy the exact on-screen / terminal message.

- [ ] `py -3.13 --version` → `Python 3.13.x`, and `import tkinter` succeeds.
- [ ] `pip install -r requirements-lock.txt` completes with **no**
      "Building wheel for …" / compiler step.
- [ ] `python app.py` starts and the dashboard loads at `127.0.0.1:8050`.
- [ ] **Browse folder…** opens a native dialog and the project loads
      (this is the main Windows-specific risk — it exercises `tkinter`).
- [ ] **Setup → Threshold**: pasting the FIJI `.exe` path shows no
      "executable not found / points to a folder" warning.
- [ ] **Setup → Prepare**: run preparation on a sample image; confirm
      `_gliaanalysis\...\Prepared\*.tif` files appear.
- [ ] **Threshold (FIJI)**: run it; confirm `ThresholdedImages\*.tif`
      appear (this exercises the FIJI subprocess + Windows path handling).
- [ ] **Setup → ROIs**: draw one ROI, switch image and back — it persists
      (exercises JSON read/write on Windows paths).
- [ ] **Run analysis** (astrocyte) or **Segment** (microglia): rows/cells
      are produced.
- [ ] **Stats → Run**, then **Download stats overview (.xlsx)** opens in
      Excel.
- [ ] **Explore → Download raw data (Prism .xlsx)** downloads and opens.

---

## 5. Known Windows caveats (current behaviour)

These are documented, not yet fixed in code:

- **Project folder needs the native dialog.** There is no "paste a path"
  field for the project folder yet — only **Browse folder…**. With
  python.org Python this works; if the button does nothing, `tkinter` is
  missing (Store/conda Python) — reinstall Python from python.org.
- **FIJI is not auto-discovered** on Windows; paste the path manually (the
  field accepts typed paths).
- **Mode isn't persisted** across restarts — re-select Microglia/Astrocyte
  each session.

---

## 6. Troubleshooting

- **pip tries to *build* a package** (you see "Building wheel for X" or a
  request for build tools): that pinned version lacks a cp313 Windows
  wheel on this machine — report **X** and its version; the pin can be
  nudged to a version that has one.
- **FIJI threshold writes 0 images**: in the Astrocyte tab the result now
  shows a collapsible **FIJI log** — open it; it names the failing macro
  line (usually a wrong path or a FIJI build missing a plugin).
- **Very deep project paths fail**: Windows' legacy `MAX_PATH` limit —
  keep the project folder reasonably shallow, or enable Win32 long paths.
- **Browse button does nothing**: `tkinter` is missing — see §1.1.
