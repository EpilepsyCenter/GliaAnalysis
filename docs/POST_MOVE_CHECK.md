# Post-move verification

Run this **after copying the project to its new (non-Dropbox) location**, in
a terminal **inside the new folder**. The code is location-independent, but
the `.venv` is pinned to the old absolute path, so the one required step is
recreating it. Everything else is a sanity check.

## 1. Confirm you're in the new location, and git is intact

```bash
pwd                       # should show the NEW path, not .../Dropbox/...
git status                # clean / your branch
git remote -v             # origin -> github .../GliaAnalysis.git
git log --oneline -1      # latest commit present
```

- [ ] `pwd` is the new path (no `Dropbox`).
- [ ] `git status` / `remote` / `log` all respond normally.

## 2. Recreate the virtual environment (required)

The copied `.venv` still points at the old path (`activate`, `pip`, and
`pyvenv.cfg` hardcode it), so delete and rebuild it. Any Python ≥3.10 works.

**macOS / Linux**
```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
```

**Windows** — see [`INSTALL_WINDOWS.md`](INSTALL_WINDOWS.md) (use
`py -3.13` and `.\.venv\Scripts\Activate.ps1`).

- [ ] Install completes with **no compiler step** (wheels only).

## 3. Run the tests

```bash
python -m pytest -q
```

- [ ] Result is **7 passed, 1 failed**. The single failure
  (`tests/test_skeleton.py::test_all_features_count`, asserts 27 vs the
  current 36 features) is **pre-existing and unrelated to the move** — it
  fails the same way in the Dropbox copy. Anything else failing is new and
  worth investigating.

## 4. Launch the app

```bash
python app.py
```

- [ ] Opens and serves at <http://127.0.0.1:8050/> without errors.

## 5. UI smoke test

- [ ] **FIJI** is still configured (Settings shows the path, with "FIJI found."). It
  lives in `~/.gliaanalysis/defaults.json` (your home folder, not the
  project), so the move doesn't touch it.
- [ ] **Open project…** opens a project (e.g. a folder under
  `sample_images/`, if you copied it) and it loads.
- [ ] Setup tabs render; in astrocyte mode the sample project's **ROIs
  load**; **Stats → Download stats overview** and **Explore → Download raw
  data** produce files.

If all boxes tick, the move is good.

## Troubleshooting

- **`bad interpreter: .../.venv/bin/python: no such file`** or pip/activate
  errors pointing at a `Dropbox` path → you're still on the old copied
  venv. Do step 2 (delete `.venv`, recreate).
- **Slow first launch** is normal (imports warm up). Subsequent loads are
  fast.

---

## Sibling apps (if you copied the whole `Software` folder)

Same idea — code moves fine, per-machine artifacts get rebuilt:

- **NED-Net** — recreate its Python env; the `NED-Net.app` launcher
  hardcodes the old path + miniconda, so fix that path (or run its
  `post_move_setup.sh`). `.git` is now ~7 MB after the cleanup.
- **TRACER** — `npm install` (rebuild `node_modules`), rebuild the
  PyInstaller backend; its remote is **SSH** (`git@github.com:...`) so add
  an SSH key on this machine or switch the remote to HTTPS.
- **xPSC thingy** — **not a git repo**; back it up / `git init` before
  relying on it elsewhere, then `pip install -r requirements.txt` in a
  fresh env.
