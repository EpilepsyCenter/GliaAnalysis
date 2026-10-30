"""Entry point of the packaged GliaAnalysis app (PyInstaller, see gliaanalysis.spec).

Besides starting the desktop app, it covers what a bundled app lacks:
- worker processes re-launch this executable: multiprocessing.freeze_support();
- Windows file dialogs run a small tkinter snippet in a second instance of the
  app (glia_dash.file_dialogs._tk_frozen): ``--glia-run-script <code>``;
- skan compiles its graph code with numba and caches it on disk; inside the
  bundle there is no writable place next to the code, so the cache goes to
  a per-user folder (NUMBA_CACHE_DIR);
- a windowed app has no console: output and errors go to a log file.
"""
import multiprocessing
import os
import sys


def _user_dir(kind: str) -> str:
    """Per-platform folder for the packaged app's log ("log") or cache ("cache")."""
    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        base = os.path.join(home, "Library", "Logs" if kind == "log" else "Caches")
        folder = os.path.join(base, "GliaAnalysis") if kind == "cache" else base
    elif sys.platform.startswith("win"):
        folder = os.path.join(os.environ.get("LOCALAPPDATA", home), "GliaAnalysis",
                              "cache" if kind == "cache" else "")
    else:
        env, default = (("XDG_CACHE_HOME", ".cache") if kind == "cache"
                        else ("XDG_STATE_HOME", os.path.join(".local", "state")))
        folder = os.path.join(os.environ.get(env, os.path.join(home, default)),
                              "gliaanalysis")
    os.makedirs(folder, exist_ok=True)
    return folder


def main() -> None:
    multiprocessing.freeze_support()

    try:
        os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(_user_dir("cache"), "numba"))
    except Exception:
        pass

    if len(sys.argv) >= 3 and sys.argv[1] == "--glia-run-script":
        # Helper mode (file dialogs on Windows): run the given snippet and exit.
        exec(compile(sys.argv[2], "<glia-helper>", "exec"), {"__name__": "__main__"})
        return

    # No console in a windowed app: keep output in a log file instead.
    try:
        log = open(os.path.join(_user_dir("log"), "GliaAnalysis.log"), "a",
                   buffering=1, encoding="utf-8")
        sys.stdout = log
        sys.stderr = log
        import datetime
        print(f"\n[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] GliaAnalysis starting "
              f"({sys.executable})", flush=True)
    except Exception:
        pass

    from glia_dash.desktop import main as desktop_main
    desktop_main()


if __name__ == "__main__":
    main()
