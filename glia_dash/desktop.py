"""Native desktop window for GliaAnalysis.

Serves the Dash app with a production WSGI server (waitress) on a background
thread and displays it inside a native OS window — WebView2 on Windows,
WKWebView on macOS, Qt WebKit on Linux — instead of a browser tab.

The window chrome follows NED-Net's desktop app: a splash with the logo while
the app loads, a native menu bar (File, Go, View, Help), a window background
that matches the app theme, and the window size remembered between launches.

Launch with:
    python -m glia_dash.desktop
or the platform launcher:
    Windows : "Start GliaAnalysis (Native).bat"
    macOS   : "Start GliaAnalysis (Native).command"
    Linux   : "start-gliaanalysis-native.sh"

Requires the optional desktop dependencies:
    pip install -r requirements-desktop.txt
"""

from __future__ import annotations

import base64
import json
import logging
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import webview
from webview.menu import Menu, MenuAction, MenuSeparator
from waitress import serve

from glia_dash import __version__
# The Dash app (and with it numpy, scikit-image, ...) is imported in _boot,
# after the window shows its splash.

HOST = "127.0.0.1"
PREFERRED_PORT = 8060  # distinct from the browser launcher (8050); falls back if busy
TITLE = "GliaAnalysis"
MIN_SIZE = (960, 640)
# Default window: a comfortable fraction of the primary screen.
WIN_SCREEN_FRACTION = 0.90
WIN_MAX = (2200, 1400)
WIN_FALLBACK = (1440, 900)
SERVER_THREADS = 8  # waitress worker threads
# Window background while the page loads (--ned-bg in style.css), so the
# window doesn't flash white on start-up.
BACKGROUND = {"light": "#F3F2EE", "dark": "#0E1114"}

_ASSETS = Path(__file__).parent / "assets"
_WIN_STATE_PATH = Path.home() / ".gliaanalysis" / "window.json"


def _system_is_dark() -> bool:
    """macOS: the system appearance is Dark (the page follows it too)."""
    if sys.platform != "darwin":
        return False
    try:
        r = subprocess.run(["defaults", "read", "-g", "AppleInterfaceStyle"],
                           capture_output=True, text=True, timeout=2)
        return r.stdout.strip().lower() == "dark"
    except Exception:
        return False


def _bring_to_front() -> None:
    """Make GliaAnalysis the active app with the window in front.

    Started from a terminal (``python -m``), macOS leaves the new window
    behind the terminal and not focused, so the first clicks only activated it.
    """
    if sys.platform != "darwin":
        return
    try:
        from AppKit import NSApplication
        from PyObjCTools import AppHelper
        AppHelper.callAfter(
            lambda: NSApplication.sharedApplication().activateIgnoringOtherApps_(True))
    except Exception:
        pass


def _default_window_size() -> tuple[int, int]:
    try:
        screen = webview.screens[0]
        w = min(int(int(screen.width) * WIN_SCREEN_FRACTION), WIN_MAX[0])
        h = min(int(int(screen.height) * WIN_SCREEN_FRACTION), WIN_MAX[1])
    except Exception:
        w, h = WIN_FALLBACK
    return (max(w, MIN_SIZE[0]), max(h, MIN_SIZE[1]))


def _load_window_size() -> tuple[int, int]:
    """The last saved window size (clamped to MIN_SIZE), else a screen-sized
    default."""
    try:
        d = json.loads(_WIN_STATE_PATH.read_text())
        return (max(int(d["width"]), MIN_SIZE[0]),
                max(int(d["height"]), MIN_SIZE[1]))
    except Exception:
        return _default_window_size()


def _save_window_size(width: int, height: int) -> None:
    try:
        _WIN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _WIN_STATE_PATH.write_text(
            json.dumps({"width": int(width), "height": int(height)}))
    except Exception:
        pass


def _icon_path() -> str | None:
    """Window/dock icon: ``.ico`` on Windows, ``.icns`` on macOS, ``.png`` on
    Linux (all built by tools/make_logo.py)."""
    if sys.platform.startswith("win"):
        name = "glia_icon.ico"
    elif sys.platform == "darwin":
        name = "glia.icns"
    else:
        name = "glia_icon.png"
    icon = _ASSETS / name
    return str(icon) if icon.is_file() else None


def _set_macos_app_identity(icon: str | None) -> None:
    """Set the macOS dock icon and app name when launched via ``python -m``.

    (Otherwise the app shows up in the Dock as a generic "Python".)
    """
    try:
        from Foundation import NSBundle
        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        if info is not None:
            info["CFBundleName"] = TITLE
            info["CFBundleDisplayName"] = TITLE
    except ImportError:
        pass

    if icon:
        try:
            from AppKit import NSApplication, NSImage
            img = NSImage.alloc().initWithContentsOfFile_(icon)
            if img:
                NSApplication.sharedApplication().setApplicationIconImage_(img)
        except ImportError:
            pass


def _asset_data_uri(name: str) -> str:
    """An asset as a data: URI (the splash shows before the server runs)."""
    try:
        data = (_ASSETS / name).read_bytes()
    except Exception:
        return ""
    mime = "image/svg+xml" if name.endswith(".svg") else "image/png"
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


# ── Native menu bar ───────────────────────────────────────────────────
# macOS shows it in the global menu bar (next to the app menu with About /
# Hide / Quit); Windows and Linux show it inside the window. Menu actions
# drive the page through window.glia (assets/app_shell.js).

# Go menu: (page id, label). Pages not in the current cell-type mode are
# not in the sidebar, so their entries do nothing.
_GO_ITEMS = [
    ("setup", "Setup"),
    (None, None),
    ("segment", "Segment"),
    ("features", "Features"),
    ("cluster", "Cluster"),
    ("astrocyte_analysis", "Astrocyte Analysis"),
    (None, None),
    ("explore", "Explore"),
    ("inflammation", "Inflammation"),
    ("stats", "Stats"),
    (None, None),
    ("export", "Export"),
]

_ABOUT_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><style>
  :root {{ --bg: #f3f2ee; --muted: #55585f; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --bg: #0e1114; --muted: #9ba2ad; }}
    .light {{ display: none; }} .dark {{ display: inline; }} }}
  body {{ margin: 0; padding: 30px 24px; text-align: center; background: var(--bg);
         font: 13px -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, Cantarell,
               Ubuntu, sans-serif; user-select: none; cursor: default; }}
  img {{ height: 56px; margin-bottom: 14px; }}
  .dark {{ display: none; }}
  .v {{ color: var(--muted); margin: 4px 0 14px; }}
  p {{ margin: 0 auto; max-width: 300px; color: var(--muted); line-height: 1.45; }}
</style></head><body>
  <img class="light" src="{logo_light}" alt="GliaAnalysis">
  <img class="dark" src="{logo_dark}" alt="GliaAnalysis">
  <div class="v">Version {version}</div>
  <p>Morphology of microglia and astrocytes from fluorescence images: segmentation,
     features, clustering and statistics.</p>
</body></html>"""


def _run_js(window, code: str) -> None:
    """Run JavaScript in the page; ignore failures (e.g. page still loading)."""
    try:
        window.evaluate_js(code)
    except Exception:
        pass


def _build_menu(window, background: str) -> list:
    def call(js):
        return lambda: _run_js(window, f"window.glia && window.glia.{js}")

    def go(page):
        return call(f"go({json.dumps(page)})")

    def show_about():
        webview.create_window(
            f"About {TITLE}",
            html=_ABOUT_HTML.format(
                logo_light=_asset_data_uri("logo-light.svg"),
                logo_dark=_asset_data_uri("logo-dark.svg"),
                version=__version__),
            width=400, height=290, resizable=False, background_color=background,
        )

    file_items = [
        MenuAction("Open Project…", call("openProject()")),
        MenuSeparator(),
        MenuAction("Settings…", go("settings")),
    ]
    if sys.platform != "darwin":  # macOS has Quit in the app menu
        file_items += [MenuSeparator(), MenuAction("Exit", window.destroy)]

    go_items = [MenuSeparator() if page is None else MenuAction(label, go(page))
                for page, label in _GO_ITEMS]

    return [
        Menu("File", file_items),
        Menu("Go", go_items),
        Menu("View", [
            MenuAction("Microglia", call('setMode("microglia")')),
            MenuAction("Astrocyte", call('setMode("astrocyte")')),
            MenuSeparator(),
            MenuAction("Toggle Dark Mode", call("toggleTheme()")),
        ]),
        Menu("Help", [MenuAction(f"About {TITLE}", show_about)]),
    ]


# ── Server ────────────────────────────────────────────────────────────


def _pick_port(preferred: int) -> int:
    """Return ``preferred`` if free, otherwise an OS-assigned free port.

    Lets the native launcher coexist with the browser launcher (or a second
    instance) instead of crashing on "address already in use".
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((HOST, preferred))
            return preferred
        except OSError:
            pass  # taken — fall through to an ephemeral port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def _run_server(app, port: int) -> None:
    """Serve the Dash WSGI app via waitress (blocking) — runs in a daemon thread."""
    logging.getLogger("waitress").setLevel(logging.ERROR)  # quiet startup chatter
    serve(app.server, host=HOST, port=port, threads=SERVER_THREADS)


def _wait_until_up(host: str, port: int, timeout: float = 30.0) -> bool:
    """Block until the server accepts TCP connections, or timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.15)
    return False


# ── Splash screen while the app starts ───────────────────────────────

_SPLASH_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><style>
  :root {{ --bg: #f3f2ee; --muted: #55585f; --accent: #13788c;
           --track: rgba(20, 22, 28, 0.10); --danger: #b5332a; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #0e1114; --muted: #9ba2ad; --accent: #6cc3d5;
             --track: rgba(255, 255, 255, 0.12); --danger: #f0857c; }}
    .light {{ display: none; }} .dark {{ display: block; }}
  }}
  html, body {{ height: 100%; margin: 0; }}
  body {{ display: flex; flex-direction: column; align-items: center; justify-content: center;
         background: var(--bg); user-select: none; cursor: default;
         font: 13px -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, Cantarell,
               Ubuntu, sans-serif; }}
  img {{ height: 72px; margin-bottom: 30px; }}
  .dark {{ display: none; }}
  .bar {{ position: relative; width: 220px; height: 3px; border-radius: 2px;
          background: var(--track); overflow: hidden; }}
  .bar::after {{ content: ""; position: absolute; top: 0; bottom: 0; width: 40%;
                 border-radius: 2px; background: var(--accent);
                 animation: slide 1.1s ease-in-out infinite; }}
  @keyframes slide {{ from {{ left: -40%; }} to {{ left: 100%; }} }}
  #status {{ margin-top: 14px; color: var(--muted); }}
  .v {{ position: fixed; bottom: 16px; font-size: 11px; color: var(--muted); }}
  body.error .bar {{ display: none; }}
  body.error #status {{ color: var(--danger); max-width: 520px; text-align: center; }}
</style></head><body>
  <img class="light" src="{logo_light}" alt="GliaAnalysis">
  <img class="dark" src="{logo_dark}" alt="GliaAnalysis">
  <div class="bar"></div>
  <div id="status">Starting GliaAnalysis…</div>
  <div class="v">Version {version}</div>
  <script>
    window.setStatus = function (t) {{ document.getElementById('status').textContent = t; }};
    window.showError = function (t) {{
      document.body.classList.add('error');
      document.getElementById('status').textContent = t;
    }};
  </script>
</body></html>"""


def _splash_html() -> str:
    return _SPLASH_HTML.format(logo_light=_asset_data_uri("logo-light.svg"),
                               logo_dark=_asset_data_uri("logo-dark.svg"),
                               version=__version__)


def _boot(window) -> None:
    """Start the app behind the splash (runs in pywebview's start-up thread):
    import the Dash app, start its server, then load it into the window."""
    def status(text):
        _run_js(window, f"window.setStatus && window.setStatus({json.dumps(text)})")

    try:
        status("Loading analysis libraries…")
        from glia_dash.main import app

        status("Starting…")
        port = _pick_port(PREFERRED_PORT)
        threading.Thread(target=_run_server, args=(app, port), daemon=True,
                         name="gliaanalysis-server").start()
        if not _wait_until_up(HOST, port):
            raise RuntimeError(f"the app's server did not start on {HOST}:{port} within 30 s")
        window.load_url(f"http://{HOST}:{port}")
    except Exception as exc:  # show it on the splash instead of a dead window
        logging.exception("GliaAnalysis failed to start")
        _run_js(window, "window.showError && window.showError("
                        + json.dumps(f"GliaAnalysis could not start: {exc}") + ")")


def main() -> None:
    icon = _icon_path()
    if sys.platform == "darwin":
        _set_macos_app_identity(icon)

    background = BACKGROUND["dark" if _system_is_dark() else "light"]
    width, height = _load_window_size()
    size = {"w": width, "h": height}

    # pywebview cancels every download by default, which silently breaks all
    # dcc.Download export buttons. Allow them (a native Save dialog appears).
    webview.settings["ALLOW_DOWNLOADS"] = True

    # The window opens at once with the splash; _boot then loads the app.
    window = webview.create_window(
        TITLE,
        html=_splash_html(),
        width=width,
        height=height,
        min_size=MIN_SIZE,
        background_color=background,
    )
    # Started from a terminal, macOS would leave the window behind it
    window.events.shown += lambda *_a: _bring_to_front()

    # File/folder pickers open on this window: fast, and with keyboard focus.
    from glia_dash import file_dialogs
    file_dialogs.set_native_window(window)

    def _on_resized(*args):  # pywebview passes (width, height)
        if len(args) >= 2:
            size["w"], size["h"] = args[0], args[1]

    try:
        window.events.resized += _on_resized
    except Exception:
        pass  # older pywebview without the resized event

    # Blocks on the native GUI event loop; returns when the window is closed.
    # The server thread is a daemon, so it dies with the process on return.
    webview.start(_boot, (window,), icon=icon, menu=_build_menu(window, background))

    _save_window_size(size["w"], size["h"])


if __name__ == "__main__":
    main()
