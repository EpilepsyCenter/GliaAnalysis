"""Native desktop window for GliaAnalysis.

Serves the Dash app with a production WSGI server (waitress) on a background
thread and displays it inside a native OS window — WebView2 on Windows,
WKWebView on macOS, Qt WebKit on Linux — instead of a browser tab.

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

import logging
import socket
import sys
import threading
import time
from pathlib import Path

import webview
from waitress import serve

from glia_dash.main import app

HOST = "127.0.0.1"
PREFERRED_PORT = 8060  # distinct from the browser launcher (8050); falls back if busy
TITLE = "GliaAnalysis"
WIN_WIDTH = 1440
WIN_HEIGHT = 900
MIN_SIZE = (960, 640)
SERVER_THREADS = 8  # waitress worker threads

_ASSETS = Path(__file__).parent / "assets"


def _icon_path() -> str | None:
    """Return the platform-appropriate window/dock icon, or None if missing.

    Windows needs a real ``.ico``; macOS (NSImage) and Linux (GTK/Qt) load the
    ``.png`` directly.
    """
    name = "glia_icon.ico" if sys.platform.startswith("win") else "glia_icon.png"
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


def _run_server(port: int) -> None:
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


def main() -> None:
    icon = _icon_path()
    if sys.platform == "darwin":
        _set_macos_app_identity(icon)

    port = _pick_port(PREFERRED_PORT)

    server_thread = threading.Thread(
        target=_run_server, args=(port,), daemon=True, name="gliaanalysis-server"
    )
    server_thread.start()

    if not _wait_until_up(HOST, port):
        raise RuntimeError(
            f"GliaAnalysis server did not come up on {HOST}:{port} within 30s."
        )

    # pywebview cancels every download by default, which silently breaks all
    # dcc.Download export buttons. Allow them (a native Save dialog appears).
    webview.settings["ALLOW_DOWNLOADS"] = True

    webview.create_window(
        TITLE,
        f"http://{HOST}:{port}",
        width=WIN_WIDTH,
        height=WIN_HEIGHT,
        min_size=MIN_SIZE,
    )
    # Blocks on the native GUI event loop; returns when the window is closed.
    # The server thread is a daemon, so it dies with the process on return.
    webview.start(icon=icon)


if __name__ == "__main__":
    main()
