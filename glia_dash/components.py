"""Shared Dash components and OS-native dialog helpers.

Mirrors NED-Net's eeg_seizure_analyzer/dash_app/components.py so both
apps share the same visual building blocks.
"""

from __future__ import annotations

import platform
import subprocess
import sys

from dash import html, dcc


# ── Native OS file / folder pickers ───────────────────────────────────
# Ported verbatim from NED-Net's upload.py. AppleScript on macOS opens
# the dialog in the foreground; tkinter is a cross-platform fallback.


def browse_folder(title: str = "Select folder") -> str | None:
    """Open a native folder picker. Returns selected path or None."""
    if platform.system() == "Darwin":
        try:
            r = subprocess.run(
                ["osascript", "-e",
                 f'POSIX path of (choose folder with prompt "{title}")'],
                capture_output=True, text=True, timeout=120,
            )
            folder = r.stdout.strip().rstrip("/")
            return folder if folder else None
        except Exception:
            pass

    try:
        r = subprocess.run(
            [sys.executable, "-c", "\n".join([
                "import tkinter as tk",
                "from tkinter import filedialog",
                "root = tk.Tk()",
                "root.withdraw()",
                "root.attributes('-topmost', True)",
                "root.update()",
                f'folder = filedialog.askdirectory(title="{title}")',
                "root.destroy()",
                "print(folder or '')",
            ])],
            capture_output=True, text=True, timeout=120,
        )
        folder = r.stdout.strip()
        return folder if folder else None
    except Exception:
        return None


def browse_file(
    title: str = "Select file",
    filetypes: list[tuple[str, str]] | None = None,
) -> str | None:
    """Open a native file picker. filetypes: list of (label, pattern) tuples."""
    filetypes = filetypes or [("TIFF images", "*.tif *.tiff"), ("All files", "*")]

    if platform.system() == "Darwin":
        try:
            exts = []
            for _, pattern in filetypes:
                for tok in pattern.split():
                    ext = tok.replace("*.", "").strip()
                    if ext and ext != "*":
                        exts.append(f'"{ext}"')
            type_clause = ""
            if exts:
                type_clause = f" of type {{{', '.join(exts)}}}"
            script = (
                f'POSIX path of (choose file with prompt "{title}"'
                f"{type_clause})"
            )
            r = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=120,
            )
            path = r.stdout.strip()
            return path if path else None
        except Exception:
            pass

    try:
        ft_lines = ",\n        ".join(
            f'("{label}", "{pat}")' for label, pat in filetypes
        )
        r = subprocess.run(
            [sys.executable, "-c", "\n".join([
                "import tkinter as tk",
                "from tkinter import filedialog",
                "root = tk.Tk()",
                "root.withdraw()",
                "root.attributes('-topmost', True)",
                "root.update()",
                "path = filedialog.askopenfilename(",
                f'    title="{title}",',
                "    filetypes=[",
                f"        {ft_lines}",
                "    ],",
                ")",
                "root.destroy()",
                "print(path)",
            ])],
            capture_output=True, text=True, timeout=120,
        )
        path = r.stdout.strip()
        return path if path else None
    except Exception:
        return None


# ── Layout helpers ────────────────────────────────────────────────────


def section_header(title: str) -> html.Div:
    """Sidebar section header."""
    return html.Div(
        className="sidebar-section",
        children=[html.Div(title, className="sidebar-section-label")],
    )


def sidebar_divider() -> html.Hr:
    return html.Hr(className="sidebar-divider")


def metric_card(label: str, value: str, accent: bool = False) -> html.Div:
    cls = "metric-value accent" if accent else "metric-value"
    return html.Div(
        className="metric-card",
        children=[
            html.Div(label, className="metric-label"),
            html.Div(value, className=cls),
        ],
    )


def empty_state(icon: str, title: str, text: str) -> html.Div:
    return html.Div(
        className="empty-state",
        children=[
            html.Div(icon, className="empty-icon"),
            html.Div(title, className="empty-title"),
            html.Div(text, className="empty-text"),
        ],
    )


def alert(message: str, variant: str = "info") -> html.Div:
    """Styled alert. variant: info, warning, danger, success."""
    cls = f"ned-alert {variant}" if variant != "info" else "ned-alert"
    return html.Div(message, className=cls)


# ── Plotly figure defaults ────────────────────────────────────────────

_FONT = "IBM Plex Sans, sans-serif"

_PLOTLY_DARK = {
    "paper_bgcolor": "#1c2128",
    "plot_bgcolor": "#0f1117",
    "font_color": "#e6edf3",
    "gridcolor": "#2d333b",
    "colorway": [
        "#58a6ff", "#3fb950", "#d29922", "#f85149",
        "#bc8cff", "#f778ba", "#79c0ff", "#56d364",
    ],
}

_PLOTLY_LIGHT = {
    "paper_bgcolor": "#ffffff",
    "plot_bgcolor": "#f6f8fa",
    "font_color": "#1f2328",
    "gridcolor": "#d0d7de",
    "colorway": [
        "#0969da", "#1a7f37", "#9a6700", "#cf222e",
        "#8250df", "#bf3989", "#0550ae", "#116329",
    ],
}

_current_theme = "light"


def set_plotly_theme(theme: str) -> None:
    global _current_theme
    _current_theme = theme


def apply_fig_theme(fig):
    p = _PLOTLY_DARK if _current_theme == "dark" else _PLOTLY_LIGHT
    fig.update_layout(
        paper_bgcolor=p["paper_bgcolor"],
        plot_bgcolor=p["plot_bgcolor"],
        font=dict(color=p["font_color"], family=_FONT, size=12),
        xaxis=dict(gridcolor=p["gridcolor"], zerolinecolor=p["gridcolor"]),
        yaxis=dict(gridcolor=p["gridcolor"], zerolinecolor=p["gridcolor"]),
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig
