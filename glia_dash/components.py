"""Shared Dash components and OS-native dialog helpers.

Mirrors NED-Net's eeg_seizure_analyzer/dash_app/components.py so both
apps share the same visual building blocks.
"""

from __future__ import annotations

from dash import html, dcc


# ── Native OS file / folder pickers ───────────────────────────────────
# Thin wrappers over glia_dash.file_dialogs (ported from NED-Net): the
# desktop window's own dialog on macOS, AppleScript in a browser on macOS,
# zenity/kdialog on Linux, tkinter on Windows — also inside the packaged app.


def browse_folder(title: str = "Select folder") -> str | None:
    """Open a native folder picker. Returns selected path or None."""
    from glia_dash.file_dialogs import pick_folder
    return pick_folder(title)


def browse_file(
    title: str = "Select file",
    filetypes: list[tuple[str, str]] | None = None,
) -> str | None:
    """Open a native file picker. filetypes: list of (label, pattern) tuples;
    the extensions in the patterns filter the dialog ("*" = any file)."""
    from glia_dash.file_dialogs import pick_file
    filetypes = filetypes or [("TIFF images", "*.tif *.tiff"), ("All files", "*")]
    exts = [tok for _, pattern in filetypes for tok in pattern.split()
            if tok not in ("*", "*.*")]
    return pick_file(title, exts)


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
    """Title + explanation for a page with nothing to show yet.

    ``icon`` is accepted for old call sites but not drawn (no emoji in
    the interface).
    """
    return html.Div(
        className="empty-state",
        children=[
            html.Div(title, className="empty-title"),
            html.Div(text, className="empty-text"),
        ],
    )


def logo(kind: str = "logo", class_name: str = "") -> html.Span:
    """The GliaAnalysis logo in both appearances; CSS shows the one that
    matches the theme. ``kind``: "logo" (detailed mark, large sizes) or
    "wordmark" (small-size mark, e.g. the sidebar). Built by
    tools/make_logo.py."""
    return html.Span(
        className=f"glia-logo {class_name}".strip(),
        children=[
            html.Img(src=f"/assets/{kind}-light.svg", alt="GliaAnalysis",
                     className="logo-light"),
            html.Img(src=f"/assets/{kind}-dark.svg", alt="GliaAnalysis",
                     className="logo-dark"),
        ],
    )


def start_panel() -> html.Div:
    """Shown on Setup › Prepare while no project folder is open."""
    return html.Div(
        className="start-panel",
        children=[
            logo("logo", "start-logo"),
            html.Div("Microglia and astrocyte morphometry",
                     className="start-tagline"),
            html.P(
                "Open a folder of microscopy images to begin. The Prepare "
                "step accepts TIFF, OME-TIFF, CZI, ND2, LIF and Olympus VSI "
                "(converted with FIJI Bio-Formats) and writes 8-bit "
                "single-channel TIFFs to _gliaanalysis/Prepared/ for the "
                "rest of the pipeline.",
                className="start-text",
            ),
            # Clicks the toolbar's Open project… button (assets/app_shell.js)
            html.Button("Open project…", id="start-open-project",
                        className="btn btn-ned-primary btn-lg"),
        ],
    )


def roi_include_control(
    tags: list, value: list, control_id: str,
    *, hint: str | None = None,
) -> html.Div:
    """A multi-select that restricts an analysis to chosen ``roi_tag`` values.

    The dropdown is always rendered (so a callback ``State`` bound to
    ``control_id`` always resolves), but its container is hidden when
    there are fewer than two ROI tags — filtering is meaningless then.
    Empty selection = all ROIs.
    """
    visible = len(tags or []) >= 2
    default_hint = ("Uncheck ROIs to drop them from this analysis — e.g. "
                    "ones underrepresented in some groups. Empty = all ROIs.")
    return html.Div([
        html.Label("ROIs to include",
                   style={"fontSize": "0.72rem",
                          "color": "var(--ned-text-muted)"}),
        dcc.Dropdown(id=control_id,
                     options=[{"label": str(t), "value": str(t)}
                              for t in (tags or [])],
                     value=list(value or []),
                     multi=True,
                     placeholder="All ROIs",
                     style={"minWidth": "320px"}),
        html.Div(hint or default_hint,
                 style={"fontSize": "0.75rem",
                        "color": "var(--ned-text-muted)",
                        "marginTop": "4px"}),
    ], style={"display": "block" if visible else "none",
              "marginBottom": "12px"})


def alert(message: str, variant: str = "info") -> html.Div:
    """Styled alert. variant: info, warning, danger, success."""
    cls = f"ned-alert {variant}" if variant != "info" else "ned-alert"
    return html.Div(message, className=cls)


# ── Plotly figure defaults ────────────────────────────────────────────
#
# One palette for every on-screen figure, matching the app theme in
# assets/style.css (warm paper in light mode, cool near-black in dark, one
# cyan accent). Exported figures keep their own colours.

FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, "
        "Cantarell, Ubuntu, 'Noto Sans', sans-serif")
FONT_MONO = ("ui-monospace, 'SF Mono', 'Cascadia Mono', Menlo, Consolas, "
             "'DejaVu Sans Mono', monospace")

_PALETTES = {
    "light": dict(
        paper="#fcfbf8", plot="#fcfbf8", fg="#15171b", muted="#55585f",
        grid="#e4e2dc", accent="#13788c", ok="#2f8a4f", warn="#c27a12",
        pos="#b5332a", neg="#2f6db5", ns="#9a9ca3",
        colorway=["#13788c", "#c46a1a", "#4f7f3a", "#b5332a",
                  "#6a4fb0", "#b0477d", "#2f6db5", "#8a7a2c"],
    ),
    "dark": dict(
        paper="#161a1f", plot="#161a1f", fg="#e6e8ec", muted="#9ba2ad",
        grid="#272c33", accent="#6cc3d5", ok="#6cc28a", warn="#e8a35c",
        pos="#f0857c", neg="#86aef0", ns="#6b717b",
        colorway=["#6cc3d5", "#e8a35c", "#8cc477", "#f0857c",
                  "#b49cf0", "#e98bbd", "#86aef0", "#d1c06a"],
    ),
}

_current_theme = "light"


def set_plotly_theme(theme: str) -> None:
    global _current_theme
    _current_theme = theme


def fig_palette(theme: str | None = None) -> dict:
    """Colours for a Plotly figure in the given theme ('light' / 'dark').

    Keys: paper, plot, fg, muted, grid, accent, ok, warn, pos, neg, ns,
    colorway (8 series colours), font.
    """
    t = theme or _current_theme or "light"
    p = dict(_PALETTES["dark" if t == "dark" else "light"])
    p["font"] = FONT
    return p


def apply_fig_theme(fig, theme: str | None = None, size: int = 12):
    """Paper, plot area, font and grid colours of the current theme."""
    p = fig_palette(theme)
    fig.update_layout(
        paper_bgcolor=p["paper"],
        plot_bgcolor=p["plot"],
        font=dict(color=p["fg"], family=FONT, size=size),
    )
    fig.update_xaxes(gridcolor=p["grid"], zerolinecolor=p["grid"])
    fig.update_yaxes(gridcolor=p["grid"], zerolinecolor=p["grid"])
    return fig
