"""Build the GliaAnalysis logo SVGs and app icons from the design reference
("A · Sholl split": a glial cell split down the middle, astrocyte processes on the
left, a skeletonised microglia on the right, over dashed Sholl rings).

    python tools/make_logo.py        # any Python with PySide6 and Pillow

Needs IBM Plex Sans installed (only here, to draw the letters); the SVGs it writes
hold the letters as outlines, so the app needs no font. Output: glia_dash/assets/

  logo-light.svg / logo-dark.svg          detailed mark on its tile + wordmark
                                          (start panel, About, splash)
  wordmark-light.svg / wordmark-dark.svg  small-size mark on its tile + wordmark
                                          (sidebar)
  icon.svg                                app icon (macOS grid: 824-unit tile in 1024)
  mark.svg                                small-size mark alone, no tile (websites)
  glia_icon.png / glia_icon.ico / glia.icns / favicon.ico

Design values (from the mockup): "Glia" in ink and "Analysis" in a muted blue-grey,
IBM Plex Sans 500, letter-spacing −0.03 em. Astrocyte #F4B24A, microglia #3BC6D6,
rings #3A475C, ink #141A26. Light: tile #141A26, text #15171b / #5A6378.
Dark: tile #1E2738, text #f3f2ee / #9FB3C8. Icons at 64 px and below use the
small-size mark (heavier strokes, no thin branches).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QByteArray, QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QFont, QFontDatabase, QFontMetricsF, QImage, QPainter, QPainterPath,
)
from PySide6.QtSvg import QSvgRenderer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "glia_dash", "assets")
S = 100.0  # font size in SVG units (1 em = 100)

ASTRO, MICRO, RING, PALE, INK = "#F4B24A", "#3BC6D6", "#3A475C", "#F4F1EA", "#141A26"
THEMES = {
    # tile, "Glia", "Analysis"
    "light": ("#141A26", "#15171b", "#5A6378"),
    "dark": ("#1E2738", "#f3f2ee", "#9FB3C8"),
}
ICON_TILE = "#1E2738"


# ── The two marks, in a 200 × 200 box ──────────────────────────────────

def mark_detailed() -> str:
    rings = "".join(
        f'<circle cx="100" cy="100" r="{r}" fill="none" stroke="{RING}" '
        f'stroke-width="1.5" stroke-dasharray="3 4"/>' for r in (34, 58, 82))
    astro_paths = [
        ("M100 100 Q80 82 52 60", 8), ("M100 100 Q74 104 38 108", 8),
        ("M100 100 Q93 74 86 30", 7), ("M100 100 Q84 124 62 152", 8),
        ("M70 79 Q60 90 50 88", 4), ("M76 134 Q66 128 56 130", 4),
    ]
    astro = (f'<g fill="none" stroke="{ASTRO}" stroke-linecap="round">'
             + "".join(f'<path d="{d}" stroke-width="{w}"/>' for d, w in astro_paths)
             + "</g>")
    astro_tips = (f'<g fill="{ASTRO}">'
                  + "".join(f'<circle cx="{x}" cy="{y}" r="{r}"/>'
                            for x, y, r in ((52, 60, 6), (38, 108, 6), (86, 30, 5.5), (62, 152, 6)))
                  + "</g>")
    micro_paths = [
        "M100 100 L132 78 L150 52 L146 30", "M132 78 L162 82", "M150 52 L172 44",
        "M100 100 L138 118 L170 112", "M138 118 L152 146 L146 172", "M152 146 L176 156",
        "M100 100 L110 140 L104 172", "M110 140 L124 160",
    ]
    micro = (f'<g fill="none" stroke="{MICRO}" stroke-width="3" stroke-linecap="round" '
             f'stroke-linejoin="round">' + "".join(f'<path d="{d}"/>' for d in micro_paths) + "</g>")
    nodes = ((132, 78), (150, 52), (146, 30), (162, 82), (172, 44), (138, 118), (170, 112),
             (152, 146), (146, 172), (176, 156), (110, 140), (104, 172), (124, 160))
    micro_nodes = (f'<g fill="{MICRO}">'
                   + "".join(f'<circle cx="{x}" cy="{y}" r="3.5"/>' for x, y in nodes) + "</g>")
    soma = (f'<circle cx="100" cy="100" r="14" fill="{PALE}"/>'
            f'<path d="M100 86 A14 14 0 0 0 100 114 Z" fill="{ASTRO}"/>'
            f'<path d="M100 86 A14 14 0 0 1 100 114 Z" fill="{MICRO}"/>'
            f'<circle cx="100" cy="100" r="5" fill="{INK}"/>')
    return rings + astro + astro_tips + micro + micro_nodes + soma


def mark_small() -> str:
    return (
        f'<circle cx="100" cy="100" r="70" fill="none" stroke="{RING}" stroke-width="6"/>'
        f'<g fill="none" stroke="{ASTRO}" stroke-width="18" stroke-linecap="round">'
        '<path d="M100 100 L48 58"/><path d="M100 100 L34 110"/><path d="M100 100 L62 156"/></g>'
        f'<g fill="none" stroke="{MICRO}" stroke-width="9" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M100 100 L138 74 L150 40"/><path d="M138 74 L172 80"/>'
        '<path d="M100 100 L144 126 L150 168"/><path d="M144 126 L176 132"/></g>'
        f'<circle cx="100" cy="100" r="22" fill="{PALE}"/>'
        f'<path d="M100 78 A22 22 0 0 0 100 122 Z" fill="{ASTRO}"/>'
        f'<path d="M100 78 A22 22 0 0 1 100 122 Z" fill="{MICRO}"/>'
    )


def tile(x: float, y: float, size: float, radius: float, fill: str, mark: str,
         mark_frac: float) -> str:
    """A rounded square with the mark centred in it (mark_frac of the tile)."""
    m = size * mark_frac
    k = m / 200.0
    mx, my = x + (size - m) / 2, y + (size - m) / 2
    return (f'<rect x="{x:.2f}" y="{y:.2f}" width="{size:.2f}" height="{size:.2f}" '
            f'rx="{radius:.2f}" fill="{fill}"/>'
            f'<g transform="translate({mx:.2f} {my:.2f}) scale({k:.4f})">{mark}</g>')


# ── Letters as outlines ────────────────────────────────────────────────

def _d(path: QPainterPath) -> str:
    """QPainterPath → SVG path data (moves, lines, cubic curves)."""
    out, i, n = [], 0, path.elementCount()
    while i < n:
        e = path.elementAt(i)
        if e.isMoveTo():
            out.append(f"M{e.x:.2f} {e.y:.2f}")
        elif e.isLineTo():
            out.append(f"L{e.x:.2f} {e.y:.2f}")
        elif e.isCurveTo():
            c2, end = path.elementAt(i + 1), path.elementAt(i + 2)
            out.append(f"C{e.x:.2f} {e.y:.2f} {c2.x:.2f} {c2.y:.2f} {end.x:.2f} {end.y:.2f}")
            i += 2
        i += 1
    return "".join(out)


def _font() -> QFont:
    font = QFont("IBM Plex Sans")
    font.setPixelSize(int(S))
    font.setWeight(QFont.Medium)
    font.setLetterSpacing(QFont.AbsoluteSpacing, -0.03 * S)
    return font


def build_lockup(detailed: bool, theme: str) -> str:
    """Mark tile on the left, "Glia" + "Analysis" on the right, centred on the cap height."""
    tile_fill, ink, muted = THEMES[theme]
    font = _font()
    fm = QFontMetricsF(font)
    if detailed:   # design: 72 px tile beside 36 px type
        size, radius, frac, gap = 2.0 * S, 0.5 * S, 0.86, 0.5 * S
        mark = mark_detailed()
    else:          # design: 32/64 px icons, heavier mark
        size, radius, frac, gap = 1.45 * S, 0.34 * S, 0.86, 0.38 * S
        mark = mark_small()

    base = size / 2 + fm.capHeight() / 2          # cap height centred on the tile
    x = size + gap
    glia, analysis = QPainterPath(), QPainterPath()
    glia.addText(QPointF(x, base), font, "Glia")
    x += fm.horizontalAdvance("Glia")
    analysis.addText(QPointF(x, base), font, "Analysis")

    text_box = glia.boundingRect().united(analysis.boundingRect())
    pad = 2.0
    left, top = -pad, min(0.0, text_box.top()) - pad
    right = text_box.right() + pad
    bottom = max(size, text_box.bottom()) + pad
    w, h = right - left, bottom - top
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{left:.2f} {top:.2f} {w:.2f} {h:.2f}" '
            f'width="{w:.1f}" height="{h:.1f}" role="img" aria-label="GliaAnalysis">'
            + tile(0, 0, size, radius, tile_fill, mark, frac)
            + f'<path d="{_d(glia)}" fill="{ink}"/>'
            + f'<path d="{_d(analysis)}" fill="{muted}"/>'
            + "</svg>\n")


def build_icon(small: bool) -> str:
    """App icon: the mark on a dark rounded square (macOS grid: 824-unit tile with
    100-unit margins in a 1024 canvas)."""
    mark = mark_small() if small else mark_detailed()
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" width="1024" height="1024">'
            + tile(100, 100, 824, 185, ICON_TILE, mark, 0.84 if small else 0.86)
            + "</svg>\n")


# ── Rasterising ────────────────────────────────────────────────────────

def render(svg: str, px: int) -> QImage:
    r = QSvgRenderer(QByteArray(svg.encode()))
    img = QImage(px, px, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    r.render(p, QRectF(0, 0, px, px))
    p.end()
    return img


def icon_for(px: int, big: str, small: str) -> QImage:
    return render(small if px <= 64 else big, px)


def write_ico(path: str, sizes: list[int], big: str, small: str) -> None:
    from PIL import Image
    with tempfile.TemporaryDirectory() as tmp:
        frames = []
        for px in sizes:
            f = os.path.join(tmp, f"{px}.png")
            icon_for(px, big, small).save(f)
            frames.append(Image.open(f).convert("RGBA"))
        frames[-1].save(path, format="ICO", sizes=[(s, s) for s in sizes],
                        append_images=frames[:-1])


def write_icns(path: str, big: str, small: str) -> None:
    if not shutil.which("iconutil"):
        print("iconutil not found: skipping", os.path.basename(path))
        return
    with tempfile.TemporaryDirectory() as tmp:
        iconset = os.path.join(tmp, "glia.iconset")
        os.makedirs(iconset)
        for size in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                px = size * scale
                icon_for(px, big, small).save(
                    os.path.join(iconset, f"icon_{size}x{size}{'@2x' if scale == 2 else ''}.png"))
        subprocess.run(["iconutil", "-c", "icns", iconset, "-o", path], check=True)


def main() -> int:
    app = QApplication(sys.argv)  # noqa: F841
    if "IBM Plex Sans" not in QFontDatabase.families():
        print("IBM Plex Sans is not installed; it is needed to draw the letters.")
        return 1
    os.makedirs(OUT, exist_ok=True)
    for theme in THEMES:
        for name, detailed in (("logo", True), ("wordmark", False)):
            with open(os.path.join(OUT, f"{name}-{theme}.svg"), "w") as fh:
                fh.write(build_lockup(detailed, theme))
    with open(os.path.join(OUT, "mark.svg"), "w") as fh:
        fh.write('<svg xmlns="http://www.w3.org/2000/svg" viewBox="22 22 156 156" '
                 'width="156" height="156" role="img" aria-label="GliaAnalysis">'
                 + mark_small() + "</svg>\n")
    big, small = build_icon(False), build_icon(True)
    with open(os.path.join(OUT, "icon.svg"), "w") as fh:
        fh.write(big)
    icon_for(512, big, small).save(os.path.join(OUT, "glia_icon.png"))
    write_ico(os.path.join(OUT, "glia_icon.ico"), [16, 24, 32, 48, 64, 128, 256], big, small)
    write_ico(os.path.join(OUT, "favicon.ico"), [16, 32, 48], big, small)
    write_icns(os.path.join(OUT, "glia.icns"), big, small)
    print("written to", os.path.abspath(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
