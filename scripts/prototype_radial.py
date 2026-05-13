"""Quick-and-dirty visual check of glia.radial on real single-cell masks.

Picks the 6 largest cells from the RAM SingleCells folder — those are
the most likely to be intact microglia with a proper soma + processes —
runs ``analyze_radial`` on each at three gap_tol_deg values to show how
the soma boundary reacts to the one tunable, and writes a PNG grid.

Run from repo root:
    PYTHONPATH=. .venv/bin/python scripts/prototype_radial.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.patches import Circle

from glia.radial import analyze_radial

SINGLECELLS = Path(
    "sample_images/RAM Iba1 GFAP/_gliaanalysis/SingleCells"
)
OUT = Path("scripts/_radial_proto_out")
OUT.mkdir(parents=True, exist_ok=True)

# Three gap tolerances side-by-side per cell so we can pick a default.
GAP_TOLS = (12.0, 20.0, 30.0)


def pick_cells(folder: Path, k: int = 6) -> list[Path]:
    """Pick the k largest masks — most likely to be intact whole cells."""
    cells = list(folder.glob("*.tif"))
    sized = []
    for p in cells:
        m = tifffile.imread(p) > 0
        sized.append((p, int(m.sum())))
    sized.sort(key=lambda t: t[1], reverse=True)
    return [p for p, _ in sized[:k]]


def render_one(ax, mask, res, title, gap_tol):
    h, w = mask.shape

    ax.imshow(mask, cmap="gray", interpolation="nearest")
    ax.imshow(np.ma.masked_where(~res.soma_mask, res.soma_mask),
              cmap="cool", alpha=0.5, interpolation="nearest")

    cy, cx = res.center_yx

    # r_out fan, color-coded by process membership
    ang_rad = np.radians(res.angles_deg)
    end_y = cy + res.r_out * np.sin(ang_rad)
    end_x = cx + res.r_out * np.cos(ang_rad)
    step = max(1, len(ang_rad) // 90)
    for i in range(0, len(ang_rad), step):
        c = "#d29922" if res.process_angle_mask[i] else "#3fb950"
        ax.plot([cx, end_x[i]], [cy, end_y[i]],
                color=c, alpha=0.25, linewidth=0.4)

    # Soma polygon outline
    poly = res.soma_polygon
    poly_closed = np.vstack([poly, poly[:1]])
    ax.plot(poly_closed[:, 1], poly_closed[:, 0],
            color="#ff7eb6", linewidth=1.8)

    # Critical-radius ring
    ax.add_patch(Circle((cx, cy), res.critical_radius,
                        fill=False, edgecolor="#f0e442",
                        linestyle="--", linewidth=1.2))

    # Center dot
    ax.plot([cx], [cy], "o", color="#f85149", markersize=6)

    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(
        f"{title}\n"
        f"gap_tol={gap_tol:.0f}°  r₀={res.critical_radius:.0f}  "
        f"proc={res.primary_process_count}  "
        f"soma/cell={res.soma_to_cell_area_ratio:.2f}",
        fontsize=8,
    )


def main():
    cells = pick_cells(SINGLECELLS, k=6)
    print("Selected cells (largest first):")
    for c in cells:
        print(f"  {c.name}")

    n_rows = len(cells)
    n_cols = len(GAP_TOLS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 4.2 * n_rows),
                             constrained_layout=True)

    for row, cell_path in enumerate(cells):
        mask = tifffile.imread(cell_path) > 0
        for col, gap in enumerate(GAP_TOLS):
            ax = axes[row, col] if n_rows > 1 else axes[col]
            try:
                res = analyze_radial(mask, gap_tol_deg=gap)
            except Exception as e:
                ax.set_title(f"FAILED: {e}", fontsize=8)
                continue
            short = cell_path.stem.split("__")[0][:18] + f"…__{cell_path.stem.split('__')[-1]}"
            render_one(ax, mask, res, short, gap)

    out_path = OUT / "radial_proto_v2.png"
    fig.savefig(out_path, dpi=110)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
