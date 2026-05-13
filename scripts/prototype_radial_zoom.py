"""Higher-detail render of the two cleanest microglia from the prototype.

Used to visually confirm the soma polygon traces the actual cell body
shape (not a circle), the r_out fan is sensible, and the Sholl profile
matches the visible process count.
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

# The two cells from the v2 grid that looked like intact microglia.
CELLS = [
    "M6_gGFP_rGFAP_frIba1_20X_zoomstack_1__all__282.tif",
    "M13CTRL_gGFP_rGFAP_frIba1_60X_zoomstack_2__all__332.tif",
]


def main():
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[2.4, 1.0, 1.4])

    for row, name in enumerate(CELLS):
        path = SINGLECELLS / name
        mask = tifffile.imread(path) > 0
        res = analyze_radial(mask, gap_tol_deg=20.0)

        # Big mask + overlay panel
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(mask, cmap="gray", interpolation="nearest")
        ax.imshow(np.ma.masked_where(~res.soma_mask, res.soma_mask),
                  cmap="cool", alpha=0.45, interpolation="nearest")

        cy, cx = res.center_yx
        ang_rad = np.radians(res.angles_deg)
        end_y = cy + res.r_out * np.sin(ang_rad)
        end_x = cx + res.r_out * np.cos(ang_rad)
        for i in range(0, len(ang_rad), 4):
            c = "#d29922" if res.process_angle_mask[i] else "#3fb950"
            ax.plot([cx, end_x[i]], [cy, end_y[i]],
                    color=c, alpha=0.35, linewidth=0.6)

        poly = res.soma_polygon
        poly_closed = np.vstack([poly, poly[:1]])
        ax.plot(poly_closed[:, 1], poly_closed[:, 0],
                color="#ff7eb6", linewidth=2.0)
        ax.add_patch(Circle((cx, cy), res.critical_radius,
                            fill=False, edgecolor="#f0e442",
                            linestyle="--", linewidth=1.4))
        ax.plot([cx], [cy], "o", color="#f85149", markersize=8)

        h, w = mask.shape
        ax.set_xlim(0, w); ax.set_ylim(h, 0)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name, fontsize=9)

        # Polar r_out(θ)
        axp = fig.add_subplot(gs[row, 1], projection="polar")
        axp.plot(np.radians(res.angles_deg), res.r_out,
                 color="#58a6ff", linewidth=1.2)
        pa = res.process_angle_mask
        if pa.any():
            axp.scatter(np.radians(res.angles_deg[pa]), res.r_out[pa],
                        color="#d29922", s=6)
        axp.axhline(res.critical_radius, color="#f0e442",
                    linestyle="--", linewidth=0.8)
        axp.set_theta_zero_location("E")
        axp.set_title(f"r_out(θ) — r₀={res.critical_radius:.0f}px",
                      fontsize=8)
        axp.tick_params(labelsize=7)

        # Sholl profile
        axs = fig.add_subplot(gs[row, 2])
        axs.plot(res.sholl_radii, res.sholl_intersections,
                 color="#58a6ff", linewidth=1.4)
        axs.axvline(res.critical_radius, color="#f0e442",
                    linestyle="--", linewidth=1.0)
        axs.set_xlabel("r (px)", fontsize=9)
        axs.set_ylabel("# intersections", fontsize=9)
        axs.set_title(
            f"Sholl — primary={res.primary_process_count}, "
            f"max int={res.max_intersections}, "
            f"RI={res.ramification_index:.1f}\n"
            f"soma area={res.soma_area:.0f}, "
            f"soma/cell={res.soma_to_cell_area_ratio:.2f}, "
            f"soma circ={res.soma_circularity:.2f}",
            fontsize=8,
        )
        axs.tick_params(labelsize=8)

    out = OUT / "radial_proto_zoom.png"
    fig.savefig(out, dpi=140)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
