"""DAPI subset smoke test on ND2 files from the RAM Iba1 GFAP project.

Steps for each chosen stack:
  1. Load the DAPI channel (channel 0, max-Z, uint8).
  2. Apply a single global Otsu threshold on the full DAPI image — this
     is the new contract: faint signal can't masquerade as a nucleus in
     a crop because there's no per-crop thresholding involved.
  3. Crop the DAPI BINARY at each cell's bbox, write
     ``<cell_id>__dapi.tif`` into a temp dir.
  4. Render an overlay grid for the 6 largest cells: DAPI binary + cell
     outline + EDT-peak and DAPI-centroid markers; EDT-centered soma;
     DAPI-centered soma. Side-by-side so the centering effect is
     visible.

Tests both a 60X stack (small cells, dense Iba1 signal) and a 20X stack
(wider field, larger cells often touching) so we see how the global
threshold behaves across magnifications.

Run from repo root:
    PYTHONPATH=. .venv/bin/python scripts/dapi_subset_test.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.patches import Circle
from skimage import measure
from skimage.filters import threshold_otsu
from skimage.segmentation import find_boundaries

from glia.io import load_image_2d_uint8
from glia.radial import analyze_radial, dapi_centroid

PROJECT = Path("sample_images/RAM Iba1 GFAP")
THRESH_DIR = PROJECT / "_gliaanalysis" / "ThresholdedImages"
CELLS_DIR = PROJECT / "_gliaanalysis" / "SingleCells"
OUT = Path("scripts/_radial_proto_out")
OUT.mkdir(parents=True, exist_ok=True)

# One 60X and one 20X stack to span magnifications.
TARGETS = [
    PROJECT / "M13CTRL_gGFP_rGFAP_frIba1_60X_zoomstack_2.nd2",
    PROJECT / "M6_gGFP_rGFAP_frIba1_20X_zoomstack_1.nd2",
]


def process_one(nd2_path: Path, td: Path, n_panels: int = 6):
    stem = nd2_path.stem
    thresh_tif = THRESH_DIR / f"{stem}.tif_thresholded.tif"
    if not thresh_tif.is_file():
        print(f"  SKIP — no thresholded image at {thresh_tif}")
        return None

    print(f"\n=== {nd2_path.name} ===")

    # 1. Load full DAPI image.
    dapi_arr, _ = load_image_2d_uint8(
        str(nd2_path), channel=0, z_projection="max",
    )
    print(f"  DAPI full image: shape={dapi_arr.shape} "
          f"dtype={dapi_arr.dtype} min={dapi_arr.min()} "
          f"max={dapi_arr.max()} mean={dapi_arr.mean():.1f}")

    # 2. Global Otsu threshold.
    try:
        t = float(threshold_otsu(dapi_arr))
    except Exception:
        print("  Otsu failed — skipping this stack")
        return None
    dapi_binary = (dapi_arr > t).astype(np.uint8) * 255
    pct = float((dapi_binary > 0).mean()) * 100
    print(f"  Global Otsu = {t:.1f} → {pct:.1f}% of pixels positive")

    # 3. Re-derive each cell's bbox by relabeling the thresholded image.
    thresh = tifffile.imread(thresh_tif) > 0
    labels = measure.label(thresh, connectivity=1)
    existing = sorted([p for p in CELLS_DIR.glob(f"{stem}__*.tif")
                       if not p.stem.endswith("__dapi")])
    print(f"  Single cells: {len(existing)}")

    out_cells = td / f"out_{stem}"
    out_cells.mkdir()
    written = 0
    for cell_path in existing:
        parts = cell_path.stem.split("__")
        if len(parts) < 3:
            continue
        try:
            lbl = int(parts[-1])
        except ValueError:
            continue
        ys, xs = np.where(labels == lbl)
        if len(ys) == 0:
            continue
        r0, c0 = int(ys.min()), int(xs.min())
        r1, c1 = int(ys.max()) + 1, int(xs.max()) + 1
        tifffile.imwrite(
            out_cells / f"{cell_path.stem}__dapi.tif",
            dapi_binary[r0:r1, c0:c1],
        )
        written += 1
    print(f"  Wrote {written} DAPI binary siblings")

    # 4. Pick n largest cells for the overlay grid.
    sized: list[tuple[Path, int]] = []
    for cell_path in existing:
        m = tifffile.imread(cell_path) > 0
        sized.append((cell_path, int(m.sum())))
    sized.sort(key=lambda t: t[1], reverse=True)
    picks = [p for p, _ in sized[:n_panels]]

    fig, axes = plt.subplots(n_panels, 3, figsize=(13, 4 * n_panels),
                             constrained_layout=True)
    if n_panels == 1:
        axes = np.array([axes])

    dapi_used = 0
    for row, cell_path in enumerate(picks):
        mask = tifffile.imread(cell_path) > 0
        dapi_sibling = out_cells / f"{cell_path.stem}__dapi.tif"
        dapi_crop = (tifffile.imread(dapi_sibling)
                     if dapi_sibling.is_file() else None)

        res_edt = analyze_radial(mask, gap_tol_deg=20.0)

        dc = (dapi_centroid(dapi_crop, cell_mask=mask)
              if dapi_crop is not None else None)
        if dc is not None:
            dapi_used += 1
        res_dapi = (analyze_radial(mask, gap_tol_deg=20.0, center_yx=dc)
                    if dc is not None else None)

        # Panel 1: DAPI binary + cell outline + both candidate centers
        ax = axes[row, 0]
        if dapi_crop is not None:
            ax.imshow(dapi_crop > 0, cmap="Blues", interpolation="nearest")
        edges = find_boundaries(mask, mode="outer")
        ax.imshow(np.ma.masked_where(~edges, edges),
                  cmap="Greys_r", alpha=0.9, interpolation="nearest")
        ax.plot([res_edt.center_yx[1]], [res_edt.center_yx[0]],
                "o", color="#3fb950", markersize=10, label="EDT peak")
        if dc is not None:
            ax.plot([dc[1]], [dc[0]], "x", color="#f85149",
                    markersize=12, mew=3, label="DAPI centroid")
        else:
            ax.text(0.5, 0.05, "no DAPI inside mask",
                    transform=ax.transAxes, ha="center",
                    color="#f85149", fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.7,
                              edgecolor="none"))
        ax.set_title(f"DAPI binary + cell outline\n{cell_path.stem}",
                     fontsize=8)
        ax.legend(loc="lower right", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")

        # Panel 2: EDT-centered soma
        ax = axes[row, 1]
        ax.imshow(mask, cmap="gray", interpolation="nearest")
        ax.imshow(np.ma.masked_where(~res_edt.soma_mask,
                                     res_edt.soma_mask),
                  cmap="cool", alpha=0.5)
        poly = res_edt.soma_polygon
        poly_closed = np.vstack([poly, poly[:1]])
        ax.plot(poly_closed[:, 1], poly_closed[:, 0],
                color="#ff7eb6", linewidth=1.8)
        ax.plot([res_edt.center_yx[1]], [res_edt.center_yx[0]],
                "o", color="#3fb950", markersize=8)
        ax.add_patch(Circle((res_edt.center_yx[1], res_edt.center_yx[0]),
                            res_edt.critical_radius, fill=False,
                            edgecolor="#f0e442", linestyle="--",
                            linewidth=1.0))
        ax.set_title(
            f"EDT-centered  "
            f"soma={res_edt.soma_area:.0f}px²  "
            f"ratio={res_edt.soma_to_cell_area_ratio:.2f}  "
            f"proc={res_edt.primary_process_count}",
            fontsize=8,
        )
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")

        # Panel 3: DAPI-centered soma (or fallback)
        ax = axes[row, 2]
        ax.imshow(mask, cmap="gray", interpolation="nearest")
        if res_dapi is not None:
            ax.imshow(np.ma.masked_where(~res_dapi.soma_mask,
                                         res_dapi.soma_mask),
                      cmap="cool", alpha=0.5)
            poly = res_dapi.soma_polygon
            poly_closed = np.vstack([poly, poly[:1]])
            ax.plot(poly_closed[:, 1], poly_closed[:, 0],
                    color="#ff7eb6", linewidth=1.8)
            ax.plot([res_dapi.center_yx[1]], [res_dapi.center_yx[0]],
                    "x", color="#f85149", markersize=12, mew=3)
            ax.add_patch(Circle((res_dapi.center_yx[1],
                                 res_dapi.center_yx[0]),
                                res_dapi.critical_radius, fill=False,
                                edgecolor="#f0e442", linestyle="--",
                                linewidth=1.0))
            title = (f"DAPI-centered  "
                     f"soma={res_dapi.soma_area:.0f}px²  "
                     f"ratio={res_dapi.soma_to_cell_area_ratio:.2f}  "
                     f"proc={res_dapi.primary_process_count}")
        else:
            title = "DAPI-centered  (fallback → EDT, no DAPI signal)"
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")

    print(f"  DAPI used for {dapi_used}/{n_panels} visualized cells "
          f"(rest fell back to EDT)")

    out_path = OUT / f"dapi_subset_{stem}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for nd2 in TARGETS:
            if not nd2.is_file():
                print(f"SKIP — missing {nd2}")
                continue
            process_one(nd2, td)


if __name__ == "__main__":
    main()
