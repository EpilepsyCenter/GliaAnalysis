"""End-to-end smoke test of the astrocyte metrics on a real GFAP image.

Pulls one ND2 from the RAM Iba1 GFAP project, loads its GFAP channel
(Cy3.5, channel 2), Otsu-thresholds in Python (skipping FIJI),
computes astrocyte metrics with and without a synthetic ROI split,
and renders an overlay PNG so the result can be eyeballed.

Run from repo root:
    PYTHONPATH=. .venv/bin/python scripts/test_astrocyte.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.filters import threshold_otsu

from glia.astrocyte import compute_astrocyte_metrics, metrics_to_row
from glia.io import load_image_2d_uint8

PROJECT = Path("sample_images/RAM Iba1 GFAP")
TARGETS = [
    PROJECT / "M13CTRL_gGFP_rGFAP_frIba1_60X_zoomstack_2.nd2",
    PROJECT / "M6_gGFP_rGFAP_frIba1_20X_zoomstack_1.nd2",
]
OUT = Path("scripts/_radial_proto_out")
OUT.mkdir(parents=True, exist_ok=True)

# Channel index for GFAP in this project's ND2 layout
# (DAPI=0, GFP=1, Cy3.5=GFAP=2, Alx647=Iba1=3).
GFAP_CHANNEL = 2


def process_one(nd2_path: Path):
    print(f"\n=== {nd2_path.name} ===")
    arr, info = load_image_2d_uint8(
        str(nd2_path), channel=GFAP_CHANNEL, z_projection="max",
    )
    print(f"GFAP loaded: shape={arr.shape} dtype={arr.dtype} "
          f"min={arr.min()} max={arr.max()} mean={arr.mean():.1f}")

    t = float(threshold_otsu(arr))
    binary = arr > t
    pct = float(binary.mean()) * 100
    print(f"Otsu threshold: {t:.1f} → {pct:.1f}% positive pixels")

    # Whole-image metrics (the "all" ROI).
    m_all = compute_astrocyte_metrics(binary, intensity_image=arr)
    print("\nWhole-image metrics:")
    for k, v in metrics_to_row(m_all).items():
        if isinstance(v, float):
            print(f"  {k:32s} = {v:,.3f}")
        else:
            print(f"  {k:32s} = {v}")

    # Synthetic split: left vs right half — emulates the per-ROI case.
    h, w = binary.shape
    left_roi = np.zeros_like(binary, dtype=bool)
    left_roi[:, : w // 2] = True
    right_roi = ~left_roi
    m_left = compute_astrocyte_metrics(
        binary, intensity_image=arr, roi_mask=left_roi,
    )
    m_right = compute_astrocyte_metrics(
        binary, intensity_image=arr, roi_mask=right_roi,
    )
    print("\nPer-ROI metrics (synthetic left/right split):")
    print(f"  LEFT  area_frac={m_left.gfap_area_fraction:.4f} "
          f"branches={m_left.n_branches} "
          f"skel_len={m_left.skeleton_length_total_px:.0f}")
    print(f"  RIGHT area_frac={m_right.gfap_area_fraction:.4f} "
          f"branches={m_right.n_branches} "
          f"skel_len={m_right.skeleton_length_total_px:.0f}")

    # Render overlay for visual sanity check.
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    axes[0].imshow(arr, cmap="gray")
    axes[0].set_title(f"GFAP raw (Otsu t={t:.0f})", fontsize=10)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].imshow(binary, cmap="gray")
    axes[1].set_title(f"GFAP binary  · area_frac={m_all.gfap_area_fraction:.3f}",
                      fontsize=10)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    from skimage.morphology import skeletonize
    skel = skeletonize(binary)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    base = arr.astype(np.float32)
    base = (base / max(base.max(), 1) * 200).clip(0, 255).astype(np.uint8)
    rgb[..., 0] = base // 2
    rgb[..., 1] = base // 2
    rgb[..., 2] = base // 2
    rgb[skel, 0] = 255
    rgb[skel, 1] = 110
    rgb[skel, 2] = 199
    axes[2].imshow(rgb)
    axes[2].set_title(
        f"Skeleton overlay  · branches={m_all.n_branches} "
        f"junctions={m_all.n_junctions} "
        f"len={m_all.skeleton_length_total_px:.0f}",
        fontsize=10,
    )
    axes[2].set_xticks([]); axes[2].set_yticks([])

    out_path = OUT / f"astrocyte_{nd2_path.stem}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved overlay to {out_path}")


def main():
    for t in TARGETS:
        if not t.is_file():
            print(f"SKIP missing {t}")
            continue
        try:
            process_one(t)
        except Exception as e:
            print(f"FAILED on {t.name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
