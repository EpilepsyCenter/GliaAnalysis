"""Generate synthetic single-cell binary masks for testing feature extraction.

The synthetic shapes have known ground-truth properties (area, circularity,
span, branch count) so we can sanity-check the scikit-image FracLac replacement
without waiting for real microscopy data.

Four canonical shapes mirror the four microglia morphology classes:
  - ameboid: filled circle, no processes
  - hypertrophic: thicker oval blob, short stubby processes
  - rod-like: highly elongated ellipse, two opposite endpoints
  - ramified: small soma with multiple long thin branches
"""
from __future__ import annotations

import numpy as np


def ameboid(size: int = 256, radius: int = 40) -> np.ndarray:
    img = np.zeros((size, size), dtype=bool)
    cy, cx = size // 2, size // 2
    y, x = np.ogrid[:size, :size]
    img[(y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2] = True
    return img


def hypertrophic(size: int = 256) -> np.ndarray:
    img = np.zeros((size, size), dtype=bool)
    cy, cx = size // 2, size // 2
    y, x = np.ogrid[:size, :size]
    img[((y - cy) / 45) ** 2 + ((x - cx) / 60) ** 2 <= 1] = True
    # add four short stubs
    for dy, dx in [(-60, 0), (60, 0), (0, -75), (0, 75)]:
        img[max(0, cy + dy - 4):cy + dy + 4, max(0, cx + dx - 4):cx + dx + 4] = True
    return img


def rod_like(size: int = 256) -> np.ndarray:
    img = np.zeros((size, size), dtype=bool)
    cy, cx = size // 2, size // 2
    y, x = np.ogrid[:size, :size]
    img[((y - cy) / 80) ** 2 + ((x - cx) / 15) ** 2 <= 1] = True
    return img


def ramified(size: int = 256, n_branches: int = 6) -> np.ndarray:
    """Small filled soma plus n_branches radial line segments."""
    img = np.zeros((size, size), dtype=bool)
    cy, cx = size // 2, size // 2

    # soma
    y, x = np.ogrid[:size, :size]
    img[(y - cy) ** 2 + (x - cx) ** 2 <= 15 ** 2] = True

    # branches
    rng = np.random.default_rng(0)
    for k in range(n_branches):
        angle = 2 * np.pi * k / n_branches + rng.uniform(-0.2, 0.2)
        length = rng.integers(70, 100)
        thickness = 2
        for r in range(length):
            yy = int(cy + r * np.sin(angle))
            xx = int(cx + r * np.cos(angle))
            if 0 <= yy < size and 0 <= xx < size:
                img[max(0, yy - thickness):yy + thickness,
                    max(0, xx - thickness):xx + thickness] = True
    return img


SHAPES = {
    "ameboid": ameboid,
    "hypertrophic": hypertrophic,
    "rod_like": rod_like,
    "ramified": ramified,
}
