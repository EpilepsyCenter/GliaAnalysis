"""Sanity tests for geometric feature extraction against synthetic shapes.

These don't validate against FracLac — that needs real images and a side-by-side
notebook. They validate that compute_geometric_features returns sensible values
on shapes with known properties (e.g. circularity of a filled disc ≈ 1).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tests.synth import ameboid, hypertrophic, ramified, rod_like


def test_ameboid_is_circular():
    from glia.features import compute_geometric_features
    mask = ameboid()
    f = compute_geometric_features(mask)
    assert f.circularity > 0.85, f"Expected near-circular ameboid; got {f.circularity}"
    assert f.span_ratio_hull < 1.15, f"Ameboid should be ~symmetric"


def test_rod_is_elongated():
    from glia.features import compute_geometric_features
    f = compute_geometric_features(rod_like())
    assert f.span_ratio_hull > 3.0, "Rod-like cell should have high span ratio"
    # The synthetic rod is a smooth 80:15 ellipse — its true 4πA/P² is ~0.4.
    # Compare against ameboid (~0.95) rather than expecting a sub-0.3 value.
    assert f.circularity < 0.5, "Rod-like cell should be much less round than ameboid"


def test_ramified_has_larger_hull_than_soma():
    from glia.features import compute_geometric_features
    f = compute_geometric_features(ramified())
    soma_area = math.pi * 15 ** 2
    # ramified shape's convex hull (the area within the hull) should be >>> soma area
    # density_in_hull = foreground / hull_area, so should be small (lots of empty hull)
    assert f.density_in_hull < 0.4, (
        f"Ramified cell hull should be sparsely filled; got density {f.density_in_hull}"
    )


def test_hypertrophic_area_exceeds_ameboid():
    from glia.features import compute_geometric_features
    f_amo = compute_geometric_features(ameboid())
    f_hyp = compute_geometric_features(hypertrophic())
    assert f_hyp.area > f_amo.area, "Hypertrophic cell should be larger than ameboid"
