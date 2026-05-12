"""Constants shared across the pipeline."""
from __future__ import annotations

# The 18 geometric features replicated from FracLac's "Hull and Circle Results"
# output, computed in Python via scikit-image + scipy + custom min-enclosing-circle.
FRACLAC_FEATURES = [
    "Foreground pixels",
    "Density of foreground pixels in hull area",
    "Span ratio of hull (major/minor axis)",
    "Maximum span across hull",
    "Area",
    "Perimeter",
    "Circularity",
    "Width of bounding rectangle",
    "Height of bounding rectangle",
    "Maximum radius from hull's center of mass",
    "Max/min radii from hull's center of mass",
    "Relative variation (CV) in radii from hull's center of mass",
    "Mean radius",
    "Diameter of bounding circle",
    "Maximum radius from circle's center of mass",
    "Max/min radii from circle's center of mass",
    "Relative variation (CV) in radii from circle's center of mass",
    "Mean radius from circle's center of mass",
]

# The 9 features from AnalyzeSkeleton (FIJI)
SKELETON_FEATURES = [
    "# of branches",
    "# of junctions",
    "# of end point voxels",
    "# of junction voxels",
    "# of slab voxels",
    "Average branch length",
    "# of triple points",
    "# of quadruple points",
    "Maximum branch length",
]

ALL_FEATURES = FRACLAC_FEATURES + SKELETON_FEATURES  # 27 total

# Diagnostic features used by the auto cluster labeler. Names must match ALL_FEATURES.
DIAGNOSTIC_FEATURES = [
    "Circularity",
    "Area",
    "# of branches",
    "# of end point voxels",
    "Span ratio of hull (major/minor axis)",
    "Maximum branch length",
]

# Morphology label templates: z-score profile of cluster mean on DIAGNOSTIC_FEATURES.
# Values are relative directional weights, not absolute z-scores. Normalized by L2.
MORPHOLOGY_TEMPLATES = {
    "Ameboid":      {"Circularity": +2, "Area": -1, "# of branches": -2,
                     "# of end point voxels": -2,
                     "Span ratio of hull (major/minor axis)": -1,
                     "Maximum branch length": -2},
    "Hypertrophic": {"Circularity":  0, "Area": +2, "# of branches": -1,
                     "# of end point voxels": -1,
                     "Span ratio of hull (major/minor axis)":  0,
                     "Maximum branch length": -1},
    "Rod-like":     {"Circularity": -1, "Area":  0, "# of branches": -1,
                     "# of end point voxels": -2,
                     "Span ratio of hull (major/minor axis)": +2,
                     "Maximum branch length":  0},
    "Ramified":     {"Circularity": -1, "Area":  0, "# of branches": +2,
                     "# of end point voxels": +2,
                     "Span ratio of hull (major/minor axis)":  0,
                     "Maximum branch length": +2},
}

DEFAULT_THRESHOLD_METHODS_GLOBAL = [
    "Huang", "Huang2", "Intermodes", "IsoData", "Li", "MaxEntropy", "Mean",
    "MinError(I)", "Minimum", "Moments", "Otsu", "Percentile", "RenyiEntropy",
    "Shanbhag", "Triangle", "Yen", "Manual",
]
DEFAULT_THRESHOLD_METHODS_LOCAL = [
    "Bernsen", "Contrast", "Mean", "Median", "MidGrey", "Niblack", "Otsu",
    "Phansalkar", "Sauvola",
]
