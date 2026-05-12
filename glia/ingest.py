"""Filename metadata parsing and Areas.csv ingestion."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def parse_metadata_from_id(
    df: pd.DataFrame, fields: list[str], sep: str = "_", source_col: str = "ID"
) -> pd.DataFrame:
    """Split a filename-derived ID column into metadata columns.

    Example: ID = "CohortA_M03_LPS_Male_CA1_xyz" with fields=
    ["Cohort", "Animal", "Condition", "Sex", "Region"] (sep="_") yields five
    columns; remaining trailing tokens are kept in the original column.
    """
    parts = df[source_col].str.split(sep, n=len(fields), expand=True)
    out = df.copy()
    for i, name in enumerate(fields):
        out[name] = parts[i]
    return out


def load_areas_csv(path: str | Path) -> pd.DataFrame:
    """Load FIJI's Areas.csv (per-image ROI area) for density calculations."""
    raw = pd.read_csv(path)
    return raw.rename(columns={"Label": "Image"})


def compute_cell_density(
    features_df: pd.DataFrame, areas_df: pd.DataFrame, image_col: str
) -> pd.DataFrame:
    """Cells per unit area, per image."""
    counts = features_df.groupby(image_col).size().reset_index(name="num_cells")
    merged = counts.merge(areas_df, left_on=image_col, right_on="Image")
    merged["density"] = merged["num_cells"] / merged["Area"]
    return merged
