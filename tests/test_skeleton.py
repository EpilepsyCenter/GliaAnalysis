"""Tests for FIJI AnalyzeSkeleton CSV ingestion and merging."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from glia.config import ALL_FEATURES, SKELETON_FEATURES
from glia.features import (
    load_skeleton_results,
    merge_geometric_and_skeleton,
)


_FIJI_HEADER = (
    "# Branches,# Junctions,# End-point voxels,# Junction voxels,"
    "# Slab voxels,Average Branch Length,# Triple points,"
    "# Quadruple points,Maximum Branch Length"
)


def _write_csv(path: Path, *rows: str) -> None:
    path.write_text(_FIJI_HEADER + "\n" + "\n".join(rows) + "\n")


def test_load_skeleton_results_single_row(tmp_path: Path):
    _write_csv(tmp_path / "cellA_results.csv",
               "12,5,7,15,200,8.4,3,1,25.1")
    _write_csv(tmp_path / "cellB_results.csv",
               "8,3,5,9,150,6.2,2,0,18.3")
    df = load_skeleton_results(tmp_path)
    assert list(df.columns) == ["ID"] + SKELETON_FEATURES
    assert set(df["ID"]) == {"cellA", "cellB"}
    a = df.set_index("ID").loc["cellA"]
    assert a["# of branches"] == 12
    assert a["# of junctions"] == 5
    assert a["Maximum branch length"] == 25.1


def test_load_skeleton_results_skips_multirow(tmp_path: Path):
    _write_csv(tmp_path / "good_results.csv",
               "10,4,6,12,180,7.5,2,1,22.0")
    _write_csv(tmp_path / "fragmented_results.csv",
               "5,2,3,6,90,4.0,1,0,12.0",
               "3,1,2,3,40,2.0,0,0,8.0")
    df = load_skeleton_results(tmp_path)
    assert list(df["ID"]) == ["good"]


def test_merge_geometric_and_skeleton(tmp_path: Path):
    _write_csv(tmp_path / "cell1_results.csv",
               "9,3,4,8,120,6.8,1,0,17.4")
    skel = load_skeleton_results(tmp_path)
    geom = pd.DataFrame([
        {"ID": "cell1", "Area": 1500, "Circularity": 0.62},
        {"ID": "orphan", "Area": 800, "Circularity": 0.91},
    ])
    merged = merge_geometric_and_skeleton(geom, skel)
    assert list(merged["ID"]) == ["cell1"]   # inner join drops orphan
    assert merged.loc[0, "# of branches"] == 9
    assert merged.loc[0, "Area"] == 1500


def test_all_features_count():
    assert len(ALL_FEATURES) == 27
