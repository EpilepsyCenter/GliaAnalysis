"""Feature-column transforms for normalization."""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_transform(df: pd.DataFrame, cols: list[str], offset: float = 0.1) -> pd.DataFrame:
    out = df.copy()
    out[cols] = np.log(out[cols].clip(lower=0) + offset)
    return out


def zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out[cols] = (out[cols] - out[cols].mean()) / out[cols].std(ddof=0)
    return out


def minmax(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    lo = out[cols].min()
    hi = out[cols].max()
    out[cols] = (out[cols] - lo) / (hi - lo)
    return out


def apply_transform(df: pd.DataFrame, cols: list[str], kind: str) -> pd.DataFrame:
    if kind == "none":
        return df
    if kind == "log":
        return log_transform(df, cols)
    if kind == "zscore":
        return zscore(df, cols)
    if kind == "minmax":
        return minmax(df, cols)
    raise ValueError(f"Unknown transform: {kind}")
