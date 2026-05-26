"""Builders for the Explore raw-data and Stats-overview exports.

Pure pandas / numpy — no Dash imports — so they're unit-testable and the
page callbacks just stream the bytes via ``dcc.send_bytes``.

* :func:`build_explore_prism_sheets` — one sheet per feature laid out so a
  Prism Column/Grouped table can be pasted straight in. Replicates the
  Explore distribution plot's aggregation (per-animal means when a split +
  Animal column are set, else per row) so the exported numbers reproduce
  the bars exactly.
* :func:`stats_results_to_sheets` — turns the captured Stats run (group
  sizes, feature tests, posthocs, LMM tables, …) into one sheet each.
* :func:`dataframes_to_xlsx_bytes` — writes a ``{sheet_name: df}`` mapping to
  a single .xlsx (openpyxl), with Excel-safe, de-duplicated sheet names.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd


_MISSING = "(missing)"
_INVALID_SHEET = re.compile(r"[\[\]:\*\?/\\]")


def _levels(df: pd.DataFrame, col: str) -> list:
    """Sorted unique levels of ``col`` (NaN → "(missing)"), as in the plot."""
    return sorted(pd.Series(df[col]).fillna(_MISSING).unique(), key=str)


def build_explore_prism_sheets(
    df: pd.DataFrame,
    features: list[str],
    group: str = "",
    split: str = "",
    animal_col: str = "",
) -> dict[str, pd.DataFrame]:
    """Per-feature, Prism-ready wide tables.

    ``df`` is assumed already inf-cleaned and transformed exactly as the
    Explore tab displays it. Returns ``{"info": …, "<feature>": …}`` where
    each feature sheet has one column per (group × split) combination and
    one row per replicate (ragged columns are NaN-padded).
    """
    group = group or ""
    split = split or ""
    has_group = bool(group) and group in df.columns
    has_split = (bool(split) and split in df.columns
                 and split != group and split != animal_col)
    use_agg = (has_split and bool(animal_col) and animal_col in df.columns
               and animal_col not in (group, split))

    unit = ("per-animal means" if use_agg
            else "per row (cells / image-ROIs)")
    sheets: dict[str, pd.DataFrame] = {
        "info": pd.DataFrame({
            "field": ["group (bar color)", "split (x-axis)",
                      "replicate unit", "features"],
            "value": [group or "(none)", split or "(none)", unit,
                      ", ".join(features)],
        })
    }

    feats = [f for f in (features or []) if f in df.columns]
    for feat in feats:
        if use_agg:
            keys = [animal_col]
            if has_group:
                keys.append(group)
            keys.append(split)
            frame = (df[[*keys, feat]].dropna(subset=[feat])
                     .groupby(keys, dropna=False)[feat]
                     .mean().reset_index())
        else:
            frame = df

        g_levels = _levels(frame, group) if has_group else [None]
        s_levels = _levels(frame, split) if has_split else [None]

        cols: dict[str, np.ndarray] = {}
        for g in g_levels:
            for s in s_levels:
                sub = frame
                if has_group:
                    sub = sub[sub[group].fillna(_MISSING) == g]
                if has_split:
                    sub = sub[sub[split].fillna(_MISSING) == s]
                vals = sub[feat].to_numpy(dtype="float64")
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue
                label = " · ".join(
                    str(x) for x in (g if has_group else None,
                                     s if has_split else None)
                    if x is not None) or "all"
                cols[label] = vals

        maxlen = max((len(v) for v in cols.values()), default=0)
        sheets[feat] = pd.DataFrame({
            k: np.concatenate([v, np.full(maxlen - len(v), np.nan)])
            for k, v in cols.items()
        })
    return sheets


def stats_results_to_sheets(results: dict) -> dict[str, pd.DataFrame]:
    """Convert a captured Stats run into ``{sheet_name: df}``.

    ``results`` is ``{"meta": {...}, "tables": {name: DataFrame}}`` as
    stashed by the Stats ``on_run`` callback. The meta dict becomes a
    two-column "overview" sheet; each non-empty table becomes its own.
    """
    sheets: dict[str, pd.DataFrame] = {}
    meta = (results or {}).get("meta", {}) or {}
    if meta:
        sheets["overview"] = pd.DataFrame(
            {"field": list(meta.keys()),
             "value": [str(v) for v in meta.values()]})
    for name, d in ((results or {}).get("tables", {}) or {}).items():
        if isinstance(d, pd.DataFrame) and not d.empty:
            sheets[name] = d.copy()
    return sheets


def _safe_sheet_name(name: str, used: set[str]) -> str:
    """Excel sheet name: ≤31 chars, no []:*?/\\, unique within the book."""
    clean = _INVALID_SHEET.sub("-", str(name)).strip() or "sheet"
    clean = clean[:31]
    base, n = clean, 1
    while clean.lower() in used:
        suffix = f"~{n}"
        clean = base[:31 - len(suffix)] + suffix
        n += 1
    used.add(clean.lower())
    return clean


def dataframes_to_xlsx_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Write ``{sheet_name: df}`` to a single .xlsx and return the bytes."""
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        used: set[str] = set()
        wrote_any = False
        for name, d in (sheets or {}).items():
            frame = d if isinstance(d, pd.DataFrame) else pd.DataFrame()
            frame.to_excel(xw, sheet_name=_safe_sheet_name(name, used),
                           index=False)
            wrote_any = True
        if not wrote_any:
            pd.DataFrame().to_excel(xw, sheet_name="empty", index=False)
    return bio.getvalue()
