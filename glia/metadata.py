"""Per-cell metadata join — Prepare-tab image_metadata → features_df.

The Setup → Prepare tab is the single point where per-image metadata
(Animal, Genotype, Treatment, …) is set. Everything downstream just
joins those columns onto every cell row via the image_stem encoded in
the cell ID (``<image_stem>__<roi_tag>__<N>``). No dedicated "Metadata"
tab any more — this function is called automatically right after
Features extracts the cell table.
"""

from __future__ import annotations

import pandas as pd


# Columns that come from the Prepare table but live in the row dict for
# reasons other than being a downstream metadata column.
_RESERVED_META_COLS = {"image", "channel", "channel_name",
                       "z_projection", "exclude"}


def join_project_image_metadata(
    df: pd.DataFrame, image_metadata: list[dict],
) -> pd.DataFrame:
    """Broadcast Prepare-tab per-image metadata onto every cell row.

    ``df`` has cell IDs like ``<image_stem>__<roi_tag>__<N>``. Each row
    in ``image_metadata`` keys on the source filename; we match its stem
    to each cell's image_stem and copy every user-defined column.
    """
    if df is None or len(df) == 0:
        return df

    # Astrocyte rows carry image_stem / roi_tag directly and don't
    # have a cell-level ``ID`` column; their metadata is joined by
    # the astrocyte page itself. Bail out so the per-cell parser
    # below doesn't KeyError on the missing column.
    if "ID" not in df.columns:
        return df

    user_cols = sorted({
        k for row in (image_metadata or []) for k, v in row.items()
        if k not in _RESERVED_META_COLS
        and not str(k).startswith("_")
        and isinstance(v, (str, int, float, bool, type(None)))
    })

    parts = df["ID"].str.split("__", n=2, expand=True)
    image_stems = parts[0]
    roi_tags = parts[1] if parts.shape[1] > 1 else ""
    cell_indices = parts[2] if parts.shape[1] > 2 else ""

    out = df.copy()
    out["roi_tag"] = roi_tags
    out["cell_index"] = cell_indices

    if not image_metadata or not user_cols:
        return out

    from glia.prepare import clean_stem
    lookup: dict[str, dict] = {}
    for row in image_metadata:
        img = row.get("image", "")
        if not img:
            continue
        # Same stem rule as the prepared-file naming, so an OME-TIFF
        # source (image="foo.ome.tif") keys on "foo" — matching the
        # cell-ID image_stem derived from the plain "foo.tif" prepared
        # file. Plain ``.tif``/``.czi``/… sources are unaffected.
        stem = clean_stem(img)
        lookup[stem] = {c: row.get(c, "") for c in user_cols}

    for c in user_cols:
        out[c] = image_stems.map(lambda s: lookup.get(s, {}).get(c, ""))
    return out


def available_roi_tags(df: pd.DataFrame) -> list[str]:
    """Sorted unique ``roi_tag`` values present in ``df`` (empty if none).

    Used to populate the analysis tabs' "ROIs to include" selector.
    """
    if df is None or "roi_tag" not in getattr(df, "columns", []):
        return []
    return sorted(str(v) for v in pd.Series(df["roi_tag"]).dropna().unique())


def filter_by_roi_tags(
    df: pd.DataFrame, include,
) -> tuple[pd.DataFrame, bool]:
    """Keep only rows whose ``roi_tag`` is in ``include``.

    Returns ``(filtered_df, applied)``. The filter is a deliberate no-op
    — ``applied=False`` — when there's no ``roi_tag`` column, when
    ``include`` is empty/None, or when ``include`` already covers every
    tag present (so "all selected" and "future ROIs" both mean analyse
    everything). Non-destructive: callers pass a working copy; the
    canonical ``state.features_df`` is never shrunk by this.
    """
    if df is None or "roi_tag" not in getattr(df, "columns", []):
        return df, False
    inc = {str(t) for t in (include or []) if str(t) != ""}
    if not inc:
        return df, False
    present = {str(v) for v in pd.Series(df["roi_tag"]).dropna().unique()}
    if inc >= present:
        return df, False
    return df[df["roi_tag"].astype(str).isin(inc)], True


def ensure_metadata_joined(state) -> None:
    """Refresh the per-image metadata join on ``state.features_df``.

    Safe to call from any view's layout — it's idempotent and a no-op
    when there's no features dataframe or no metadata table. Keeps
    Explore/Cluster/Stats in sync with whatever the user last typed
    into the Prepare-tab metadata table.
    """
    if state is None or getattr(state, "features_df", None) is None:
        return
    from glia.settings import get_image_metadata
    try:
        meta = get_image_metadata(getattr(state, "project_dir", ""))
        if not meta:
            return
        state.features_df = join_project_image_metadata(state.features_df, meta)
    except Exception:
        pass
