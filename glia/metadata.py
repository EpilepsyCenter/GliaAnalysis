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

    lookup: dict[str, dict] = {}
    for row in image_metadata:
        img = row.get("image", "")
        if not img:
            continue
        stem = img.rsplit(".", 1)[0]
        lookup[stem] = {c: row.get(c, "") for c in user_cols}

    for c in user_cols:
        out[c] = image_stems.map(lambda s: lookup.get(s, {}).get(c, ""))
    return out


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
