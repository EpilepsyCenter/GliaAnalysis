"""Per-project + per-user settings persistence.

Two scopes:

* **Project** — `<project>/.gliaanalysis_settings.json`. Captures analysis
  parameters tied to *this dataset* (threshold method/band, area bounds,
  metadata field names, cluster k, stats factor selection). Travels with
  the folder so reopening it restores the analysis state.

* **User** — `~/.gliaanalysis/defaults.json`. Captures preferences tied to
  *this machine / user* (FIJI executable path). Survives across projects.

Both files are best-effort: a write or read failure should never break the
UI. Callers swallow errors.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


# Single process-wide lock guarding read-modify-write of the project
# settings JSON. Dash callbacks run in a thread pool, so two writers
# (e.g. on_table_edit committing a tag edit and a slider mirror writing
# the gap tolerance) can both load the same baseline, modify different
# keys, and the later writer overwrites the earlier one. Serializing
# the read-modify-write step removes that race.
_SETTINGS_FILE_LOCK = threading.Lock()


PROJECT_SETTINGS_FILENAME = ".gliaanalysis_settings.json"
USER_SETTINGS_DIR = Path.home() / ".gliaanalysis"
USER_SETTINGS_PATH = USER_SETTINGS_DIR / "defaults.json"

# State fields that live on SessionState and belong in the project file.
_PROJECT_FIELDS = (
    "threshold_method",
    "threshold_kind",
    "threshold_lower",
    "threshold_upper",
    "local_radius",
    "preprocess",
    "area_min",
    "area_max",
    "metadata_cols",
    "metadata_sep",
    "k",
    "pca_n_components",
    "cluster_labels",
    "transform",
    "animal_id_col",
    "factor_cols",
    "soma_gap_tol_deg",
    "use_dapi",
)

# Extra fields persisted from SessionState.extra. These don't live as
# typed attributes on SessionState but we still want them to round-trip.
_PROJECT_EXTRA_FIELDS = (
    "inflammation_model",
    # Explore tab selections
    "explore_features",
    "explore_group",
    "explore_split",
    "explore_show_points",
    # Stats tab selections (animal_id_col + factor_cols already on state)
    "stats_features",
    "stats_method",
    "stats_padjust",
    "stats_aggregate",
)

# State fields that live on SessionState and belong in the user file.
_USER_FIELDS = ("fiji_path",)


# ── Project settings ────────────────────────────────────────────────


def _state_to_project_dict(state) -> dict:
    out: dict[str, Any] = {"version": 1}
    for name in _PROJECT_FIELDS:
        if hasattr(state, name):
            out[name] = getattr(state, name)
    # cluster_labels keys may be ints; JSON keys must be strings.
    cl = out.get("cluster_labels")
    if isinstance(cl, dict):
        out["cluster_labels"] = {str(k): v for k, v in cl.items()}
    # Pull selected extra-store keys into the JSON so they survive reloads.
    extra = getattr(state, "extra", None) or {}
    for name in _PROJECT_EXTRA_FIELDS:
        if name in extra:
            out[name] = extra[name]
    return out


def save_project_settings(project_dir: str, state) -> str | None:
    """Write a snapshot of the project-scoped state fields to JSON.

    Merges into the existing JSON rather than overwriting it, so keys
    that don't live on ``SessionState`` (notably ``image_metadata``,
    updated by a sibling write path) survive every slider tweak.

    The whole read-modify-write block is serialized under
    ``_SETTINGS_FILE_LOCK`` so a concurrent ``set_image_metadata``
    write can't race past us between load and write.
    """
    if not project_dir or not Path(project_dir).is_dir():
        return None
    path = Path(project_dir) / PROJECT_SETTINGS_FILENAME
    with _SETTINGS_FILE_LOCK:
        try:
            existing = load_project_settings(project_dir)
        except Exception:
            existing = {}
        merged = {**existing, **_state_to_project_dict(state)}
        try:
            path.write_text(json.dumps(merged, indent=2))
        except Exception:
            return None
    return str(path)


def load_project_settings(project_dir: str) -> dict:
    if not project_dir or not Path(project_dir).is_dir():
        return {}
    path = Path(project_dir) / PROJECT_SETTINGS_FILENAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text()) or {}
    except Exception:
        return {}


def apply_project_settings(state, settings: dict) -> None:
    """Copy known project-scoped keys from ``settings`` onto ``state``."""
    if not settings:
        return
    for name in _PROJECT_FIELDS:
        if name not in settings:
            continue
        val = settings[name]
        if name == "cluster_labels" and isinstance(val, dict):
            val = {int(k): v for k, v in val.items()
                   if str(k).lstrip("-").isdigit()}
        try:
            setattr(state, name, val)
        except Exception:
            pass
    # Extra-store keys: drop them on state.extra so the Inflammation
    # page can find a previously trained model after a folder reopen.
    extra = getattr(state, "extra", None)
    if extra is not None:
        for name in _PROJECT_EXTRA_FIELDS:
            if name in settings:
                extra[name] = settings[name]


# ── User settings ───────────────────────────────────────────────────


def save_user_settings(state) -> str | None:
    """Persist user-scoped fields to ~/.gliaanalysis/defaults.json."""
    USER_SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"version": 1}
    for name in _USER_FIELDS:
        if hasattr(state, name):
            out[name] = getattr(state, name)
    try:
        USER_SETTINGS_PATH.write_text(json.dumps(out, indent=2))
    except Exception:
        return None
    return str(USER_SETTINGS_PATH)


def load_user_settings() -> dict:
    if not USER_SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(USER_SETTINGS_PATH.read_text()) or {}
    except Exception:
        return {}


def apply_user_settings(state, settings: dict) -> None:
    if not settings:
        return
    for name in _USER_FIELDS:
        if name in settings:
            try:
                setattr(state, name, settings[name])
            except Exception:
                pass


# ── Per-image metadata table ────────────────────────────────────────
#
# Lives inside the same .gliaanalysis_settings.json under the key
# ``image_metadata``. Schema is a list of rows; each row has at least an
# ``image`` (filename, basename only) key plus whatever user-defined
# columns. ``prepare_channel`` and ``prepare_z`` are optional per-image
# overrides for the Prepare step.


_IMAGE_METADATA_KEY = "image_metadata"


def get_image_metadata(project_dir: str) -> list[dict]:
    raw = list(load_project_settings(project_dir).get(
        _IMAGE_METADATA_KEY, []
    ))
    # Defensive: filter any private/internal keys that might have been
    # persisted by an older build (e.g. ``_channel_options``).
    return [
        {k: v for k, v in r.items() if not str(k).startswith("_")}
        for r in raw
    ]


def set_image_metadata(project_dir: str, rows: list[dict]) -> str | None:
    """Replace the project's image-metadata table on disk.

    Strips internal/private keys (anything beginning with ``_``) so that
    UI-only state — like the per-row dropdown options the Prepare table
    keeps under ``_channel_options`` — doesn't leak into the persisted
    metadata and then onto every cell's row in features_df.

    Serialized under ``_SETTINGS_FILE_LOCK`` so a concurrent
    ``save_project_settings`` can't race past us between load and write.
    """
    if not project_dir or not Path(project_dir).is_dir():
        return None
    clean: list[dict] = []
    for r in rows or []:
        clean.append({k: v for k, v in r.items() if not str(k).startswith("_")})
    path = Path(project_dir) / PROJECT_SETTINGS_FILENAME
    with _SETTINGS_FILE_LOCK:
        settings = load_project_settings(project_dir)
        settings[_IMAGE_METADATA_KEY] = clean
        settings.setdefault("version", 1)
        try:
            path.write_text(json.dumps(settings, indent=2))
        except Exception:
            return None
    return str(path)


def upsert_image_metadata_row(
    project_dir: str, image: str, fields: dict,
) -> None:
    """Merge ``fields`` into the row for ``image``, appending if absent."""
    rows = get_image_metadata(project_dir)
    for r in rows:
        if r.get("image") == image:
            r.update(fields)
            break
    else:
        rows.append({"image": image, **fields})
    set_image_metadata(project_dir, rows)
