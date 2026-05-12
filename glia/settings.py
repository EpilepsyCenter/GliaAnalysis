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
from pathlib import Path
from typing import Any


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
    return out


def save_project_settings(project_dir: str, state) -> str | None:
    """Write a snapshot of the project-scoped state fields to JSON."""
    if not project_dir or not Path(project_dir).is_dir():
        return None
    path = Path(project_dir) / PROJECT_SETTINGS_FILENAME
    try:
        path.write_text(json.dumps(_state_to_project_dict(state), indent=2))
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
