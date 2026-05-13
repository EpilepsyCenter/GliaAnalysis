"""Server-side session state manager.

Stores per-session state (project directory, current image, extracted
features dataframe) keyed by a session UUID. The UUID lives in a
lightweight dcc.Store on the client; the actual data lives here so we
don't serialise megabytes of feature tables through Dash callbacks.

Mirrors NED-Net's eeg_seizure_analyzer/dash_app/server_state.py.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionState:
    """All mutable state for one browser session."""

    # Mode + project
    mode: str = "microglia"       # microglia | astrocyte
    project_dir: str = ""
    fiji_path: str = ""

    # Setup — thresholding
    threshold_method: str = "Otsu"
    threshold_kind: str = "global"        # global | local
    local_radius: int = 100
    threshold_lower: float = 0.0          # manual / override band [lo, hi]
    threshold_upper: float = 255.0
    preprocess: bool = True               # FIJI-style preprocessing chain
    area_min: float = 0.0
    area_max: float = 0.0

    # Setup — soma (radial soma + Sholl analysis)
    soma_gap_tol_deg: float = 20.0
    # Optional: use the DAPI nucleus centroid as the radial scan center.
    # Off by default; opt-in per project. When the DAPI signal is poor
    # the user can leave this off and stay on the EDT-peak fallback.
    use_dapi: bool = False

    # Data
    features_df: Any = None               # cell-level 27-feature dataframe
    metadata_cols: list = field(default_factory=list)
    metadata_sep: str = "_"

    # Cluster
    k: int = 4
    pca_n_components: int = 5
    cluster_labels: dict = field(default_factory=dict)   # {cluster_id: label}
    transform: str = "none"               # none | log | zscore | minmax

    # Stats
    animal_id_col: str = ""
    factor_cols: list = field(default_factory=list)

    # Arbitrary key-value store for extensibility
    extra: dict = field(default_factory=dict)


_sessions: dict[str, SessionState] = {}


def create_session() -> str:
    sid = str(uuid.uuid4())
    _sessions[sid] = SessionState()
    return sid


def get_session(session_id: str | None) -> SessionState:
    """Return the SessionState for ``session_id``, creating it if missing.

    Crucially: when ``session_id`` is given but not registered (e.g. after a
    debug-mode reload wiped ``_sessions`` while the client still holds the
    old sid), we register a fresh SessionState **under that same sid**.
    Otherwise every callback in the same request would create a new state
    under a different uuid, and state set by one would be invisible to the
    next.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())
    if session_id not in _sessions:
        _sessions[session_id] = SessionState()
    return _sessions[session_id]


def get(session_id: str | None, key: str, default: Any = None) -> Any:
    state = get_session(session_id)
    if hasattr(state, key):
        return getattr(state, key)
    return state.extra.get(key, default)


def put(session_id: str | None, key: str, value: Any) -> None:
    state = get_session(session_id)
    if hasattr(state, key):
        setattr(state, key, value)
    else:
        state.extra[key] = value
