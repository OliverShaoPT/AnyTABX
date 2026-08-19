"""Field names and IO helpers for env-centric / agent-centric records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

OWN_FEATURE_DIM = 14
OTHER_FEATURE_DIM = 16  # env / obs_flat; agent-centric comm may append unit_id
ZONE_FEATURE_DIM = 6
# other-slot layout (TABX _get_obs): index 12 = is_alive
OTHER_IS_ALIVE_IDX = 12
# own-slot layout: index 12 = is_alive
OWN_IS_ALIVE_IDX = 12
# Appended only in comm agent-centric / airsoul wrapper — not in TABX.get_obs.
OTHER_UNIT_ID_IDX = 16
OTHER_FEATURE_DIM_WITH_UNIT_ID = 17

ENV_CENTRIC_ARRAYS = (
    "actions_behavior",
    "actions_reference",
    "actions_reference_distribution",
    "reward_team",
    "reward_individual",
    "done",
    "truncation",
    "is_win",
    "reset",
    "episode_id",
    "behavior_policy_id",
    "policy_tag",
    "visible_matrix",
    "unit_position",
    "unit_rotation",
    "unit_health",
    "unit_max_health",
    "unit_is_alive",
    "unit_team",
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_meta(path: Path, meta: dict[str, Any]) -> None:
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_meta(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_npy(path: Path, array: np.ndarray) -> None:
    np.save(path, array)


def load_npy(path: Path) -> np.ndarray:
    return np.load(path)


def write_env_centric_record(record_dir: Path, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
    ensure_dir(record_dir)
    for name, array in arrays.items():
        save_npy(record_dir / f"{name}.npy", np.asarray(array))
    save_meta(record_dir / "meta.json", meta)


def read_env_centric_record(record_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    record_dir = Path(record_dir)
    meta = load_meta(record_dir / "meta.json")
    arrays: dict[str, np.ndarray] = {}
    for path in sorted(record_dir.glob("*.npy")):
        arrays[path.stem] = load_npy(path)
    return arrays, meta


def discover_agent_centric_dirs(records_root: Path) -> list[Path]:
    """Find agent-centric sequence dirs (flat output_root or nested layout).

    Preference:
    1. Flat: immediate children of ``records_root`` that contain ``obs_static.npy``
    2. Nested: ``**/agent_centric/unit_*`` then ``**/agent_centric/ally_*``
    """

    root = Path(records_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")
    flat = sorted(
        p for p in root.iterdir() if p.is_dir() and (p / "obs_static.npy").is_file()
    )
    if flat:
        return flat
    nested = sorted(root.glob("**/agent_centric/unit_*"))
    if not nested:
        nested = sorted(root.glob("**/agent_centric/ally_*"))
    return nested


def split_flat_obs(
    obs: np.ndarray,
    *,
    n_units: int,
    max_n_zone: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split one flat agent obs into static / dynamic / dynamic_mask."""

    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    n_other = n_units - 1
    own_end = OWN_FEATURE_DIM
    other_end = own_end + OTHER_FEATURE_DIM * n_other
    own = obs[:own_end]
    other = obs[own_end:other_end].reshape(n_other, OTHER_FEATURE_DIM)
    if max_n_zone > 0:
        zones = obs[other_end : other_end + ZONE_FEATURE_DIM * max_n_zone].reshape(
            max_n_zone, ZONE_FEATURE_DIM
        )
        static = np.concatenate([own, zones.reshape(-1)], axis=0)
    else:
        static = own
    # Alive & in-FOV (invisible / padding / dead slots have is_alive=0).
    # Do not use any-nonzero: disabled ghosts at the origin have rel_pos != 0.
    mask = (other[..., OTHER_IS_ALIVE_IDX] > 0.5).astype(np.uint8)
    return static.astype(np.float32), other.astype(np.float32), mask


def rolled_other_unit_ids(ego_index: int, n_units: int) -> np.ndarray:
    """Global ``unit_keys`` index for each dyn slot after ``roll(-ego)[:, 1:]``.

    Slot ``j`` is unit ``(ego_index + 1 + j) % n_units``. Env ``get_obs`` is
    unchanged; this is reconstructed from the same roll convention.
    """

    n = int(n_units)
    if n < 2:
        return np.zeros((0,), dtype=np.float32)
    ego = int(ego_index) % n
    return ((np.arange(n - 1, dtype=np.int32) + ego + 1) % n).astype(np.float32)


def append_other_unit_ids(
    dyn: np.ndarray, ego_index: int, n_units: int
) -> np.ndarray:
    """Append unit_keys index as last feature. Idempotent if already present."""

    dyn = np.asarray(dyn, dtype=np.float32)
    squeeze = False
    if dyn.ndim == 2:
        dyn = dyn[None]
        squeeze = True
    if dyn.ndim != 3:
        raise ValueError(f"dyn expected [T, M, F] or [M, F], got {dyn.shape}")
    _t, m, f = dyn.shape
    if f >= OTHER_FEATURE_DIM_WITH_UNIT_ID:
        out = dyn
    elif f != OTHER_FEATURE_DIM:
        raise ValueError(
            f"dyn feature dim {f}, expected {OTHER_FEATURE_DIM} or "
            f"{OTHER_FEATURE_DIM_WITH_UNIT_ID}"
        )
    else:
        ids = rolled_other_unit_ids(ego_index, n_units)
        if ids.shape[0] < m:
            ids = np.pad(ids, (0, m - int(ids.shape[0])))
        elif ids.shape[0] > m:
            ids = ids[:m]
        ch = np.broadcast_to(ids.reshape(1, m, 1), (_t, m, 1)).astype(np.float32)
        out = np.concatenate([dyn, np.array(ch, copy=True)], axis=-1)
    if squeeze:
        return out[0]
    return out
