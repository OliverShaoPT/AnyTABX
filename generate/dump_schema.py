"""Field names and IO helpers for env-centric / agent-centric records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

OWN_FEATURE_DIM = 14
OTHER_FEATURE_DIM = 16
ZONE_FEATURE_DIM = 6

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
    # Invisible slots are zeroed in TABX.get_obs.
    mask = np.any(np.abs(other) > 1e-8, axis=-1).astype(np.uint8)
    return static.astype(np.float32), other.astype(np.float32), mask
