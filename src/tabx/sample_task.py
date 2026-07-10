"""Sample, save, and load fixed TABX task banks.

The task-bank JSON format stores many tasks in one file.  Every task keeps the
same ``scenario`` and ``zone_scenario`` schema used by the existing scenario
assets, so loading a bank only adds batching/padding around TABX's normal
``env.reset(key, env_params)`` interface.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from src.tabx.config import TABXConfig
from src.tabx.heuristic_policy.utils import build_batched_heuristic_params
from src.tabx.physics.utils import build_batched_physics_params
from src.tabx.scenarios.constants import CHALLENGES, UNIT_SCENARIOS, ZONE_SCENARIOS
from src.tabx.scenarios.scenario import VectorizedScenario, ZoneScenario
from src.tabx.scenarios.utils import (
    generate_padded_unit_scenario,
    generate_padded_zone_scenario,
)
from src.tabx.task_generators import (
    COMPOSITION_ARCHETYPES,
    COMPOSITION_MATCH_MODES,
    DISTANCE_BUCKETS,
    LAYOUT_ARCHETYPES,
    SPREAD_BUCKETS,
    ZONE_ARCHETYPES,
    ZONE_INTENSITY_BUCKETS,
    ZONE_RELATION_BUCKETS,
    generate_programmatic_task,
)

TASK_BANK_VERSION = "1.0"
SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"
SUPPORTED_TRANSFORMS = ("identity", "mirror_x", "mirror_y", "rotate_180")


@dataclass(frozen=True)
class SampleConfig:
    """Command-line configuration for offline task sampling."""

    output: str = "sampled_tasks.json"
    n_tasks: int = 100
    seed: int = 0
    programmatic_ratio: float = 1.0
    max_n_ally: int = 10
    max_n_enemy: int = 10
    max_n_zone: int = 4
    composition_archetypes: tuple[str, ...] = COMPOSITION_ARCHETYPES
    composition_match_modes: tuple[str, ...] = COMPOSITION_MATCH_MODES
    composition_price_rel_tol: float = 0.15
    composition_effective_rel_tol: float = 0.15
    # Per-unit HP as a fraction of template max HP; B is projected to match
    # sum(price * hp_frac) after price-matched roster sampling.
    health_frac_buckets: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9, 1.0)
    health_frac_min: float = 0.5
    health_frac_max: float = 1.0
    layout_archetypes: tuple[str, ...] = LAYOUT_ARCHETYPES
    distance_buckets: tuple[str, ...] = DISTANCE_BUCKETS
    spread_buckets: tuple[str, ...] = SPREAD_BUCKETS
    zone_archetypes: tuple[str, ...] = ZONE_ARCHETYPES
    zone_intensity_buckets: tuple[str, ...] = ZONE_INTENSITY_BUCKETS
    zone_relation_buckets: tuple[str, ...] = ZONE_RELATION_BUCKETS
    include_challenges: bool = True
    transforms: tuple[str, ...] = SUPPORTED_TRANSFORMS
    map_scales: tuple[float, ...] = (0.85, 1.0, 1.15)
    stat_scales: tuple[float, ...] = (0.9, 1.0, 1.1)
    # Keep ally/enemy stat scales coupled so balance comes from roster matching,
    # while scenario-type buckets carry most of the diversity.
    couple_stat_scales: bool = True
    zone_strength_scales: tuple[float, ...] = (0.75, 1.0, 1.25)
    # Prefer under-covered (composition, layout, distance, spread, zone, ...) buckets.
    bucket_coverage: bool = True
    bucket_coverage_candidates: int = 16
    physics: str = "default"
    heuristic: str = "expert"
    # Keep only tasks whose both-team heuristic win rate falls in this band.
    # With epsilon=0 the policy is nearly deterministic, so win rates collapse to
    # 0/1; use a small positive filter_epsilon so intermediate rates are possible.
    filter_by_win_rate: bool = True
    win_rate_min: float = 0.4
    win_rate_max: float = 0.6
    filter_num_seeds: int = 16
    filter_epsilon: float = 0.05
    filter_max_episode_steps: int = 512
    filter_reject_all_truncated: bool = True
    filter_max_attempts: int = 20


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _to_jax_array(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_jax_array(item) for key, item in value.items()}
    if isinstance(value, list):
        return jnp.asarray(value)
    return jnp.asarray([value]).reshape(-1, 1)


def task_to_structs(task: dict[str, Any]) -> tuple[VectorizedScenario, ZoneScenario]:
    """Convert one JSON task into the structs consumed by TABX."""

    if "scenario" not in task or "zone_scenario" not in task:
        raise ValueError("Each task must contain 'scenario' and 'zone_scenario'.")
    scenario = VectorizedScenario(**_to_jax_array(task["scenario"]))
    zone_scenario = ZoneScenario(**_to_jax_array(task["zone_scenario"]))
    return scenario, zone_scenario


def load_task_bank(path: str | Path) -> dict[str, Any]:
    """Load and validate a single-file task bank."""

    bank = _read_json(Path(path))
    if bank.get("schema_version") != TASK_BANK_VERSION:
        raise ValueError(
            f"Unsupported task-bank version {bank.get('schema_version')!r}; "
            f"expected {TASK_BANK_VERSION!r}."
        )
    tasks = bank.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("A task bank must contain a non-empty 'tasks' list.")
    for task in tasks:
        validate_task(task)
    return bank


def save_task_bank(
    path: str | Path,
    tasks: Sequence[dict[str, Any]],
    *,
    seed: int,
    physics: str = "default",
    heuristic: str = "expert",
    max_n_ally: int | None = None,
    max_n_enemy: int | None = None,
    max_n_zone: int | None = None,
    filter_protocol: dict[str, Any] | None = None,
) -> Path:
    """Save sampled tasks and their manifest into one JSON file."""

    if not tasks:
        raise ValueError("Cannot save an empty task bank.")
    for task in tasks:
        validate_task(task)

    inferred = infer_task_limits(tasks)
    max_n_ally = inferred[0] if max_n_ally is None else max_n_ally
    max_n_enemy = inferred[1] if max_n_enemy is None else max_n_enemy
    max_n_zone = inferred[2] if max_n_zone is None else max_n_zone
    requested = (max_n_ally, max_n_enemy, max_n_zone)
    if any(actual < required for actual, required in zip(requested, inferred)):
        raise ValueError(
            f"Task-bank schema {requested} is smaller than required limits {inferred}."
        )
    manifest: dict[str, Any] = {
        "generator": "src.tabx.sample_task",
        "seed": seed,
        "n_tasks": len(tasks),
        "physics": physics,
        "heuristic": heuristic,
        "schema": {
            "max_n_ally": max_n_ally,
            "max_n_enemy": max_n_enemy,
            "max_n_zone": max_n_zone,
        },
    }
    if filter_protocol is not None:
        manifest["filter_protocol"] = filter_protocol
    bank = {
        "schema_version": TASK_BANK_VERSION,
        "manifest": manifest,
        "tasks": list(tasks),
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(bank, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    return output_path


def infer_task_limits(tasks: Sequence[dict[str, Any]]) -> tuple[int, int, int]:
    """Infer common padding limits from unpadded tasks."""

    max_n_ally = 0
    max_n_enemy = 0
    max_n_zone = 0
    for task in tasks:
        teams = np.asarray(task["scenario"]["teams"]).reshape(-1)
        max_n_ally = max(max_n_ally, int(np.count_nonzero(teams == 0)))
        max_n_enemy = max(max_n_enemy, int(np.count_nonzero(teams == 1)))
        max_n_zone = max(max_n_zone, int(task["zone_scenario"]["n_zone"]))
    return max_n_ally, max_n_enemy, max_n_zone


def build_batched_env_params_from_tasks(
    tasks: Sequence[dict[str, Any]],
    *,
    physics: str = "default",
    heuristic: str = "expert",
    max_n_ally: int | None = None,
    max_n_enemy: int | None = None,
    max_n_zone: int | None = None,
) -> tuple[dict[str, Any], TABXConfig]:
    """Build normal TABX env_params from in-memory task dictionaries."""

    if not tasks:
        raise ValueError("At least one task is required.")
    inferred = infer_task_limits(tasks)
    max_n_ally = inferred[0] if max_n_ally is None else max_n_ally
    max_n_enemy = inferred[1] if max_n_enemy is None else max_n_enemy
    max_n_zone = inferred[2] if max_n_zone is None else max_n_zone
    requested = (max_n_ally, max_n_enemy, max_n_zone)
    if any(actual < required for actual, required in zip(requested, inferred)):
        raise ValueError(
            "Requested task limits are smaller than the task bank requires: "
            f"requested={requested}, required={inferred}."
        )

    scenarios: list[VectorizedScenario] = []
    zone_scenarios: list[ZoneScenario] = []
    for task in tasks:
        validate_task(task)
        scenario, zone_scenario = task_to_structs(task)
        scenarios.append(generate_padded_unit_scenario(scenario, max_n_ally, max_n_enemy))
        zone_scenarios.append(generate_padded_zone_scenario(zone_scenario, max_n_zone))

    batched_scenario = jax.tree.map(lambda *values: jnp.stack(values), *scenarios)
    batched_zones = jax.tree.map(lambda *values: jnp.stack(values), *zone_scenarios)
    n_tasks = len(tasks)
    physics_params = build_batched_physics_params(
        [physics] * n_tasks, squeeze_when_single_physics=False
    )
    heuristic_params = build_batched_heuristic_params(
        [heuristic] * n_tasks, squeeze_when_single_heuristic=False
    )
    config = TABXConfig(
        max_n_ally=max_n_ally,
        max_n_enemy=max_n_enemy,
        max_n_zone=max_n_zone,
    )
    return {
        "scenario": batched_scenario,
        "zone_scenario": batched_zones,
        "physics_params": physics_params,
        "heuristic_params": heuristic_params,
    }, config


def build_batched_env_params_from_task_file(
    path: str | Path,
    *,
    physics: str | None = None,
    heuristic: str | None = None,
) -> tuple[dict[str, Any], TABXConfig]:
    """Load a task file and build env_params accepted by ``TABX.reset``."""

    bank = load_task_bank(path)
    manifest = bank["manifest"]
    schema = manifest["schema"]
    return build_batched_env_params_from_tasks(
        bank["tasks"],
        physics=physics or manifest["physics"],
        heuristic=heuristic or manifest["heuristic"],
        max_n_ally=int(schema["max_n_ally"]),
        max_n_enemy=int(schema["max_n_enemy"]),
        max_n_zone=int(schema["max_n_zone"]),
    )


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _transform_task(task: dict[str, Any], transform: str) -> None:
    if transform not in SUPPORTED_TRANSFORMS:
        raise ValueError(f"Unknown transform {transform!r}; choose from {SUPPORTED_TRANSFORMS}.")

    scenario = task["scenario"]
    zones = task["zone_scenario"]
    positions = scenario["positions"]
    rotations = scenario["rotations"]
    pos_min = scenario["pos_min"]
    pos_max = scenario["pos_max"]

    if transform == "identity":
        return
    if transform == "mirror_x":
        for position in positions:
            position[0] *= -1
        for position in zones["position"]:
            position[0] *= -1
        for rotation in rotations:
            rotation[0] = _wrap_angle(math.pi - rotation[0])
        for lower, upper in zip(pos_min, pos_max):
            lower[0], upper[0] = -upper[0], -lower[0]
    elif transform == "mirror_y":
        for position in positions:
            position[1] *= -1
        for position in zones["position"]:
            position[1] *= -1
        for rotation in rotations:
            rotation[0] = _wrap_angle(-rotation[0])
        for lower, upper in zip(pos_min, pos_max):
            lower[1], upper[1] = -upper[1], -lower[1]
    else:
        for position in positions:
            position[0] *= -1
            position[1] *= -1
        for position in zones["position"]:
            position[0] *= -1
            position[1] *= -1
        for rotation in rotations:
            rotation[0] = _wrap_angle(rotation[0] + math.pi)
        for lower, upper in zip(pos_min, pos_max):
            lower[0], upper[0] = -upper[0], -lower[0]
            lower[1], upper[1] = -upper[1], -lower[1]


def _scale_map(task: dict[str, Any], scale: float) -> None:
    if scale <= 0:
        raise ValueError("Map scale must be positive.")
    scenario = task["scenario"]
    zones = task["zone_scenario"]
    for field in ("positions", "pos_min", "pos_max"):
        scenario[field] = (np.asarray(scenario[field], dtype=float) * scale).tolist()
    zones["position"] = (np.asarray(zones["position"], dtype=float) * scale).tolist()
    zones["axes"] = (np.asarray(zones["axes"], dtype=float) * scale).tolist()
    grid_info = task.get("grid_info", {})
    for field in ("max_field_width", "max_field_height", "margin_width", "margin_height"):
        if field in grid_info:
            grid_info[field] *= scale


def _scale_team_stats(task: dict[str, Any], ally_scale: float, enemy_scale: float) -> None:
    teams = np.asarray(task["scenario"]["teams"]).reshape(-1)
    scales = np.where(teams == 0, ally_scale, enemy_scale).reshape(-1, 1)
    for field in ("healths", "speeds", "attack_damages"):
        values = np.asarray(task["scenario"][field], dtype=float)
        task["scenario"][field] = (values * scales).tolist()


def _scale_zone_strength(task: dict[str, Any], scale: float) -> None:
    values = np.asarray(task["zone_scenario"]["effect_value"], dtype=float)
    task["zone_scenario"]["effect_value"] = (values * scale).tolist()


def _load_composed_task(unit_name: str, zone_name: str) -> dict[str, Any]:
    unit_asset = _read_json(SCENARIO_DIR / "units" / f"{unit_name}.json")
    zone_asset = _read_json(SCENARIO_DIR / "zones" / f"{zone_name}.json")
    return {
        "grid_info": copy.deepcopy(unit_asset.get("grid_info", {})),
        "scenario": copy.deepcopy(unit_asset["scenario"]),
        "zone_scenario": copy.deepcopy(zone_asset["zone_scenario"]),
    }


def _load_challenge_task(challenge_name: str) -> dict[str, Any]:
    asset = _read_json(SCENARIO_DIR / "challenges" / f"{challenge_name}.json")
    return {
        "grid_info": copy.deepcopy(asset.get("grid_info", {})),
        "scenario": copy.deepcopy(asset["scenario"]),
        "zone_scenario": copy.deepcopy(asset["zone_scenario"]),
    }


def _task_hash(task: dict[str, Any]) -> str:
    payload = {
        "scenario": task["scenario"],
        "zone_scenario": task["zone_scenario"],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (int, float, bool)):
        return math.isfinite(float(value))
    return True


def validate_task(task: dict[str, Any], *, collision_margin: float = 1e-5) -> None:
    """Raise ``ValueError`` when a task cannot safely enter the current environment."""

    if "scenario" not in task or "zone_scenario" not in task:
        raise ValueError("Task is missing scenario data.")
    if not _all_finite(task["scenario"]) or not _all_finite(task["zone_scenario"]):
        raise ValueError("Task contains NaN or infinite values.")

    scenario = task["scenario"]
    required_fields = set(VectorizedScenario.__dataclass_fields__)
    missing = required_fields.difference(scenario)
    if missing:
        raise ValueError(f"Scenario is missing fields: {sorted(missing)}.")

    teams = np.asarray(scenario["teams"]).reshape(-1)
    alive = np.asarray(scenario["is_alive"], dtype=bool).reshape(-1)
    disabled = np.asarray(scenario["is_disabled"], dtype=bool).reshape(-1)
    active = alive & ~disabled
    if not np.any(active & (teams == 0)) or not np.any(active & (teams == 1)):
        raise ValueError("Both teams need at least one active unit.")

    n_units = len(teams)
    for field in required_fields:
        if len(scenario[field]) != n_units:
            raise ValueError(f"Scenario field {field!r} has inconsistent unit count.")

    positions = np.asarray(scenario["positions"], dtype=float)
    lower = np.asarray(scenario["pos_min"], dtype=float)
    upper = np.asarray(scenario["pos_max"], dtype=float)
    if np.any(positions[active] < lower[active]) or np.any(positions[active] > upper[active]):
        raise ValueError("An active unit is outside its movement boundary.")

    radii = np.asarray(scenario["body_radiuss"], dtype=float).reshape(-1)
    active_indices = np.flatnonzero(active)
    for offset, first in enumerate(active_indices):
        for second in active_indices[offset + 1 :]:
            distance = float(np.linalg.norm(positions[first] - positions[second]))
            if distance + collision_margin < radii[first] + radii[second]:
                raise ValueError(f"Units {first} and {second} overlap at spawn.")

    zones = task["zone_scenario"]
    zone_fields = set(ZoneScenario.__dataclass_fields__)
    missing_zones = zone_fields.difference(zones)
    if missing_zones:
        raise ValueError(f"Zone scenario is missing fields: {sorted(missing_zones)}.")
    n_zone = int(zones["n_zone"])
    for field in ("zone_type", "position", "axes", "effect_value"):
        if len(zones[field]) != n_zone:
            raise ValueError(f"Zone field {field!r} does not match n_zone={n_zone}.")
    if n_zone and np.any(np.asarray(zones["axes"], dtype=float) <= 0):
        raise ValueError("Zone axes must be positive.")

    if task.get("metadata", {}).get("source", {}).get("kind") == "programmatic":
        zone_types = np.asarray(zones["zone_type"], dtype=int).reshape(-1)
        effects = np.asarray(zones["effect_value"], dtype=float).reshape(-1)
        if np.any((zone_types == 2) & (np.abs(effects) > 1e-8)):
            raise ValueError("Programmatic bush zones must have zero effect_value.")
        if np.any((zone_types == 3) & ((effects <= 0) | (effects > 1))):
            raise ValueError("Programmatic swamp effect_value must be in (0, 1].")

        zone_positions = np.asarray(zones["position"], dtype=float)
        zone_axes = np.asarray(zones["axes"], dtype=float)
        lava_indices = np.flatnonzero(zone_types == 1)
        for zone_index in lava_indices:
            normalized = (
                (positions[active] - zone_positions[zone_index])
                / (zone_axes[zone_index] + radii[active, None])
            ) ** 2
            if np.any(normalized.sum(axis=1) <= 1.0):
                raise ValueError("An active unit intersects programmatic lava at spawn.")


def _progress_bar(current: int, total: int, width: int = 24) -> str:
    filled = int(width * current / max(total, 1))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _scenario_bucket_key(
    composition: str,
    layout: str,
    distance: str,
    spread: str,
    zone: str,
    zone_intensity: str,
    zone_relation: str,
    match_mode: str,
) -> tuple[str, ...]:
    return (
        composition,
        layout,
        distance,
        spread,
        zone,
        zone_intensity,
        zone_relation,
        match_mode,
    )


def _pick_scenario_buckets(
    rng: random.Random,
    config: SampleConfig,
    coverage: Counter[tuple[str, ...]],
) -> tuple[str, ...]:
    """Pick scenario-type buckets, preferring under-covered combinations."""

    def draw() -> tuple[str, ...]:
        return _scenario_bucket_key(
            rng.choice(config.composition_archetypes),
            rng.choice(config.layout_archetypes),
            rng.choice(config.distance_buckets),
            rng.choice(config.spread_buckets),
            rng.choice(config.zone_archetypes),
            rng.choice(config.zone_intensity_buckets),
            rng.choice(config.zone_relation_buckets),
            rng.choice(config.composition_match_modes),
        )

    if not config.bucket_coverage:
        return draw()

    best = draw()
    best_count = coverage[best]
    n_draw = max(1, config.bucket_coverage_candidates)
    for _ in range(n_draw - 1):
        candidate = draw()
        count = coverage[candidate]
        if count < best_count or (count == best_count and rng.random() < 0.5):
            best = candidate
            best_count = count
    return best


def _sample_coupled_stat_scales(
    rng: random.Random, scales: Sequence[float], *, coupled: bool
) -> tuple[float, float]:
    """Sample ally/enemy stat scales; coupled mode keeps them equal or one step apart."""

    if not coupled:
        return rng.choice(tuple(scales)), rng.choice(tuple(scales))
    ordered = sorted(scales)
    base = rng.choice(ordered)
    if len(ordered) == 1 or rng.random() < 0.75:
        return base, base
    index = ordered.index(base)
    neighbors = []
    if index > 0:
        neighbors.append(ordered[index - 1])
    if index + 1 < len(ordered):
        neighbors.append(ordered[index + 1])
    other = rng.choice(neighbors) if neighbors else base
    if rng.random() < 0.5:
        return base, other
    return other, base


def _passes_win_rate_filter(
    task: dict[str, Any],
    *,
    physics: str,
    heuristic: str,
    num_seeds: int,
    seed: int,
    epsilon: float,
    max_episode_steps: int,
    win_rate_min: float,
    win_rate_max: float,
    reject_all_truncated: bool,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate one candidate and decide whether its win rate is balanced enough."""

    # Lazy import avoids a circular dependency with eval_task.
    from src.tabx.eval_task import evaluate_tasks

    result = evaluate_tasks(
        [task],
        physics=physics,
        heuristic=heuristic,
        num_seeds=num_seeds,
        seed=seed,
        epsilon_override=epsilon,
        max_episode_steps=max_episode_steps,
    )[0]
    win_rate = float(result["win_rate"])
    accepted = win_rate_min <= win_rate <= win_rate_max
    reject_reason = None
    if reject_all_truncated and result["quality_flags"]["all_truncated"]:
        accepted = False
        reject_reason = "all_truncated"
    elif not accepted:
        reject_reason = "win_rate_out_of_band"
    result["reject_reason"] = reject_reason
    return accepted, result


def sample_tasks(config: SampleConfig) -> list[dict[str, Any]]:
    """Sample unique low-cost task variants from existing TABX assets."""

    if config.n_tasks <= 0:
        raise ValueError("n_tasks must be positive.")
    if not 0.0 <= config.programmatic_ratio <= 1.0:
        raise ValueError("programmatic_ratio must be in [0, 1].")
    if config.max_n_ally < 1 or config.max_n_enemy < 1 or config.max_n_zone < 0:
        raise ValueError("max_n_ally/max_n_enemy must be positive and max_n_zone non-negative.")
    if not config.transforms or not config.map_scales or not config.stat_scales:
        raise ValueError("Transform and scale buckets must not be empty.")
    if not config.composition_match_modes:
        raise ValueError("composition_match_modes must not be empty.")
    if config.composition_price_rel_tol < 0.0:
        raise ValueError("composition_price_rel_tol must be non-negative.")
    if config.composition_effective_rel_tol < 0.0:
        raise ValueError("composition_effective_rel_tol must be non-negative.")
    if not config.health_frac_buckets:
        raise ValueError("health_frac_buckets must not be empty.")
    if any(not 0.0 < float(value) <= 1.0 for value in config.health_frac_buckets):
        raise ValueError("health_frac_buckets must lie in (0, 1].")
    if not 0.0 < config.health_frac_min <= config.health_frac_max <= 1.0:
        raise ValueError("Require 0 < health_frac_min <= health_frac_max <= 1.")
    if config.bucket_coverage_candidates <= 0:
        raise ValueError("bucket_coverage_candidates must be positive.")
    if not 0.0 <= config.win_rate_min <= config.win_rate_max <= 1.0:
        raise ValueError("Require 0 <= win_rate_min <= win_rate_max <= 1.")
    if config.filter_by_win_rate:
        if config.filter_num_seeds <= 0:
            raise ValueError("filter_num_seeds must be positive when filtering.")
        if not 0.0 <= config.filter_epsilon <= 1.0:
            raise ValueError("filter_epsilon must be in [0, 1].")
        if config.filter_epsilon <= 0.0:
            raise ValueError(
                "filter_epsilon must be > 0 when filtering by win rate; "
                "epsilon=0 makes expert vs expert nearly deterministic (win rates collapse to 0/1)."
            )
        if config.filter_max_attempts <= 0:
            raise ValueError("filter_max_attempts must be positive when filtering.")
    generator_buckets = (
        config.composition_archetypes,
        config.composition_match_modes,
        config.layout_archetypes,
        config.distance_buckets,
        config.spread_buckets,
        config.zone_archetypes,
        config.zone_intensity_buckets,
        config.zone_relation_buckets,
    )
    if any(not bucket for bucket in generator_buckets):
        raise ValueError("Programmatic generator buckets must not be empty.")
    unknown = set(config.transforms).difference(SUPPORTED_TRANSFORMS)
    if unknown:
        raise ValueError(f"Unsupported transforms: {sorted(unknown)}.")
    supported_buckets = (
        (config.composition_archetypes, COMPOSITION_ARCHETYPES, "composition"),
        (config.composition_match_modes, COMPOSITION_MATCH_MODES, "composition match"),
        (config.layout_archetypes, LAYOUT_ARCHETYPES, "layout"),
        (config.distance_buckets, DISTANCE_BUCKETS, "distance"),
        (config.spread_buckets, SPREAD_BUCKETS, "spread"),
        (config.zone_archetypes, ZONE_ARCHETYPES, "zone"),
        (config.zone_intensity_buckets, ZONE_INTENSITY_BUCKETS, "zone intensity"),
        (config.zone_relation_buckets, ZONE_RELATION_BUCKETS, "zone relation"),
    )
    for selected, supported, label in supported_buckets:
        unsupported = set(selected).difference(supported)
        if unsupported:
            raise ValueError(f"Unsupported {label} buckets: {sorted(unsupported)}.")

    rng = random.Random(config.seed)
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    bucket_coverage: Counter[tuple[str, ...]] = Counter()
    if config.filter_by_win_rate:
        # filter_max_attempts is a per-accepted-task budget.
        max_attempts = config.filter_max_attempts * config.n_tasks
    else:
        max_attempts = max(1_000, config.n_tasks * 100)
    n_candidates = 0
    n_rejected_by_win_rate = 0
    n_generation_failures = 0

    if config.filter_by_win_rate:
        print(
            "Win-rate filter enabled: "
            f"target={config.n_tasks}, band=[{config.win_rate_min:.2f}, {config.win_rate_max:.2f}], "
            f"seeds={config.filter_num_seeds}, epsilon={config.filter_epsilon}, "
            f"max_attempts_per_task={config.filter_max_attempts}, "
            f"max_attempts_total={max_attempts}"
        )
    print(
        "Balance-first sampling: "
        f"match_modes={list(config.composition_match_modes)}, "
        f"price_rel_tol={config.composition_price_rel_tol}, "
        f"effective_rel_tol={config.composition_effective_rel_tol}, "
        f"health_frac_buckets={list(config.health_frac_buckets)}, "
        f"couple_stat_scales={config.couple_stat_scales}, "
        f"bucket_coverage={config.bucket_coverage}"
    )

    for attempt in range(1, max_attempts + 1):
        is_programmatic = rng.random() < config.programmatic_ratio
        map_scale = rng.choice(config.map_scales)
        ally_stat_scale, enemy_stat_scale = _sample_coupled_stat_scales(
            rng, config.stat_scales, coupled=config.couple_stat_scales
        )
        transform = rng.choice(config.transforms)

        if is_programmatic:
            (
                composition,
                layout,
                distance,
                spread,
                zone,
                zone_intensity,
                zone_relation,
                match_mode,
            ) = _pick_scenario_buckets(rng, config, bucket_coverage)
            try:
                task = generate_programmatic_task(
                    rng,
                    composition=composition,
                    layout=layout,
                    distance=distance,
                    spread=spread,
                    zone=zone,
                    zone_intensity=zone_intensity,
                    zone_relation=zone_relation,
                    map_scale=map_scale,
                    max_n_ally=config.max_n_ally,
                    max_n_enemy=config.max_n_enemy,
                    match_mode=match_mode,
                    price_rel_tol=config.composition_price_rel_tol,
                    effective_rel_tol=config.composition_effective_rel_tol,
                    health_frac_buckets=config.health_frac_buckets,
                    health_frac_min=config.health_frac_min,
                    health_frac_max=config.health_frac_max,
                )
            except ValueError as exc:
                n_generation_failures += 1
                if config.filter_by_win_rate:
                    print(
                        f"try {attempt}/{max_attempts} {_progress_bar(attempt, max_attempts)} "
                        f"GENERATE_FAIL accepted={len(tasks)}/{config.n_tasks} "
                        f"reason={exc}"
                    )
                continue
            source = task["metadata"]["source"]
            zone_strength_scale = 1.0
            scenario_key = _scenario_bucket_key(
                composition,
                layout,
                distance,
                spread,
                zone,
                zone_intensity,
                zone_relation,
                match_mode,
            )
        else:
            use_challenge = config.include_challenges and rng.random() < 0.25
            if use_challenge:
                source_name = rng.choice(sorted(CHALLENGES))
                task = _load_challenge_task(source_name)
                source = {"kind": "challenge", "name": source_name}
            else:
                unit_name = rng.choice(sorted(UNIT_SCENARIOS))
                zone_name = rng.choice(sorted(ZONE_SCENARIOS))
                task = _load_composed_task(unit_name, zone_name)
                source = {"kind": "composed", "unit": unit_name, "zone": zone_name}
            _scale_map(task, map_scale)
            zone_strength_scale = rng.choice(config.zone_strength_scales)
            scenario_key = None

        _transform_task(task, transform)
        _scale_team_stats(task, ally_stat_scale, enemy_stat_scale)
        _scale_zone_strength(task, zone_strength_scale)
        try:
            validate_task(task)
        except ValueError as exc:
            n_generation_failures += 1
            if config.filter_by_win_rate:
                print(
                    f"try {attempt}/{max_attempts} {_progress_bar(attempt, max_attempts)} "
                    f"INVALID accepted={len(tasks)}/{config.n_tasks} reason={exc}"
                )
            continue
        if int(task["zone_scenario"]["n_zone"]) > config.max_n_zone:
            n_generation_failures += 1
            if config.filter_by_win_rate:
                print(
                    f"try {attempt}/{max_attempts} {_progress_bar(attempt, max_attempts)} "
                    f"INVALID accepted={len(tasks)}/{config.n_tasks} reason=too_many_zones"
                )
            continue

        digest = _task_hash(task)
        if digest in seen:
            if config.filter_by_win_rate:
                print(
                    f"try {attempt}/{max_attempts} {_progress_bar(attempt, max_attempts)} "
                    f"DUPLICATE accepted={len(tasks)}/{config.n_tasks}"
                )
            continue
        seen.add(digest)
        n_candidates += 1
        task["task_id"] = f"task_{len(tasks):06d}_{digest[:10]}"
        metadata = task.get("metadata", {})
        metadata.update(
            {
                "source": source,
                "transform": transform,
                "map_scale": map_scale,
                "ally_stat_scale": ally_stat_scale,
                "enemy_stat_scale": enemy_stat_scale,
                "zone_strength_scale": zone_strength_scale,
                "canonical_hash": digest,
            }
        )
        task["metadata"] = metadata

        if config.filter_by_win_rate:
            match_info = metadata.get("composition_match") or {}
            print(
                f"try {attempt}/{max_attempts} {_progress_bar(attempt, max_attempts)} "
                f"EVALUATING accepted={len(tasks)}/{config.n_tasks} "
                f"comp={metadata.get('composition_archetype')} "
                f"match={match_info.get('match_mode')} "
                f"price_diff={match_info.get('price_rel_diff', float('nan')):.3f} "
                f"eff_diff={match_info.get('effective_rel_diff', float('nan')):.3f} "
                f"layout={metadata.get('layout_archetype')} "
                f"zone={metadata.get('zone_archetype')} ..."
            )
            accepted, eval_result = _passes_win_rate_filter(
                task,
                physics=config.physics,
                heuristic=config.heuristic,
                num_seeds=config.filter_num_seeds,
                seed=config.seed + n_candidates,
                epsilon=config.filter_epsilon,
                max_episode_steps=config.filter_max_episode_steps,
                win_rate_min=config.win_rate_min,
                win_rate_max=config.win_rate_max,
                reject_all_truncated=config.filter_reject_all_truncated,
            )
            task["metadata"]["filter_eval"] = {
                "win_rate": eval_result["win_rate"],
                "win_rate_ci95": eval_result["win_rate_ci95"],
                "episode_length": eval_result["episode_length"],
                "truncation_rate": eval_result["truncation_rate"],
                "hp_margin": eval_result["hp_margin"],
                "quality_flags": eval_result["quality_flags"],
                "n_rollouts": eval_result["n_rollouts"],
                "accepted": accepted,
                "reject_reason": eval_result.get("reject_reason"),
            }
            status = "ACCEPT" if accepted else "REJECT"
            print(
                f"try {attempt}/{max_attempts} {_progress_bar(attempt, max_attempts)} "
                f"{status} win_rate={eval_result['win_rate']:.3f} "
                f"ci95=[{eval_result['win_rate_ci95'][0]:.3f}, {eval_result['win_rate_ci95'][1]:.3f}] "
                f"len={eval_result['episode_length']['mean']:.1f} "
                f"trunc={eval_result['truncation_rate']:.3f} "
                f"hp_margin={eval_result['hp_margin']['mean']:.3f} "
                f"accepted={len(tasks) + int(accepted)}/{config.n_tasks}"
                + (
                    f" reason={eval_result.get('reject_reason')}"
                    if not accepted and eval_result.get("reject_reason")
                    else ""
                )
            )
            if not accepted:
                n_rejected_by_win_rate += 1
                # Keep the hash reserved so we do not re-evaluate the same candidate.
                continue

        tasks.append(task)
        if scenario_key is not None:
            bucket_coverage[scenario_key] += 1
        if len(tasks) == config.n_tasks:
            if config.filter_by_win_rate:
                print(
                    f"Done. Accepted {len(tasks)}/{n_candidates} unique candidates "
                    f"(rejected_by_win_rate={n_rejected_by_win_rate}, "
                    f"generation_failures={n_generation_failures}, "
                    f"covered_buckets={len(bucket_coverage)})."
                )
            elif config.bucket_coverage:
                print(
                    f"Done. Sampled {len(tasks)} tasks across {len(bucket_coverage)} scenario buckets."
                )
            return tasks

    raise RuntimeError(
        f"Could only produce {len(tasks)} valid unique tasks after {max_attempts} attempts "
        f"(unique_candidates={n_candidates}, rejected_by_win_rate={n_rejected_by_win_rate}, "
        f"generation_failures={n_generation_failures})."
    )


def task_ids(tasks: Iterable[dict[str, Any]]) -> list[str]:
    """Return stable IDs for display and result joins."""

    return [str(task.get("task_id", _task_hash(task)[:16])) for task in tasks]


def main() -> None:
    config = tyro.cli(SampleConfig)
    tasks = sample_tasks(config)
    filter_protocol = None
    if config.filter_by_win_rate:
        filter_protocol = {
            "enabled": True,
            "win_rate_min": config.win_rate_min,
            "win_rate_max": config.win_rate_max,
            "num_seeds": config.filter_num_seeds,
            "epsilon": config.filter_epsilon,
            "max_episode_steps": config.filter_max_episode_steps,
            "reject_all_truncated": config.filter_reject_all_truncated,
            "heuristic": config.heuristic,
            "physics": config.physics,
        }
    output = save_task_bank(
        config.output,
        tasks,
        seed=config.seed,
        physics=config.physics,
        heuristic=config.heuristic,
        max_n_ally=config.max_n_ally if config.programmatic_ratio > 0 else None,
        max_n_enemy=config.max_n_enemy if config.programmatic_ratio > 0 else None,
        max_n_zone=config.max_n_zone if config.programmatic_ratio > 0 else None,
        filter_protocol=filter_protocol,
    )
    saved_bank = load_task_bank(output)
    schema = saved_bank["manifest"]["schema"]
    limits = (schema["max_n_ally"], schema["max_n_enemy"], schema["max_n_zone"])
    print(f"Saved {len(tasks)} tasks to {output} with schema limits {limits}.")


if __name__ == "__main__":
    main()
