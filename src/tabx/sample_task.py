"""Sample, save, and load fixed TABX task banks.

The task-bank JSON format stores many tasks in one file.  Every task keeps the
same ``scenario`` and ``zone_scenario`` schema used by the existing scenario
assets, so loading a bank only adds batching/padding around TABX's normal
``env.reset(key, env_params)`` interface.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import multiprocessing
import os
import random
import time
from collections import Counter
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from contextlib import redirect_stdout
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

# The sampling CLI is CPU-only. Configure JAX before importing it so an
# installed CUDA plugin is never initialized in either the parent process or
# the forkserver workers. Keep library imports neutral for GPU callers that
# reuse the task-bank helpers from this module.
if __name__ == "__main__":
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_SKIP_CUDA_CONSTRAINTS_CHECK"] = "1"

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
    crossover_parent_tasks,
    generate_free_task,
    generate_programmatic_task,
    mutate_parent_task,
)
from src.tabx.units import get_all_unit_spec

TASK_BANK_VERSION = "1.0"
SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"
SUPPORTED_TRANSFORMS = ("identity", "mirror_x", "mirror_y", "rotate_180")
_EVALUATION_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_EVALUATION_CACHE_MAX_ENTRIES = 512
_CPU_WORKER_ID = -1


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
    open_ended_generation: bool = True
    free_generation_ratio: float = 0.35
    parent_mutation_ratio: float = 0.25
    parent_crossover_ratio: float = 0.10
    parameter_distance_threshold: float = 0.04
    behavior_distance_threshold: float = 0.03
    map_elites: bool = True
    map_elites_per_cell: int = 1
    surrogate_enabled: bool = True
    surrogate_min_samples: int = 8
    surrogate_ridge: float = 1.0
    acquisition_min_score: float = 0.25
    acquisition_novelty_weight: float = 0.55
    acquisition_uncertainty_weight: float = 0.45
    continuous_collection: bool = False
    collection_candidate_budget: int = 1000
    collection_rollout_budget: int = 0
    collection_time_budget_seconds: float = 0.0
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
    distance_scales: tuple[float, ...] = (0.8, 1.2)
    spread_buckets: tuple[str, ...] = SPREAD_BUCKETS
    spread_scales: tuple[float, ...] = (0.8, 1.2)
    zone_archetypes: tuple[str, ...] = ZONE_ARCHETYPES
    zone_intensity_buckets: tuple[str, ...] = ZONE_INTENSITY_BUCKETS
    zone_relation_buckets: tuple[str, ...] = ZONE_RELATION_BUCKETS
    include_challenges: bool = True
    transforms: tuple[str, ...] = SUPPORTED_TRANSFORMS
    map_scales: tuple[float, ...] = (0.85, 1.0, 1.15)
    stat_scales: tuple[float, ...] = (0.9, 1.0, 1.1)
    # Legacy master switch. Per-dimension overrides below take precedence when
    # set, so callers can keep buckets for one axis and sample another freely.
    continuous_sampling: bool = True
    continuous_map_sampling: bool | None = None
    continuous_stat_sampling: bool | None = None
    continuous_distance_sampling: bool | None = None
    continuous_spread_sampling: bool | None = None
    continuous_zone_strength_sampling: bool | None = None
    # buckets | continuous | mixed. ``mixed`` chooses per generated task.
    health_fraction_sampling: str = "continuous"
    # Keep ally/enemy stat scales coupled so balance comes from roster matching,
    # while scenario-type buckets carry most of the diversity.
    couple_stat_scales: bool = True
    zone_strength_scales: tuple[float, ...] = (0.75, 1.0, 1.25)
    static_balance_repair: bool = True
    static_balance_max_gap: float = 0.20
    static_repair_max_scale: float = 1.55
    simulation_balance_repair: bool = True
    simulation_repair_rounds: int = 4
    simulation_repair_max_scale: float = 1.55
    balance_hp_margin_weight: float = 0.20
    balance_damage_share_weight: float = 0.15
    balance_truncation_weight: float = 0.10
    balance_no_interaction_weight: float = 0.20
    max_truncation_rate: float = 0.20
    max_no_interaction_rate: float = 0.0
    static_engagement_filter: bool = True
    static_engagement_dt: float = 0.5
    static_engagement_max_step_fraction: float = 0.45
    static_stomp_filter: bool = True
    static_stomp_min_episode_fraction: float = 0.05
    static_stomp_min_ttk_ratio: float = 2.5
    # Prefer under-covered (composition, layout, distance, spread, zone, ...) buckets.
    bucket_coverage: bool = True
    bucket_coverage_candidates: int = 16
    physics: str = "default"
    heuristic: str = "expert"
    # Keep only tasks whose both-team heuristic win rate falls in this band.
    # With epsilon=0 the policy is nearly deterministic, so win rates collapse to
    # 0/1; use a small positive evaluation epsilon so intermediate rates are possible.
    filter_by_win_rate: bool = True
    win_rate_min: float = 0.4
    win_rate_max: float = 0.6
    # Shared evaluation protocol. A discovery, B confirmation, and standalone
    # eval runs should use the same epsilon.
    evaluation_epsilon: float = 0.05
    # A: adaptive discovery/repair on one expandable seed prefix.
    discovery_seed_offset: int = 0
    discovery_seed_stages: tuple[int, ...] = (8, 16, 32, 64)
    discovery_min_accept_seeds: int = 32
    discovery_candidate_win_rate_min: float = 0.30
    discovery_candidate_win_rate_max: float = 0.70
    discovery_max_abs_hp_margin: float = 0.30
    # Deprecated CLI alias. Existing scripts may still pass it, but it must
    # agree with evaluation_epsilon so A and B cannot silently diverge.
    filter_epsilon: float | None = None
    filter_max_episode_steps: int = 512
    filter_reject_all_truncated: bool = True
    filter_max_attempts: int = 20
    # B: frozen-task confirmation on fresh paired original/flipped seeds.
    confirmation_enabled: bool = True
    confirmation_seed_stages: tuple[int, ...] = (32, 64, 128)
    confirmation_min_accept_seeds: int = 64
    confirmation_balance_probability: float = 0.85
    confirmation_early_reject_probability: float = 0.15
    confirmation_seed_offset: int = 100_000
    confirmation_win_rate_min: float = 0.40
    confirmation_win_rate_max: float = 0.60
    confirmation_team_win_rate_min: float = 0.45
    confirmation_team_win_rate_max: float = 0.55
    confirmation_max_side_bias: float = 0.20
    confirmation_max_abs_hp_margin: float = 0.20
    confirmation_hp_ci_max_abs_bound: float = 0.30
    directed_repair_distance_factor: float = 0.75
    directed_repair_map_factor: float = 0.90
    directed_repair_damage_factor: float = 1.20
    directed_repair_max_offense_scale: float = 1.50
    adaptive_sampling: bool = True
    adaptive_sampling_learning_rate: float = 0.25
    adaptive_sampling_exploration: float = 0.15
    # Number of CPU cores selected with ``--cpu-cores``. Each sampler process
    # is restricted to one CPU thread, so this is also the worker count.
    cpu_cores: int = 1
    # Completed candidates are validated against an immutable archive snapshot
    # in micro-batches. Zero selects ``2 * cpu_cores`` automatically.
    commit_batch_size: int = 0
    commit_workers: int = 3
    commit_timeout_ms: int = 100


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _normalize_scenario_radius_field(scenario: dict[str, Any]) -> dict[str, Any]:
    """Accept the correctly spelled radius alias at every public boundary.

    The environment struct still uses the historical ``body_radiuss`` field,
    so normalization keeps that spelling internally. If both names are
    supplied they must agree; this prevents silently loading ambiguous data.
    """

    legacy = scenario.get("body_radiuss")
    canonical = scenario.get("body_radii")
    if legacy is None and canonical is None:
        return scenario
    if legacy is not None and canonical is not None:
        if not np.array_equal(np.asarray(legacy), np.asarray(canonical)):
            raise ValueError("scenario body_radii and body_radiuss disagree.")
    normalized = dict(scenario)
    normalized["body_radiuss"] = copy.deepcopy(
        legacy if legacy is not None else canonical
    )
    normalized.pop("body_radii", None)
    return normalized


def normalize_task_schema(task: dict[str, Any]) -> dict[str, Any]:
    """Normalize supported task aliases in place and return ``task``."""

    if "scenario" in task:
        task["scenario"] = _normalize_scenario_radius_field(task["scenario"])
    return task


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
    scenario_data = _normalize_scenario_radius_field(task["scenario"])
    scenario = VectorizedScenario(**_to_jax_array(scenario_data))
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
        normalize_task_schema(task)
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
    generator_protocol: dict[str, Any] | None = None,
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
            "radius_field": "body_radii",
            "accepted_radius_aliases": ["body_radiuss", "body_radii"],
        },
    }
    if filter_protocol is not None:
        manifest["filter_protocol"] = filter_protocol
    if generator_protocol is not None:
        manifest["generator_protocol"] = generator_protocol
    stored_tasks = copy.deepcopy(list(tasks))
    for stored_task in stored_tasks:
        scenario = stored_task["scenario"]
        # Write the canonical spelling for external tools and retain the
        # historical alias so older TABX consumers remain compatible.
        scenario["body_radii"] = copy.deepcopy(scenario["body_radiuss"])
    bank = {
        "schema_version": TASK_BANK_VERSION,
        "manifest": manifest,
        "tasks": stored_tasks,
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


def _sample_scale(rng: random.Random, values: Sequence[float], continuous: bool) -> float:
    """Sample within the configured range while retaining discrete-mode compatibility."""

    if not values:
        raise ValueError("Scale values must not be empty.")
    if continuous and len(values) > 1:
        return rng.uniform(float(min(values)), float(max(values)))
    return float(rng.choice(tuple(values)))


def estimate_dynamic_team_values(task: dict[str, Any]) -> dict[str, Any]:
    """Cheap context-aware strength estimate used only for candidate pre-balancing."""

    scenario = task["scenario"]
    specs = {key: np.asarray(value, dtype=float).reshape(-1) for key, value in get_all_unit_spec().items()}
    teams = np.asarray(scenario["teams"], dtype=int).reshape(-1)
    unit_ids = np.asarray(scenario["unit_ids"], dtype=int).reshape(-1)
    health = np.asarray(scenario["healths"], dtype=float).reshape(-1)
    damage = np.asarray(scenario["attack_damages"], dtype=float).reshape(-1)
    cooldown = np.maximum(np.asarray(scenario["attack_cooldowns"], dtype=float).reshape(-1), 1e-6)
    speed = np.asarray(scenario["speeds"], dtype=float).reshape(-1)
    attack_range = np.asarray(scenario["attack_ranges"], dtype=float).reshape(-1)
    positions = np.asarray(scenario["positions"], dtype=float)

    base_hp = np.maximum(specs["healths"][unit_ids], 1e-6)
    base_damage = specs["attack_damages"][unit_ids]
    base_cooldown = np.maximum(specs["attack_cooldown"][unit_ids], 1e-6)
    base_rate = np.maximum(np.abs(base_damage) / base_cooldown, 1e-6)
    current_rate = np.maximum(np.abs(damage) / cooldown, 1e-6)
    hp_ratio = np.maximum(health / base_hp, 1e-3)
    output_ratio = np.maximum(current_rate / base_rate, 1e-3)
    prices = specs["prices"][unit_ids]
    # Weighted geometric mean: bounded growth for simultaneous stat changes.
    unit_base = prices * np.power(hp_ratio, 0.45) * np.power(output_ratio, 0.55)

    centers = [positions[teams == team].mean(axis=0) for team in (0, 1)]
    initial_distance = float(np.linalg.norm(centers[0] - centers[1]))
    map_width = float(task.get("grid_info", {}).get("max_field_width", 121.0))
    distance_factor = float(np.clip(initial_distance / max(map_width * 0.35, 1.0), 0.0, 1.5))
    zone_types = np.asarray(task["zone_scenario"]["zone_type"], dtype=int).reshape(-1)
    zone_positions = np.asarray(task["zone_scenario"]["position"], dtype=float).reshape(-1, 2)
    zone_axes = np.asarray(task["zone_scenario"]["axes"], dtype=float).reshape(-1, 2)

    values: list[float] = []
    details: list[dict[str, float]] = []
    for team in (0, 1):
        mask = teams == team
        raw = float(unit_base[mask].sum())
        default_speed = np.maximum(specs["speeds"][unit_ids[mask]], 1e-6)
        default_range = np.maximum(specs["attack_ranges"][unit_ids[mask]], 1e-6)
        speed_log = float(np.average(np.log(np.maximum(speed[mask] / default_speed, 1e-3)), weights=unit_base[mask]))
        range_log = float(np.average(np.log(np.maximum(attack_range[mask] / default_range, 1e-3)), weights=unit_base[mask]))
        mobility_bonus = 0.08 * speed_log * (0.5 + distance_factor)
        range_bonus = 0.12 * range_log * distance_factor
        terrain_bonus = 0.0
        for zone_type, center, axes in zip(zone_types, zone_positions, zone_axes):
            if zone_type == 0:
                continue
            inside = (((positions[mask] - center) / np.maximum(axes, 1e-6)) ** 2).sum(axis=1) <= 1.0
            exposure = float(inside.mean()) if len(inside) else 0.0
            if zone_type in (1, 3):
                terrain_bonus -= 0.10 * exposure
            elif zone_type == 2:
                terrain_bonus += 0.04 * exposure
        healer_ratio = float((damage[mask] < 0).mean())
        combat_hp = float(health[mask & (damage >= 0)].sum())
        healer_synergy = min(0.15, healer_ratio * math.log1p(max(combat_hp, 0.0)) * 0.02)
        context = float(np.clip(1.0 + mobility_bonus + range_bonus + terrain_bonus + healer_synergy, 0.5, 2.0))
        values.append(raw * context)
        details.append({"raw": raw, "context_multiplier": context, "healer_ratio": healer_ratio})
    gap = abs(values[0] - values[1]) / max(max(values), 1.0)
    return {"ally": values[0], "enemy": values[1], "relative_gap": gap, "details": details}


def _scale_team_combat(task: dict[str, Any], team: int, scale: float) -> None:
    """Scale HP and attack/heal throughput by sqrt(scale), preserving unit roles."""

    scenario = task["scenario"]
    teams = np.asarray(scenario["teams"], dtype=int).reshape(-1)
    factor = math.sqrt(max(scale, 1e-6))
    for field in ("healths", "attack_damages"):
        values = np.asarray(scenario[field], dtype=float)
        values[teams == team] *= factor
        scenario[field] = values.tolist()


def static_balance_repair(task: dict[str, Any], config: SampleConfig) -> dict[str, Any]:
    """Move a candidate toward equal estimated strength before expensive rollouts."""

    before = estimate_dynamic_team_values(task)
    record: dict[str, Any] = {"before": before, "applied": False}
    if before["relative_gap"] > config.static_balance_max_gap:
        weaker = 0 if before["ally"] < before["enemy"] else 1
        stronger_value = before["enemy"] if weaker == 0 else before["ally"]
        weaker_value = before["ally"] if weaker == 0 else before["enemy"]
        scale = float(
            np.clip(stronger_value / max(weaker_value, 1e-6), 1.0, config.static_repair_max_scale)
        )
        _scale_team_combat(task, weaker, scale)
        record.update({"applied": True, "team": weaker, "scale": scale})
    record["after"] = estimate_dynamic_team_values(task)
    return record


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


def _evaluation_epsilon(config: SampleConfig) -> float:
    """Return the single epsilon used by every sampling evaluation phase."""

    if config.filter_epsilon is not None and not math.isclose(
        float(config.filter_epsilon),
        float(config.evaluation_epsilon),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "filter_epsilon is a deprecated alias and must equal evaluation_epsilon."
        )
    return float(config.evaluation_epsilon)


def _discovery_num_seeds(config: SampleConfig) -> int:
    """Return the maximum A-stage rollout budget per task version."""

    return max(int(value) for value in config.discovery_seed_stages)


def _phase_seed_base(task: dict[str, Any], seed: int) -> int:
    """Hash a worker seed and canonical task identity into a rollout seed."""

    payload = f"{int(seed)}:{_task_hash(task)}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % (2**31)


def _parameter_signature(task: dict[str, Any]) -> np.ndarray:
    """Fixed-size normalized descriptor for approximate parameter-level deduplication."""

    scenario = task["scenario"]
    teams = np.asarray(scenario["teams"], dtype=int).reshape(-1)
    ids = np.asarray(scenario["unit_ids"], dtype=int).reshape(-1)
    positions = np.asarray(scenario["positions"], dtype=float)
    health = np.asarray(scenario["healths"], dtype=float).reshape(-1)
    damage = np.asarray(scenario["attack_damages"], dtype=float).reshape(-1)
    speed = np.asarray(scenario["speeds"], dtype=float).reshape(-1)
    width = float(task.get("grid_info", {}).get("max_field_width", 121.0))
    height = float(task.get("grid_info", {}).get("max_field_height", 78.0))
    diagonal = max(math.hypot(width, height), 1.0)
    features: list[float] = []
    for team in (0, 1):
        mask = teams == team
        counts = np.bincount(ids[mask], minlength=9).astype(float)
        features.extend((counts / max(counts.sum(), 1.0)).tolist())
        team_positions = positions[mask]
        center = team_positions.mean(axis=0)
        spread = float(np.linalg.norm(team_positions - center, axis=1).mean())
        features.extend(
            [
                len(team_positions) / 10.0,
                center[0] / max(width, 1.0),
                center[1] / max(height, 1.0),
                spread / diagonal,
                float(np.mean(health[mask])) / 500.0,
                float(np.mean(np.abs(damage[mask]))) / 100.0,
                float(np.mean(speed[mask])) / 2.0,
            ]
        )
    centers = [positions[teams == team].mean(axis=0) for team in (0, 1)]
    features.append(float(np.linalg.norm(centers[0] - centers[1])) / diagonal)
    zones = task["zone_scenario"]
    zone_types = np.asarray(zones["zone_type"], dtype=int).reshape(-1)
    features.append(float(zones["n_zone"]) / 4.0)
    for zone_type in (1, 2, 3):
        features.append(float(np.count_nonzero(zone_types == zone_type)) / 4.0)
    if int(zones["n_zone"]) > 0:
        axes = np.asarray(zones["axes"], dtype=float).reshape(-1, 2)
        coverage = float(np.sum(math.pi * axes[:, 0] * axes[:, 1]) / max(width * height, 1.0))
    else:
        coverage = 0.0
    features.append(min(coverage, 2.0) / 2.0)
    return np.asarray(features, dtype=float)


def _behavior_signature(result: dict[str, Any], max_episode_steps: int) -> np.ndarray:
    first = result.get("first_interaction_step", {}).get("mean")
    first_value = 1.0 if first is None else float(first) / max(max_episode_steps, 1)
    damage_share = result.get("damage_share", [0.5, 0.5])
    return np.asarray(
        [
            float(result["win_rate"]),
            (float(result["hp_margin"]["mean"]) + 1.0) * 0.5,
            float(damage_share[0]),
            float(result["truncation_rate"]),
            float(result["quality_flags"]["no_interaction_rate"]),
            float(result["episode_length"]["mean"]) / max(max_episode_steps, 1),
            first_value,
        ],
        dtype=float,
    )


def _flip_task_for_validation(task: dict[str, Any]) -> dict[str, Any]:
    """Swap team identities while preserving physical placement and task attributes."""

    flipped = copy.deepcopy(task)
    scenario = flipped["scenario"]
    teams = np.asarray(scenario["teams"], dtype=int).reshape(-1)
    order = np.concatenate([np.flatnonzero(teams == 1), np.flatnonzero(teams == 0)])
    for field in scenario:
        scenario[field] = np.asarray(scenario[field])[order].tolist()
    scenario["teams"] = (1 - np.asarray(scenario["teams"], dtype=int)).tolist()
    return flipped


def _nearest_distance(signature: np.ndarray, archive: Sequence[np.ndarray]) -> float:
    if not archive:
        return math.inf
    return min(float(np.sqrt(np.mean(np.square(signature - other)))) for other in archive)


def _nearest_distances_batch(
    signatures: Sequence[np.ndarray], archive: Sequence[np.ndarray]
) -> np.ndarray:
    """Return nearest RMS distances using one vectorized archive query."""

    if not signatures:
        return np.empty(0, dtype=float)
    if not archive:
        return np.full(len(signatures), math.inf, dtype=float)
    queries = np.asarray(signatures, dtype=float)
    references = np.asarray(archive, dtype=float)
    # Avoid materializing [batch, archive, dimensions]. The squared-distance
    # identity keeps memory bounded while BLAS handles the expensive product.
    dimensions = max(queries.shape[1], 1)
    query_norm = np.sum(np.square(queries), axis=1, keepdims=True)
    reference_norm = np.sum(np.square(references), axis=1)[None, :]
    squared = query_norm + reference_norm - 2.0 * (queries @ references.T)
    return np.sqrt(np.maximum(np.min(squared, axis=1), 0.0) / dimensions)


def _bucket(value: float, boundaries: Sequence[float]) -> int:
    return int(np.searchsorted(np.asarray(boundaries, dtype=float), value, side="right"))


def _map_elites_cell(
    task: dict[str, Any], result: dict[str, Any] | None = None
) -> tuple[int, ...]:
    """Interpretable MAP-Elites cell spanning roster, geometry, terrain, and behavior."""

    scenario = task["scenario"]
    teams = np.asarray(scenario["teams"], dtype=int).reshape(-1)
    ids = np.asarray(scenario["unit_ids"], dtype=int).reshape(-1)
    positions = np.asarray(scenario["positions"], dtype=float)
    counts = [int(np.count_nonzero(teams == team)) for team in (0, 1)]
    ranged = [float(np.mean(np.isin(ids[teams == team], (4, 5, 6)))) for team in (0, 1)]
    healers = [float(np.mean(np.isin(ids[teams == team], (7, 8)))) for team in (0, 1)]
    centers = [positions[teams == team].mean(axis=0) for team in (0, 1)]
    width = float(task.get("grid_info", {}).get("max_field_width", 121.0))
    distance = float(np.linalg.norm(centers[0] - centers[1])) / max(width, 1.0)
    n_zone = int(task["zone_scenario"]["n_zone"])
    if result is None:
        behavior_bins = (-1, -1)
    else:
        duration = float(result["episode_length"]["mean"])
        damage_balance = abs(float(result.get("damage_share", [0.5])[0]) - 0.5)
        behavior_bins = (_bucket(duration, (128, 256, 384)), _bucket(damage_balance, (0.1, 0.25)))
    return (
        _bucket(counts[0], (3, 6)),
        _bucket(counts[1], (3, 6)),
        _bucket(ranged[0], (0.25, 0.6)),
        _bucket(ranged[1], (0.25, 0.6)),
        _bucket(healers[0], (0.01, 0.3)),
        _bucket(healers[1], (0.01, 0.3)),
        _bucket(distance, (0.2, 0.45)),
        _bucket(n_zone, (0.5, 2.5)),
        *behavior_bins,
    )


class _OnlineOutcomePredictor:
    """Small dependency-free ridge surrogate for win rate and balance quality."""

    def __init__(self, min_samples: int, ridge: float):
        self.min_samples = min_samples
        self.ridge = ridge
        self.features: list[np.ndarray] = []
        self.win_rates: list[float] = []
        self.qualities: list[float] = []

    @property
    def ready(self) -> bool:
        return len(self.features) >= self.min_samples

    def update(self, feature: np.ndarray, result: dict[str, Any], quality: float) -> None:
        self.features.append(np.asarray(feature, dtype=float))
        self.win_rates.append(float(result["win_rate"]))
        self.qualities.append(float(quality))

    def update_many(
        self,
        samples: Sequence[tuple[np.ndarray, dict[str, Any], float]],
    ) -> None:
        """Publish one accepted micro-batch without fitting between samples."""

        for feature, result, quality in samples:
            self.update(feature, result, quality)

    def _predict_target(self, feature: np.ndarray, targets: Sequence[float]) -> float:
        x = np.asarray(self.features, dtype=float)
        x = np.column_stack([x, np.ones(len(x))])
        query = np.append(np.asarray(feature, dtype=float), 1.0)
        regularizer = np.eye(x.shape[1]) * self.ridge
        regularizer[-1, -1] = 0.0
        weights = np.linalg.solve(x.T @ x + regularizer, x.T @ np.asarray(targets))
        return float(query @ weights)

    def predict(self, feature: np.ndarray) -> dict[str, float] | None:
        if not self.ready:
            return None
        win_rate = float(np.clip(self._predict_target(feature, self.win_rates), 0.0, 1.0))
        quality = max(0.0, self._predict_target(feature, self.qualities))
        nearest = _nearest_distance(feature, self.features)
        uncertainty = float(np.clip(nearest / 0.20, 0.0, 1.0))
        return {
            "win_rate": win_rate,
            "balance_error": quality,
            "uncertainty": uncertainty,
        }

    def predict_many(
        self, features: Sequence[np.ndarray]
    ) -> list[dict[str, float] | None]:
        """Predict a snapshot batch with one ridge solve per target."""

        if not features:
            return []
        if not self.ready:
            return [None] * len(features)
        training = np.asarray(self.features, dtype=float)
        design = np.column_stack([training, np.ones(len(training))])
        queries = np.column_stack(
            [np.asarray(features, dtype=float), np.ones(len(features))]
        )
        regularizer = np.eye(design.shape[1]) * self.ridge
        regularizer[-1, -1] = 0.0
        normal = design.T @ design + regularizer
        targets = np.column_stack([self.win_rates, self.qualities])
        weights = np.linalg.solve(normal, design.T @ targets)
        predictions = queries @ weights
        uncertainties = np.clip(
            _nearest_distances_batch(features, self.features) / 0.20,
            0.0,
            1.0,
        )
        return [
            {
                "win_rate": float(np.clip(values[0], 0.0, 1.0)),
                "balance_error": max(0.0, float(values[1])),
                "uncertainty": float(uncertainty),
            }
            for values, uncertainty in zip(predictions, uncertainties)
        ]


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (int, float, bool)):
        return math.isfinite(float(value))
    return True


def _continuous_option(config: SampleConfig, name: str) -> bool:
    """Resolve a per-axis continuous flag with the legacy master fallback."""

    value = getattr(config, name)
    return config.continuous_sampling if value is None else bool(value)


def _health_fraction_buckets(
    rng: random.Random, config: SampleConfig
) -> tuple[float, ...]:
    """Build the HP fraction source without coupling it to other dimensions."""

    mode = config.health_fraction_sampling
    if mode == "mixed":
        mode = "continuous" if rng.random() < 0.5 else "buckets"
    if mode == "buckets":
        return config.health_frac_buckets
    if mode == "continuous":
        return tuple(
            rng.uniform(config.health_frac_min, config.health_frac_max)
            for _ in range(32)
        )
    raise ValueError(
        "health_fraction_sampling must be 'buckets', 'continuous', or 'mixed'."
    )


def estimate_static_engagement(
    task: dict[str, Any], *, dt: float = 0.5
) -> dict[str, Any]:
    """Conservatively estimate the earliest possible hostile interaction.

    This is a cheap reachability/contact-time screen, not a path planner. TABX
    zones do not form solid obstacles, so the straight-line lower bound is a
    useful way to reject immobile/non-offensive or excessively distant teams
    before paying for JAX rollouts.
    """

    scenario = _normalize_scenario_radius_field(task["scenario"])
    positions = np.asarray(scenario["positions"], dtype=float)
    teams = np.asarray(scenario["teams"], dtype=int).reshape(-1)
    alive = np.asarray(scenario["is_alive"], dtype=bool).reshape(-1)
    disabled = np.asarray(scenario["is_disabled"], dtype=bool).reshape(-1)
    active = alive & ~disabled
    radii = np.asarray(scenario["body_radiuss"], dtype=float).reshape(-1)
    speeds = np.maximum(np.asarray(scenario["speeds"], dtype=float).reshape(-1), 0.0)
    ranges = np.maximum(
        np.asarray(scenario["attack_ranges"], dtype=float).reshape(-1), 0.0
    )
    damages = np.asarray(scenario["attack_damages"], dtype=float).reshape(-1)
    cooldowns = np.maximum(
        np.asarray(scenario["attack_cooldowns"], dtype=float).reshape(-1), 1e-6
    )
    healths = np.maximum(np.asarray(scenario["healths"], dtype=float).reshape(-1), 0.0)

    # Apply spawn-local swamp slowdown. This is intentionally conservative;
    # later path changes remain the simulator's responsibility.
    zones = task.get("zone_scenario", {})
    if int(zones.get("n_zone", 0)):
        zone_types = np.asarray(zones["zone_type"], dtype=int).reshape(-1)
        centers = np.asarray(zones["position"], dtype=float).reshape(-1, 2)
        axes = np.asarray(zones["axes"], dtype=float).reshape(-1, 2)
        effects = np.asarray(zones["effect_value"], dtype=float).reshape(-1)
        for index in np.flatnonzero(zone_types == 3):
            inside = np.sum(((positions - centers[index]) / axes[index]) ** 2, axis=1) <= 1.0
            speeds[inside] *= np.clip(1.0 - effects[index], 0.0, 1.0)

    ally = np.flatnonzero(active & (teams == 0))
    enemy = np.flatnonzero(active & (teams == 1))
    best_steps = math.inf
    best_pair: list[int] | None = None
    already_in_range = False
    for first in ally:
        for second in enemy:
            attack_range = max(
                ranges[first] if damages[first] > 0 else 0.0,
                ranges[second] if damages[second] > 0 else 0.0,
            )
            if attack_range <= 0.0:
                continue
            center_distance = float(np.linalg.norm(positions[first] - positions[second]))
            gap = max(0.0, center_distance - radii[first] - radii[second] - attack_range)
            closing_speed = speeds[first] + speeds[second]
            if gap <= 1e-8:
                steps = 0.0
                already_in_range = True
            elif closing_speed <= 1e-8 or dt <= 0:
                continue
            else:
                steps = gap / (closing_speed * dt)
            if steps < best_steps:
                best_steps = steps
                best_pair = [int(first), int(second)]

    team_hp = [float(healths[indexes].sum()) for indexes in (ally, enemy)]
    team_dps = [
        float((np.maximum(damages[indexes], 0.0) / cooldowns[indexes]).sum())
        for indexes in (ally, enemy)
    ]
    elimination_steps = [
        (
            team_hp[0] / max(team_dps[1] * dt, 1e-8)
            if team_dps[1] > 0
            else math.inf
        ),
        (
            team_hp[1] / max(team_dps[0] * dt, 1e-8)
            if team_dps[0] > 0
            else math.inf
        ),
    ]
    finite_ttk = [value for value in elimination_steps if math.isfinite(value)]
    ttk_ratio = (
        max(finite_ttk) / max(min(finite_ttk), 1e-6)
        if len(finite_ttk) == 2
        else math.inf
    )
    earliest_elimination = min(finite_ttk) if finite_ttk else math.inf
    if math.isfinite(best_steps) and math.isfinite(earliest_elimination):
        earliest_elimination += best_steps

    return {
        "reachable": math.isfinite(best_steps),
        "estimated_first_contact_steps": (
            float(best_steps) if math.isfinite(best_steps) else None
        ),
        "best_pair": best_pair,
        "already_in_range": already_in_range,
        "dt": float(dt),
        "method": "straight_line_lower_bound_with_spawn_swamp",
        "team_hp": team_hp,
        "team_dps": team_dps,
        "estimated_team_elimination_steps": [
            float(value) if math.isfinite(value) else None for value in elimination_steps
        ],
        "estimated_earliest_elimination_steps": (
            float(earliest_elimination) if math.isfinite(earliest_elimination) else None
        ),
        "estimated_ttk_ratio": float(ttk_ratio) if math.isfinite(ttk_ratio) else None,
    }


def validate_task(task: dict[str, Any], *, collision_margin: float = 1e-5) -> None:
    """Raise ``ValueError`` when a task cannot safely enter the current environment."""

    if "scenario" not in task or "zone_scenario" not in task:
        raise ValueError("Task is missing scenario data.")
    if not _all_finite(task["scenario"]) or not _all_finite(task["zone_scenario"]):
        raise ValueError("Task contains NaN or infinite values.")

    scenario = _normalize_scenario_radius_field(task["scenario"])
    task["scenario"] = scenario
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


_BUCKET_DIMENSIONS = (
    "composition",
    "layout",
    "distance",
    "spread",
    "zone",
    "zone_intensity",
    "zone_relation",
    "match_mode",
)


def _record_bucket_feedback(
    feedback: dict[tuple[str, str], Counter[str]],
    bucket: tuple[str, ...] | None,
    outcome: str,
) -> None:
    if bucket is None:
        return
    for dimension, value in zip(_BUCKET_DIMENSIONS, bucket):
        feedback.setdefault((dimension, value), Counter())[outcome] += 1


def _bucket_feedback_utility(
    feedback: dict[tuple[str, str], Counter[str]], bucket: tuple[str, ...]
) -> float:
    utilities = []
    for dimension, value in zip(_BUCKET_DIMENSIONS, bucket):
        stats = feedback.get((dimension, value), Counter())
        generated = stats["generated"]
        success = (stats["accepted"] + 1.0) / (generated + 2.0)
        static_failures = stats["generation_failure"] + stats["static_invalid"]
        interaction_failures = sum(
            count
            for reason, count in stats.items()
            if "truncation" in reason or "interaction" in reason
        )
        imbalance_failures = sum(
            count for reason, count in stats.items() if "win_rate" in reason
        )
        penalty = (
            1.0 * static_failures
            + 0.7 * interaction_failures
            + 0.35 * imbalance_failures
        ) / max(generated, 1)
        utilities.append(success / (1.0 + penalty))
    return float(np.mean(utilities)) if utilities else 0.5


def _draw_source_kind(
    rng: random.Random,
    config: SampleConfig,
    feedback: dict[str, Counter[str]],
) -> str:
    base_weights = {
        "crossover": config.parent_crossover_ratio,
        "mutation": config.parent_mutation_ratio,
        "free": config.free_generation_ratio,
        "base": max(
            0.0,
            1.0
            - config.parent_crossover_ratio
            - config.parent_mutation_ratio
            - config.free_generation_ratio,
        ),
    }
    weighted: list[tuple[str, float]] = []
    for kind, base in base_weights.items():
        stats = feedback.get(kind, Counter())
        utility = (stats["accepted"] + 1.0) / (stats["generated"] + 2.0)
        multiplier = (
            1.0 - config.adaptive_sampling_learning_rate
            + config.adaptive_sampling_learning_rate * 2.0 * utility
        )
        weight = base * max(multiplier, config.adaptive_sampling_exploration)
        weighted.append((kind, weight))
    total = sum(weight for _, weight in weighted)
    if total <= 0:
        return "base"
    target = rng.random() * total
    cumulative = 0.0
    for kind, weight in weighted:
        cumulative += weight
        if target <= cumulative:
            return kind
    return weighted[-1][0]


def _pick_scenario_buckets(
    rng: random.Random,
    config: SampleConfig,
    coverage: Counter[tuple[str, ...]],
    feedback: dict[tuple[str, str], Counter[str]] | None = None,
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

    if not config.bucket_coverage and not config.adaptive_sampling:
        return draw()

    feedback = feedback or {}
    best = draw()

    def score(candidate: tuple[str, ...]) -> float:
        coverage_score = 1.0 / (1.0 + coverage[candidate])
        adaptive_score = _bucket_feedback_utility(feedback, candidate)
        learned = (
            (1.0 - config.adaptive_sampling_learning_rate) * coverage_score
            + config.adaptive_sampling_learning_rate * adaptive_score
        )
        return (
            (1.0 - config.adaptive_sampling_exploration) * learned
            + config.adaptive_sampling_exploration * rng.random()
        )

    best_score = score(best)
    n_draw = max(1, config.bucket_coverage_candidates)
    for _ in range(n_draw - 1):
        candidate = draw()
        candidate_score = score(candidate)
        if candidate_score > best_score:
            best = candidate
            best_score = candidate_score
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
    max_truncation_rate: float = 1.0,
    max_no_interaction_rate: float = 1.0,
    max_n_ally: int | None = None,
    max_n_enemy: int | None = None,
    max_n_zone: int | None = None,
    seed_stages: Sequence[int] | None = None,
    min_accept_seeds: int | None = None,
    candidate_win_rate_min: float | None = None,
    candidate_win_rate_max: float | None = None,
    max_abs_hp_margin: float = 1.0,
) -> tuple[bool, dict[str, Any]]:
    """Adaptively evaluate A-stage seeds, running only newly requested episodes."""

    # Lazy import avoids a circular dependency with eval_task.
    from src.tabx.eval_task import aggregate_episode_samples, evaluate_tasks

    stages = sorted(
        set(
            int(value)
            for value in (
                seed_stages
                if seed_stages is not None
                else (num_seeds,)
            )
            if value is not None and int(value) > 0
        )
    )
    if not stages:
        raise ValueError("At least one positive A-stage seed count is required.")
    min_accept = int(min_accept_seeds or min(num_seeds, stages[-1]))
    candidate_min = float(
        win_rate_min if candidate_win_rate_min is None else candidate_win_rate_min
    )
    candidate_max = float(
        win_rate_max if candidate_win_rate_max is None else candidate_win_rate_max
    )
    accumulated_samples: dict[str, list[Any]] | None = None
    completed = 0
    stage_history: list[dict[str, Any]] = []

    def evaluate_chunk(chunk_seed: int, seed_count: int) -> dict[str, Any]:
        cache_key = (
            _task_hash(task),
            physics,
            heuristic,
            seed_count,
            chunk_seed,
            epsilon,
            max_episode_steps,
            max_n_ally,
            max_n_enemy,
            max_n_zone,
        )
        cached = _EVALUATION_CACHE.get(cache_key)
        if cached is not None:
            result = copy.deepcopy(cached)
            result["evaluation_cache_hit"] = True
            return result
        result = evaluate_tasks(
            [task],
            physics=physics,
            heuristic=heuristic,
            num_seeds=seed_count,
            seed=chunk_seed,
            epsilon_override=epsilon,
            max_episode_steps=max_episode_steps,
            max_n_ally=max_n_ally,
            max_n_enemy=max_n_enemy,
            max_n_zone=max_n_zone,
            include_episode_samples=True,
        )[0]
        result["evaluation_cache_hit"] = False
        if len(_EVALUATION_CACHE) >= _EVALUATION_CACHE_MAX_ENTRIES:
            _EVALUATION_CACHE.pop(next(iter(_EVALUATION_CACHE)))
        _EVALUATION_CACHE[cache_key] = copy.deepcopy(result)
        return result

    result: dict[str, Any] | None = None
    for target in stages:
        delta = target - completed
        if delta <= 0:
            continue
        chunk = evaluate_chunk(seed + completed, delta)
        chunk_samples = chunk.pop("episode_samples")
        if accumulated_samples is None:
            accumulated_samples = {key: list(values) for key, values in chunk_samples.items()}
        else:
            for key, values in chunk_samples.items():
                accumulated_samples[key].extend(values)
        completed = target
        result = aggregate_episode_samples(accumulated_samples, max_episode_steps)
        result.update(
            {
                "task_id": task.get("task_id", _task_hash(task)[:16]),
                "ally_heuristic": heuristic,
                "enemy_heuristic": heuristic,
                "evaluation_stage": "A",
                "n_rollouts_consumed": completed,
                "seed_stages_completed": list(stages[: stages.index(target) + 1]),
            }
        )
        win_rate = float(result["win_rate"])
        hp_margin = abs(float(result["hp_margin"]["mean"]))
        quality_reason = None
        if reject_all_truncated and result["quality_flags"]["all_truncated"]:
            quality_reason = "all_truncated"
        else:
            truncation_rate = float(result["truncation_rate"])
            truncations = int(round(truncation_rate * completed))
            if truncation_rate > max_truncation_rate and (
                completed == stages[-1]
                or _wilson_lower(truncations, completed) > max_truncation_rate
            ):
                quality_reason = "truncation_rate"
            no_interaction_rate = float(
                result["quality_flags"]["no_interaction_rate"]
            )
            no_interactions = int(round(no_interaction_rate * completed))
            if max_no_interaction_rate <= 0.0:
                if no_interactions > 0:
                    quality_reason = "no_interaction"
            elif no_interaction_rate > max_no_interaction_rate and (
                completed == stages[-1]
                or _wilson_lower(no_interactions, completed)
                > max_no_interaction_rate
            ):
                quality_reason = "no_interaction"
        stage_history.append(
            {
                "n_rollouts": completed,
                "win_rate": win_rate,
                "win_rate_ci95": result["win_rate_ci95"],
                "hp_margin": float(result["hp_margin"]["mean"]),
                "quality_reason": quality_reason,
            }
        )
        if quality_reason is not None:
            result["reject_reason"] = quality_reason
            result["adaptive_stage_history"] = stage_history
            return False, result
        ci_low, ci_high = map(float, result["win_rate_ci95"])
        if completed == stages[0] and (win_rate <= 0.125 or win_rate >= 0.875):
            result["reject_reason"] = "screening_win_rate_extreme"
            result["adaptive_stage_history"] = stage_history
            return False, result
        if ci_high < candidate_min or ci_low > candidate_max:
            result["reject_reason"] = "win_rate_confidently_outside_candidate_band"
            result["adaptive_stage_history"] = stage_history
            return False, result
        if (
            completed >= min_accept
            and win_rate_min <= win_rate <= win_rate_max
            and hp_margin <= max_abs_hp_margin
        ):
            result["reject_reason"] = None
            result["adaptive_stage_history"] = stage_history
            return True, result

    assert result is not None
    accepted = (
        candidate_min <= float(result["win_rate"]) <= candidate_max
        and abs(float(result["hp_margin"]["mean"])) <= max_abs_hp_margin
    )
    result["reject_reason"] = None if accepted else "A_candidate_band"
    result["adaptive_stage_history"] = stage_history
    return accepted, result


def _balance_error(result: dict[str, Any], config: SampleConfig) -> float:
    damage_share = float(result.get("damage_share", [0.5, 0.5])[0])
    return (
        abs(float(result["win_rate"]) - 0.5)
        + config.balance_hp_margin_weight * abs(float(result["hp_margin"]["mean"]))
        + config.balance_damage_share_weight * abs(damage_share - 0.5)
        + config.balance_truncation_weight * float(result["truncation_rate"])
        + config.balance_no_interaction_weight
        * float(result["quality_flags"]["no_interaction_rate"])
    )


def _move_teams_closer(task: dict[str, Any], factor: float) -> None:
    """Reduce only inter-team separation while preserving each formation."""

    scenario = task["scenario"]
    positions = np.asarray(scenario["positions"], dtype=float)
    teams = np.asarray(scenario["teams"], dtype=int).reshape(-1)
    centers = [positions[teams == team].mean(axis=0) for team in (0, 1)]
    midpoint = (centers[0] + centers[1]) * 0.5
    for team in (0, 1):
        mask = teams == team
        target_center = midpoint + (centers[team] - midpoint) * factor
        positions[mask] += target_center - centers[team]
    lower = np.asarray(scenario["pos_min"], dtype=float)
    upper = np.asarray(scenario["pos_max"], dtype=float)
    scenario["positions"] = np.clip(positions, lower, upper).tolist()


def _scale_all_offense(task: dict[str, Any], factor: float) -> None:
    """Accelerate high-interaction stalemates without amplifying healing."""

    values = np.asarray(task["scenario"]["attack_damages"], dtype=float)
    values[values > 0] *= factor
    task["scenario"]["attack_damages"] = values.tolist()


def _classify_simulation_failure(
    result: dict[str, Any], config: SampleConfig
) -> str:
    """Map rollout symptoms to one repair family."""

    no_interaction = float(result["quality_flags"]["no_interaction_rate"])
    truncation = float(result["truncation_rate"])
    interaction = result.get("first_interaction_step", {})
    interaction_rate = float(interaction.get("rate", 0.0))
    interaction_step = interaction.get("mean")
    if no_interaction > config.max_no_interaction_rate:
        return "no_interaction"
    if truncation > config.max_truncation_rate:
        late = (
            interaction_step is None
            or float(interaction_step) > config.filter_max_episode_steps * 0.35
        )
        return (
            "low_interaction_truncation"
            if interaction_rate < 0.8 or late
            else "high_interaction_truncation"
        )
    if (
        float(result["episode_length"]["mean"])
        < config.filter_max_episode_steps * 0.15
        and abs(float(result["hp_margin"]["mean"])) > 0.4
    ):
        return "fast_stomp"
    win_rate = float(result["win_rate"])
    if win_rate < config.win_rate_min or win_rate > config.win_rate_max:
        return "combat_imbalance"
    return "unrepairable_quality"


def _repair_by_simulation(
    task: dict[str, Any], config: SampleConfig, *, seed: int
) -> tuple[dict[str, Any], bool, dict[str, Any], list[dict[str, Any]]]:
    """Diagnose rollout failures and apply a bounded repair for that symptom."""

    accepted, initial = _passes_win_rate_filter(
        task,
        physics=config.physics,
        heuristic=config.heuristic,
        num_seeds=_discovery_num_seeds(config),
        seed=seed,
        epsilon=_evaluation_epsilon(config),
        max_episode_steps=config.filter_max_episode_steps,
        win_rate_min=config.win_rate_min,
        win_rate_max=config.win_rate_max,
        reject_all_truncated=config.filter_reject_all_truncated,
        max_truncation_rate=config.max_truncation_rate,
        max_no_interaction_rate=config.max_no_interaction_rate,
        max_n_ally=config.max_n_ally,
        max_n_enemy=config.max_n_enemy,
        max_n_zone=config.max_n_zone,
        seed_stages=config.discovery_seed_stages,
        min_accept_seeds=config.discovery_min_accept_seeds,
        candidate_win_rate_min=config.discovery_candidate_win_rate_min,
        candidate_win_rate_max=config.discovery_candidate_win_rate_max,
        max_abs_hp_margin=config.discovery_max_abs_hp_margin,
    )
    failure_type = None if accepted else _classify_simulation_failure(initial, config)
    history = [
        {
            "round": 0,
            "action": "evaluate",
            "failure_type": failure_type,
            "accepted": accepted,
            "rollouts_consumed": int(
                initial.get("n_rollouts_consumed", _discovery_num_seeds(config))
            ),
            "result": initial,
        }
    ]
    if accepted or not config.simulation_balance_repair:
        return task, accepted, initial, history
    best_task = task
    best_result = initial
    best_accepted = False
    best_error = _balance_error(initial, config)
    no_improvement = 0
    cumulative_combat_scale = {0: 1.0, 1: 1.0}
    cumulative_offense_scale = 1.0
    for round_index in range(config.simulation_repair_rounds):
        failure_type = _classify_simulation_failure(best_result, config)
        trial = copy.deepcopy(best_task)
        action: dict[str, Any] = {"kind": failure_type}
        if failure_type in {"no_interaction", "low_interaction_truncation"}:
            _move_teams_closer(trial, config.directed_repair_distance_factor)
            action.update(
                {
                    "distance_factor": config.directed_repair_distance_factor,
                    "map_factor": config.directed_repair_map_factor,
                }
            )
            map_trial = copy.deepcopy(trial)
            _scale_map(map_trial, config.directed_repair_map_factor)
            try:
                validate_task(map_trial)
                trial = map_trial
            except ValueError:
                action["map_factor_skipped"] = True
        elif failure_type == "high_interaction_truncation":
            remaining = (
                config.directed_repair_max_offense_scale
                / cumulative_offense_scale
            )
            offense_factor = min(config.directed_repair_damage_factor, remaining)
            if offense_factor <= 1.0 + 1e-6:
                break
            _scale_all_offense(trial, offense_factor)
            action.update(
                {
                    "offense_factor": offense_factor,
                    "cumulative_offense_scale": (
                        cumulative_offense_scale * offense_factor
                    ),
                }
            )
        elif failure_type in {"combat_imbalance", "fast_stomp"}:
            win_rate = float(best_result["win_rate"])
            weak_team = 1 if win_rate > config.win_rate_max else 0
            imbalance = max(
                min(abs(win_rate - 0.5) / 0.5, 1.0),
                min(abs(float(best_result["hp_margin"]["mean"])), 1.0),
            )
            suggested_scale = 1.0 + (
                config.simulation_repair_max_scale - 1.0
            ) * max(imbalance, 0.25)
            remaining = (
                config.simulation_repair_max_scale
                / cumulative_combat_scale[weak_team]
            )
            scale = min(suggested_scale, remaining)
            if scale <= 1.0 + 1e-6:
                break
            _scale_team_combat(trial, weak_team, scale)
            action.update(
                {
                    "team": weak_team,
                    "combat_scale": scale,
                    "cumulative_combat_scale": (
                        cumulative_combat_scale[weak_team] * scale
                    ),
                }
            )
        else:
            break
        try:
            validate_task(trial)
        except ValueError as exc:
            history.append(
                {
                    "round": round_index + 1,
                    "action": action,
                    "failure_type": failure_type,
                    "accepted": False,
                    "rollouts_consumed": 0,
                    "repair_error": str(exc),
                    "result": best_result,
                }
            )
            break
        trial_accepted, result = _passes_win_rate_filter(
            trial,
            physics=config.physics,
            heuristic=config.heuristic,
            num_seeds=_discovery_num_seeds(config),
            seed=seed,
            epsilon=_evaluation_epsilon(config),
            max_episode_steps=config.filter_max_episode_steps,
            win_rate_min=config.win_rate_min,
            win_rate_max=config.win_rate_max,
            reject_all_truncated=config.filter_reject_all_truncated,
            max_truncation_rate=config.max_truncation_rate,
            max_no_interaction_rate=config.max_no_interaction_rate,
            max_n_ally=config.max_n_ally,
            max_n_enemy=config.max_n_enemy,
            max_n_zone=config.max_n_zone,
            seed_stages=config.discovery_seed_stages,
            min_accept_seeds=config.discovery_min_accept_seeds,
            candidate_win_rate_min=config.discovery_candidate_win_rate_min,
            candidate_win_rate_max=config.discovery_candidate_win_rate_max,
            max_abs_hp_margin=config.discovery_max_abs_hp_margin,
        )
        error = _balance_error(result, config)
        history.append(
            {
                "round": round_index + 1,
                "action": action,
                "failure_type": failure_type,
                "accepted": trial_accepted,
                "rollouts_consumed": int(
                    result.get("n_rollouts_consumed", _discovery_num_seeds(config))
                ),
                "balance_error": error,
                "result": result,
            }
        )
        if error < best_error or trial_accepted:
            best_task, best_result, best_accepted, best_error = trial, result, trial_accepted, error
            if failure_type == "high_interaction_truncation":
                cumulative_offense_scale *= float(action["offense_factor"])
            elif failure_type in {"combat_imbalance", "fast_stomp"}:
                repaired_team = int(action["team"])
                cumulative_combat_scale[repaired_team] *= float(
                    action["combat_scale"]
                )
            no_improvement = 0
        else:
            no_improvement += 1
        if trial_accepted:
            break
        if no_improvement >= 2:
            break
    return best_task, best_accepted, best_result, history


def _beta_cdf_integer(x: float, alpha: int, beta: int) -> float:
    """Regularized beta CDF for positive integer parameters."""

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    n = alpha + beta - 1
    log_x = math.log(x)
    log_one_minus_x = math.log1p(-x)
    terms = [
        math.lgamma(n + 1)
        - math.lgamma(index + 1)
        - math.lgamma(n - index + 1)
        + index * log_x
        + (n - index) * log_one_minus_x
        for index in range(alpha, n + 1)
    ]
    maximum = max(terms)
    return float(math.exp(maximum) * sum(math.exp(value - maximum) for value in terms))


def _balance_posterior_probability(
    wins: int, total: int, lower: float, upper: float
) -> float:
    """Probability that a Bernoulli rate lies in the target interval."""

    alpha = int(wins) + 1
    beta = int(total - wins) + 1
    return float(
        np.clip(
            _beta_cdf_integer(upper, alpha, beta)
            - _beta_cdf_integer(lower, alpha, beta),
            0.0,
            1.0,
        )
    )


def _wilson_upper(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 1.0
    observed = successes / total
    denominator = 1.0 + z * z / total
    center = (observed + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            observed * (1.0 - observed) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return min(1.0, center + half_width)


def _wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 0.0
    observed = successes / total
    denominator = 1.0 + z * z / total
    center = (observed + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            observed * (1.0 - observed) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half_width)


def _run_confirmation(
    task: dict[str, Any], config: SampleConfig, *, seed: int
) -> tuple[bool, dict[str, Any]]:
    """Run adaptive high-confidence B certification on a frozen task."""

    from src.tabx.eval_task import aggregate_episode_samples, evaluate_tasks

    team_max = max(config.max_n_ally, config.max_n_enemy)
    stages = sorted(set(int(value) for value in config.confirmation_seed_stages if value > 0))
    if not stages:
        raise ValueError("confirmation_seed_stages must contain a positive value.")
    states = {
        "original": {"task": task, "samples": None, "completed": 0},
        "flipped": {
            "task": _flip_task_for_validation(task),
            "samples": None,
            "completed": 0,
        },
    }
    stage_history: list[dict[str, Any]] = []
    last: dict[str, Any] = {}

    def extend(label: str, target: int) -> dict[str, Any]:
        state = states[label]
        delta = target - int(state["completed"])
        chunk = evaluate_tasks(
            [state["task"]],
            physics=config.physics,
            heuristic=config.heuristic,
            num_seeds=delta,
            seed=seed + int(state["completed"]),
            epsilon_override=_evaluation_epsilon(config),
            max_episode_steps=config.filter_max_episode_steps,
            max_n_ally=team_max,
            max_n_enemy=team_max,
            max_n_zone=config.max_n_zone,
            include_episode_samples=True,
        )[0]
        samples = chunk["episode_samples"]
        if state["samples"] is None:
            state["samples"] = {key: list(values) for key, values in samples.items()}
        else:
            for key, values in samples.items():
                state["samples"][key].extend(values)
        state["completed"] = target
        result = aggregate_episode_samples(
            state["samples"], config.filter_max_episode_steps
        )
        result.update(
            {
                "task_id": state["task"].get("task_id", label),
                "ally_heuristic": config.heuristic,
                "enemy_heuristic": config.heuristic,
            }
        )
        return result

    accepted = False
    reasons: list[str] = []
    for target in stages:
        original = extend("original", target)
        flipped = extend("flipped", target)
        original_wins = int(round(float(original["win_rate"]) * target))
        flipped_wins = int(round(float(flipped["win_rate"]) * target))
        original_probability = _balance_posterior_probability(
            original_wins,
            target,
            config.confirmation_win_rate_min,
            config.confirmation_win_rate_max,
        )
        flipped_probability = _balance_posterior_probability(
            flipped_wins,
            target,
            config.confirmation_win_rate_min,
            config.confirmation_win_rate_max,
        )
        original_win_rate = float(original["win_rate"])
        flipped_win_rate = float(flipped["win_rate"])
        team_win_rate = float((original_win_rate + 1.0 - flipped_win_rate) * 0.5)
        side_bias = float(original_win_rate + flipped_win_rate - 1.0)
        team_hp_margin = float(
            (
                float(original["hp_margin"]["mean"])
                - float(flipped["hp_margin"]["mean"])
            )
            * 0.5
        )
        reasons = []
        for label, result, probability in (
            ("original", original, original_probability),
            ("flipped", flipped, flipped_probability),
        ):
            win_rate = float(result["win_rate"])
            if not config.confirmation_win_rate_min <= win_rate <= config.confirmation_win_rate_max:
                reasons.append(f"{label}_win_rate")
            if probability < config.confirmation_balance_probability:
                reasons.append(f"{label}_balance_probability")
            hp_ci_low, hp_ci_high = map(float, result["hp_margin"]["ci95"])
            if (
                hp_ci_low < -config.confirmation_hp_ci_max_abs_bound
                or hp_ci_high > config.confirmation_hp_ci_max_abs_bound
            ):
                reasons.append(f"{label}_hp_margin_confidence")
            truncations = int(round(float(result["truncation_rate"]) * target))
            no_interactions = int(
                round(float(result["quality_flags"]["no_interaction_rate"]) * target)
            )
            if (
                _wilson_upper(truncations, target) > config.max_truncation_rate
                and float(result["truncation_rate"]) > 0.0
            ):
                reasons.append(f"{label}_truncation_confidence")
            if config.max_no_interaction_rate <= 0.0:
                if no_interactions > 0:
                    reasons.append(f"{label}_no_interaction")
            elif _wilson_upper(no_interactions, target) > config.max_no_interaction_rate:
                reasons.append(f"{label}_no_interaction_confidence")
        if not (
            config.confirmation_team_win_rate_min
            <= team_win_rate
            <= config.confirmation_team_win_rate_max
        ):
            reasons.append("team_invariant_win_rate")
        if abs(side_bias) > config.confirmation_max_side_bias:
            reasons.append("side_bias")
        if abs(team_hp_margin) > config.confirmation_max_abs_hp_margin:
            reasons.append("team_hp_margin")
        stage_record = {
            "n_seeds_per_orientation": target,
            "accepted": not reasons and target >= config.confirmation_min_accept_seeds,
            "original_win_rate": original_win_rate,
            "flipped_win_rate": flipped_win_rate,
            "original_balance_probability": original_probability,
            "flipped_balance_probability": flipped_probability,
            "team_invariant_win_rate": team_win_rate,
            "side_bias": side_bias,
            "team_hp_margin": team_hp_margin,
            "pending_reasons": reasons,
        }
        stage_history.append(stage_record)
        last = {
            "original": original,
            "flipped": flipped,
            "original_probability": original_probability,
            "flipped_probability": flipped_probability,
            "team_win_rate": team_win_rate,
            "side_bias": side_bias,
            "team_hp_margin": team_hp_margin,
            "target": target,
        }
        if target >= config.confirmation_min_accept_seeds and not reasons:
            accepted = True
            break
        clearly_unbalanced = (
            target >= stages[0]
            and (
                original_probability < config.confirmation_early_reject_probability
                or flipped_probability < config.confirmation_early_reject_probability
            )
            and (
                not config.confirmation_win_rate_min
                <= original_win_rate
                <= config.confirmation_win_rate_max
                or not config.confirmation_win_rate_min
                <= flipped_win_rate
                <= config.confirmation_win_rate_max
            )
        )
        if clearly_unbalanced:
            reasons.append("posterior_early_reject")
            break
        irreversible_no_interaction = (
            config.max_no_interaction_rate <= 0.0
            and (
                float(original["quality_flags"]["no_interaction_rate"]) > 0.0
                or float(flipped["quality_flags"]["no_interaction_rate"]) > 0.0
            )
        )
        if irreversible_no_interaction:
            reasons.append("no_interaction_early_reject")
            break

    payload = {
        "phase": "B",
        "accepted": accepted,
        "reject_reasons": [] if accepted else reasons,
        "seed": seed,
        "num_seeds": last["target"],
        "rollouts_consumed": last["target"] * 2,
        "epsilon": _evaluation_epsilon(config),
        "original_balance_probability": last["original_probability"],
        "flipped_balance_probability": last["flipped_probability"],
        "team_invariant_win_rate": last["team_win_rate"],
        "side_bias": last["side_bias"],
        "team_hp_margin": last["team_hp_margin"],
        "stage_history": stage_history,
        "original": last["original"],
        "flipped": last["flipped"],
    }
    return accepted, payload


def _sample_tasks_serial(
    config: SampleConfig,
    *,
    initial_parents: Sequence[dict[str, Any]] = (),
    initial_bucket_coverage: dict[tuple[str, ...], int] | None = None,
    forced_source: str | None = None,
    max_attempts_override: int | None = None,
) -> list[dict[str, Any]]:
    """Sample unique low-cost task variants from existing TABX assets."""

    if config.n_tasks <= 0:
        raise ValueError("n_tasks must be positive.")
    if config.cpu_cores <= 0:
        raise ValueError("cpu_cores must be positive.")
    if config.commit_batch_size < 0:
        raise ValueError("commit_batch_size must be non-negative.")
    if config.commit_workers <= 0:
        raise ValueError("commit_workers must be positive.")
    if config.commit_timeout_ms < 0:
        raise ValueError("commit_timeout_ms must be non-negative.")
    if not 0.0 <= config.programmatic_ratio <= 1.0:
        raise ValueError("programmatic_ratio must be in [0, 1].")
    open_ratios = (
        config.free_generation_ratio,
        config.parent_mutation_ratio,
        config.parent_crossover_ratio,
    )
    if any(not 0.0 <= ratio <= 1.0 for ratio in open_ratios) or sum(open_ratios) > 1.0:
        raise ValueError("Open-ended source ratios must be in [0, 1] and sum to at most 1.")
    if config.parameter_distance_threshold < 0 or config.behavior_distance_threshold < 0:
        raise ValueError("Deduplication distance thresholds must be non-negative.")
    if config.map_elites_per_cell <= 0:
        raise ValueError("map_elites_per_cell must be positive.")
    if config.surrogate_min_samples <= 0 or config.surrogate_ridge <= 0:
        raise ValueError("Surrogate sample count and ridge must be positive.")
    if config.collection_candidate_budget <= 0:
        raise ValueError("collection_candidate_budget must be positive.")
    if config.collection_rollout_budget < 0 or config.collection_time_budget_seconds < 0:
        raise ValueError("Collection rollout/time budgets must be non-negative.")
    if (
        config.acquisition_novelty_weight < 0
        or config.acquisition_uncertainty_weight < 0
        or config.acquisition_min_score < 0
    ):
        raise ValueError("Acquisition weights and threshold must be non-negative.")
    if config.max_n_ally < 1 or config.max_n_enemy < 1 or config.max_n_zone < 0:
        raise ValueError("max_n_ally/max_n_enemy must be positive and max_n_zone non-negative.")
    if (
        not config.transforms
        or not config.map_scales
        or not config.stat_scales
        or not config.distance_scales
        or not config.spread_scales
    ):
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
    if config.health_fraction_sampling not in {"buckets", "continuous", "mixed"}:
        raise ValueError(
            "health_fraction_sampling must be buckets, continuous, or mixed."
        )
    if config.static_engagement_dt <= 0:
        raise ValueError("static_engagement_dt must be positive.")
    if not 0.0 < config.static_engagement_max_step_fraction <= 1.0:
        raise ValueError("static_engagement_max_step_fraction must be in (0, 1].")
    if not 0.0 <= config.static_stomp_min_episode_fraction <= 1.0:
        raise ValueError("static_stomp_min_episode_fraction must be in [0, 1].")
    if config.static_stomp_min_ttk_ratio < 1.0:
        raise ValueError("static_stomp_min_ttk_ratio must be at least 1.")
    if not 0.0 < config.directed_repair_distance_factor < 1.0:
        raise ValueError("directed_repair_distance_factor must be in (0, 1).")
    if not 0.0 < config.directed_repair_map_factor <= 1.0:
        raise ValueError("directed_repair_map_factor must be in (0, 1].")
    if config.directed_repair_damage_factor <= 1.0:
        raise ValueError("directed_repair_damage_factor must exceed 1.")
    if config.directed_repair_max_offense_scale < config.directed_repair_damage_factor:
        raise ValueError(
            "directed_repair_max_offense_scale must be at least the per-action factor."
        )
    if not 0.0 <= config.adaptive_sampling_learning_rate <= 1.0:
        raise ValueError("adaptive_sampling_learning_rate must be in [0, 1].")
    if not 0.0 <= config.adaptive_sampling_exploration <= 1.0:
        raise ValueError("adaptive_sampling_exploration must be in [0, 1].")
    if config.bucket_coverage_candidates <= 0:
        raise ValueError("bucket_coverage_candidates must be positive.")
    if not 0.0 <= config.static_balance_max_gap <= 1.0:
        raise ValueError("static_balance_max_gap must be in [0, 1].")
    if config.simulation_repair_rounds < 0:
        raise ValueError("simulation_repair_rounds must be non-negative.")
    if config.static_repair_max_scale < 1.0 or config.simulation_repair_max_scale < 1.0:
        raise ValueError("Repair max scales must be at least 1.0.")
    if not 0.0 <= config.max_truncation_rate <= 1.0:
        raise ValueError("max_truncation_rate must be in [0, 1].")
    if not 0.0 <= config.max_no_interaction_rate <= 1.0:
        raise ValueError("max_no_interaction_rate must be in [0, 1].")
    if not 0.0 <= config.win_rate_min <= config.win_rate_max <= 1.0:
        raise ValueError("Require 0 <= win_rate_min <= win_rate_max <= 1.")
    if config.filter_by_win_rate:
        discovery_num_seeds = _discovery_num_seeds(config)
        epsilon = _evaluation_epsilon(config)
        if discovery_num_seeds <= 0:
            raise ValueError("discovery_num_seeds must be positive when filtering.")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("evaluation epsilon must be in [0, 1].")
        if epsilon <= 0.0:
            raise ValueError(
                "evaluation epsilon must be > 0 when filtering by win rate; "
                "epsilon=0 makes expert vs expert nearly deterministic (win rates collapse to 0/1)."
            )
        if config.filter_max_attempts <= 0:
            raise ValueError("filter_max_attempts must be positive when filtering.")
        if not config.confirmation_enabled:
            raise ValueError("confirmation_enabled must remain true in the A+B protocol.")
        discovery_stages = tuple(sorted(set(config.discovery_seed_stages)))
        confirmation_stages = tuple(sorted(set(config.confirmation_seed_stages)))
        if not discovery_stages or any(value <= 0 for value in discovery_stages):
            raise ValueError("discovery_seed_stages must contain positive values.")
        if not confirmation_stages or any(value <= 0 for value in confirmation_stages):
            raise ValueError("confirmation_seed_stages must contain positive values.")
        if config.discovery_min_accept_seeds not in discovery_stages:
            raise ValueError("discovery_min_accept_seeds must be one of discovery_seed_stages.")
        if config.confirmation_min_accept_seeds not in confirmation_stages:
            raise ValueError("confirmation_min_accept_seeds must be one of confirmation_seed_stages.")
        if not (
            0.0
            <= config.discovery_candidate_win_rate_min
            <= config.win_rate_min
            <= config.win_rate_max
            <= config.discovery_candidate_win_rate_max
            <= 1.0
        ):
            raise ValueError("Invalid A candidate and target win-rate bands.")
        if config.discovery_max_abs_hp_margin < 0:
            raise ValueError("discovery_max_abs_hp_margin must be non-negative.")
        if not 0.0 <= config.confirmation_balance_probability <= 1.0:
            raise ValueError("confirmation_balance_probability must be in [0, 1].")
        if not 0.0 <= config.confirmation_early_reject_probability <= 1.0:
            raise ValueError("confirmation_early_reject_probability must be in [0, 1].")
        if config.discovery_seed_offset < 0:
            raise ValueError("discovery_seed_offset must be non-negative.")
        if config.confirmation_seed_offset <= 0:
            raise ValueError("confirmation_seed_offset must be positive.")
        if not (
            0.0
            <= config.confirmation_win_rate_min
            <= config.confirmation_win_rate_max
            <= 1.0
        ):
            raise ValueError("Invalid confirmation win-rate band.")
        if not (
            0.0
            <= config.confirmation_team_win_rate_min
            <= config.confirmation_team_win_rate_max
            <= 1.0
        ):
            raise ValueError("Invalid confirmation team-invariant win-rate band.")
        if config.confirmation_max_side_bias < 0 or config.confirmation_max_abs_hp_margin < 0:
            raise ValueError("Confirmation bias and HP-margin limits must be non-negative.")
        if (
            config.confirmation_hp_ci_max_abs_bound
            < config.confirmation_max_abs_hp_margin
        ):
            raise ValueError(
                "confirmation_hp_ci_max_abs_bound must be at least the HP mean limit."
            )
        if config.discovery_seed_offset == config.confirmation_seed_offset:
            raise ValueError("A and B seed offsets must be distinct.")
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

    rng = random.Random()
    tasks: list[dict[str, Any]] = []
    # Keep a dedicated pool so B-confirmed candidates can immediately seed
    # mutation/crossover without coupling parent selection to output ordering.
    parent_pool: list[dict[str, Any]] = []
    parent_hashes: set[str] = set()
    seen: set[str] = set()
    bucket_coverage: Counter[tuple[str, ...]] = Counter(initial_bucket_coverage or {})
    parameter_archive: list[np.ndarray] = []
    behavior_archive: list[np.ndarray] = []
    source_counts: Counter[str] = Counter()
    map_archive: dict[tuple[int, ...], list[tuple[float, dict[str, Any]]]] = {}
    map_cell_counts: Counter[tuple[int, ...]] = Counter()
    bucket_feedback: dict[tuple[str, str], Counter[str]] = {}
    source_feedback: dict[str, Counter[str]] = {}
    predictor = _OnlineOutcomePredictor(config.surrogate_min_samples, config.surrogate_ridge)
    collection_started = time.monotonic()
    simulation_rollouts = 0
    skipped_by_acquisition = 0

    def add_parent(candidate: dict[str, Any]) -> None:
        digest = _task_hash(candidate)
        if digest in parent_hashes:
            return
        parent_hashes.add(digest)
        parent_pool.append(copy.deepcopy(candidate))

    def select_parent() -> dict[str, Any]:
        if config.map_elites and map_archive:
            minimum = min(map_cell_counts[cell] for cell in map_archive)
            cells = [cell for cell in map_archive if map_cell_counts[cell] == minimum]
            return rng.choice(map_archive[rng.choice(cells)])[1]
        return rng.choice(parent_pool)
    for initial_parent in initial_parents:
        add_parent(initial_parent)
    if config.filter_by_win_rate:
        # filter_max_attempts is a per-accepted-task budget.
        max_attempts = config.filter_max_attempts * config.n_tasks
    else:
        max_attempts = max(1_000, config.n_tasks * 100)
    if config.continuous_collection:
        max_attempts = config.collection_candidate_budget
    if max_attempts_override is not None:
        max_attempts = max_attempts_override
    n_candidates = 0
    n_rejected_by_win_rate = 0
    n_generation_failures = 0

    if config.filter_by_win_rate:
        print(
            "Win-rate filter enabled: "
            f"target={config.n_tasks}, band=[{config.win_rate_min:.2f}, {config.win_rate_max:.2f}], "
            f"A_stages={list(config.discovery_seed_stages)}, "
            f"B_stages={list(config.confirmation_seed_stages)}, "
            f"B_probability={config.confirmation_balance_probability:.2f}, "
            f"epsilon={_evaluation_epsilon(config)}, "
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
    if config.open_ended_generation:
        print(
            "Open-ended sampling: "
            f"free={config.free_generation_ratio:.2f}, "
            f"mutation={config.parent_mutation_ratio:.2f}, "
            f"crossover={config.parent_crossover_ratio:.2f}, "
            f"parameter_threshold={config.parameter_distance_threshold:.3f}, "
            f"behavior_threshold={config.behavior_distance_threshold:.3f}"
        )

    for attempt in range(1, max_attempts + 1):
        if (
            config.continuous_collection
            and config.collection_rollout_budget > 0
            and simulation_rollouts >= config.collection_rollout_budget
        ):
            break
        if (
            config.continuous_collection
            and config.collection_time_budget_seconds > 0
            and time.monotonic() - collection_started >= config.collection_time_budget_seconds
        ):
            break
        requested_source = forced_source or (
            _draw_source_kind(rng, config, source_feedback)
            if config.adaptive_sampling
            else _draw_source_kind(rng, config, {})
        )
        crossover_requested = requested_source == "crossover"
        mutation_requested = requested_source == "mutation"
        free_requested = requested_source == "free"
        use_crossover = (
            config.open_ended_generation
            and crossover_requested
            and len(parent_pool) >= 2
        )
        use_mutation = (
            config.open_ended_generation
            and mutation_requested
            and len(parent_pool) >= 1
        )
        # A requested parent operation falls back to free generation until enough
        # accepted parents exist; it must never leak into another source interval.
        use_free = config.open_ended_generation and (
            free_requested
            or (crossover_requested and len(parent_pool) < 2)
            or (mutation_requested and len(parent_pool) < 1)
        )
        is_programmatic = rng.random() < config.programmatic_ratio
        map_scale = _sample_scale(
            rng, config.map_scales, _continuous_option(config, "continuous_map_sampling")
        )
        if _continuous_option(config, "continuous_stat_sampling"):
            ally_stat_scale = _sample_scale(rng, config.stat_scales, True)
            enemy_stat_scale = ally_stat_scale if config.couple_stat_scales else _sample_scale(rng, config.stat_scales, True)
        else:
            ally_stat_scale, enemy_stat_scale = _sample_coupled_stat_scales(
                rng, config.stat_scales, coupled=config.couple_stat_scales
            )
        transform = rng.choice(config.transforms)
        distance_scale = _sample_scale(
            rng,
            config.distance_scales,
            _continuous_option(config, "continuous_distance_sampling"),
        )
        spread_scale = _sample_scale(
            rng,
            config.spread_scales,
            _continuous_option(config, "continuous_spread_sampling"),
        )

        if use_crossover:
            first = select_parent()
            first_hash = _task_hash(first)
            second = rng.choice(
                [
                    candidate
                    for candidate in parent_pool
                    if _task_hash(candidate) != first_hash
                ]
            )
            task = crossover_parent_tasks(
                rng,
                first,
                second,
                max_n_ally=config.max_n_ally,
                max_n_enemy=config.max_n_enemy,
                max_n_zone=config.max_n_zone,
                map_scale=map_scale,
            )
            source = task["metadata"]["source"]
            zone_strength_scale = _sample_scale(
                rng,
                config.zone_strength_scales,
                _continuous_option(config, "continuous_zone_strength_sampling"),
            )
            scenario_key = None
        elif use_mutation:
            parent = select_parent()
            task = mutate_parent_task(
                rng,
                parent,
                max_n_ally=config.max_n_ally,
                max_n_enemy=config.max_n_enemy,
                max_n_zone=config.max_n_zone,
                map_scale=map_scale,
            )
            source = task["metadata"]["source"]
            zone_strength_scale = _sample_scale(
                rng,
                config.zone_strength_scales,
                _continuous_option(config, "continuous_zone_strength_sampling"),
            )
            scenario_key = None
        elif use_free:
            task = generate_free_task(
                rng,
                max_n_ally=config.max_n_ally,
                max_n_enemy=config.max_n_enemy,
                max_n_zone=config.max_n_zone,
                map_scale=map_scale,
            )
            source = task["metadata"]["source"]
            zone_strength_scale = _sample_scale(
                rng,
                config.zone_strength_scales,
                _continuous_option(config, "continuous_zone_strength_sampling"),
            )
            scenario_key = None
        elif is_programmatic:
            (
                composition,
                layout,
                distance,
                spread,
                zone,
                zone_intensity,
                zone_relation,
                match_mode,
            ) = _pick_scenario_buckets(
                rng, config, bucket_coverage, bucket_feedback
            )
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
            _record_bucket_feedback(bucket_feedback, scenario_key, "generated")
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
                    health_frac_buckets=_health_fraction_buckets(rng, config),
                    health_frac_min=config.health_frac_min,
                    health_frac_max=config.health_frac_max,
                    distance_scale=distance_scale,
                    spread_scale=spread_scale,
                )
            except ValueError as exc:
                n_generation_failures += 1
                source_feedback.setdefault("base", Counter())["generated"] += 1
                source_feedback.setdefault("base", Counter())[
                    "generation_failure"
                ] += 1
                _record_bucket_feedback(
                    bucket_feedback, scenario_key, "generation_failure"
                )
                if config.filter_by_win_rate:
                    print(
                        f"try {attempt}/{max_attempts} {_progress_bar(attempt, max_attempts)} "
                        f"GENERATE_FAIL accepted={len(tasks)}/{config.n_tasks} "
                        f"reason={exc}"
                    )
                continue
            source = task["metadata"]["source"]
            zone_strength_scale = _sample_scale(
                rng,
                config.zone_strength_scales,
                _continuous_option(config, "continuous_zone_strength_sampling"),
            )
        else:
            distance_scale = 1.0
            spread_scale = 1.0
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
            zone_strength_scale = _sample_scale(
                rng,
                config.zone_strength_scales,
                _continuous_option(config, "continuous_zone_strength_sampling"),
            )
            scenario_key = None

        actual_source_kind = str(source.get("kind", "base"))
        source_feedback_kind = (
            actual_source_kind
            if actual_source_kind in {"free", "mutation", "crossover"}
            else "base"
        )
        source_feedback.setdefault(source_feedback_kind, Counter())["generated"] += 1
        _transform_task(task, transform)
        _scale_team_stats(task, ally_stat_scale, enemy_stat_scale)
        _scale_zone_strength(task, zone_strength_scale)
        if config.static_balance_repair:
            task.setdefault("metadata", {})["static_balance_repair"] = static_balance_repair(
                task, config
            )
        try:
            validate_task(task)
            engagement = estimate_static_engagement(
                task, dt=config.static_engagement_dt
            )
            task.setdefault("metadata", {})["static_engagement"] = engagement
            max_contact_steps = (
                config.filter_max_episode_steps
                * config.static_engagement_max_step_fraction
            )
            if config.static_engagement_filter and not engagement["reachable"]:
                raise ValueError("teams have no statically reachable hostile interaction")
            if (
                config.static_engagement_filter
                and engagement["estimated_first_contact_steps"] is not None
                and engagement["estimated_first_contact_steps"] > max_contact_steps
            ):
                raise ValueError(
                    "estimated first contact exceeds static engagement threshold: "
                    f"{engagement['estimated_first_contact_steps']:.1f} > "
                    f"{max_contact_steps:.1f} steps"
                )
            earliest_elimination = engagement["estimated_earliest_elimination_steps"]
            ttk_ratio = engagement["estimated_ttk_ratio"]
            if (
                config.static_stomp_filter
                and earliest_elimination is not None
                and ttk_ratio is not None
                and earliest_elimination
                < config.filter_max_episode_steps
                * config.static_stomp_min_episode_fraction
                and ttk_ratio > config.static_stomp_min_ttk_ratio
            ):
                raise ValueError(
                    "static estimate predicts an early one-sided elimination: "
                    f"steps={earliest_elimination:.1f}, ttk_ratio={ttk_ratio:.2f}"
                )
        except ValueError as exc:
            n_generation_failures += 1
            source_feedback[source_feedback_kind]["static_invalid"] += 1
            _record_bucket_feedback(bucket_feedback, scenario_key, "static_invalid")
            if config.filter_by_win_rate:
                print(
                    f"try {attempt}/{max_attempts} {_progress_bar(attempt, max_attempts)} "
                    f"INVALID accepted={len(tasks)}/{config.n_tasks} reason={exc}"
                )
            continue
        if int(task["zone_scenario"]["n_zone"]) > config.max_n_zone:
            n_generation_failures += 1
            source_feedback[source_feedback_kind]["too_many_zones"] += 1
            _record_bucket_feedback(bucket_feedback, scenario_key, "too_many_zones")
            if config.filter_by_win_rate:
                print(
                    f"try {attempt}/{max_attempts} {_progress_bar(attempt, max_attempts)} "
                    f"INVALID accepted={len(tasks)}/{config.n_tasks} reason=too_many_zones"
                )
            continue

        parameter_signature = _parameter_signature(task)
        parameter_distance = _nearest_distance(parameter_signature, parameter_archive)
        if parameter_distance < config.parameter_distance_threshold:
            if config.filter_by_win_rate:
                print(
                    f"try {attempt}/{max_attempts} {_progress_bar(attempt, max_attempts)} "
                    f"PARAM_DUPLICATE distance={parameter_distance:.4f} "
                    f"accepted={len(tasks)}/{config.n_tasks}"
                )
            continue

        surrogate_prediction = (
            predictor.predict(parameter_signature) if config.surrogate_enabled else None
        )
        novelty_score = (
            1.0
            if not math.isfinite(parameter_distance)
            else float(np.clip(parameter_distance / 0.20, 0.0, 1.0))
        )
        uncertainty_score = (
            1.0 if surrogate_prediction is None else surrogate_prediction["uncertainty"]
        )
        acquisition_score = (
            config.acquisition_novelty_weight * novelty_score
            + config.acquisition_uncertainty_weight * uncertainty_score
        )
        if (
            config.filter_by_win_rate
            and surrogate_prediction is not None
            and acquisition_score < config.acquisition_min_score
        ):
            skipped_by_acquisition += 1
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
                "distance_scale": distance_scale,
                "spread_scale": spread_scale,
                "canonical_hash": digest,
                "map_elites_candidate_cell": list(_map_elites_cell(task)),
                "surrogate_prediction": surrogate_prediction,
                "acquisition_score": acquisition_score,
            }
        )
        task["metadata"] = metadata
        task["metadata"]["novelty"] = {
            "parameter_distance": (
                parameter_distance if math.isfinite(parameter_distance) else None
            ),
            "behavior_distance": None,
        }

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
            discovery_seed = (
                _phase_seed_base(task, config.seed) + config.discovery_seed_offset
            )
            task, accepted, eval_result, repair_history = _repair_by_simulation(
                task, config, seed=discovery_seed
            )
            task["metadata"]["static_engagement_after_repair"] = (
                estimate_static_engagement(task, dt=config.static_engagement_dt)
            )
            simulation_rollouts += sum(
                int(
                    item.get(
                        "rollouts_consumed",
                        item["result"].get(
                            "n_rollouts_consumed", _discovery_num_seeds(config)
                        ),
                    )
                )
                for item in repair_history
            )
            fast_eval_result = copy.deepcopy(eval_result)
            repaired_digest = _task_hash(task)
            if repaired_digest != digest:
                if repaired_digest in seen:
                    continue
                seen.add(repaired_digest)
                digest = repaired_digest
                task["task_id"] = f"task_{len(tasks):06d}_{digest[:10]}"
                task["metadata"]["canonical_hash"] = digest
            parameter_signature = _parameter_signature(task)
            parameter_distance = _nearest_distance(
                parameter_signature, parameter_archive
            )
            if parameter_distance < config.parameter_distance_threshold:
                accepted = False
                eval_result["reject_reason"] = "parameter_duplicate_after_repair"
            behavior_signature = _behavior_signature(
                eval_result, config.filter_max_episode_steps
            )
            behavior_distance = _nearest_distance(
                behavior_signature, behavior_archive
            )
            if accepted and behavior_distance < config.behavior_distance_threshold:
                accepted = False
                eval_result["reject_reason"] = "behavior_duplicate"
            confirmation = None
            validation_gate = {
                "selected": False,
                "reason": "A_rejected",
                "policy": "A_then_B",
            }
            frozen_seed_base = _phase_seed_base(task, config.seed)
            if accepted and config.confirmation_enabled:
                accepted, confirmation = _run_confirmation(
                    task,
                    config,
                    seed=frozen_seed_base + config.confirmation_seed_offset,
                )
                simulation_rollouts += int(confirmation["rollouts_consumed"])
                if not accepted:
                    validation_gate["reason"] = "B_rejected"
                    eval_result["reject_reason"] = "confirmation:"
                    eval_result["reject_reason"] += ",".join(
                        confirmation["reject_reasons"]
                    )
                else:
                    # From this point onward, archive quality, surrogate updates,
                    # and saved filter metrics use the independent B result rather
                    # than the A seeds that were used to tune the task.
                    eval_result = confirmation["original"]
                    behavior_signature = _behavior_signature(
                        eval_result, config.filter_max_episode_steps
                    )
                    behavior_distance = _nearest_distance(
                        behavior_signature, behavior_archive
                    )
                    if behavior_distance < config.behavior_distance_threshold:
                        accepted = False
                        eval_result["reject_reason"] = (
                            "behavior_duplicate_after_confirmation"
                        )
                    if accepted:
                        add_parent(task)
                        validation_gate = {
                            "selected": True,
                            "reason": "B_confirmed",
                            "policy": "A_then_B",
                        }
            task["metadata"]["validation_gate"] = validation_gate
            task["metadata"]["validation_status"] = (
                "confirmed"
                if confirmation is not None and accepted
                else "rejected"
            )
            task["metadata"]["simulation_balance_repair"] = {
                "applied": len(repair_history) > 1,
                "history": repair_history,
                "final_balance_error": _balance_error(eval_result, config),
            }
            task["metadata"]["fast_filter_eval"] = fast_eval_result
            task["metadata"]["confirmation"] = confirmation
            task["metadata"]["evaluation_protocol"] = {
                "epsilon": _evaluation_epsilon(config),
                "discovery_seed": discovery_seed,
                "discovery_num_seeds": fast_eval_result["n_rollouts"],
                "confirmation_seed": (
                    confirmation["seed"] if confirmation is not None else None
                ),
                "confirmation_num_seeds": (
                    confirmation["num_seeds"] if confirmation is not None else None
                ),
                "phases": {
                    "A": {
                        "purpose": "discovery_and_repair",
                        "seed": discovery_seed,
                        "num_seeds": fast_eval_result["n_rollouts"],
                        "seed_stages": list(config.discovery_seed_stages),
                    },
                    "B": (
                        {
                            "purpose": "frozen_candidate_confirmation",
                            "seed": confirmation["seed"],
                            "num_seeds": confirmation["num_seeds"],
                            "seed_stages": list(config.confirmation_seed_stages),
                            "balance_probability_threshold": (
                                config.confirmation_balance_probability
                            ),
                        }
                        if confirmation is not None
                        else None
                    ),
                },
            }
            task["metadata"]["filter_eval"] = {
                "win_rate": eval_result["win_rate"],
                "win_rate_ci95": eval_result["win_rate_ci95"],
                "episode_length": eval_result["episode_length"],
                "truncation_rate": eval_result["truncation_rate"],
                "hp_margin": eval_result["hp_margin"],
                "damage_share": eval_result["damage_share"],
                "damage_mean": eval_result["damage_mean"],
                "damage_mean_ci95": eval_result["damage_mean_ci95"],
                "healing_mean": eval_result["healing_mean"],
                "healing_mean_ci95": eval_result["healing_mean_ci95"],
                "attack_success_rate": eval_result["attack_success_rate"],
                "attack_success_rate_ci95": eval_result[
                    "attack_success_rate_ci95"
                ],
                "first_interaction_step": eval_result["first_interaction_step"],
                "quality_flags": eval_result["quality_flags"],
                "n_rollouts": eval_result["n_rollouts"],
                "accepted": accepted,
                "reject_reason": eval_result.get("reject_reason"),
            }
            predictor.update(
                parameter_signature,
                eval_result,
                _balance_error(eval_result, config),
            )
            task["metadata"]["novelty"] = {
                "parameter_distance": (
                    parameter_distance if math.isfinite(parameter_distance) else None
                ),
                "behavior_distance": (
                    behavior_distance if math.isfinite(behavior_distance) else None
                ),
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
                reason = str(eval_result.get("reject_reason") or "filter_reject")
                _record_bucket_feedback(bucket_feedback, scenario_key, reason)
                source_feedback[source_feedback_kind][reason] += 1
                # Keep the hash reserved so we do not re-evaluate the same candidate.
                continue

        source_feedback[source_feedback_kind]["accepted"] += 1
        add_parent(task)
        task.setdefault("metadata", {}).setdefault(
            "validation_status", "unfiltered" if not config.filter_by_win_rate else "provisional"
        )
        task["metadata"]["adaptive_sampling"] = {
            "enabled": config.adaptive_sampling,
            "requested_source": requested_source,
            "source_group": source_feedback_kind,
            "source_stats": dict(source_feedback[source_feedback_kind]),
            "bucket_utility": (
                _bucket_feedback_utility(bucket_feedback, scenario_key)
                if scenario_key is not None
                else None
            ),
        }
        tasks.append(task)
        source_counts[str(task["metadata"]["source"].get("kind", "unknown"))] += 1
        parameter_archive.append(parameter_signature)
        if config.filter_by_win_rate:
            behavior_archive.append(behavior_signature)
            elite_quality = -_balance_error(eval_result, config)
            elite_cell = _map_elites_cell(task, eval_result)
        else:
            static_record = task["metadata"].get("static_balance_repair")
            static_gap = (
                float(static_record["after"]["relative_gap"])
                if static_record is not None
                else float(estimate_dynamic_team_values(task)["relative_gap"])
            )
            elite_quality = -static_gap
            elite_cell = _map_elites_cell(task)
        task["metadata"]["map_elites"] = {
            "cell": list(elite_cell),
            "quality": elite_quality,
        }
        map_cell_counts[elite_cell] += 1
        elites = map_archive.setdefault(elite_cell, [])
        elites.append((elite_quality, task))
        elites.sort(key=lambda item: item[0], reverse=True)
        del elites[config.map_elites_per_cell :]
        if scenario_key is not None:
            bucket_coverage[scenario_key] += 1
            _record_bucket_feedback(bucket_feedback, scenario_key, "accepted")
        if not config.continuous_collection and len(tasks) == config.n_tasks:
            if config.filter_by_win_rate:
                validation_counts = Counter(
                    str(task["metadata"].get("validation_status", "unknown"))
                    for task in tasks
                )
                print(
                    f"Done. Accepted {len(tasks)}/{n_candidates} unique candidates "
                    f"(rejected_by_win_rate={n_rejected_by_win_rate}, "
                    f"generation_failures={n_generation_failures}, "
                    f"covered_buckets={len(bucket_coverage)}, sources={dict(source_counts)}, "
                    f"confirmed_parent_pool={len(parent_pool)}, "
                    f"validation={dict(validation_counts)})."
                )
            elif config.bucket_coverage:
                print(
                    f"Done. Sampled {len(tasks)} tasks across {len(bucket_coverage)} scenario buckets; "
                    f"sources={dict(source_counts)}."
                )
            return tasks

    if config.continuous_collection and tasks:
        validation_counts = Counter(
            str(task["metadata"].get("validation_status", "unknown"))
            for task in tasks
        )
        print(
            f"Budget exhausted. Collected {len(tasks)} tasks from {n_candidates} candidates; "
            f"rollouts={simulation_rollouts}, elapsed={time.monotonic() - collection_started:.1f}s, "
            f"map_cells={len(map_archive)}, acquisition_skips={skipped_by_acquisition}, "
            f"sources={dict(source_counts)}, confirmed_parent_pool={len(parent_pool)}, "
            f"validation={dict(validation_counts)}."
        )
        return tasks

    raise RuntimeError(
        f"Could only produce {len(tasks)} valid unique tasks after {max_attempts} attempts "
        f"(unique_candidates={n_candidates}, rejected_by_win_rate={n_rejected_by_win_rate}, "
        f"generation_failures={n_generation_failures}, "
        f"confirmed_parent_pool={len(parent_pool)})."
    )


@dataclass(frozen=True)
class _CandidateJob:
    """One immutable candidate assignment created by the coordinator."""

    candidate_id: int
    global_seed: int
    candidate_seed: int
    source_kind: str
    parent_ids: tuple[str, ...]
    parents: tuple[dict[str, Any], ...]
    bucket_coverage: dict[tuple[str, ...], int]
    config: SampleConfig


@dataclass(frozen=True)
class _CandidateResult:
    """A worker result with immutable features for batched validation."""

    candidate_id: int
    candidate_seed: int
    source_kind: str
    parent_ids: tuple[str, ...]
    worker_id: int
    worker_pid: int
    task: dict[str, Any] | None
    reject_reason: str | None
    task_hash: str | None = None
    parameter_signature: np.ndarray | None = None
    behavior_signature: np.ndarray | None = None
    elite_cell: tuple[int, ...] | None = None
    elite_quality: float | None = None


def _derive_candidate_seed(global_seed: int, candidate_id: int) -> int:
    """Derive a scheduling-independent seed from a global candidate ID."""

    payload = f"tabx-candidate:{int(global_seed)}:{int(candidate_id)}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _initialize_cpu_worker(counter: Any) -> None:
    global _CPU_WORKER_ID
    with counter.get_lock():
        _CPU_WORKER_ID = int(counter.value)
        counter.value += 1


def _sample_candidate_cpu_worker(job: _CandidateJob) -> _CandidateResult:
    """Generate and evaluate one candidate without owning global state."""

    jax.config.update("jax_platform_name", "cpu")
    try:
        with redirect_stdout(io.StringIO()):
            tasks = _sample_tasks_serial(
                job.config,
                initial_parents=job.parents,
                initial_bucket_coverage=job.bucket_coverage,
                forced_source=job.source_kind,
                max_attempts_override=1,
            )
    except RuntimeError as exc:
        if not str(exc).startswith("Could only produce"):
            raise
        return _CandidateResult(
            candidate_id=job.candidate_id,
            candidate_seed=job.candidate_seed,
            source_kind=job.source_kind,
            parent_ids=job.parent_ids,
            worker_id=_CPU_WORKER_ID,
            worker_pid=os.getpid(),
            task=None,
            reject_reason="worker_filter_or_generation_reject",
        )
    task = tasks[0]
    task.setdefault("metadata", {})["parallel_sampling"] = {
        "candidate_id": job.candidate_id,
        "global_seed": job.global_seed,
        "candidate_seed": job.candidate_seed,
        "worker_id": _CPU_WORKER_ID,
        "worker_pid": os.getpid(),
        "parent_ids": list(job.parent_ids),
        "seed_derivation": (
            "sha256(sha256(global_seed, candidate_id), canonical_task_hash)"
        ),
    }
    parameter_signature = _parameter_signature(task)
    eval_result = task.get("metadata", {}).get("filter_eval")
    behavior_signature = None
    if job.config.filter_by_win_rate and isinstance(eval_result, dict):
        behavior_signature = _behavior_signature(
            eval_result, job.config.filter_max_episode_steps
        )
        elite_quality = -_balance_error(eval_result, job.config)
        elite_cell = _map_elites_cell(task, eval_result)
    else:
        static_record = task.get("metadata", {}).get("static_balance_repair")
        static_gap = (
            float(static_record["after"]["relative_gap"])
            if static_record is not None
            else float(estimate_dynamic_team_values(task)["relative_gap"])
        )
        elite_quality = -static_gap
        elite_cell = _map_elites_cell(task)
    return _CandidateResult(
        candidate_id=job.candidate_id,
        candidate_seed=job.candidate_seed,
        source_kind=job.source_kind,
        parent_ids=job.parent_ids,
        worker_id=_CPU_WORKER_ID,
        worker_pid=os.getpid(),
        task=task,
        reject_reason=None,
        task_hash=_task_hash(task),
        parameter_signature=parameter_signature,
        behavior_signature=behavior_signature,
        elite_cell=elite_cell,
        elite_quality=elite_quality,
    )


def _parallel_worker_count(config: SampleConfig) -> int:
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    if config.cpu_cores > available:
        raise ValueError(
            f"Requested {config.cpu_cores} CPU cores, but only {available} are "
            "available to this process."
        )
    requested = config.cpu_cores
    work_items = (
        config.collection_candidate_budget
        if config.continuous_collection
        else config.n_tasks
    )
    if config.continuous_collection and config.collection_rollout_budget > 0:
        work_items = min(work_items, config.collection_rollout_budget)
    return max(1, min(requested, available, work_items))


def _candidate_rollouts(task: dict[str, Any]) -> int:
    metadata = task.get("metadata", {})
    repair = metadata.get("simulation_balance_repair", {})
    repair_rollouts = sum(
        int(item.get("rollouts_consumed", 0)) for item in repair.get("history", [])
    )
    confirmation = metadata.get("confirmation")
    confirmation_rollouts = (
        int(confirmation.get("rollouts_consumed", 0))
        if isinstance(confirmation, dict)
        else 0
    )
    return repair_rollouts + confirmation_rollouts


def _task_bucket_key(task: dict[str, Any]) -> tuple[str, ...] | None:
    metadata = task.get("metadata", {})
    match_mode = (metadata.get("composition_match") or {}).get("match_mode")
    values = (
        metadata.get("composition_archetype"),
        metadata.get("layout_archetype"),
        metadata.get("distance_bucket"),
        metadata.get("spread_bucket"),
        metadata.get("zone_archetype"),
        metadata.get("zone_intensity"),
        metadata.get("zone_relation"),
        match_mode,
    )
    if any(value is None for value in values):
        return None
    return tuple(str(value) for value in values)


def _assign_candidate_parents(
    source_kind: str,
    parent_pool: Sequence[dict[str, Any]],
    parent_leases: Counter[str],
    map_cell_counts: Counter[tuple[int, ...]],
    rng: random.Random,
) -> tuple[str, tuple[str, ...], tuple[dict[str, Any], ...]]:
    required = 2 if source_kind == "crossover" else 1 if source_kind == "mutation" else 0
    if required == 0:
        return source_kind, (), ()
    if len(parent_pool) < required:
        return "free", (), ()

    ranked = sorted(
        parent_pool,
        key=lambda task: (
            parent_leases[_task_hash(task)],
            map_cell_counts[tuple(task.get("metadata", {}).get("map_elites", {}).get("cell", ()))],
            _task_hash(task),
        ),
    )
    first_rank = (
        parent_leases[_task_hash(ranked[0])],
        map_cell_counts[
            tuple(ranked[0].get("metadata", {}).get("map_elites", {}).get("cell", ()))
        ],
    )
    first_group = [
        task
        for task in ranked
        if (
            parent_leases[_task_hash(task)],
            map_cell_counts[
                tuple(task.get("metadata", {}).get("map_elites", {}).get("cell", ()))
            ],
        )
        == first_rank
    ]
    first = rng.choice(first_group)
    selected = [first]
    if required == 2:
        remaining = [task for task in ranked if _task_hash(task) != _task_hash(first)]
        next_rank = (
            parent_leases[_task_hash(remaining[0])],
            map_cell_counts[
                tuple(
                    remaining[0].get("metadata", {}).get("map_elites", {}).get("cell", ())
                )
            ],
        )
        second_group = [
            task
            for task in remaining
            if (
                parent_leases[_task_hash(task)],
                map_cell_counts[
                    tuple(task.get("metadata", {}).get("map_elites", {}).get("cell", ()))
                ],
            )
            == next_rank
        ]
        selected.append(rng.choice(second_group))
    parent_ids = tuple(_task_hash(task) for task in selected)
    for parent_id in parent_ids:
        parent_leases[parent_id] += 1
    return source_kind, parent_ids, tuple(copy.deepcopy(task) for task in selected)


def sample_tasks(config: SampleConfig) -> list[dict[str, Any]]:
    """Sample in processes and publish deterministic micro-batch transactions."""

    if config.n_tasks <= 0:
        raise ValueError("n_tasks must be positive.")
    if config.cpu_cores <= 0:
        raise ValueError("cpu_cores must be positive.")
    if config.commit_batch_size < 0:
        raise ValueError("commit_batch_size must be non-negative.")
    if config.commit_workers <= 0:
        raise ValueError("commit_workers must be positive.")
    if config.commit_timeout_ms < 0:
        raise ValueError("commit_timeout_ms must be non-negative.")
    if config.filter_by_win_rate and config.filter_max_attempts <= 0:
        raise ValueError("filter_max_attempts must be positive when filtering.")
    if config.continuous_collection and config.collection_candidate_budget <= 0:
        raise ValueError("collection_candidate_budget must be positive.")
    worker_count = _parallel_worker_count(config)
    commit_batch_size = config.commit_batch_size or max(1, 2 * worker_count)
    commit_workers = min(config.commit_workers, 3, commit_batch_size)
    commit_timeout_seconds = config.commit_timeout_ms / 1_000.0
    max_candidates = (
        config.collection_candidate_budget
        if config.continuous_collection
        else (
            config.filter_max_attempts * config.n_tasks
            if config.filter_by_win_rate
            else max(1_000, config.n_tasks * 100)
        )
    )
    accepted_tasks: list[dict[str, Any]] = []
    evaluated_hashes: set[str] = set()
    parameter_archive: list[np.ndarray] = []
    behavior_archive: list[np.ndarray] = []
    parent_pool: list[dict[str, Any]] = []
    parent_leases: Counter[str] = Counter()
    source_feedback: dict[str, Counter[str]] = {}
    predictor = _OnlineOutcomePredictor(config.surrogate_min_samples, config.surrogate_ridge)
    map_archive: dict[tuple[int, ...], list[tuple[float, dict[str, Any]]]] = {}
    map_cell_counts: Counter[tuple[int, ...]] = Counter()
    bucket_coverage: Counter[tuple[str, ...]] = Counter()
    reject_counts: Counter[str] = Counter()
    simulation_rollouts = 0
    started = time.monotonic()

    # Forkserver interpreters read these before importing JAX, which both forces
    # CPU execution and prevents every process from creating a full CPU thread
    # pool of its own.
    managed_environment = {
        "JAX_PLATFORMS": "cpu",
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_SKIP_CUDA_CONSTRAINTS_CHECK": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "XLA_FLAGS": (
            "--xla_cpu_multi_thread_eigen=false "
            "intra_op_parallelism_threads=1"
        ),
    }
    previous_environment = {name: os.environ.get(name) for name in managed_environment}
    os.environ.update(managed_environment)
    print(
        f"Central CPU sampling: workers={worker_count}, target={config.n_tasks}, "
        f"candidate_budget={max_candidates}, threads_per_worker=1, "
        f"commit_batch_size={commit_batch_size}, commit_workers={commit_workers}"
    )

    def target_reached() -> bool:
        return not config.continuous_collection and len(accepted_tasks) >= config.n_tasks

    def budget_reached(next_candidate_id: int) -> bool:
        if next_candidate_id >= max_candidates:
            return True
        if (
            config.continuous_collection
            and config.collection_rollout_budget > 0
            and simulation_rollouts >= config.collection_rollout_budget
        ):
            return True
        return bool(
            config.continuous_collection
            and config.collection_time_budget_seconds > 0
            and time.monotonic() - started >= config.collection_time_budget_seconds
        )

    def make_job(candidate_id: int) -> _CandidateJob:
        candidate_seed = _derive_candidate_seed(config.seed, candidate_id)
        rng = random.Random(candidate_seed)
        requested_source = (
            _draw_source_kind(
                rng,
                config,
                source_feedback if config.adaptive_sampling else {},
            )
            if config.open_ended_generation
            else "base"
        )
        available_parents = parent_pool
        if config.map_elites and map_archive:
            available_parents = [
                elite_task
                for elites in map_archive.values()
                for _, elite_task in elites
            ]
        source_kind, parent_ids, parents = _assign_candidate_parents(
            requested_source,
            available_parents,
            parent_leases,
            map_cell_counts,
            rng,
        )
        job_config = replace(
            config,
            n_tasks=1,
            seed=candidate_seed,
            cpu_cores=1,
            continuous_collection=False,
            filter_max_attempts=1,
        )
        return _CandidateJob(
            candidate_id=candidate_id,
            global_seed=config.seed,
            candidate_seed=candidate_seed,
            source_kind=source_kind,
            parent_ids=parent_ids,
            parents=parents,
            bucket_coverage=dict(bucket_coverage),
            config=job_config,
        )

    def commit_batch(
        results: Sequence[_CandidateResult],
        validation_executor: ThreadPoolExecutor,
    ) -> None:
        """Validate one snapshot batch and publish accepted candidates atomically."""

        nonlocal simulation_rollouts
        ordered = sorted(results, key=lambda item: item.candidate_id)
        candidates: list[_CandidateResult] = []
        for result in ordered:
            for parent_id in result.parent_ids:
                parent_leases[parent_id] -= 1
            source_stats = source_feedback.setdefault(result.source_kind, Counter())
            source_stats["generated"] += 1
            if result.task is None:
                reason = result.reject_reason or "worker_reject"
                source_stats[reason] += 1
                reject_counts[reason] += 1
                continue
            if (
                result.task_hash is None
                or result.parameter_signature is None
                or result.elite_cell is None
                or result.elite_quality is None
            ):
                raise RuntimeError("Worker returned incomplete candidate features.")
            simulation_rollouts += _candidate_rollouts(result.task)
            candidates.append(result)

        if not candidates:
            return

        # The three expensive, read-only queries share one immutable snapshot.
        parameter_future = validation_executor.submit(
            _nearest_distances_batch,
            [item.parameter_signature for item in candidates],
            parameter_archive,
        )
        behavior_candidates = [
            item for item in candidates if item.behavior_signature is not None
        ]
        behavior_future = validation_executor.submit(
            _nearest_distances_batch,
            [item.behavior_signature for item in behavior_candidates],
            behavior_archive,
        )
        prediction_future = validation_executor.submit(
            predictor.predict_many,
            [item.parameter_signature for item in candidates],
        )
        parameter_distances = parameter_future.result()
        behavior_distances = behavior_future.result()
        predictions = prediction_future.result()
        behavior_by_id = {
            item.candidate_id: float(distance)
            for item, distance in zip(behavior_candidates, behavior_distances)
        }

        batch_hashes: set[str] = set()
        new_evaluated_hashes: set[str] = set()
        batch_parameters: list[np.ndarray] = []
        batch_behaviors: list[np.ndarray] = []
        accepted_records: list[
            tuple[_CandidateResult, float, float, dict[str, float] | None, float]
        ] = []

        def reject(result: _CandidateResult, reason: str) -> None:
            reject_counts[reason] += 1
            source_feedback.setdefault(result.source_kind, Counter())[reason] += 1

        for result, archive_parameter_distance, prediction in zip(
            candidates, parameter_distances, predictions
        ):
            if target_reached() or (
                not config.continuous_collection
                and len(accepted_tasks) + len(accepted_records) >= config.n_tasks
            ):
                break
            digest = result.task_hash
            parameter_signature = result.parameter_signature
            assert digest is not None and parameter_signature is not None

            # The earliest candidate ID owns an exact hash even if a later
            # parameter/behavior filter rejects it, matching sequential semantics.
            if digest in evaluated_hashes or digest in batch_hashes:
                reject(result, "exact_duplicate")
                continue
            batch_hashes.add(digest)
            new_evaluated_hashes.add(digest)

            parameter_distance = float(archive_parameter_distance)
            if batch_parameters:
                parameter_distance = min(
                    parameter_distance,
                    _nearest_distance(parameter_signature, batch_parameters),
                )
            if parameter_distance < config.parameter_distance_threshold:
                reject(result, "parameter_duplicate")
                continue

            behavior_signature = result.behavior_signature
            behavior_distance = behavior_by_id.get(result.candidate_id, math.inf)
            if behavior_signature is not None and batch_behaviors:
                behavior_distance = min(
                    behavior_distance,
                    _nearest_distance(behavior_signature, batch_behaviors),
                )
            if behavior_distance < config.behavior_distance_threshold:
                reject(result, "behavior_duplicate")
                continue

            novelty_score = (
                1.0
                if not math.isfinite(parameter_distance)
                else float(np.clip(parameter_distance / 0.20, 0.0, 1.0))
            )
            uncertainty_score = 1.0 if prediction is None else prediction["uncertainty"]
            acquisition_score = (
                config.acquisition_novelty_weight * novelty_score
                + config.acquisition_uncertainty_weight * uncertainty_score
            )
            if (
                config.filter_by_win_rate
                and prediction is not None
                and acquisition_score < config.acquisition_min_score
            ):
                reject(result, "acquisition_reject")
                continue

            batch_parameters.append(parameter_signature)
            if behavior_signature is not None:
                batch_behaviors.append(behavior_signature)
            accepted_records.append(
                (
                    result,
                    parameter_distance,
                    behavior_distance,
                    prediction,
                    acquisition_score,
                )
            )

        # Atomic publication: no worker or validation thread can observe a
        # partially updated archive version.
        evaluated_hashes.update(new_evaluated_hashes)
        predictor_samples: list[tuple[np.ndarray, dict[str, Any], float]] = []
        for (
            result,
            parameter_distance,
            behavior_distance,
            prediction,
            acquisition_score,
        ) in accepted_records:
            task = result.task
            parameter_signature = result.parameter_signature
            digest = result.task_hash
            elite_cell = result.elite_cell
            elite_quality = result.elite_quality
            assert (
                task is not None
                and parameter_signature is not None
                and digest is not None
                and elite_cell is not None
                and elite_quality is not None
            )
            source_stats = source_feedback.setdefault(result.source_kind, Counter())
            source_stats["accepted"] += 1
            metadata = task.setdefault("metadata", {})
            metadata["canonical_hash"] = digest
            metadata["novelty"] = {
                "parameter_distance": (
                    parameter_distance if math.isfinite(parameter_distance) else None
                ),
                "behavior_distance": (
                    behavior_distance if math.isfinite(behavior_distance) else None
                ),
            }
            metadata["surrogate_prediction"] = prediction
            metadata["acquisition_score"] = acquisition_score
            metadata["adaptive_sampling"] = {
                "enabled": config.adaptive_sampling,
                "source_group": result.source_kind,
                "source_stats": dict(source_stats),
                "scope": "global_microbatch_coordinator",
            }
            metadata["map_elites"] = {
                "cell": list(elite_cell),
                "quality": elite_quality,
                "scope": "global_microbatch_coordinator",
            }
            task["task_id"] = f"task_{len(accepted_tasks):06d}_{digest[:10]}"
            accepted_tasks.append(task)
            parameter_archive.append(parameter_signature)
            if result.behavior_signature is not None:
                behavior_archive.append(result.behavior_signature)
            eval_result = metadata.get("filter_eval")
            if isinstance(eval_result, dict):
                predictor_samples.append(
                    (parameter_signature, eval_result, -elite_quality)
                )
            parent_pool.append(copy.deepcopy(task))
            scenario_key = _task_bucket_key(task)
            if scenario_key is not None:
                bucket_coverage[scenario_key] += 1
            map_cell_counts[elite_cell] += 1
            elites = map_archive.setdefault(elite_cell, [])
            elites.append((elite_quality, task))
            elites.sort(key=lambda item: item[0], reverse=True)
            del elites[config.map_elites_per_cell :]
        predictor.update_many(predictor_samples)

    try:
        # ``forkserver`` is Linux-friendly and avoids directly forking a
        # multithreaded JAX runtime.
        context = multiprocessing.get_context("forkserver")
        worker_counter = context.Value("i", 0)
        with (
            ThreadPoolExecutor(max_workers=commit_workers) as validation_executor,
            ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=context,
                initializer=_initialize_cpu_worker,
                initargs=(worker_counter,),
            ) as executor,
        ):
            futures: dict[Any, _CandidateJob] = {}
            completed: dict[int, _CandidateResult] = {}
            next_candidate_id = 0
            batch_started_at: float | None = None
            pipeline_limit = max(worker_count, commit_batch_size)
            while not target_reached():
                while (
                    len(futures) + len(completed) < pipeline_limit
                    and not budget_reached(next_candidate_id)
                ):
                    job = make_job(next_candidate_id)
                    futures[executor.submit(_sample_candidate_cpu_worker, job)] = job
                    next_candidate_id += 1

                batch_timed_out = bool(
                    completed
                    and batch_started_at is not None
                    and time.monotonic() - batch_started_at >= commit_timeout_seconds
                )
                if completed and (
                    len(completed) >= commit_batch_size
                    or not futures
                    or batch_timed_out
                ):
                    batch = list(completed.values())
                    completed.clear()
                    batch_started_at = None
                    commit_batch(batch, validation_executor)
                    print(
                        f"candidates_dispatched={next_candidate_id}/{max_candidates} "
                        f"batch_committed={len(batch)} "
                        f"accepted={len(accepted_tasks)}/{config.n_tasks} "
                        f"rejected={sum(reject_counts.values())}",
                        flush=True,
                    )
                    continue
                if not futures:
                    break
                wait_timeout = None
                if completed and batch_started_at is not None:
                    wait_timeout = max(
                        0.0,
                        commit_timeout_seconds
                        - (time.monotonic() - batch_started_at),
                    )
                done, _ = wait(
                    tuple(futures),
                    timeout=wait_timeout,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    job = futures.pop(future)
                    result = future.result()
                    if result.candidate_id != job.candidate_id:
                        raise RuntimeError("Worker returned a mismatched candidate_id.")
                    completed[result.candidate_id] = result
                    if batch_started_at is None:
                        batch_started_at = time.monotonic()
            for future in futures:
                future.cancel()
    finally:
        for name, previous in previous_environment.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous

    if not config.continuous_collection and len(accepted_tasks) != config.n_tasks:
        raise RuntimeError(
            f"Central sampler produced {len(accepted_tasks)} unique tasks, expected "
            f"{config.n_tasks}, after {max_candidates} candidate IDs; "
            f"rejections={dict(reject_counts)}. Increase filter_max_attempts."
        )
    if config.continuous_collection and not accepted_tasks:
        raise RuntimeError(
            "Continuous collection exhausted its budget without an accepted task; "
            f"rejections={dict(reject_counts)}."
        )
    print(
        f"Central CPU sampling complete: {len(accepted_tasks)} unique tasks; "
        f"rejections={dict(reject_counts)}, rollouts={simulation_rollouts}."
    )
    return accepted_tasks


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
            "num_seeds": _discovery_num_seeds(config),
            "epsilon": _evaluation_epsilon(config),
            "evaluation_epsilon": _evaluation_epsilon(config),
            "discovery_num_seeds": _discovery_num_seeds(config),
            "discovery_seed_offset": config.discovery_seed_offset,
            "seed_phases": {
                "A": "discovery_and_repair",
                "B": "adaptive_high_confidence_confirmation",
            },
            "discovery_seed_stages": list(config.discovery_seed_stages),
            "discovery_min_accept_seeds": config.discovery_min_accept_seeds,
            "discovery_candidate_win_rate_min": config.discovery_candidate_win_rate_min,
            "discovery_candidate_win_rate_max": config.discovery_candidate_win_rate_max,
            "discovery_max_abs_hp_margin": config.discovery_max_abs_hp_margin,
            "max_episode_steps": config.filter_max_episode_steps,
            "reject_all_truncated": config.filter_reject_all_truncated,
            "max_truncation_rate": config.max_truncation_rate,
            "max_no_interaction_rate": config.max_no_interaction_rate,
            "continuous_sampling": config.continuous_sampling,
            "continuous_map_sampling": config.continuous_map_sampling,
            "continuous_stat_sampling": config.continuous_stat_sampling,
            "continuous_distance_sampling": config.continuous_distance_sampling,
            "continuous_spread_sampling": config.continuous_spread_sampling,
            "continuous_zone_strength_sampling": config.continuous_zone_strength_sampling,
            "health_fraction_sampling": config.health_fraction_sampling,
            "static_engagement_filter": config.static_engagement_filter,
            "static_engagement_dt": config.static_engagement_dt,
            "static_engagement_max_step_fraction": config.static_engagement_max_step_fraction,
            "static_stomp_filter": config.static_stomp_filter,
            "static_stomp_min_episode_fraction": config.static_stomp_min_episode_fraction,
            "static_stomp_min_ttk_ratio": config.static_stomp_min_ttk_ratio,
            "static_balance_repair": config.static_balance_repair,
            "static_balance_max_gap": config.static_balance_max_gap,
            "simulation_balance_repair": config.simulation_balance_repair,
            "simulation_repair_rounds": config.simulation_repair_rounds,
            "directed_repair_distance_factor": config.directed_repair_distance_factor,
            "directed_repair_map_factor": config.directed_repair_map_factor,
            "directed_repair_damage_factor": config.directed_repair_damage_factor,
            "directed_repair_max_offense_scale": config.directed_repair_max_offense_scale,
            "confirmation_enabled": config.confirmation_enabled,
            "confirmation_seed_stages": list(config.confirmation_seed_stages),
            "confirmation_min_accept_seeds": config.confirmation_min_accept_seeds,
            "confirmation_balance_probability": config.confirmation_balance_probability,
            "confirmation_early_reject_probability": config.confirmation_early_reject_probability,
            "confirmation_seed_offset": config.confirmation_seed_offset,
            "confirmation_win_rate_min": config.confirmation_win_rate_min,
            "confirmation_win_rate_max": config.confirmation_win_rate_max,
            "confirmation_team_win_rate_min": config.confirmation_team_win_rate_min,
            "confirmation_team_win_rate_max": config.confirmation_team_win_rate_max,
            "confirmation_max_side_bias": config.confirmation_max_side_bias,
            "confirmation_max_abs_hp_margin": config.confirmation_max_abs_hp_margin,
            "confirmation_hp_ci_max_abs_bound": config.confirmation_hp_ci_max_abs_bound,
            "heuristic": config.heuristic,
            "physics": config.physics,
        }
    generator_protocol = {
        "execution_backend": "cpu",
        "cpu_cores": _parallel_worker_count(config),
        "parallel_scheduler": "deterministic_microbatch_coordinator",
        "commit_batch_size": (
            config.commit_batch_size
            if config.commit_batch_size > 0
            else 2 * _parallel_worker_count(config)
        ),
        "commit_workers": config.commit_workers,
        "commit_timeout_ms": config.commit_timeout_ms,
        "archive_query": "vectorized_snapshot_rms_distance",
        "predictor_update": "frozen_snapshot_prediction_then_batch_update",
        "candidate_seed_derivation": "sha256(global_seed, candidate_id)",
        "global_deduplication": ["task_hash", "parameter_signature", "behavior_signature"],
        "parent_assignment": "global_map_elites_least_leased",
        "open_ended_generation": config.open_ended_generation,
        "free_generation_ratio": config.free_generation_ratio,
        "parent_mutation_ratio": config.parent_mutation_ratio,
        "parent_crossover_ratio": config.parent_crossover_ratio,
        "parameter_distance_threshold": config.parameter_distance_threshold,
        "behavior_distance_threshold": config.behavior_distance_threshold,
        "max_n_ally": config.max_n_ally,
        "max_n_enemy": config.max_n_enemy,
        "max_n_zone": config.max_n_zone,
        "map_elites": config.map_elites,
        "map_elites_per_cell": config.map_elites_per_cell,
        "surrogate_enabled": config.surrogate_enabled,
        "surrogate_min_samples": config.surrogate_min_samples,
        "surrogate_ridge": config.surrogate_ridge,
        "acquisition_min_score": config.acquisition_min_score,
        "acquisition_novelty_weight": config.acquisition_novelty_weight,
        "acquisition_uncertainty_weight": config.acquisition_uncertainty_weight,
        "continuous_collection": config.continuous_collection,
        "adaptive_sampling": config.adaptive_sampling,
        "adaptive_sampling_learning_rate": config.adaptive_sampling_learning_rate,
        "adaptive_sampling_exploration": config.adaptive_sampling_exploration,
        "collection_candidate_budget": config.collection_candidate_budget,
        "collection_rollout_budget": config.collection_rollout_budget,
        "collection_time_budget_seconds": config.collection_time_budget_seconds,
    }
    output = save_task_bank(
        config.output,
        tasks,
        seed=config.seed,
        physics=config.physics,
        heuristic=config.heuristic,
        max_n_ally=(
            config.max_n_ally
            if config.programmatic_ratio > 0 or config.open_ended_generation
            else None
        ),
        max_n_enemy=(
            config.max_n_enemy
            if config.programmatic_ratio > 0 or config.open_ended_generation
            else None
        ),
        max_n_zone=(
            config.max_n_zone
            if config.programmatic_ratio > 0 or config.open_ended_generation
            else None
        ),
        filter_protocol=filter_protocol,
        generator_protocol=generator_protocol,
    )
    saved_bank = load_task_bank(output)
    schema = saved_bank["manifest"]["schema"]
    limits = (schema["max_n_ally"], schema["max_n_enemy"], schema["max_n_zone"])
    print(f"Saved {len(tasks)} tasks to {output} with schema limits {limits}.")


if __name__ == "__main__":
    main()
