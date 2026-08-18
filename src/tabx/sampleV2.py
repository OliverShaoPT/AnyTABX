"""Distribution-expanding, CPU-parallel task sampling for TABX.

``sampleV2`` is intentionally implemented in a new module.  It does not
modify the legacy generators.  The sampler combines four independent gates:

* explicit quotas over six mutually exclusive team/zone scale strata;
* a fixed 827-D, symmetry-invariant environment descriptor for 20 units per
  team and 20 zones;
* reference-only novelty thresholds, so candidates must leave the existing
  task archive before they consume rollout time;
* orientation-paired A/B balance certification on CPU worker processes.

Run with::

    python -m src.tabx.sampleV2 \
        --reference-task-file task_files/balanced_tasks_v1.json \
        --output outputs/sampleV2_tasks.json \
        --n-tasks 300 \
        --cpu-cores 8
"""

from __future__ import annotations

import copy
import io
import json
import math
import os
import random
import secrets
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import redirect_stdout
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

# sampleV2 is CPU-only even when imported by a launcher instead of executed
# with ``python -m``.  These values must be set before importing the JAX-backed
# legacy certification interfaces below.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_SKIP_CUDA_CONSTRAINTS_CHECK"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)

import numpy as np
import tyro

from src.tabx import sampleV1 as v1
from src.tabx import sample_task as base
from src.tabx import sample_task_balanced_diverse as qd
from src.tabx import task_generators as generators


N_UNIT_TYPES = 9
N_ZONE_TYPES = 4
ATTACK_TYPE_VALUES = (0, 1)
PROPOSAL_KINDS = (
    "global_constructive",
    "radical_mutation",
    "programmatic",
)
SCALE_STRATA = (
    "legacy_scale",
    "large_team",
    "xlarge_team",
    "high_zone",
    "xhigh_zone",
    "joint_ood",
)
GROUP_WEIGHTS = {
    "roster": 0.25,
    "initial_geometry": 0.30,
    "unit_attributes": 0.20,
    "terrain_and_map": 0.25,
}


@dataclass(frozen=True)
class DescriptorLayout:
    group_slices: dict[str, slice]
    dimension: int


@dataclass(frozen=True)
class ReferenceScaler:
    center: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        return (array - self.center) / self.scale


@dataclass(frozen=True)
class SampleV2Config(v1.SampleV1Config):
    """Configuration for reference-distant, scale-stratified collection."""

    output: str = "outputs/sampleV2_tasks.json"
    n_tasks: int = 300
    cpu_cores: int = 8

    max_n_ally: int = 20
    max_n_enemy: int = 20
    max_n_zone: int = 20

    win_rate_min: float = 0.30
    win_rate_max: float = 0.70
    discovery_candidate_win_rate_min: float = 0.20
    discovery_candidate_win_rate_max: float = 0.80
    confirmation_win_rate_min: float = 0.30
    confirmation_win_rate_max: float = 0.70
    confirmation_team_win_rate_min: float = 0.30
    confirmation_team_win_rate_max: float = 0.70

    # Every incremental rollout call has 16 seeds.  This keeps one JIT shape.
    discovery_seed_stages: tuple[int, ...] = (16, 32, 48, 64)
    discovery_min_accept_seeds: int = 32
    confirmation_seed_stages: tuple[int, ...] = (
        16,
        32,
        48,
        64,
        80,
        96,
        112,
        128,
    )
    confirmation_min_accept_seeds: int = 64
    jax_persistent_cache: bool = True
    jax_cache_dir: str = ".jax_cache/sampleV2_cpu"
    jax_cache_prewarm: bool = True

    # 65% fresh restarts, 30% radical restarts from a parent, and only 5%
    # legacy programmatic generation.  Local mutation/crossover are disabled.
    programmatic_ratio: float = 0.05
    free_generation_ratio: float = 0.65
    parent_mutation_ratio: float = 0.30
    parent_crossover_ratio: float = 0.0
    radical_mutation_fraction: float = 1.0

    # Mutually exclusive final-bank proportions.  The default places 80% of
    # tasks beyond the old 10-unit/10-zone support.
    scale_stratum_ratios: tuple[float, ...] = (
        0.20,
        0.20,
        0.15,
        0.15,
        0.10,
        0.20,
    )

    reference_task_file: str = "task_files/balanced_tasks_v1.json"
    environment_min_distance: float = 0.0
    environment_nn_quantile: float = 0.90
    compact_anchor_min_distance: float = 0.0
    compact_anchor_nn_quantile: float = 0.75
    pool_environment_distance_scale: float = 0.50

    diversity_candidate_budget_multiplier: int = 60
    certified_pool_multiplier: float = 1.50
    collection_time_budget_seconds: float = 0.0
    progress_interval_candidates: int = 250

    global_map_scale_max: float = 1.50
    radical_replace_fraction_min: float = 0.40
    radical_replace_fraction_max: float = 0.80

    # Final selector: environmental novelty is the leading soft objective;
    # existing coverage and behavior constraints remain hard/secondary.
    environment_novelty_weight: float = 0.60
    coverage_selection_weight: float = 0.25
    parameter_selection_weight: float = 0.075
    behavior_selection_weight: float = 0.075


@dataclass(frozen=True)
class _ReferenceArchive:
    compact_signatures: tuple[np.ndarray, ...]
    compact_threshold: float
    environment_raw: np.ndarray
    environment_scaled: np.ndarray
    environment_threshold: float
    scaler: ReferenceScaler
    layout: DescriptorLayout
    report: dict[str, Any]
    prewarm_task: dict[str, Any]


@dataclass(frozen=True)
class _V2ParallelJob:
    candidate_id: int
    target: tuple[str, ...]
    proposal_kind: str
    certification_source_kind: str
    scale_stratum: str
    parent_ids: tuple[str, ...]
    raw_task: dict[str, Any]
    config: SampleV2Config


@dataclass
class _V2ParallelResult:
    candidate_id: int
    target: tuple[str, ...]
    proposal_kind: str
    scale_stratum: str
    parent_ids: tuple[str, ...]
    worker_id: int
    worker_pid: int
    outcome: qd._CandidateOutcome


def _flat(
    scenario: dict[str, Any],
    name: str,
    *,
    dtype: Any = float,
) -> np.ndarray:
    return np.asarray(scenario[name], dtype=dtype).reshape(-1)


def _descriptor_variant(
    task: dict[str, Any],
    *,
    mirror_x: bool,
    mirror_y: bool,
    flip_teams: bool,
) -> tuple[np.ndarray, DescriptorLayout]:
    """Describe one symmetry variant in a fixed 20/20 layout."""

    scenario = task["scenario"]
    teams = _flat(scenario, "teams", dtype=int)
    unit_ids = _flat(scenario, "unit_ids", dtype=int)
    if np.any(unit_ids < 0) or np.any(unit_ids >= N_UNIT_TYPES):
        raise ValueError("sampleV2 descriptor expects unit IDs in [0, 8].")

    positions = np.asarray(scenario["positions"], dtype=float).reshape(-1, 2).copy()
    rotations = _flat(scenario, "rotations")
    sx = -1.0 if mirror_x else 1.0
    sy = -1.0 if mirror_y else 1.0
    positions[:, 0] *= sx
    positions[:, 1] *= sy
    heading_x = np.cos(rotations) * sx
    heading_y = np.sin(rotations) * sy

    grid = task.get("grid_info", {})
    width = max(float(grid.get("max_field_width", 121.0)), 1e-6)
    height = max(float(grid.get("max_field_height", 78.0)), 1e-6)
    normalized_positions = positions / np.asarray([width, height])

    health = _flat(scenario, "healths")
    damage = _flat(scenario, "attack_damages")
    cooldown = _flat(scenario, "attack_cooldowns")
    attack_range = _flat(scenario, "attack_ranges")
    attack_type = _flat(scenario, "attack_types", dtype=int)
    speed = _flat(scenario, "speeds")
    sight_angle = _flat(scenario, "sight_angles")
    radius_key = "body_radii" if "body_radii" in scenario else "body_radiuss"
    body_radius = _flat(scenario, radius_key)
    body_weight = _flat(scenario, "body_weights")
    attack_type_index = {
        value: index for index, value in enumerate(ATTACK_TYPE_VALUES)
    }

    roster_features: list[float] = []
    geometry_features: list[float] = []
    attribute_features: list[float] = []
    physical_team_order = (1, 0) if flip_teams else (0, 1)
    for physical_team in physical_team_order:
        indexes = np.flatnonzero(teams == physical_team)
        if len(indexes) > 20:
            raise ValueError(f"Team has {len(indexes)} units; sampleV2 maximum is 20.")
        order = np.lexsort(
            (
                health[indexes],
                normalized_positions[indexes, 1],
                normalized_positions[indexes, 0],
                unit_ids[indexes],
            )
        )
        indexes = indexes[order]
        counts = np.bincount(unit_ids[indexes], minlength=N_UNIT_TYPES)[:N_UNIT_TYPES]
        roster_features.extend(counts.astype(float).tolist())
        roster_features.append(float(len(indexes)))

        for slot in range(20):
            if slot < len(indexes):
                index = indexes[slot]
                geometry_features.extend(
                    [
                        1.0,
                        float(normalized_positions[index, 0]),
                        float(normalized_positions[index, 1]),
                        float(heading_x[index]),
                        float(heading_y[index]),
                    ]
                )
                attack_one_hot = [0.0] * len(ATTACK_TYPE_VALUES)
                kind = int(attack_type[index])
                if kind not in attack_type_index:
                    raise ValueError(f"Unsupported attack type {kind}.")
                attack_one_hot[attack_type_index[kind]] = 1.0
                attribute_features.extend(
                    [
                        float(health[index]),
                        float(damage[index]),
                        float(cooldown[index]),
                        float(attack_range[index]),
                        *attack_one_hot,
                        float(speed[index]),
                        float(sight_angle[index]),
                        float(body_radius[index]),
                        float(body_weight[index]),
                    ]
                )
            else:
                geometry_features.extend([0.0] * 5)
                attribute_features.extend([0.0] * 10)

    zones = task["zone_scenario"]
    n_zone = int(zones["n_zone"])
    if n_zone > 20:
        raise ValueError(f"Scene has {n_zone} zones; sampleV2 maximum is 20.")
    zone_types = np.asarray(zones["zone_type"], dtype=int).reshape(-1)[:n_zone]
    zone_positions = (
        np.asarray(zones["position"], dtype=float).reshape(-1, 2)[:n_zone].copy()
    )
    zone_axes = np.asarray(zones["axes"], dtype=float).reshape(-1, 2)[:n_zone]
    zone_effects = (
        np.asarray(zones["effect_value"], dtype=float).reshape(-1)[:n_zone]
    )
    zone_positions[:, 0] *= sx
    zone_positions[:, 1] *= sy
    zone_positions /= np.asarray([width, height])
    normalized_axes = zone_axes / np.asarray([width, height])
    if n_zone:
        zone_order = np.lexsort(
            (
                zone_effects,
                normalized_axes[:, 1],
                normalized_axes[:, 0],
                zone_positions[:, 1],
                zone_positions[:, 0],
                zone_types,
            )
        )
    else:
        zone_order = np.asarray([], dtype=int)

    terrain_features: list[float] = [
        float(grid.get("grid_width", 28.0)),
        float(grid.get("grid_height", 18.0)),
        width,
        height,
        float(grid.get("margin_width", 0.0)),
        float(grid.get("margin_height", 0.0)),
        float(n_zone),
    ]
    for slot in range(20):
        if slot < n_zone:
            index = int(zone_order[slot])
            zone_type = int(zone_types[index])
            terrain_features.extend(
                [
                    1.0,
                    *[float(zone_type == value) for value in range(N_ZONE_TYPES)],
                    float(zone_positions[index, 0]),
                    float(zone_positions[index, 1]),
                    float(normalized_axes[index, 0]),
                    float(normalized_axes[index, 1]),
                    float(zone_effects[index]),
                ]
            )
        else:
            terrain_features.extend([0.0] * 10)

    groups = {
        "roster": np.asarray(roster_features, dtype=float),
        "initial_geometry": np.asarray(geometry_features, dtype=float),
        "unit_attributes": np.asarray(attribute_features, dtype=float),
        "terrain_and_map": np.asarray(terrain_features, dtype=float),
    }
    values: list[np.ndarray] = []
    group_slices: dict[str, slice] = {}
    start = 0
    for name in GROUP_WEIGHTS:
        group = groups[name]
        values.append(group)
        group_slices[name] = slice(start, start + len(group))
        start += len(group)
    return (
        np.concatenate(values),
        DescriptorLayout(group_slices=group_slices, dimension=start),
    )


def _environment_descriptor(
    task: dict[str, Any],
) -> tuple[np.ndarray, DescriptorLayout]:
    variants = [
        _descriptor_variant(
            task,
            mirror_x=mirror_x,
            mirror_y=mirror_y,
            flip_teams=flip_teams,
        )
        for mirror_x in (False, True)
        for mirror_y in (False, True)
        for flip_teams in (False, True)
    ]
    quantized = [(np.round(values, 12), layout) for values, layout in variants]
    return min(quantized, key=lambda item: tuple(item[0].tolist()))


def _descriptor_matrix(
    tasks: Iterable[dict[str, Any]],
) -> tuple[np.ndarray, DescriptorLayout]:
    rows: list[np.ndarray] = []
    layout: DescriptorLayout | None = None
    for task in tasks:
        row, current_layout = _environment_descriptor(task)
        if layout is None:
            layout = current_layout
        elif current_layout != layout:
            raise AssertionError("sampleV2 environment descriptor layout changed.")
        rows.append(row)
    if layout is None:
        raise ValueError("Cannot describe an empty task collection.")
    return np.stack(rows), layout


def _robust_scale(values: np.ndarray) -> np.ndarray:
    q25, q75 = np.percentile(values, [25, 75], axis=0)
    interquartile = q75 - q25
    standard_deviation = np.std(values, axis=0)
    return np.where(interquartile > 1e-9, interquartile, standard_deviation)


def _pooled_feature_scales(
    values: np.ndarray,
    presence: np.ndarray,
    *,
    default: float,
) -> np.ndarray:
    """Estimate repeated-slot scales from all occupied reference slots."""

    width = values.shape[-1]
    result = np.full(width, float(default), dtype=float)
    for feature in range(width):
        observed = values[..., feature][presence]
        if len(observed) <= 1:
            continue
        q25, q75 = np.percentile(observed, [25, 75])
        scale = max(float(q75 - q25), float(np.std(observed)))
        if scale > 1e-9:
            result[feature] = scale
    return result


def _fit_reference_scaler(
    values: np.ndarray,
    layout: DescriptorLayout,
) -> ReferenceScaler:
    """Fit only on references while retaining unseen slots 11-20."""

    center = np.median(values, axis=0)
    scale = _robust_scale(values)
    fallback = np.ones(layout.dimension, dtype=float)

    geometry_slice = layout.group_slices["initial_geometry"]
    geometry = values[:, geometry_slice].reshape(len(values), 2, 20, 5)
    unit_presence = geometry[..., 0] > 0.5
    geometry_fallback = _pooled_feature_scales(
        geometry,
        unit_presence,
        default=1.0,
    )
    geometry_fallback[0] = 1.0
    fallback[geometry_slice] = np.tile(geometry_fallback, 40)

    attribute_slice = layout.group_slices["unit_attributes"]
    attributes = values[:, attribute_slice].reshape(len(values), 2, 20, 10)
    attribute_fallback = _pooled_feature_scales(
        attributes,
        unit_presence,
        default=1.0,
    )
    fallback[attribute_slice] = np.tile(attribute_fallback, 40)

    terrain_slice = layout.group_slices["terrain_and_map"]
    terrain = values[:, terrain_slice]
    zone_slots = terrain[:, 7:].reshape(len(values), 20, 10)
    zone_presence = zone_slots[..., 0] > 0.5
    zone_fallback = _pooled_feature_scales(
        zone_slots,
        zone_presence,
        default=1.0,
    )
    zone_fallback[:5] = 1.0
    terrain_fallback = np.ones(207, dtype=float)
    terrain_fallback[7:] = np.tile(zone_fallback, 20)
    fallback[terrain_slice] = terrain_fallback

    scale = np.where(scale > 1e-9, scale, fallback)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return ReferenceScaler(center=center, scale=scale)


def _group_distance_matrix(
    first: np.ndarray,
    second: np.ndarray,
    layout: DescriptorLayout,
) -> np.ndarray:
    """Group-weighted RMS distance without a 3-D broadcast tensor."""

    total = np.zeros((len(first), len(second)), dtype=float)
    for name, weight in GROUP_WEIGHTS.items():
        section = layout.group_slices[name]
        left = first[:, section]
        right = second[:, section]
        dimension = max(1, left.shape[1])
        squared = (
            np.sum(np.square(left), axis=1)[:, None]
            + np.sum(np.square(right), axis=1)[None, :]
            - 2.0 * left @ right.T
        )
        total += weight * np.sqrt(np.maximum(squared, 0.0) / dimension)
    return total


def _nearest_environment_distance(
    scaled: np.ndarray,
    references: np.ndarray,
    layout: DescriptorLayout,
) -> float:
    if not len(references):
        return math.inf
    distances = _group_distance_matrix(
        np.asarray(scaled, dtype=float).reshape(1, -1),
        references,
        layout,
    )
    return float(np.min(distances))


def _reference_nn_values(
    scaled: np.ndarray,
    layout: DescriptorLayout,
) -> np.ndarray:
    distances = _group_distance_matrix(scaled, scaled, layout)
    np.fill_diagonal(distances, math.inf)
    return np.min(distances, axis=1)


def _largest_remainder_quotas(
    total: int,
    ratios: Sequence[float],
) -> dict[str, int]:
    raw = np.asarray(ratios, dtype=float) * total / float(sum(ratios))
    floors = np.floor(raw).astype(int)
    remainder = total - int(np.sum(floors))
    order = sorted(
        range(len(SCALE_STRATA)),
        key=lambda index: (-(raw[index] - floors[index]), index),
    )
    for index in order[:remainder]:
        floors[index] += 1
    return {
        name: int(floors[index])
        for index, name in enumerate(SCALE_STRATA)
    }


def _scale_quotas(config: SampleV2Config) -> dict[str, int]:
    return _largest_remainder_quotas(config.n_tasks, config.scale_stratum_ratios)


def _count_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value <= 3:
        return "1-3"
    if value <= 6:
        return "4-6"
    if value <= 10:
        return "7-10"
    if value <= 15:
        return "11-15"
    return "16-20"


def _task_scale_stratum(task: dict[str, Any]) -> str:
    teams = np.asarray(task["scenario"]["teams"], dtype=int).reshape(-1)
    team_size = max(int(np.count_nonzero(teams == team)) for team in (0, 1))
    zone_count = int(task["zone_scenario"]["n_zone"])
    if team_size > 10 and zone_count > 10:
        return "joint_ood"
    if team_size >= 16:
        return "xlarge_team"
    if team_size > 10:
        return "large_team"
    if zone_count >= 16:
        return "xhigh_zone"
    if zone_count > 10:
        return "high_zone"
    return "legacy_scale"


def _stratum_bounds(
    scale_stratum: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if scale_stratum == "legacy_scale":
        return (1, 10), (0, 10)
    if scale_stratum == "large_team":
        return (11, 15), (0, 10)
    if scale_stratum == "xlarge_team":
        return (16, 20), (0, 10)
    if scale_stratum == "high_zone":
        return (1, 10), (11, 15)
    if scale_stratum == "xhigh_zone":
        return (1, 10), (16, 20)
    if scale_stratum == "joint_ood":
        return (11, 20), (11, 20)
    raise ValueError(f"Unknown scale stratum: {scale_stratum!r}.")


def _build_exact_free_zones(
    rng: random.Random,
    scenario: dict[str, Any],
    grid_info: dict[str, Any],
    n_zone: int,
) -> dict[str, Any]:
    """Build exactly ``n_zone`` zones without changing the legacy generator."""

    if not 0 <= n_zone <= 20:
        raise ValueError("sampleV2 exact zone count must lie in [0, 20].")
    width = float(grid_info["max_field_width"])
    height = float(grid_info["max_field_height"])
    positions = np.asarray(scenario["positions"], dtype=float).reshape(-1, 2)
    radius_key = (
        "body_radii" if "body_radii" in scenario else "body_radiuss"
    )
    radii = np.asarray(scenario[radius_key], dtype=float).reshape(-1)

    zone_types: list[list[int]] = []
    centers: list[list[float]] = []
    axes_values: list[list[float]] = []
    effects: list[list[float]] = []
    for _ in range(n_zone):
        zone_type = rng.choice((1, 2, 3))
        axes = np.asarray(
            [
                rng.uniform(2.0, max(2.01, width * 0.18)),
                rng.uniform(2.0, max(2.01, height * 0.22)),
            ],
            dtype=float,
        )
        center: np.ndarray | None = None
        for _attempt in range(64):
            candidate = np.asarray(
                [
                    rng.uniform(-width / 2 + axes[0], width / 2 - axes[0]),
                    rng.uniform(-height / 2 + axes[1], height / 2 - axes[1]),
                ],
                dtype=float,
            )
            if zone_type != 1:
                center = candidate
                break
            normalized = ((positions - candidate) / (axes + radii[:, None])) ** 2
            if np.all(normalized.sum(axis=1) > 1.0):
                center = candidate
                break
        if center is None:
            # Lava needs spawn clearance.  Falling back to a non-lethal zone
            # preserves the exact structural quota without creating invalid
            # initial damage.
            zone_type = rng.choice((2, 3))
            center = np.asarray(
                [
                    rng.uniform(-width / 2 + axes[0], width / 2 - axes[0]),
                    rng.uniform(-height / 2 + axes[1], height / 2 - axes[1]),
                ],
                dtype=float,
            )
        if zone_type == 1:
            effect = rng.uniform(4.0, 16.0)
        elif zone_type == 3:
            effect = rng.uniform(0.1, 0.5)
        else:
            effect = 0.0
        zone_types.append([zone_type])
        centers.append(center.tolist())
        axes_values.append(axes.tolist())
        effects.append([effect])
    return {
        "n_zone": n_zone,
        "zone_type": zone_types,
        "position": centers,
        "axes": axes_values,
        "effect_value": effects,
    }


def _sample_bounded_roster(
    rng: random.Random,
    bounds: tuple[int, int],
    *,
    parent_roster: Sequence[int] | None,
    config: SampleV2Config,
) -> tuple[list[int], dict[str, Any] | None]:
    count = rng.randint(bounds[0], bounds[1])
    if parent_roster is None:
        return v1._sample_palette_roster(rng, count), None

    values = list(parent_roster)
    rng.shuffle(values)
    values = values[:count]
    while len(values) < count:
        values.append(rng.randrange(N_UNIT_TYPES))
    fraction = rng.uniform(
        config.radical_replace_fraction_min,
        config.radical_replace_fraction_max,
    )
    replace_count = min(
        len(values),
        max(1, int(math.ceil(len(values) * fraction))),
    )
    for index in rng.sample(range(len(values)), k=replace_count):
        values[index] = rng.randrange(N_UNIT_TYPES)
    return values, {
        "target_count": count,
        "replace_count": replace_count,
        "replace_fraction": fraction,
    }


def _match_bounded_opponent(
    rng: random.Random,
    ally: Sequence[int],
    ally_health: Sequence[float],
) -> tuple[list[int], list[float], dict[str, Any]]:
    """Keep counts within the selected stratum while favoring static balance."""

    target_price = float(generators.team_price(ally))
    target_effective = float(
        generators.team_effective_value(ally, ally_health)
    )
    draw = rng.random()
    if draw < 0.30:
        enemy = list(ally)
        enemy_health = list(ally_health)
        mode = "mirror"
    elif draw < 0.55:
        enemy = generators.unit_swap_team(
            rng,
            ally,
            len(ally),
            n_replace=max(1, int(math.ceil(len(ally) * 0.30))),
        )
        enemy_health = generators.project_health_fractions(
            rng,
            generators.unit_prices_list(enemy),
            target_effective,
            f_min=0.5,
            f_max=1.0,
        )
        mode = "unit_swap"
    else:
        best: list[int] | None = None
        best_difference = math.inf
        for _ in range(128):
            candidate = v1._sample_palette_roster(rng, len(ally))
            difference = abs(
                float(generators.team_price(candidate)) - target_price
            ) / max(target_price, 1.0)
            if difference < best_difference:
                best = candidate
                best_difference = difference
            if difference <= 0.06:
                break
        assert best is not None
        enemy = best
        enemy_health = generators.project_health_fractions(
            rng,
            generators.unit_prices_list(enemy),
            target_effective,
            f_min=0.5,
            f_max=1.0,
        )
        mode = "cost_match"

    enemy_price = float(generators.team_price(enemy))
    enemy_effective = float(
        generators.team_effective_value(enemy, enemy_health)
    )
    return enemy, enemy_health, {
        "match_mode": mode,
        "ally_price": target_price,
        "enemy_price": enemy_price,
        "price_rel_diff": abs(target_price - enemy_price)
        / max(target_price, 1.0),
        "ally_effective": target_effective,
        "enemy_effective": enemy_effective,
        "effective_rel_diff": abs(target_effective - enemy_effective)
        / max(target_effective, 1.0),
    }


def _generate_constructive_task(
    config: SampleV2Config,
    *,
    candidate_id: int,
    generation_seed: int,
    target: tuple[str, ...],
    scale_stratum: str,
    parent_roster: Sequence[int] | None = None,
    parent_digest: str | None = None,
) -> dict[str, Any]:
    seed = qd._derive_seed(generation_seed, candidate_id, "sampleV2-constructive")
    rng = random.Random(seed)
    descriptor = qd._target_descriptor(target)
    team_bounds, zone_bounds = _stratum_bounds(scale_stratum)

    ally, radical_info = _sample_bounded_roster(
        rng,
        team_bounds,
        parent_roster=parent_roster,
        config=config,
    )
    ally_health = [
        rng.uniform(config.health_frac_min, config.health_frac_max)
        for _ in ally
    ]
    enemy, enemy_health, match_info = _match_bounded_opponent(
        rng,
        ally,
        ally_health,
    )

    base_map_scale = qd._sample_scale(
        rng,
        config.map_scales,
        config.continuous_map_sampling,
        config.continuous_sampling,
    )
    density_scale = math.sqrt(max(len(ally) + len(enemy), 1) / 12.0)
    map_scale = min(
        config.global_map_scale_max,
        max(base_map_scale, base_map_scale * density_scale),
    )
    distance_scale = qd._sample_scale(
        rng,
        config.distance_scales,
        config.continuous_distance_sampling,
        config.continuous_sampling,
    )
    if len(ally) >= 16:
        distance_scale *= 0.65
    elif len(ally) > 10:
        distance_scale *= 0.80
    spread_scale = qd._sample_scale(
        rng,
        config.spread_scales,
        config.continuous_spread_sampling,
        config.continuous_sampling,
    )
    scenario, grid_info = generators.build_scenario(
        rng,
        ally,
        enemy,
        descriptor["layout"],
        descriptor["distance"],
        descriptor["spread"],
        map_scale,
        ally_health_fracs=ally_health,
        enemy_health_fracs=enemy_health,
        distance_scale=distance_scale,
        spread_scale=spread_scale,
    )
    requested_zone_count = rng.randint(zone_bounds[0], zone_bounds[1])
    zone_scenario = _build_exact_free_zones(
        rng,
        scenario,
        grid_info,
        requested_zone_count,
    )
    source: dict[str, Any]
    if parent_roster is None:
        source = {
            "kind": "global",
            "subtype": "sampleV2_constructive_restart",
        }
    else:
        source = {
            "kind": "mutation",
            "subtype": "sampleV2_radical_restart",
            "parent": parent_digest,
            "radical_mutation": radical_info,
        }
    task = {
        "grid_info": grid_info,
        "scenario": scenario,
        "zone_scenario": zone_scenario,
        "metadata": v1._custom_task_metadata(
            source=source,
            ally=ally,
            enemy=enemy,
            ally_health=ally_health,
            enemy_health=enemy_health,
            match_info=match_info,
            target=target,
            map_scale=map_scale,
            distance_scale=distance_scale,
            spread_scale=spread_scale,
        ),
    }
    task["metadata"]["sample_v2"] = {
        "scale_stratum": scale_stratum,
        "requested_team_count": len(ally),
        "requested_zone_count": requested_zone_count,
        "generation_policy": "scale_targeted_constructive",
    }
    return v1._finalize_custom_task(
        rng,
        task,
        config,
        target=target,
        candidate_id=candidate_id,
        generation_seed=generation_seed,
    )


def _generate_proposal(
    config: SampleV2Config,
    *,
    candidate_id: int,
    generation_seed: int,
    target: tuple[str, ...],
    proposal_kind: str,
    scale_stratum: str,
    parents: Sequence[qd._ParentPayload],
) -> tuple[dict[str, Any], str]:
    if proposal_kind == "global_constructive":
        return (
            _generate_constructive_task(
                config,
                candidate_id=candidate_id,
                generation_seed=generation_seed,
                target=target,
                scale_stratum=scale_stratum,
            ),
            "global_constructive",
        )
    if proposal_kind == "radical_mutation":
        if not parents:
            raise ValueError("Radical mutation requires one certified parent.")
        parent = parents[0]
        seed = qd._derive_seed(
            generation_seed,
            candidate_id,
            "sampleV2-radical-parent",
        )
        team = random.Random(seed).choice((0, 1))
        return (
            _generate_constructive_task(
                config,
                candidate_id=candidate_id,
                generation_seed=generation_seed,
                target=target,
                scale_stratum=scale_stratum,
                parent_roster=v1._parent_roster(parent.task, team),
                parent_digest=parent.digest,
            ),
            "mutation",
        )
    if proposal_kind != "programmatic":
        raise ValueError(f"Unknown sampleV2 proposal kind: {proposal_kind!r}.")
    if scale_stratum != "legacy_scale":
        raise ValueError("Programmatic proposals are restricted to legacy_scale.")
    legacy_config = replace(
        config,
        max_n_ally=min(10, config.max_n_ally),
        max_n_enemy=min(10, config.max_n_enemy),
        max_n_zone=min(10, config.max_n_zone),
    )
    task, source_kind = qd._generate_raw_candidate(
        legacy_config,
        candidate_id=candidate_id,
        generation_seed=generation_seed,
        target=target,
        parents=parents,
        forced_source_kind="programmatic",
    )
    task.setdefault("metadata", {}).setdefault("sample_v2", {}).update(
        {
            "scale_stratum": "legacy_scale",
            "generation_policy": "legacy_programmatic",
        }
    )
    return task, source_kind


def _derive_threshold(
    nearest_values: Sequence[float],
    *,
    explicit: float,
    quantile: float,
    floor: float,
) -> tuple[float, str]:
    if explicit > 0:
        return float(explicit), "explicit"
    if not nearest_values:
        return float(floor), "single_reference_fallback"
    threshold = max(
        float(np.quantile(np.asarray(nearest_values, dtype=float), quantile)),
        float(floor),
    )
    return threshold, "reference_leave_one_out_quantile"


def _load_reference_archive(config: SampleV2Config) -> _ReferenceArchive:
    reference_path = Path(config.reference_task_file)
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Reference task bank does not exist: {reference_path.resolve()}"
        )
    bank = base.load_task_bank(reference_path)
    tasks = list(bank["tasks"])
    if not tasks:
        raise ValueError("Reference task bank must contain at least one task.")

    compact_signatures = tuple(
        qd._canonical_parameter_signature(task) for task in tasks
    )
    compact_nearest: list[float] = []
    if len(compact_signatures) > 1:
        for index, signature in enumerate(compact_signatures):
            distances = v1._parameter_distances(
                signature,
                compact_signatures,
            )
            distances[index] = math.inf
            compact_nearest.append(float(np.min(distances)))
    compact_threshold, compact_policy = _derive_threshold(
        compact_nearest,
        explicit=config.compact_anchor_min_distance,
        quantile=config.compact_anchor_nn_quantile,
        floor=config.parameter_distance_threshold,
    )

    environment_raw, layout = _descriptor_matrix(tasks)
    if layout.dimension != 827:
        raise AssertionError(
            f"sampleV2 fixed descriptor must be 827-D, got {layout.dimension}."
        )
    scaler = _fit_reference_scaler(environment_raw, layout)
    environment_scaled = scaler.transform(environment_raw)
    environment_nearest = (
        _reference_nn_values(environment_scaled, layout)
        if len(tasks) > 1
        else np.empty(0, dtype=float)
    )
    environment_threshold, environment_policy = _derive_threshold(
        environment_nearest.tolist(),
        explicit=config.environment_min_distance,
        quantile=config.environment_nn_quantile,
        floor=0.0,
    )
    report = {
        "file": str(reference_path),
        "n_tasks": len(tasks),
        "descriptor_dimension": layout.dimension,
        "descriptor_limits": {
            "max_units_per_team": 20,
            "max_zones": 20,
            "attack_type_values": list(ATTACK_TYPE_VALUES),
        },
        "scaler_fit": "reference_archive_only_with_semantic_slot_fallback",
        "compact_gate": {
            "threshold": compact_threshold,
            "threshold_policy": compact_policy,
            "nearest_neighbor_quantile": config.compact_anchor_nn_quantile,
            "reference_nn_min": (
                min(compact_nearest) if compact_nearest else None
            ),
            "reference_nn_median": (
                float(np.median(compact_nearest))
                if compact_nearest
                else None
            ),
            "reference_nn_max": (
                max(compact_nearest) if compact_nearest else None
            ),
        },
        "environment_gate": {
            "threshold": environment_threshold,
            "threshold_policy": environment_policy,
            "nearest_neighbor_quantile": config.environment_nn_quantile,
            "reference_nn_min": (
                float(np.min(environment_nearest))
                if len(environment_nearest)
                else None
            ),
            "reference_nn_median": (
                float(np.median(environment_nearest))
                if len(environment_nearest)
                else None
            ),
            "reference_nn_max": (
                float(np.max(environment_nearest))
                if len(environment_nearest)
                else None
            ),
        },
    }
    return _ReferenceArchive(
        compact_signatures=compact_signatures,
        compact_threshold=compact_threshold,
        environment_raw=environment_raw,
        environment_scaled=environment_scaled,
        environment_threshold=environment_threshold,
        scaler=scaler,
        layout=layout,
        report=report,
        prewarm_task=copy.deepcopy(tasks[0]),
    )


def _scaled_task_descriptor(
    task: dict[str, Any],
    reference: _ReferenceArchive,
) -> np.ndarray:
    raw, layout = _environment_descriptor(task)
    if layout != reference.layout:
        raise AssertionError("Candidate and reference descriptor layouts differ.")
    return reference.scaler.transform(raw)


def _prefilter_candidate(
    task: dict[str, Any],
    config: SampleV2Config,
    reference: _ReferenceArchive,
    pool_environment: Sequence[np.ndarray],
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    try:
        base.validate_task(task)
        compact_signature = qd._canonical_parameter_signature(task)
    except ValueError as exc:
        return None, None, f"prefilter_invalid:{exc}"

    compact_distance = v1._nearest_parameter_distance(
        compact_signature,
        reference.compact_signatures,
    )
    if compact_distance < reference.compact_threshold:
        return None, None, "prefilter_compact_reference_near"

    try:
        static_reason, static_values, engagement = qd._static_screen(task, config)
    except ValueError as exc:
        return None, None, f"prefilter_static_invalid:{exc}"
    if static_reason in {"fast_stomp", "static_imbalance_hard"}:
        return None, None, f"prefilter_{static_reason}"

    try:
        environment = _scaled_task_descriptor(task, reference)
    except ValueError as exc:
        return None, None, f"prefilter_descriptor_invalid:{exc}"
    reference_distance = _nearest_environment_distance(
        environment,
        reference.environment_scaled,
        reference.layout,
    )
    if reference_distance < reference.environment_threshold:
        return None, None, "prefilter_environment_reference_near"

    pool_distance = _nearest_environment_distance(
        environment,
        (
            np.asarray(pool_environment, dtype=float)
            if pool_environment
            else np.empty((0, reference.layout.dimension), dtype=float)
        ),
        reference.layout,
    )
    required_pool_distance = (
        reference.environment_threshold
        * config.pool_environment_distance_scale
    )
    if pool_distance < required_pool_distance:
        return None, None, "prefilter_environment_pool_near"

    metadata = task.setdefault("metadata", {}).setdefault(
        "sample_v2_prefilter",
        {},
    )
    metadata.update(
        {
            "compact_reference_distance": compact_distance,
            "required_compact_reference_distance": reference.compact_threshold,
            "environment_reference_distance": reference_distance,
            "required_environment_reference_distance": (
                reference.environment_threshold
            ),
            "environment_pool_distance": (
                None if not math.isfinite(pool_distance) else pool_distance
            ),
            "required_environment_pool_distance": required_pool_distance,
            "static_reason": static_reason,
            "static_team_values": static_values,
            "static_engagement": engagement,
            "rollout_skipped": False,
        }
    )
    return compact_signature, environment, None


def _proposal_base_weights(config: SampleV2Config) -> dict[str, float]:
    return {
        "global_constructive": float(config.free_generation_ratio),
        "radical_mutation": float(config.parent_mutation_ratio),
        "programmatic": float(config.programmatic_ratio),
    }


def _draw_proposal_kind(
    rng: random.Random,
    config: SampleV2Config,
    scale_stratum: str,
    pool: Sequence[qd._CertifiedCandidate],
    generated: Counter[str],
    confirmed: Counter[str],
) -> str:
    weights = _proposal_base_weights(config)
    if not pool:
        weights["radical_mutation"] = 0.0
    if scale_stratum != "legacy_scale":
        weights["programmatic"] = 0.0

    adapted: list[tuple[str, float]] = []
    for kind in PROPOSAL_KINDS:
        base_weight = weights[kind]
        if base_weight <= 0:
            adapted.append((kind, 0.0))
            continue
        success_rate = (confirmed[kind] + 1.0) / (generated[kind] + 2.0)
        multiplier = (
            1.0
            - config.source_adaptation_rate
            + config.source_adaptation_rate * 2.0 * success_rate
        )
        adapted.append(
            (
                kind,
                base_weight
                * max(multiplier, config.source_exploration_floor),
            )
        )
    index = v1._weighted_choice_index(
        rng,
        [weight for _, weight in adapted],
    )
    return adapted[index][0]


def _draw_scale_stratum(
    rng: random.Random,
    quotas: dict[str, int],
    confirmed: Counter[str],
    leases: Counter[str],
) -> str:
    unmet = {
        name
        for name in SCALE_STRATA
        if confirmed[name] + leases[name] < quotas[name]
    }
    weights: list[float] = []
    names: list[str] = []
    for name in SCALE_STRATA:
        target = quotas[name]
        if target <= 0:
            continue
        if unmet and name not in unmet:
            continue
        current = confirmed[name] + leases[name]
        deficit = max(target - current, 0)
        # Once all hard quotas have been leased, this floor creates the
        # surplus pool needed by farthest-point selection.  Until then, no CPU
        # rollout is spent on a stratum whose minimum is already satisfied.
        weights.append(
            float(deficit)
            if unmet
            else 0.08 * target
        )
        names.append(name)
    if not names:
        raise RuntimeError("No active sampleV2 scale stratum.")
    return names[v1._weighted_choice_index(rng, weights)]


def _choose_target(
    rng: random.Random,
    config: SampleV2Config,
    coverage: qd._CoverageState,
    feedback: qd._BucketFeedback,
) -> tuple[str, ...]:
    best: tuple[str, ...] | None = None
    best_score = -math.inf
    for _ in range(config.target_draws):
        target = qd._draw_target(rng, config)
        deficit = coverage.gain(qd._target_descriptor(target))
        accept_rate = max(
            feedback.estimated_accept_rate(target),
            config.target_accept_rate_floor,
        )
        score = deficit * math.sqrt(accept_rate)
        score += config.target_exploration * rng.random()
        if score > best_score:
            best = target
            best_score = score
    assert best is not None
    return best


def _assign_parent(
    proposal_kind: str,
    pool: Sequence[qd._CertifiedCandidate],
    parent_leases: Counter[str],
    rng: random.Random,
) -> tuple[qd._ParentPayload, ...]:
    if proposal_kind != "radical_mutation":
        return ()
    if not pool:
        raise RuntimeError("Radical mutation requires a certified parent.")
    least = min(parent_leases[candidate.digest] for candidate in pool)
    candidates = [
        candidate
        for candidate in pool
        if parent_leases[candidate.digest] == least
    ]
    parent = rng.choice(candidates)
    parent_leases[parent.digest] += 1
    return (
        qd._ParentPayload(
            task=copy.deepcopy(parent.task),
            digest=parent.digest,
        ),
    )


def _extend_candidate_cell(candidate: qd._CertifiedCandidate) -> str:
    scale_stratum = _task_scale_stratum(candidate.task)
    teams = np.asarray(
        candidate.task["scenario"]["teams"],
        dtype=int,
    ).reshape(-1)
    team_size = max(
        int(np.count_nonzero(teams == team)) for team in (0, 1)
    )
    zone_count = int(candidate.task["zone_scenario"]["n_zone"])
    candidate.map_cell = (
        *candidate.map_cell,
        scale_stratum,
        _count_bucket(team_size),
        _count_bucket(zone_count),
    )
    candidate.task.setdefault("metadata", {}).setdefault(
        "sample_v2",
        {},
    ).update(
        {
            "scale_stratum": scale_stratum,
            "actual_max_team_size": team_size,
            "actual_zone_count": zone_count,
            "team_size_bucket": _count_bucket(team_size),
            "zone_count_bucket": _count_bucket(zone_count),
        }
    )
    return scale_stratum


def _sample_candidate_cpu_worker(job: _V2ParallelJob) -> _V2ParallelResult:
    """A/B-certify one already-prefiltered task on a CPU worker."""

    try:
        with redirect_stdout(io.StringIO()):
            outcome = qd._certify_candidate(
                job.raw_task,
                job.certification_source_kind,
                job.config,
                candidate_id=job.candidate_id,
            )
        if outcome.candidate is not None:
            outcome.candidate.source_kind = job.proposal_kind
            _extend_candidate_cell(outcome.candidate)
    except (RuntimeError, ValueError) as exc:
        outcome = qd._CandidateOutcome(
            None,
            f"worker:{type(exc).__name__}:{exc}",
            "worker_error",
        )
    return _V2ParallelResult(
        candidate_id=job.candidate_id,
        target=job.target,
        proposal_kind=job.proposal_kind,
        scale_stratum=job.scale_stratum,
        parent_ids=job.parent_ids,
        worker_id=0 if qd._CPU_WORKER_ID < 0 else qd._CPU_WORKER_ID,
        worker_pid=os.getpid(),
        outcome=outcome,
    )


def _rollout_batch_sizes(config: SampleV2Config) -> tuple[int, ...]:
    return v1._rollout_batch_sizes(config)


def _configure_jax_persistent_cache(
    config: SampleV2Config,
) -> Path | None:
    if not config.jax_persistent_cache:
        return None
    cache_dir = Path(config.jax_cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    base.jax.config.update("jax_enable_compilation_cache", True)
    base.jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    base.jax.config.update(
        "jax_persistent_cache_min_compile_time_secs",
        0.0,
    )
    base.jax.config.update(
        "jax_persistent_cache_min_entry_size_bytes",
        -1,
    )
    return cache_dir


def _prewarm_rollout_cache(
    config: SampleV2Config,
    reference_task: dict[str, Any],
) -> dict[str, Any]:
    batch_sizes = _rollout_batch_sizes(config)
    if not config.jax_persistent_cache or not config.jax_cache_prewarm:
        return {
            "enabled": False,
            "batch_sizes": list(batch_sizes),
            "elapsed_seconds": 0.0,
        }

    _configure_jax_persistent_cache(config)
    from src.tabx import eval_task

    env_params, tabx_config = base.build_batched_env_params_from_tasks(
        [reference_task],
        physics=config.physics,
        heuristic=config.heuristic,
        max_n_ally=config.max_n_ally,
        max_n_enemy=config.max_n_enemy,
        max_n_zone=config.max_n_zone,
    )
    heuristic_params = eval_task.load_heuristic_params(
        config.heuristic,
        epsilon_override=base._evaluation_epsilon(config),
    )
    task_params = base.jax.tree.map(lambda value: value[0], env_params)
    task_params = eval_task._with_team_heuristics(
        task_params,
        heuristic_params,
        heuristic_params,
    )
    runner = eval_task._cached_episode_runner(
        tabx_config.max_n_ally,
        tabx_config.max_n_enemy,
        tabx_config.max_n_zone,
        config.filter_max_episode_steps,
    )

    started = time.monotonic()
    compile_seconds: dict[str, float] = {}
    for index, batch_size in enumerate(batch_sizes, start=1):
        shape_started = time.monotonic()
        print(
            "sampleV2 JAX cache warm-up: "
            f"shape={index}/{len(batch_sizes)}, seed_batch={batch_size}. "
            "A cold cache can take several minutes once.",
            flush=True,
        )
        seeds = np.arange(batch_size, dtype=np.uint32)
        keys = base.jax.vmap(base.jax.random.key)(base.jnp.asarray(seeds))
        executable = runner.lower(keys, task_params).compile()
        compile_seconds[str(batch_size)] = (
            time.monotonic() - shape_started
        )
        del executable
    elapsed = time.monotonic() - started
    print(
        "sampleV2 JAX cache warm-up complete: "
        f"batch_sizes={list(batch_sizes)}, elapsed={elapsed:.1f}s. "
        "Starting CPU workers.",
        flush=True,
    )
    return {
        "enabled": True,
        "batch_sizes": list(batch_sizes),
        "compile_seconds": compile_seconds,
        "elapsed_seconds": elapsed,
    }


def _prepare_rollout_cache(
    config: SampleV2Config,
    reference_task: dict[str, Any],
    worker_count: int,
) -> dict[str, Any]:
    if (
        worker_count <= 1
        or not config.jax_persistent_cache
        or not config.jax_cache_prewarm
    ):
        return _prewarm_rollout_cache(config, reference_task)
    context = qd._multiprocessing_context()
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=context,
    ) as warmup_executor:
        return warmup_executor.submit(
            _prewarm_rollout_cache,
            config,
            reference_task,
        ).result()


def _validate_config(config: SampleV2Config) -> None:
    qd._validate_config(config)
    if config.hard_task_mode:
        raise ValueError(
            "sampleV2 collects balanced tasks; hard_task_mode is unsupported."
        )
    if (
        config.max_n_ally != 20
        or config.max_n_enemy != 20
        or config.max_n_zone != 20
    ):
        raise ValueError(
            "sampleV2 uses a fixed 20-ally/20-enemy/20-zone descriptor and "
            "requires all three limits to equal 20."
        )
    if len(config.scale_stratum_ratios) != len(SCALE_STRATA):
        raise ValueError(
            f"scale_stratum_ratios must contain {len(SCALE_STRATA)} values."
        )
    if any(value < 0 for value in config.scale_stratum_ratios):
        raise ValueError("scale_stratum_ratios must be non-negative.")
    if sum(config.scale_stratum_ratios) <= 0:
        raise ValueError("At least one scale stratum ratio must be positive.")
    if not math.isclose(
        sum(config.scale_stratum_ratios),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("scale_stratum_ratios must sum to one.")
    if not math.isclose(
        config.programmatic_ratio
        + config.free_generation_ratio
        + config.parent_mutation_ratio,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "sampleV2 proposal ratios must sum to one."
        )
    if config.parent_crossover_ratio != 0:
        raise ValueError("sampleV2 disables parent crossover.")
    if not 0.0 <= config.environment_nn_quantile <= 1.0:
        raise ValueError("environment_nn_quantile must lie in [0, 1].")
    if not 0.0 <= config.compact_anchor_nn_quantile <= 1.0:
        raise ValueError("compact_anchor_nn_quantile must lie in [0, 1].")
    if config.environment_min_distance < 0:
        raise ValueError("environment_min_distance must be non-negative.")
    if config.compact_anchor_min_distance < 0:
        raise ValueError("compact_anchor_min_distance must be non-negative.")
    if not 0.0 <= config.pool_environment_distance_scale <= 1.0:
        raise ValueError(
            "pool_environment_distance_scale must lie in [0, 1]."
        )
    if config.certified_pool_multiplier < 1.0:
        raise ValueError("certified_pool_multiplier must be at least one.")
    if config.progress_interval_candidates <= 0:
        raise ValueError("progress_interval_candidates must be positive.")
    selection_weight = (
        config.environment_novelty_weight
        + config.coverage_selection_weight
        + config.parameter_selection_weight
        + config.behavior_selection_weight
    )
    if not math.isclose(
        selection_weight,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("sampleV2 final-selection weights must sum to one.")
    if any(value <= 0 for value in _rollout_batch_sizes(config)):
        raise ValueError("Rollout stage increments must be positive.")
    if len(set(_rollout_batch_sizes(config))) != 1:
        raise ValueError(
            "sampleV2 rollout increments must use one fixed JAX batch size."
        )


def _parameter_pairwise_matrix(
    signatures: np.ndarray,
) -> np.ndarray:
    total = np.zeros((len(signatures), len(signatures)), dtype=float)
    for indexes, weight in v1._DISTANCE_GROUPS:
        values = signatures[:, indexes]
        squared = (
            np.sum(np.square(values), axis=1)[:, None]
            + np.sum(np.square(values), axis=1)[None, :]
            - 2.0 * values @ values.T
        )
        total += weight * np.sqrt(
            np.maximum(squared, 0.0) / max(values.shape[1], 1)
        )
    np.fill_diagonal(total, 0.0)
    return total


def _behavior_pairwise_matrix(
    signatures: np.ndarray,
) -> np.ndarray:
    squared = (
        np.sum(np.square(signatures), axis=1)[:, None]
        + np.sum(np.square(signatures), axis=1)[None, :]
        - 2.0 * signatures @ signatures.T
    )
    result = np.sqrt(
        np.maximum(squared, 0.0) / max(signatures.shape[1], 1)
    )
    np.fill_diagonal(result, 0.0)
    return result


def _select_v2_candidates(
    pool: Sequence[qd._CertifiedCandidate],
    environments: dict[str, np.ndarray],
    reference_distances: dict[str, float],
    reference: _ReferenceArchive,
    config: SampleV2Config,
) -> list[qd._CertifiedCandidate]:
    """Quota-constrained farthest-point selection over certified tasks."""

    quotas = _scale_quotas(config)
    candidates = list(pool)
    environment_matrix = np.asarray(
        [environments[candidate.digest] for candidate in candidates],
        dtype=float,
    )
    environment_pairwise = _group_distance_matrix(
        environment_matrix,
        environment_matrix,
        reference.layout,
    )
    parameter_pairwise = _parameter_pairwise_matrix(
        np.asarray(
            [candidate.parameter_signature for candidate in candidates],
            dtype=float,
        )
    )
    behavior_pairwise = _behavior_pairwise_matrix(
        np.asarray(
            [candidate.behavior_signature for candidate in candidates],
            dtype=float,
        )
    )
    remaining = list(range(len(candidates)))
    nearest_environment = np.full(len(candidates), math.inf, dtype=float)
    nearest_parameter = np.full(len(candidates), math.inf, dtype=float)
    nearest_behavior = np.full(len(candidates), math.inf, dtype=float)
    selected: list[qd._CertifiedCandidate] = []
    selected_counts: Counter[str] = Counter()
    coverage = qd._CoverageState(config)

    while remaining and len(selected) < config.n_tasks:
        best: qd._CertifiedCandidate | None = None
        best_pool_index: int | None = None
        best_score = -math.inf
        best_environment_distance = math.inf
        for pool_index in remaining:
            candidate = candidates[pool_index]
            stratum = _task_scale_stratum(candidate.task)
            if selected_counts[stratum] >= quotas[stratum]:
                continue
            if not qd._online_append_allows(candidate, selected, config):
                continue

            environment_distance = nearest_environment[pool_index]
            if math.isfinite(environment_distance):
                environment_novelty = min(
                    environment_distance
                    / max(reference.environment_threshold, 1e-9),
                    2.0,
                ) / 2.0
            else:
                environment_novelty = min(
                    reference_distances[candidate.digest]
                    / max(reference.environment_threshold, 1e-9),
                    2.0,
                ) / 2.0
            parameter_distance = nearest_parameter[pool_index]
            behavior_distance = nearest_behavior[pool_index]
            parameter_novelty = (
                1.0
                if not math.isfinite(parameter_distance)
                else min(
                    parameter_distance / config.parameter_novelty_scale,
                    1.0,
                )
            )
            behavior_novelty = (
                1.0
                if not math.isfinite(behavior_distance)
                else min(
                    behavior_distance / config.behavior_novelty_scale,
                    1.0,
                )
            )
            quota_deficit = (
                quotas[stratum] - selected_counts[stratum]
            ) / max(quotas[stratum], 1)
            coverage_gain = coverage.gain(candidate.descriptor)
            coverage_score = min(coverage_gain, 1.0)
            score = (
                config.environment_novelty_weight * environment_novelty
                + config.coverage_selection_weight * coverage_score
                + config.parameter_selection_weight * parameter_novelty
                + config.behavior_selection_weight * behavior_novelty
                + 1e-6 * quota_deficit
            )
            tie_break = int(candidate.digest[:12], 16) / float(16**12)
            score += tie_break * 1e-12
            if score > best_score:
                best = candidate
                best_pool_index = pool_index
                best_score = score
                best_environment_distance = environment_distance
        if best is None or best_pool_index is None:
            break

        remaining.remove(best_pool_index)
        selected.append(best)
        nearest_environment = np.minimum(
            nearest_environment,
            environment_pairwise[:, best_pool_index],
        )
        nearest_parameter = np.minimum(
            nearest_parameter,
            parameter_pairwise[:, best_pool_index],
        )
        nearest_behavior = np.minimum(
            nearest_behavior,
            behavior_pairwise[:, best_pool_index],
        )
        stratum = _task_scale_stratum(best.task)
        selected_counts[stratum] += 1
        coverage.add(best.descriptor)
        best.task.setdefault("metadata", {})["sample_v2_selection"] = {
            "score_at_selection": best_score,
            "scale_stratum": stratum,
            "stratum_quota": quotas[stratum],
            "reference_environment_distance": reference_distances[best.digest],
            "nearest_selected_environment_distance": (
                None
                if not math.isfinite(best_environment_distance)
                else best_environment_distance
            ),
            "scope": "B_confirmed_surplus_pool",
            "policy": "quota_constrained_environment_farthest_point",
        }
    return selected


def _percentiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def sample_tasks_with_report(
    config: SampleV2Config,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect scale-stratified, reference-distant tasks on CPU workers."""

    _validate_config(config)
    print(
        "sampleV2 loading the reference archive and fitting the fixed "
        "827-D descriptor...",
        flush=True,
    )
    reference = _load_reference_archive(config)
    quotas = _scale_quotas(config)
    budget = qd._collection_budget(config)
    generation_seed = (
        secrets.randbits(32)
        if config.generation_seed is None
        else int(config.generation_seed)
    )
    generation_seed_policy = (
        "random_per_run"
        if config.generation_seed is None
        else "explicit"
    )
    pool_target = max(
        config.n_tasks,
        int(math.ceil(config.n_tasks * config.certified_pool_multiplier)),
    )

    pool: list[qd._CertifiedCandidate] = []
    pool_environments: list[np.ndarray] = []
    environment_by_digest: dict[str, np.ndarray] = {}
    reference_distance_by_digest: dict[str, float] = {}
    feedback = qd._BucketFeedback()
    reject_counts: Counter[str] = Counter()
    prefilter_reject_counts: Counter[str] = Counter()
    source_generated: Counter[str] = Counter()
    source_submitted: Counter[str] = Counter()
    source_confirmed: Counter[str] = Counter()
    scale_confirmed: Counter[str] = Counter()
    scale_leases: Counter[str] = Counter()
    parent_leases: Counter[str] = Counter()
    target_leases: Counter[tuple[str, ...]] = Counter()

    worker_count = qd._parallel_worker_count(config, budget)
    max_in_flight = max(
        worker_count,
        config.commit_batch_size or 2 * worker_count,
    )
    next_candidate_id = 0
    started = 0.0

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
    configured_cache_dir = (
        Path(config.jax_cache_dir).expanduser().resolve()
        if config.jax_persistent_cache
        else None
    )
    if configured_cache_dir is not None:
        managed_environment.update(
            {
                "JAX_ENABLE_COMPILATION_CACHE": "true",
                "JAX_COMPILATION_CACHE_DIR": str(configured_cache_dir),
                "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
                "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
            }
        )
    previous_environment = {
        name: os.environ.get(name) for name in managed_environment
    }
    os.environ.update(managed_environment)

    print(
        "sampleV2 CPU sampling: "
        f"workers={worker_count}, max_in_flight={max_in_flight}, "
        f"target={config.n_tasks}, surplus_pool={pool_target}, "
        f"raw_budget={budget}, reference_tasks={reference.report['n_tasks']}, "
        f"environment_threshold={reference.environment_threshold:.4f}, "
        f"compact_threshold={reference.compact_threshold:.4f}, "
        f"win_rate_band=[{config.win_rate_min:.2f}, {config.win_rate_max:.2f}], "
        f"quotas={quotas}, generation_seed={generation_seed}",
        flush=True,
    )

    def time_budget_exhausted() -> bool:
        return bool(
            config.collection_time_budget_seconds > 0
            and started > 0
            and time.monotonic() - started
            >= config.collection_time_budget_seconds
        )

    def scale_minimums_met() -> bool:
        return all(
            scale_confirmed[name] >= quota
            for name, quota in quotas.items()
        )

    def collection_complete() -> bool:
        return len(pool) >= pool_target and scale_minimums_met()

    def release_parents(parents: Sequence[qd._ParentPayload]) -> None:
        for parent in parents:
            parent_leases[parent.digest] -= 1

    def print_progress() -> None:
        elapsed = 0.0 if started <= 0 else time.monotonic() - started
        print(
            "sampleV2 progress: "
            f"raw={next_candidate_id}/{budget}, "
            f"submitted={sum(source_submitted.values())}, "
            f"B_confirmed={len(pool)}/{pool_target}, "
            f"strata={dict(scale_confirmed)}, "
            f"prefilter_rejected={sum(prefilter_reject_counts.values())}, "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    def build_job() -> _V2ParallelJob | None:
        nonlocal next_candidate_id

        while next_candidate_id < budget and not time_budget_exhausted():
            candidate_id = next_candidate_id
            next_candidate_id += 1
            if (
                next_candidate_id % config.progress_interval_candidates
                == 0
            ):
                print_progress()
            scheduler_rng = random.Random(
                qd._derive_seed(
                    generation_seed,
                    candidate_id,
                    "sampleV2-scheduler",
                )
            )
            coverage = qd._leased_coverage(pool, target_leases, config)
            target = _choose_target(
                scheduler_rng,
                config,
                coverage,
                feedback,
            )
            scale_stratum = _draw_scale_stratum(
                scheduler_rng,
                quotas,
                scale_confirmed,
                scale_leases,
            )
            proposal_kind = _draw_proposal_kind(
                scheduler_rng,
                config,
                scale_stratum,
                pool,
                source_generated,
                source_confirmed,
            )
            parents = _assign_parent(
                proposal_kind,
                pool,
                parent_leases,
                scheduler_rng,
            )
            source_generated[proposal_kind] += 1
            feedback.record_generated(target)

            try:
                raw_task, certification_source_kind = _generate_proposal(
                    config,
                    candidate_id=candidate_id,
                    generation_seed=generation_seed,
                    target=target,
                    proposal_kind=proposal_kind,
                    scale_stratum=scale_stratum,
                    parents=parents,
                )
                actual_stratum = _task_scale_stratum(raw_task)
                if actual_stratum != scale_stratum:
                    raise ValueError(
                        "Generated scale stratum mismatch: "
                        f"requested={scale_stratum}, actual={actual_stratum}."
                    )
                _, _, rejection = _prefilter_candidate(
                    raw_task,
                    config,
                    reference,
                    pool_environments,
                )
            except (RuntimeError, ValueError) as exc:
                rejection = (
                    f"prefilter_generation:{type(exc).__name__}:{exc}"
                )
                raw_task = {}
                certification_source_kind = proposal_kind

            if rejection is not None:
                release_parents(parents)
                reject_counts[rejection] += 1
                prefilter_reject_counts[rejection] += 1
                continue

            target_leases[target] += 1
            scale_leases[scale_stratum] += 1
            source_submitted[proposal_kind] += 1
            return _V2ParallelJob(
                candidate_id=candidate_id,
                target=target,
                proposal_kind=proposal_kind,
                certification_source_kind=certification_source_kind,
                scale_stratum=scale_stratum,
                parent_ids=tuple(parent.digest for parent in parents),
                raw_task=raw_task,
                config=config,
            )
        return None

    def commit_result(result: _V2ParallelResult) -> None:
        for parent_id in result.parent_ids:
            parent_leases[parent_id] -= 1
        target_leases[result.target] -= 1
        scale_leases[result.scale_stratum] -= 1

        outcome = result.outcome
        if outcome.candidate is None:
            reject_counts[outcome.reason] += 1
            return
        candidate = outcome.candidate
        actual_stratum = _task_scale_stratum(candidate.task)
        if actual_stratum != result.scale_stratum:
            reject_counts["post_certification_scale_stratum_changed"] += 1
            return

        try:
            final_environment = _scaled_task_descriptor(
                candidate.task,
                reference,
            )
        except ValueError as exc:
            reject_counts[
                f"post_certification_descriptor_invalid:{exc}"
            ] += 1
            return
        final_reference_distance = _nearest_environment_distance(
            final_environment,
            reference.environment_scaled,
            reference.layout,
        )
        if final_reference_distance < reference.environment_threshold:
            reject_counts["post_certification_environment_reference_near"] += 1
            return
        final_pool_distance = _nearest_environment_distance(
            final_environment,
            (
                np.asarray(pool_environments, dtype=float)
                if pool_environments
                else np.empty(
                    (0, reference.layout.dimension),
                    dtype=float,
                )
            ),
            reference.layout,
        )
        required_pool_distance = (
            reference.environment_threshold
            * config.pool_environment_distance_scale
        )
        if final_pool_distance < required_pool_distance:
            reject_counts["post_certification_environment_pool_near"] += 1
            return

        duplicate_reason = qd._is_pool_duplicate(candidate, pool, config)
        if duplicate_reason is not None:
            reject_counts[duplicate_reason] += 1
            return

        candidate.task.setdefault("metadata", {}).setdefault(
            "sample_v2",
            {},
        ).update(
            {
                "proposal_kind": result.proposal_kind,
                "scale_stratum": actual_stratum,
                "post_certification_environment_reference_distance": (
                    final_reference_distance
                ),
                "required_environment_reference_distance": (
                    reference.environment_threshold
                ),
                "post_certification_environment_pool_distance": (
                    None
                    if not math.isfinite(final_pool_distance)
                    else final_pool_distance
                ),
                "required_environment_pool_distance": required_pool_distance,
                "reference_task_file": config.reference_task_file,
            }
        )
        candidate.task["metadata"]["parallel_sampling"] = {
            "candidate_id": result.candidate_id,
            "worker_id": result.worker_id,
            "worker_pid": result.worker_pid,
            "parent_ids": list(result.parent_ids),
            "execution_backend": "cpu",
            "commit_policy": "asynchronous_completion_order",
            "prefilter_location": "coordinator_before_worker_submission",
        }
        pool.append(candidate)
        pool_environments.append(final_environment)
        environment_by_digest[candidate.digest] = final_environment
        reference_distance_by_digest[candidate.digest] = (
            final_reference_distance
        )
        scale_confirmed[actual_stratum] += 1
        source_confirmed[result.proposal_kind] += 1
        feedback.record_confirmed(result.target)
        print(
            f"candidate={result.candidate_id + 1}/{budget} B_CONFIRMED "
            f"pool={len(pool)}/{pool_target} "
            f"stratum={actual_stratum} "
            f"source={result.proposal_kind} "
            f"environment_distance={final_reference_distance:.4f} "
            f"worker={result.worker_id}",
            flush=True,
        )

    executor: ProcessPoolExecutor | None = None
    prewarm_report: dict[str, Any] = {
        "enabled": False,
        "batch_sizes": list(_rollout_batch_sizes(config)),
        "elapsed_seconds": 0.0,
    }
    cache_dir: Path | None = None
    try:
        cache_dir = _configure_jax_persistent_cache(config)
        prewarm_report = _prepare_rollout_cache(
            config,
            reference.prewarm_task,
            worker_count,
        )
        started = time.monotonic()
        if worker_count == 1:
            while not collection_complete():
                job = build_job()
                if job is None:
                    break
                commit_result(_sample_candidate_cpu_worker(job))
        else:
            context = qd._multiprocessing_context()
            worker_counter = context.Value("i", 0)
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=context,
                initializer=qd._initialize_cpu_worker,
                initargs=(worker_counter,),
            )
            pending: dict[Any, _V2ParallelJob] = {}

            def refill_workers() -> None:
                while (
                    len(pending) < max_in_flight
                    and not collection_complete()
                    and not time_budget_exhausted()
                ):
                    job = build_job()
                    if job is None:
                        break
                    future = executor.submit(
                        _sample_candidate_cpu_worker,
                        job,
                    )
                    pending[future] = job

            refill_workers()
            while pending and not collection_complete():
                if time_budget_exhausted():
                    reject_counts["time_budget_exhausted"] += 1
                    break
                completed, _ = wait(
                    tuple(pending),
                    return_when=FIRST_COMPLETED,
                )
                for future in sorted(
                    completed,
                    key=lambda item: pending[item].candidate_id,
                ):
                    job = pending.pop(future)
                    result = future.result()
                    if result.candidate_id != job.candidate_id:
                        raise RuntimeError(
                            "Worker returned a mismatched candidate_id."
                        )
                    commit_result(result)
                refill_workers()
            for future in pending:
                future.cancel()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        for name, previous in previous_environment.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous

    selected = _select_v2_candidates(
        pool,
        environment_by_digest,
        reference_distance_by_digest,
        reference,
        config,
    )
    if not selected:
        raise RuntimeError(
            "sampleV2 could not collect any certified task: "
            f"raw_budget={budget}, rejections={dict(reject_counts)}."
        )
    if len(selected) < config.n_tasks:
        print(
            "WARNING: sampleV2 exhausted its budget before satisfying all "
            f"quotas; saving {len(selected)}/{config.n_tasks} tasks.",
            flush=True,
        )

    tasks = [candidate.task for candidate in selected[: config.n_tasks]]
    for index, (task, candidate) in enumerate(zip(tasks, selected)):
        task["task_id"] = f"task_{index:06d}_{candidate.digest[:10]}"
        task["metadata"]["canonical_hash"] = candidate.digest

    selected_scale_counts = Counter(
        _task_scale_stratum(candidate.task) for candidate in selected
    )
    selected_reference_distances = [
        reference_distance_by_digest[candidate.digest]
        for candidate in selected
    ]
    selected_mutual_distances: list[float] = []
    if len(selected) > 1:
        selected_matrix = np.asarray(
            [environment_by_digest[candidate.digest] for candidate in selected],
            dtype=float,
        )
        pairwise = _group_distance_matrix(
            selected_matrix,
            selected_matrix,
            reference.layout,
        )
        np.fill_diagonal(pairwise, math.inf)
        selected_mutual_distances = np.min(pairwise, axis=1).tolist()

    report = qd._build_report(
        selected[: config.n_tasks],
        pool,
        reject_counts,
        source_generated,
        source_confirmed,
        feedback,
        config,
        budget,
        0.0 if started <= 0 else time.monotonic() - started,
    )
    report.update(
        {
            "generation_seed": generation_seed,
            "generation_seed_policy": generation_seed_policy,
            "evaluation_seed": config.seed,
            "complete": len(tasks) == config.n_tasks,
            "partial_result": len(tasks) != config.n_tasks,
            "requested_tasks": config.n_tasks,
            "saved_tasks": len(tasks),
            "task_shortfall": max(0, config.n_tasks - len(tasks)),
            "termination_reason": (
                "target_and_quotas_reached"
                if len(tasks) == config.n_tasks
                else (
                    "time_budget_exhausted"
                    if time_budget_exhausted()
                    else "candidate_budget_or_quota_selection_exhausted"
                )
            ),
            "reference_archive": reference.report,
            "prefilter_rejections": dict(prefilter_reject_counts),
            "proposal_sources": {
                "configured_ratios": _proposal_base_weights(config),
                "generated": dict(source_generated),
                "submitted_to_rollout": dict(source_submitted),
                "confirmed": dict(source_confirmed),
            },
            "scale_strata": {
                "ratios": {
                    name: config.scale_stratum_ratios[index]
                    for index, name in enumerate(SCALE_STRATA)
                },
                "target_quotas": quotas,
                "certified_pool": {
                    name: scale_confirmed[name] for name in SCALE_STRATA
                },
                "selected": {
                    name: selected_scale_counts[name]
                    for name in SCALE_STRATA
                },
                "outside_legacy_selected_fraction": (
                    1.0
                    - selected_scale_counts["legacy_scale"]
                    / max(len(selected), 1)
                ),
            },
            "environment_novelty": {
                "descriptor_dimension": reference.layout.dimension,
                "reference_threshold": reference.environment_threshold,
                "pool_mutual_threshold": (
                    reference.environment_threshold
                    * config.pool_environment_distance_scale
                ),
                "selected_to_reference": _percentiles(
                    selected_reference_distances
                ),
                "selected_nearest_mutual": _percentiles(
                    selected_mutual_distances
                ),
                "selection_policy": (
                    "quota_constrained_environment_farthest_point"
                ),
            },
            "parallel_sampling": {
                "execution_backend": "cpu",
                "workers": worker_count,
                "threads_per_worker": 1,
                "max_in_flight": max_in_flight,
                "scheduling": "asynchronous_refill",
                "multiprocessing_start_method": (
                    "local"
                    if worker_count == 1
                    else qd._multiprocessing_context().get_start_method()
                ),
                "prefilter": "coordinator_before_rollout_submission",
                "commit_policy": "asynchronous_completion_order",
            },
            "jax_compilation_cache": {
                "enabled": cache_dir is not None,
                "directory": (
                    None if cache_dir is None else str(cache_dir)
                ),
                "prewarm": prewarm_report,
                "rollout_batch_sizes": list(
                    _rollout_batch_sizes(config)
                ),
            },
        }
    )
    return tasks, report


def sample_tasks(config: SampleV2Config) -> list[dict[str, Any]]:
    """Compatibility wrapper returning only the sampled tasks."""

    tasks, _ = sample_tasks_with_report(config)
    return tasks


def save_task_bank(
    config: SampleV2Config,
    tasks: Sequence[dict[str, Any]],
    report: dict[str, Any],
) -> Path:
    """Save a sampleV2 bank using the repository's stable task schema."""

    generator_protocol = {
        "strategy": "reference_distant_scale_stratified_quality_diversity",
        "task_profile": "balanced_wide_band",
        "generation_seed": report["generation_seed"],
        "generation_seed_policy": report["generation_seed_policy"],
        "evaluation_seed": config.seed,
        "reference_archive": report["reference_archive"],
        "proposal_mix": _proposal_base_weights(config),
        "scale_strata": report["scale_strata"],
        "environment_novelty": report["environment_novelty"],
        "rollout_prefilter": {
            "compact_reference_distance_before_rollout": True,
            "environment_reference_distance_before_rollout": True,
            "environment_pool_distance_before_rollout": True,
            "static_screen_before_rollout": True,
            "post_certification_environment_recheck": True,
        },
        "limits": {
            "max_n_ally": config.max_n_ally,
            "max_n_enemy": config.max_n_enemy,
            "max_n_zone": config.max_n_zone,
        },
        "collection_report": report,
    }
    output = base.save_task_bank(
        config.output,
        tasks,
        seed=config.seed,
        physics=config.physics,
        heuristic=config.heuristic,
        max_n_ally=config.max_n_ally,
        max_n_enemy=config.max_n_enemy,
        max_n_zone=config.max_n_zone,
        filter_protocol=qd._filter_protocol(config),
        generator_protocol=generator_protocol,
    )
    bank = json.loads(output.read_text(encoding="utf-8"))
    bank["manifest"]["generator"] = "src.tabx.sampleV2"
    output.write_text(
        json.dumps(
            bank,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    config = tyro.cli(SampleV2Config)
    tasks, report = sample_tasks_with_report(config)
    output = save_task_bank(config, tasks, report)
    completion = (
        "complete"
        if report["complete"]
        else f"partial={len(tasks)}/{report['requested_tasks']}"
    )
    print(
        f"Saved {len(tasks)} sampleV2 tasks to {output}; "
        f"collection={completion}, "
        f"strata={report['scale_strata']['selected']}, "
        f"prefilter_rejections="
        f"{sum(report['prefilter_rejections'].values())}, "
        f"sources={report['proposal_sources']['confirmed']}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
