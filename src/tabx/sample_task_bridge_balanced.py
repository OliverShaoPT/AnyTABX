"""CPU-parallel, balance-certified quality-diversity sampling for TABX zones.

This module is deliberately self-contained.  It reuses the stable unit/layout
generator and the A/B rollout certification protocol from :mod:`sample_task`,
but owns its zone generator, zone descriptors, adaptive scheduling, duplicate
policy, and final constrained selection.

The sampler treats zone diversity as a conditional hierarchy rather than as
three independent metadata labels:

* count: 0, 1--2, 3--4, 5--7, or 8--10;
* topology: focal, pair, linear, ring, grid, nested, or distributed;
* type composition: one lava/bush/swamp type, two types, or all three;
* effect profile: homogeneous, gradient, alternating, focal-strong, or bimodal;
* actual geometric relation and spatial role.

Every descriptor is recomputed from the final ``zone_scenario``.  Inapplicable
values (for example relation for one zone or intensity for bushes) are excluded
from coverage targets.  Accepted tasks must pass both adaptive A discovery and
fresh original/flipped B confirmation.  Worker processes are CPU-only and
single-threaded.

Example::

    python -m src.tabx.sample_task_bridge_balanced \
        --n-tasks 200 --cpu-cores 8 \
        --output outputs/zone_diverse_balanced_tasks.json
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import random
import secrets
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# Workers must not claim a GPU or create hidden BLAS thread pools.  These are
# set before importing sample_task because that module imports JAX transitively.
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

from src.tabx import sample_task as base
from src.tabx.task_generators import (
    COMPOSITION_ARCHETYPES,
    COMPOSITION_MATCH_MODES,
    DISTANCE_BUCKETS,
    LAYOUT_ARCHETYPES,
    SPREAD_BUCKETS,
    generate_programmatic_task,
)


COUNT_BUCKET_RANGES: dict[str, tuple[int, int]] = {
    "0": (0, 0),
    "1-2": (1, 2),
    "3-4": (3, 4),
    "5-7": (5, 7),
    "8-10": (8, 10),
}
TOPOLOGIES = (
    "void",
    "focal",
    "pair",
    "linear",
    "ring",
    "grid",
    "nested",
    "distributed",
)
TYPE_MIXES = (
    "none",
    "single_lava",
    "single_bush",
    "single_swamp",
    "dual",
    "triple",
)
INTENSITY_BINS = ("very_low", "low", "medium", "high", "very_high")
RELATIONS = ("none", "separated", "tangent", "overlap", "nested", "mixed")
EFFECT_PROFILES = (
    "none",
    "not_applicable",
    "homogeneous",
    "gradient",
    "alternating",
    "focal_strong",
    "bimodal",
)
SPATIAL_ROLES = (
    "none",
    "center",
    "flanks",
    "team_symmetric",
    "lanes",
    "retreat",
    "distributed",
)
COVERAGE_BUCKETS = ("none", "tiny", "small", "medium", "large")

ZONE_TYPE_NAMES = {1: "lava", 2: "bush", 3: "swamp"}
ZONE_TYPE_IDS = {value: key for key, value in ZONE_TYPE_NAMES.items()}
BASE_EFFECTS: dict[str, dict[str, float]] = {
    "lava": dict(zip(INTENSITY_BINS, (5.0, 7.5, 10.0, 12.5, 15.0))),
    "swamp": dict(zip(INTENSITY_BINS, (0.12, 0.21, 0.30, 0.39, 0.48))),
}
ACTIVE_COVERAGE_VALUES: dict[str, tuple[str, ...]] = {
    "zone_count": tuple(COUNT_BUCKET_RANGES),
    "zone_topology": TOPOLOGIES,
    "zone_type_mix": TYPE_MIXES,
    "zone_intensity": ("none", "not_applicable", *INTENSITY_BINS),
    "zone_relation": RELATIONS,
    "zone_effect_profile": EFFECT_PROFILES,
    "zone_spatial_role": SPATIAL_ROLES,
    "zone_coverage": COVERAGE_BUCKETS,
}
COVERAGE_DIMENSIONS = tuple(ACTIVE_COVERAGE_VALUES)
PAIR_DIMENSIONS = (
    ("zone_count", "zone_topology"),
    ("zone_type_mix", "zone_intensity"),
    ("zone_topology", "zone_relation"),
    ("zone_topology", "zone_spatial_role"),
)
INAPPLICABLE_VALUES = frozenset(("none", "not_applicable"))
_CPU_WORKER_ID = -1


@dataclass(frozen=True)
class BridgeBalancedConfig(base.SampleConfig):
    """Configuration for zone-semantic, balanced quality-diversity sampling."""

    output: str = "outputs/zone_diverse_balanced_tasks.json"
    n_tasks: int = 200
    cpu_cores: int = 4
    max_n_zone: int = 10
    programmatic_ratio: float = 1.0
    open_ended_generation: bool = False
    include_challenges: bool = False

    # The physical task is frozen before confirmation.  No combat-stat repair
    # is allowed: balance comes from symmetric construction and hard A/B gates.
    static_balance_repair: bool = False
    simulation_balance_repair: bool = False
    simulation_repair_rounds: int = 0
    stat_scales: tuple[float, ...] = (1.0,)
    couple_stat_scales: bool = True
    confirmation_enabled: bool = True

    # Mirror is intentionally repeated to make half of target draws use a
    # roster-identical matchup while preserving other composition modes.
    composition_match_modes: tuple[str, ...] = (
        "mirror",
        "mirror",
        "unit_swap",
        "cost_match",
    )
    include_void_tasks: bool = True
    zone_count_target_weights: tuple[float, ...] = (0.08, 0.22, 0.25, 0.25, 0.20)
    target_draws: int = 64
    target_exploration: float = 0.10
    target_accept_rate_floor: float = 0.002
    target_weight_cap: float = 500.0

    # Geometry and strength ranges.  Total coverage is sampled first, so high
    # zone counts naturally use smaller individual ellipses.
    zone_coverage_fraction_min: float = 0.05
    zone_coverage_fraction_max: float = 0.38
    zone_axis_min: float = 1.5
    zone_axis_width_fraction_max: float = 0.22
    zone_axis_height_fraction_max: float = 0.28
    zone_position_jitter_fraction: float = 0.05
    per_zone_effect_jitter: float = 0.08

    # Candidate collection and final constrained greedy selection.
    generation_seed: int | None = None
    candidate_budget: int = 0
    candidate_budget_multiplier: int = 2000
    certified_pool_multiplier: float = 1.50
    max_in_flight: int = 0
    collection_time_budget_seconds: float = 0.0
    save_partial: bool = True

    zone_marginal_fraction: float = 0.70
    zone_pair_fraction: float = 0.30
    coverage_score_weight: float = 0.55
    parameter_novelty_weight: float = 0.30
    behavior_novelty_weight: float = 0.15
    parameter_novelty_scale: float = 0.18
    behavior_novelty_scale: float = 0.15
    zone_parameter_distance_threshold: float = 0.018
    zone_behavior_distance_threshold: float = 0.012
    max_zone_cell_fraction: float = 0.05
    max_exact_roster_fraction: float = 0.03
    max_family_tasks: int = 3


@dataclass(frozen=True)
class _ZoneTarget:
    count_bucket: str
    n_zone: int
    topology: str
    type_mix: str
    intensity: str
    relation: str
    effect_profile: str
    spatial_role: str
    coverage: str
    composition: str
    layout: str
    distance: str
    spread: str
    match_mode: str


@dataclass(frozen=True)
class _CandidateJob:
    job_id: int
    generation_seed: int
    target: _ZoneTarget
    config: BridgeBalancedConfig


@dataclass
class _CertifiedCandidate:
    task: dict[str, Any]
    digest: str
    descriptor: dict[str, str]
    parameter_signature: np.ndarray
    behavior_signature: np.ndarray
    roster_key: tuple[tuple[int, ...], tuple[int, ...]]
    family_id: str


@dataclass
class _CandidateResult:
    job_id: int
    worker_id: int
    worker_pid: int
    target: _ZoneTarget
    candidate: _CertifiedCandidate | None
    reason: str


class _BucketFeedback:
    """Smoothed target acceptance estimates for adaptive scheduling."""

    def __init__(self) -> None:
        self.generated: Counter[tuple[str, str]] = Counter()
        self.confirmed: Counter[tuple[str, str]] = Counter()

    def record_generated(self, descriptor: dict[str, str]) -> None:
        for dimension in COVERAGE_DIMENSIONS:
            self.generated[(dimension, descriptor[dimension])] += 1

    def record_confirmed(self, descriptor: dict[str, str]) -> None:
        for dimension in COVERAGE_DIMENSIONS:
            self.confirmed[(dimension, descriptor[dimension])] += 1

    def accept_rate(self, descriptor: dict[str, str]) -> float:
        rates = []
        for dimension in COVERAGE_DIMENSIONS:
            key = (dimension, descriptor[dimension])
            rates.append((self.confirmed[key] + 1.0) / (self.generated[key] + 2.0))
        return float(np.mean(rates)) if rates else 0.5


class _CoverageState:
    """Conditional marginal and pair coverage for final, actual descriptors."""

    def __init__(self, config: BridgeBalancedConfig) -> None:
        self.config = config
        self.marginals: Counter[tuple[str, str]] = Counter()
        self.pairs: Counter[tuple[str, str, str, str]] = Counter()

    def copy(self) -> "_CoverageState":
        result = _CoverageState(self.config)
        result.marginals.update(self.marginals)
        result.pairs.update(self.pairs)
        return result

    def add(self, descriptor: dict[str, str]) -> None:
        for dimension in COVERAGE_DIMENSIONS:
            self.marginals[(dimension, descriptor[dimension])] += 1
        for first, second in PAIR_DIMENSIONS:
            first_value = descriptor[first]
            second_value = descriptor[second]
            if _pair_is_applicable(first, first_value, second, second_value):
                self.pairs[(first, first_value, second, second_value)] += 1

    def marginal_target(self, dimension: str, value: str) -> int:
        if value in INAPPLICABLE_VALUES:
            return 0
        if dimension == "zone_count":
            index = tuple(COUNT_BUCKET_RANGES).index(value)
            return max(
                1,
                int(
                    round(
                        self.config.n_tasks
                        * self.config.zone_count_target_weights[index]
                    )
                ),
            )
        if dimension == "zone_topology" and value == "void":
            return self.marginal_target("zone_count", "0")
        active = [
            item
            for item in ACTIVE_COVERAGE_VALUES[dimension]
            if item not in INAPPLICABLE_VALUES
        ]
        return max(
            1,
            int(
                math.floor(
                    self.config.n_tasks
                    * self.config.zone_marginal_fraction
                    / max(len(active), 1)
                )
            ),
        )

    def pair_target(
        self,
        first: str,
        first_value: str,
        second: str,
        second_value: str,
    ) -> int:
        if not _pair_is_applicable(first, first_value, second, second_value):
            return 0
        combinations = sum(
            _pair_is_applicable(first, a, second, b)
            for a in ACTIVE_COVERAGE_VALUES[first]
            for b in ACTIVE_COVERAGE_VALUES[second]
        )
        raw = self.config.n_tasks * self.config.zone_pair_fraction / max(combinations, 1)
        return max(1, int(math.floor(raw))) if raw >= 1.0 else 0

    def gain(self, descriptor: dict[str, str]) -> float:
        gain = 0.0
        for dimension in COVERAGE_DIMENSIONS:
            value = descriptor[dimension]
            target = self.marginal_target(dimension, value)
            if target and self.marginals[(dimension, value)] < target:
                gain += 1.0
        for first, second in PAIR_DIMENSIONS:
            first_value = descriptor[first]
            second_value = descriptor[second]
            target = self.pair_target(first, first_value, second, second_value)
            key = (first, first_value, second, second_value)
            if target and self.pairs[key] < target:
                gain += 0.5
        return gain


def _derive_seed(seed: int, *parts: object) -> int:
    payload = ":".join(["zone-diverse-balanced", str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def _weighted_choice(
    rng: random.Random,
    values: Sequence[str],
    weights: Sequence[float],
) -> str:
    threshold = rng.random() * float(sum(weights))
    cumulative = 0.0
    for value, weight in zip(values, weights):
        cumulative += float(weight)
        if threshold <= cumulative:
            return value
    return values[-1]


def _sample_scale(
    rng: random.Random,
    values: Sequence[float],
    continuous: bool | None,
    fallback: bool,
) -> float:
    enabled = fallback if continuous is None else continuous
    return base._sample_scale(rng, values, enabled)


def _pair_is_applicable(
    first: str,
    first_value: str,
    second: str,
    second_value: str,
) -> bool:
    if first_value in INAPPLICABLE_VALUES or second_value in INAPPLICABLE_VALUES:
        return False
    if (first, second) == ("zone_count", "zone_topology"):
        if first_value == "0":
            return second_value == "void"
        if first_value == "1-2":
            return second_value in ("focal", "pair", "nested", "distributed")
        return second_value not in ("void", "focal", "pair")
    if (first, second) == ("zone_type_mix", "zone_intensity"):
        return first_value not in ("none", "single_bush")
    if (first, second) == ("zone_topology", "zone_relation"):
        if first_value in ("void", "focal"):
            return False
        if first_value == "nested":
            return second_value == "nested"
        return second_value != "nested"
    if (first, second) == ("zone_topology", "zone_spatial_role"):
        return first_value != "void"
    return True


def _valid_topologies(count_bucket: str) -> tuple[str, ...]:
    if count_bucket == "0":
        return ("void",)
    if count_bucket == "1-2":
        return ("focal", "pair", "nested", "distributed")
    return ("linear", "ring", "grid", "nested", "distributed")


def _draw_target(rng: random.Random, config: BridgeBalancedConfig) -> _ZoneTarget:
    count_values = tuple(COUNT_BUCKET_RANGES)
    weights = config.zone_count_target_weights
    if not config.include_void_tasks:
        count_values = count_values[1:]
        weights = weights[1:]
    count_bucket = _weighted_choice(rng, count_values, weights)
    topology = rng.choice(_valid_topologies(count_bucket))
    lower, upper = COUNT_BUCKET_RANGES[count_bucket]
    if topology == "void":
        n_zone = 0
    elif topology == "focal":
        n_zone = 1
    elif topology == "pair":
        n_zone = 2
    else:
        n_zone = rng.randint(max(lower, 2), upper)

    if n_zone == 0:
        type_mix = "none"
        intensity = "none"
        relation = "none"
        effect_profile = "none"
        spatial_role = "none"
        coverage = "none"
    else:
        type_mix = rng.choice(TYPE_MIXES[1:])
        intensity = (
            "not_applicable"
            if type_mix == "single_bush"
            else rng.choice(INTENSITY_BINS)
        )
        relation = (
            "none"
            if n_zone < 2
            else "nested"
            if topology == "nested"
            else rng.choice(
                ("separated", "tangent", "overlap")
                if n_zone == 2
                else ("separated", "tangent", "overlap", "mixed")
            )
        )
        effect_profile = (
            "not_applicable"
            if type_mix == "single_bush"
            else rng.choice(EFFECT_PROFILES[2:])
        )
        spatial_role = rng.choice(SPATIAL_ROLES[1:])
        coverage = rng.choice(COVERAGE_BUCKETS[1:])

    return _ZoneTarget(
        count_bucket=count_bucket,
        n_zone=n_zone,
        topology=topology,
        type_mix=type_mix,
        intensity=intensity,
        relation=relation,
        effect_profile=effect_profile,
        spatial_role=spatial_role,
        coverage=coverage,
        composition=rng.choice(config.composition_archetypes),
        layout=rng.choice(config.layout_archetypes),
        distance=rng.choice(config.distance_buckets),
        spread=rng.choice(config.spread_buckets),
        match_mode=rng.choice(config.composition_match_modes),
    )


def _target_descriptor(target: _ZoneTarget) -> dict[str, str]:
    return {
        "zone_count": target.count_bucket,
        "zone_topology": target.topology,
        "zone_type_mix": target.type_mix,
        "zone_intensity": target.intensity,
        "zone_relation": target.relation,
        "zone_effect_profile": target.effect_profile,
        "zone_spatial_role": target.spatial_role,
        "zone_coverage": target.coverage,
    }


def _choose_target(
    rng: random.Random,
    config: BridgeBalancedConfig,
    coverage: _CoverageState,
    feedback: _BucketFeedback,
) -> _ZoneTarget:
    best: _ZoneTarget | None = None
    best_score = -math.inf
    for _ in range(config.target_draws):
        target = _draw_target(rng, config)
        descriptor = _target_descriptor(target)
        deficit = coverage.gain(descriptor)
        accept_rate = max(
            feedback.accept_rate(descriptor),
            config.target_accept_rate_floor,
        )
        directed = min(deficit / accept_rate, config.target_weight_cap)
        score = directed + config.target_exploration * rng.random()
        if score > best_score:
            best = target
            best_score = score
    assert best is not None
    return best


def _team_geometry(task: dict[str, Any]) -> dict[str, np.ndarray]:
    scenario = task["scenario"]
    positions = np.asarray(scenario["positions"], dtype=float).reshape(-1, 2)
    teams = np.asarray(scenario["teams"], dtype=int).reshape(-1)
    ally = positions[teams == 0]
    enemy = positions[teams == 1]
    ally_center = ally.mean(axis=0)
    enemy_center = enemy.mean(axis=0)
    direction = enemy_center - ally_center
    norm = float(np.linalg.norm(direction))
    direction = direction / norm if norm > 1e-8 else np.array([1.0, 0.0])
    perpendicular = np.array([-direction[1], direction[0]])
    return {
        "positions": positions,
        "teams": teams,
        "ally_center": ally_center,
        "enemy_center": enemy_center,
        "midpoint": (ally_center + enemy_center) * 0.5,
        "direction": direction,
        "perpendicular": perpendicular,
    }


def _fit_zone(
    center: np.ndarray,
    axes: np.ndarray,
    width: float,
    height: float,
    config: BridgeBalancedConfig,
) -> tuple[np.ndarray, np.ndarray]:
    maximum = np.array(
        [
            width * config.zone_axis_width_fraction_max,
            height * config.zone_axis_height_fraction_max,
        ],
        dtype=float,
    )
    axes = np.clip(
        np.asarray(axes, dtype=float),
        config.zone_axis_min,
        maximum,
    )
    lower = np.array([-width * 0.5, -height * 0.5]) + axes + 0.5
    upper = np.array([width * 0.5, height * 0.5]) - axes - 0.5
    center = np.minimum(np.maximum(np.asarray(center, dtype=float), lower), upper)
    return center, axes


def _zone_axes(
    rng: random.Random,
    target: _ZoneTarget,
    width: float,
    height: float,
    config: BridgeBalancedConfig,
) -> list[np.ndarray]:
    if target.n_zone == 0:
        return []
    coverage_ranges = {
        "tiny": (0.03, 0.08),
        "small": (0.08, 0.18),
        "medium": (0.18, 0.35),
        "large": (0.35, 0.50),
    }
    requested_lower, requested_upper = coverage_ranges[target.coverage]
    lower = max(requested_lower, config.zone_coverage_fraction_min)
    upper = min(requested_upper, config.zone_coverage_fraction_max)
    if upper < lower:
        lower = upper = min(
            max((requested_lower + requested_upper) * 0.5, config.zone_coverage_fraction_min),
            config.zone_coverage_fraction_max,
        )
    coverage = rng.uniform(lower, upper)
    # Nested ellipses overlap heavily, so nominal summed area may be larger
    # without covering the whole map.
    if target.topology == "nested":
        coverage *= 1.35
    per_zone_area = width * height * coverage / target.n_zone
    result: list[np.ndarray] = []
    for index in range(target.n_zone):
        aspect = math.exp(rng.uniform(math.log(0.55), math.log(1.80)))
        product = per_zone_area / math.pi
        axes = np.array(
            [
                math.sqrt(product * aspect),
                math.sqrt(product / aspect),
            ],
            dtype=float,
        )
        if target.topology == "nested":
            factor = 1.35 - 0.70 * index / max(target.n_zone - 1, 1)
            axes *= factor
        axes *= rng.uniform(0.88, 1.12)
        _, fitted = _fit_zone(np.zeros(2), axes, width, height, config)
        result.append(fitted)
    return result


def _relation_spacing(relation: str) -> float:
    return {
        "none": 1.0,
        "separated": 1.55,
        "tangent": 1.02,
        "overlap": 0.62,
        "nested": 0.12,
        "mixed": 1.10,
    }[relation]


def _base_centers(
    rng: random.Random,
    target: _ZoneTarget,
    geometry: dict[str, np.ndarray],
    axes: Sequence[np.ndarray],
    width: float,
    height: float,
) -> list[np.ndarray]:
    n_zone = target.n_zone
    midpoint = geometry["midpoint"]
    direction = geometry["direction"]
    perpendicular = geometry["perpendicular"]
    spacing = _relation_spacing(target.relation)
    mean_radius = float(np.mean([np.mean(value) for value in axes])) if axes else 3.0
    gap = max(2.5, mean_radius * 2.0 * spacing)

    if target.topology == "focal":
        return [midpoint.copy()]
    if target.topology == "pair":
        return [
            midpoint - perpendicular * gap * 0.5,
            midpoint + perpendicular * gap * 0.5,
        ]
    if target.topology == "linear":
        values = np.linspace(-(n_zone - 1) * 0.5, (n_zone - 1) * 0.5, n_zone)
        return [midpoint + direction * gap * value for value in values]
    if target.topology == "ring":
        ring_radius = min(
            width * 0.27,
            height * 0.31,
            max(gap * n_zone / (2.0 * math.pi), mean_radius * 1.6),
        )
        return [
            midpoint
            + direction * ring_radius * math.cos(2.0 * math.pi * index / n_zone)
            + perpendicular * ring_radius * math.sin(2.0 * math.pi * index / n_zone)
            for index in range(n_zone)
        ]
    if target.topology == "grid":
        columns = int(math.ceil(math.sqrt(n_zone)))
        rows = int(math.ceil(n_zone / columns))
        result = []
        for index in range(n_zone):
            row, column = divmod(index, columns)
            along = (column - (columns - 1) * 0.5) * gap
            across = (row - (rows - 1) * 0.5) * gap
            result.append(midpoint + direction * along + perpendicular * across)
        return result
    if target.topology == "nested":
        return [
            midpoint
            + direction * rng.uniform(-0.10, 0.10) * mean_radius
            + perpendicular * rng.uniform(-0.10, 0.10) * mean_radius
            for _ in range(n_zone)
        ]

    # A deterministic low-discrepancy layout produces broad map coverage
    # without the accidental clumps of independent uniform coordinates.
    golden = math.pi * (3.0 - math.sqrt(5.0))
    radius_max = min(width * 0.32, height * 0.36)
    return [
        midpoint
        + direction
        * radius_max
        * math.sqrt((index + 0.5) / n_zone)
        * math.cos(index * golden)
        + perpendicular
        * radius_max
        * math.sqrt((index + 0.5) / n_zone)
        * math.sin(index * golden)
        for index in range(n_zone)
    ]


def _apply_spatial_role(
    centers: Sequence[np.ndarray],
    target: _ZoneTarget,
    geometry: dict[str, np.ndarray],
    width: float,
    height: float,
) -> list[np.ndarray]:
    if not centers:
        return []
    midpoint = geometry["midpoint"]
    direction = geometry["direction"]
    perpendicular = geometry["perpendicular"]
    ally_center = geometry["ally_center"]
    enemy_center = geometry["enemy_center"]
    n_zone = len(centers)

    if target.spatial_role == "center":
        return [midpoint + (center - midpoint) * 0.72 for center in centers]
    if target.spatial_role == "flanks":
        return [
            midpoint
            + direction * float(np.dot(center - midpoint, direction)) * 0.75
            + perpendicular
            * (
                float(np.dot(center - midpoint, perpendicular)) * 1.25
                + (1 if index % 2 else -1) * height * 0.08
            )
            for index, center in enumerate(centers)
        ]
    if target.spatial_role == "team_symmetric":
        result = []
        for index in range(n_zone):
            anchor = ally_center if index % 2 == 0 else enemy_center
            sign = -1.0 if index % 2 == 0 else 1.0
            ring = index // 2
            result.append(
                anchor
                + perpendicular * sign * (4.0 + 3.0 * (ring % 3))
                + direction * sign * 1.5 * (ring // 3)
            )
        return result
    if target.spatial_role == "lanes":
        return [
            midpoint
            + direction
            * (
                (index // 2) - max((n_zone - 1) // 4, 0)
            )
            * max(width * 0.08, 4.0)
            + perpendicular * (-1.0 if index % 2 == 0 else 1.0) * height * 0.16
            for index in range(n_zone)
        ]
    if target.spatial_role == "retreat":
        result = []
        for index in range(n_zone):
            if index % 2 == 0:
                result.append(ally_center - direction * (5.0 + 2.0 * (index // 2)))
            else:
                result.append(enemy_center + direction * (5.0 + 2.0 * (index // 2)))
        return result
    if target.spatial_role == "distributed":
        return [midpoint + (center - midpoint) * 1.22 for center in centers]
    return [np.asarray(center, dtype=float) for center in centers]


def _zone_types(rng: random.Random, target: _ZoneTarget) -> list[int]:
    if target.n_zone == 0:
        return []
    if target.type_mix.startswith("single_"):
        zone_type = ZONE_TYPE_IDS[target.type_mix.removeprefix("single_")]
        return [zone_type] * target.n_zone
    if target.type_mix == "dual":
        chosen = rng.sample((1, 2, 3), 2)
    else:
        chosen = [1, 2, 3]
    rng.shuffle(chosen)
    return [chosen[index % len(chosen)] for index in range(target.n_zone)]


def _effect_multipliers(profile: str, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=float)
    if profile == "homogeneous":
        return np.ones(count, dtype=float)
    if profile == "gradient":
        return np.linspace(0.65, 1.35, count)
    if profile == "alternating":
        return np.asarray([0.72 if index % 2 == 0 else 1.28 for index in range(count)])
    if profile == "focal_strong":
        values = np.full(count, 0.75, dtype=float)
        values[count // 2] = 1.55
        return values
    if profile == "bimodal":
        return np.asarray([0.72] * (count // 2) + [1.28] * (count - count // 2))
    return np.ones(count, dtype=float)


def _lava_is_clear(
    center: np.ndarray,
    axes: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
) -> bool:
    normalized = ((positions - center) / (axes + radii[:, None])) ** 2
    return bool(np.all(normalized.sum(axis=1) > 1.0))


def _place_lava_safely(
    center: np.ndarray,
    axes: np.ndarray,
    geometry: dict[str, np.ndarray],
    radii: np.ndarray,
    width: float,
    height: float,
    config: BridgeBalancedConfig,
) -> tuple[np.ndarray, np.ndarray]:
    positions = geometry["positions"]
    direction = geometry["direction"]
    perpendicular = geometry["perpendicular"]
    for shrink in (1.0, 0.86, 0.72, 0.60, 0.50):
        candidate_axes = np.maximum(axes * shrink, config.zone_axis_min)
        offsets = (
            np.zeros(2),
            perpendicular * (candidate_axes[1] + 2.0),
            -perpendicular * (candidate_axes[1] + 2.0),
            direction * (candidate_axes[0] + 2.0),
            -direction * (candidate_axes[0] + 2.0),
        )
        for offset in offsets:
            fitted_center, fitted_axes = _fit_zone(
                center + offset,
                candidate_axes,
                width,
                height,
                config,
            )
            if _lava_is_clear(fitted_center, fitted_axes, positions, radii):
                return fitted_center, fitted_axes
    raise ValueError("could_not_place_lava_clear_of_spawns")


def _build_zone_scenario(
    rng: random.Random,
    task: dict[str, Any],
    target: _ZoneTarget,
    config: BridgeBalancedConfig,
) -> dict[str, Any]:
    if target.n_zone == 0:
        return {
            "n_zone": 0,
            "zone_type": [],
            "position": [],
            "axes": [],
            "effect_value": [],
        }

    grid = task["grid_info"]
    width = float(grid["max_field_width"])
    height = float(grid["max_field_height"])
    geometry = _team_geometry(task)
    axes = _zone_axes(rng, target, width, height, config)
    centers = _base_centers(rng, target, geometry, axes, width, height)
    centers = _apply_spatial_role(centers, target, geometry, width, height)
    zone_types = _zone_types(rng, target)
    multipliers = _effect_multipliers(target.effect_profile, target.n_zone)
    radii = np.asarray(
        task["scenario"]["body_radiuss"],
        dtype=float,
    ).reshape(-1)

    fitted_centers: list[list[float]] = []
    fitted_axes: list[list[float]] = []
    effects: list[list[float]] = []
    for index, (center, zone_axes, zone_type) in enumerate(
        zip(centers, axes, zone_types)
    ):
        jitter_scale = (
            config.zone_position_jitter_fraction * 0.10
            if target.topology == "nested"
            else config.zone_position_jitter_fraction
        )
        jitter = np.array(
            [
                rng.uniform(-width, width),
                rng.uniform(-height, height),
            ]
        ) * jitter_scale
        center, zone_axes = _fit_zone(
            np.asarray(center) + jitter,
            zone_axes,
            width,
            height,
            config,
        )
        if zone_type == 1:
            center, zone_axes = _place_lava_safely(
                center,
                zone_axes,
                geometry,
                radii,
                width,
                height,
                config,
            )
        fitted_centers.append(center.tolist())
        fitted_axes.append(zone_axes.tolist())

        if zone_type == 2:
            effect = 0.0
        else:
            type_name = ZONE_TYPE_NAMES[zone_type]
            effect = BASE_EFFECTS[type_name][target.intensity]
            effect *= float(multipliers[index])
            effect *= rng.uniform(
                1.0 - config.per_zone_effect_jitter,
                1.0 + config.per_zone_effect_jitter,
            )
            if zone_type == 3:
                effect = float(np.clip(effect, 1e-4, 1.0))
        effects.append([float(effect)])

    return {
        "n_zone": target.n_zone,
        "zone_type": [[int(value)] for value in zone_types],
        "position": fitted_centers,
        "axes": fitted_axes,
        "effect_value": effects,
    }


def _count_bucket(n_zone: int) -> str:
    for label, (lower, upper) in COUNT_BUCKET_RANGES.items():
        if lower <= n_zone <= upper:
            return label
    raise ValueError(f"n_zone={n_zone} lies outside the supported range.")


def _type_mix(zone_types: np.ndarray) -> str:
    unique = sorted(set(int(value) for value in zone_types))
    if not unique:
        return "none"
    if len(unique) == 1:
        return f"single_{ZONE_TYPE_NAMES[unique[0]]}"
    return "dual" if len(unique) == 2 else "triple"


def _normalized_effects(zone_types: np.ndarray, effects: np.ndarray) -> np.ndarray:
    values = []
    for zone_type, effect in zip(zone_types, effects):
        if int(zone_type) == 1:
            values.append(float(effect) / 10.0)
        elif int(zone_type) == 3:
            values.append(float(effect) / 0.30)
    return np.asarray(values, dtype=float)


def _intensity_bin(zone_types: np.ndarray, effects: np.ndarray) -> str:
    if not len(zone_types):
        return "none"
    normalized = _normalized_effects(zone_types, effects)
    if not len(normalized):
        return "not_applicable"
    value = float(np.mean(normalized))
    if value < 0.65:
        return "very_low"
    if value < 0.85:
        return "low"
    if value < 1.15:
        return "medium"
    if value < 1.35:
        return "high"
    return "very_high"


def _pair_relation(
    positions: np.ndarray,
    axes: np.ndarray,
) -> str:
    if len(positions) < 2:
        return "none"
    center_spread = float(
        np.max(np.linalg.norm(positions - positions.mean(axis=0), axis=1))
    )
    axis_sizes = np.mean(axes, axis=1)
    size_ratio = float(np.max(axis_sizes) / max(np.min(axis_sizes), 1e-8))
    if center_spread < 0.30 * max(float(np.mean(axes)), 1e-8) and size_ratio > 1.25:
        return "nested"

    # Classify the topology graph rather than every possible pair.  Non-neighbour
    # pairs in a valid overlapping ring are naturally separated and should not
    # force the whole structure into the mixed bucket.
    neighbour_pairs: set[tuple[int, int]] = set()
    for first in range(len(positions)):
        ratios = np.full(len(positions), np.inf, dtype=float)
        for second in range(len(positions)):
            if first == second:
                continue
            radius_sum = max(
                float(np.mean(axes[first]) + np.mean(axes[second])),
                1e-6,
            )
            ratios[second] = float(
                np.linalg.norm(positions[first] - positions[second])
            ) / radius_sum
        second = int(np.argmin(ratios))
        neighbour_pairs.add(tuple(sorted((first, second))))

    labels: list[str] = []
    for first, second in sorted(neighbour_pairs):
        distance = float(np.linalg.norm(positions[first] - positions[second]))
        radius_sum = max(
            float(np.mean(axes[first]) + np.mean(axes[second])),
            1e-6,
        )
        if distance < 0.78 * radius_sum:
            labels.append("overlap")
        elif distance < 1.18 * radius_sum:
            labels.append("tangent")
        else:
            labels.append("separated")
    unique = set(labels)
    return labels[0] if len(unique) == 1 else "mixed"


def _infer_topology(positions: np.ndarray, axes: np.ndarray) -> str:
    count = len(positions)
    if count == 0:
        return "void"
    if count == 1:
        return "focal"
    center_spread = float(
        np.max(np.linalg.norm(positions - positions.mean(axis=0), axis=1))
    )
    mean_axis = float(np.mean(axes))
    axis_sizes = np.mean(axes, axis=1)
    size_ratio = float(np.max(axis_sizes) / max(np.min(axis_sizes), 1e-8))
    if center_spread < 0.30 * max(mean_axis, 1e-8) and size_ratio > 1.25:
        return "nested"
    relation = _pair_relation(positions, axes)
    if relation == "nested":
        return "nested"
    if count == 2:
        return "pair"

    centered = positions - positions.mean(axis=0)
    radii = np.linalg.norm(centered, axis=1)
    if count >= 4 and float(np.mean(radii)) > 1e-6:
        radial_cv = float(np.std(radii) / np.mean(radii))
        angular = np.sort(np.arctan2(centered[:, 1], centered[:, 0]))
        gaps = np.diff(np.r_[angular, angular[0] + 2.0 * math.pi])
        if radial_cv < 0.28 and float(np.std(gaps) / max(np.mean(gaps), 1e-6)) < 0.65:
            return "ring"

    singular = np.linalg.svd(centered, compute_uv=False)
    if len(singular) > 1 and singular[0] > 3.5 * max(singular[1], 1e-8):
        return "linear"

    if count >= 4:
        # A grid has repeated nearest-neighbour scales and occupies both PCA
        # axes.  This remains rotation invariant.
        covariance = centered.T @ centered / count
        eigenvalues = np.linalg.eigvalsh(covariance)
        if eigenvalues[0] > 0.12 * max(eigenvalues[1], 1e-8):
            distances = []
            for index in range(count):
                other = np.delete(positions, index, axis=0)
                distances.append(float(np.min(np.linalg.norm(other - positions[index], axis=1))))
            if float(np.std(distances) / max(np.mean(distances), 1e-6)) < 0.55:
                return "grid"
    return "distributed"


def _effect_profile(
    zone_types: np.ndarray,
    effects: np.ndarray,
    positions: np.ndarray,
) -> str:
    if not len(zone_types):
        return "none"
    values = _normalized_effects(zone_types, effects)
    if not len(values):
        return "not_applicable"
    if len(values) == 1 or float(np.std(values) / max(np.mean(values), 1e-6)) < 0.10:
        return "homogeneous"
    if float(np.max(values)) > 1.45 * max(float(np.median(values)), 1e-6):
        return "focal_strong"
    if len(values) >= 3:
        # Effects correspond only to lava/swamp.  Use their original order for
        # alternating detection and a position-sorted order for gradients.
        signs = np.sign(values - float(np.median(values)))
        if sum(signs[index] != signs[index - 1] for index in range(1, len(signs))) >= len(signs) - 2:
            return "alternating"
        active_positions = positions[np.isin(zone_types, (1, 3))]
        centered = active_positions - active_positions.mean(axis=0)
        if len(active_positions) >= 3:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            order = np.argsort(centered @ vh[0])
            correlation = np.corrcoef(np.arange(len(values)), values[order])[0, 1]
            if math.isfinite(float(correlation)) and abs(float(correlation)) > 0.72:
                return "gradient"
    return "bimodal"


def _spatial_role(task: dict[str, Any], positions: np.ndarray) -> str:
    if not len(positions):
        return "none"
    geometry = _team_geometry(task)
    midpoint = geometry["midpoint"]
    direction = geometry["direction"]
    perpendicular = geometry["perpendicular"]
    width = float(task["grid_info"]["max_field_width"])
    height = float(task["grid_info"]["max_field_height"])
    relative = positions - midpoint
    along = np.abs(relative @ direction)
    across = np.abs(relative @ perpendicular)
    if float(np.mean(np.linalg.norm(relative, axis=1))) < min(width, height) * 0.11:
        return "center"
    if float(np.mean(across)) > height * 0.20:
        return "flanks"
    team_centers = np.vstack((geometry["ally_center"], geometry["enemy_center"]))
    nearest_team = np.min(
        np.linalg.norm(positions[:, None, :] - team_centers[None, :, :], axis=2),
        axis=1,
    )
    if float(np.mean(nearest_team)) < min(width, height) * 0.14:
        outward = relative @ direction
        return "retreat" if np.any(outward < 0) and np.any(outward > 0) else "team_symmetric"
    if float(np.std(along)) > 1.35 * max(float(np.std(across)), 1e-6):
        return "lanes"
    return "distributed"


def _coverage_bucket(task: dict[str, Any], axes: np.ndarray) -> str:
    if not len(axes):
        return "none"
    width = float(task["grid_info"]["max_field_width"])
    height = float(task["grid_info"]["max_field_height"])
    value = float(np.sum(math.pi * axes[:, 0] * axes[:, 1]) / max(width * height, 1.0))
    if value < 0.08:
        return "tiny"
    if value < 0.18:
        return "small"
    if value < 0.35:
        return "medium"
    return "large"


def _zone_descriptor(task: dict[str, Any]) -> dict[str, str]:
    zones = task["zone_scenario"]
    n_zone = int(zones["n_zone"])
    zone_types = np.asarray(zones["zone_type"], dtype=int).reshape(-1)
    effects = np.asarray(zones["effect_value"], dtype=float).reshape(-1)
    positions = np.asarray(zones["position"], dtype=float).reshape(n_zone, 2)
    axes = np.asarray(zones["axes"], dtype=float).reshape(n_zone, 2)
    return {
        "zone_count": _count_bucket(n_zone),
        "zone_topology": _infer_topology(positions, axes),
        "zone_type_mix": _type_mix(zone_types),
        "zone_intensity": _intensity_bin(zone_types, effects),
        "zone_relation": _pair_relation(positions, axes),
        "zone_effect_profile": _effect_profile(zone_types, effects, positions),
        "zone_spatial_role": _spatial_role(task, positions),
        "zone_coverage": _coverage_bucket(task, axes),
    }


def _zone_parameter_signature(task: dict[str, Any]) -> np.ndarray:
    """Canonical, fixed-size signature for up to ten zones.

    Coordinates are expressed in the team-relative frame.  Four sign variants
    remove team-label and reflection choices before lexicographic canonicalization.
    """

    zones = task["zone_scenario"]
    n_zone = int(zones["n_zone"])
    geometry = _team_geometry(task)
    width = float(task["grid_info"]["max_field_width"])
    height = float(task["grid_info"]["max_field_height"])
    positions = np.asarray(zones["position"], dtype=float).reshape(n_zone, 2)
    axes = np.asarray(zones["axes"], dtype=float).reshape(n_zone, 2)
    zone_types = np.asarray(zones["zone_type"], dtype=int).reshape(-1)
    effects = np.asarray(zones["effect_value"], dtype=float).reshape(-1)
    relative = positions - geometry["midpoint"]
    local = np.column_stack(
        (
            relative @ geometry["direction"] / max(width, 1.0),
            relative @ geometry["perpendicular"] / max(height, 1.0),
        )
    )

    variants: list[np.ndarray] = []
    for sign_x, sign_y in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        rows = []
        for index in range(n_zone):
            normalized_effect = (
                effects[index] / 15.0
                if zone_types[index] == 1
                else effects[index]
                if zone_types[index] == 3
                else 0.0
            )
            rows.append(
                [
                    float(zone_types[index] == 1),
                    float(zone_types[index] == 2),
                    float(zone_types[index] == 3),
                    local[index, 0] * sign_x,
                    local[index, 1] * sign_y,
                    axes[index, 0] / max(width, 1.0),
                    axes[index, 1] / max(height, 1.0),
                    normalized_effect,
                ]
            )
        rows.sort(key=lambda row: tuple(round(value, 10) for value in row))
        while len(rows) < 10:
            rows.append([0.0] * 8)
        counts = [float(np.count_nonzero(zone_types == value)) / 10.0 for value in (1, 2, 3)]
        coverage = 0.0
        if n_zone:
            coverage = float(
                np.sum(math.pi * axes[:, 0] * axes[:, 1])
                / max(width * height, 1.0)
            )
        prefix = [n_zone / 10.0, *counts, min(coverage, 1.5) / 1.5]
        variants.append(np.asarray(prefix + [item for row in rows for item in row], dtype=float))
    return min(variants, key=lambda value: tuple(np.round(value, 10).tolist()))


def _mean_confirmation_scalar(
    confirmation: dict[str, Any],
    path: Sequence[str],
    default: float = 0.0,
) -> float:
    values = []
    for label in ("original", "flipped"):
        current: Any = confirmation[label]
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = default
                break
            current = current[key]
        if current is None:
            values.append(float(default))
        elif isinstance(current, (list, tuple, np.ndarray)):
            array = np.asarray(current, dtype=float).reshape(-1)
            values.append(float(np.mean(array)) if len(array) else float(default))
        else:
            values.append(float(current))
    return float(np.mean(values))


def _zone_behavior_signature(
    task: dict[str, Any],
    confirmation: dict[str, Any],
    max_episode_steps: int,
) -> np.ndarray:
    """Outcome signature augmented with static zone-exposure proxies."""

    duration = _mean_confirmation_scalar(
        confirmation,
        ("episode_length", "mean"),
        float(max_episode_steps),
    ) / max(max_episode_steps, 1)
    first_contact = _mean_confirmation_scalar(
        confirmation,
        ("first_interaction_step", "mean"),
        float(max_episode_steps),
    ) / max(max_episode_steps, 1)
    interaction_rate = _mean_confirmation_scalar(
        confirmation,
        ("first_interaction_step", "rate"),
        0.0,
    )
    damage = math.log1p(
        max(_mean_confirmation_scalar(confirmation, ("damage_mean",), 0.0), 0.0)
    ) / math.log1p(10_000.0)
    healing = math.log1p(
        max(_mean_confirmation_scalar(confirmation, ("healing_mean",), 0.0), 0.0)
    ) / math.log1p(10_000.0)

    zones = task["zone_scenario"]
    n_zone = int(zones["n_zone"])
    zone_types = np.asarray(zones["zone_type"], dtype=int).reshape(-1)
    effects = np.asarray(zones["effect_value"], dtype=float).reshape(-1)
    axes = np.asarray(zones["axes"], dtype=float).reshape(n_zone, 2)
    width = float(task["grid_info"]["max_field_width"])
    height = float(task["grid_info"]["max_field_height"])
    coverage = (
        float(np.sum(math.pi * axes[:, 0] * axes[:, 1]) / max(width * height, 1.0))
        if n_zone
        else 0.0
    )
    counts = [float(np.count_nonzero(zone_types == value)) / 10.0 for value in (1, 2, 3)]
    normalized_effect = _normalized_effects(zone_types, effects)
    return np.asarray(
        [
            duration,
            first_contact,
            interaction_rate,
            damage,
            healing,
            min(coverage, 1.5) / 1.5,
            *counts,
            float(np.mean(normalized_effect)) / 2.0 if len(normalized_effect) else 0.0,
            float(np.std(normalized_effect)) / 2.0 if len(normalized_effect) else 0.0,
        ],
        dtype=float,
    )


def _roster_key(task: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    teams = np.asarray(task["scenario"]["teams"], dtype=int).reshape(-1)
    unit_ids = np.asarray(task["scenario"]["unit_ids"], dtype=int).reshape(-1)
    rosters = [tuple(sorted(unit_ids[teams == team].tolist())) for team in (0, 1)]
    return tuple(sorted(rosters))  # type: ignore[return-value]


def _family_id(task: dict[str, Any], descriptor: dict[str, str]) -> str:
    payload = {
        "roster": _roster_key(task),
        "layout": task.get("metadata", {}).get("layout_archetype"),
        "zone_count": descriptor["zone_count"],
        "zone_topology": descriptor["zone_topology"],
        "zone_type_mix": descriptor["zone_type_mix"],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def _generate_task(job: _CandidateJob) -> dict[str, Any]:
    config = job.config
    target = job.target
    rng = random.Random(_derive_seed(job.generation_seed, job.job_id, "generate"))
    map_scale = _sample_scale(
        rng,
        config.map_scales,
        config.continuous_map_sampling,
        config.continuous_sampling,
    )
    task = generate_programmatic_task(
        rng,
        composition=target.composition,
        layout=target.layout,
        distance=target.distance,
        spread=target.spread,
        zone="void",
        zone_intensity="low",
        zone_relation="separated",
        map_scale=map_scale,
        max_n_ally=config.max_n_ally,
        max_n_enemy=config.max_n_enemy,
        match_mode=target.match_mode,
        price_rel_tol=config.composition_price_rel_tol,
        effective_rel_tol=config.composition_effective_rel_tol,
        health_frac_buckets=base._health_fraction_buckets(rng, config),
        health_frac_min=config.health_frac_min,
        health_frac_max=config.health_frac_max,
    )
    transform = rng.choice(config.transforms)
    symmetric_scale = _sample_scale(
        rng,
        config.stat_scales,
        config.continuous_stat_sampling,
        config.continuous_sampling,
    )
    base._transform_task(task, transform)
    base._scale_team_stats(task, symmetric_scale, symmetric_scale)
    task["zone_scenario"] = _build_zone_scenario(rng, task, target, config)
    base.validate_task(task)

    descriptor = _zone_descriptor(task)
    metadata = task.setdefault("metadata", {})
    metadata.update(
        {
            "task_profile": "balanced_zone_diverse",
            "source": {
                "kind": "programmatic",
                "subtype": "structured_zone_quality_diversity",
            },
            "transform": transform,
            "map_scale": map_scale,
            "symmetric_stat_scale": symmetric_scale,
            "zone_archetype": descriptor["zone_topology"],
            "zone_intensity": descriptor["zone_intensity"],
            "zone_relation": descriptor["zone_relation"],
            "zone_generation": {
                "target": {
                    key: value
                    for key, value in target.__dict__.items()
                },
                "actual_descriptor": descriptor,
                "descriptor_source": "recomputed_from_final_zone_scenario",
                "conditional_inapplicable_values_excluded": True,
            },
        }
    )
    return task


def _certify_task(
    task: dict[str, Any],
    config: BridgeBalancedConfig,
    *,
    job_id: int,
) -> tuple[bool, dict[str, Any], str]:
    discovery_seed = _derive_seed(config.seed, job_id, "A")
    accepted_a, result_a = base._passes_win_rate_filter(
        task,
        physics=config.physics,
        heuristic=config.heuristic,
        num_seeds=base._discovery_num_seeds(config),
        seed=discovery_seed,
        epsilon=base._evaluation_epsilon(config),
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
    if not accepted_a:
        reason = str(result_a.get("reject_reason") or "A_rejected")
        return False, {"A": result_a}, f"A_rejected:{reason}"

    frozen_hash = base._task_hash(task)
    confirmation_seed = (
        base._phase_seed_base(task, config.seed)
        + config.confirmation_seed_offset
    )
    accepted_b, confirmation = base._run_confirmation(
        task,
        config,
        seed=confirmation_seed,
    )
    if base._task_hash(task) != frozen_hash:
        raise RuntimeError("B confirmation mutated the frozen task.")
    if not accepted_b:
        reasons = ",".join(confirmation.get("reject_reasons", ()))
        return False, {"A": result_a, "B": confirmation}, f"B_rejected:{reasons}"
    return True, {"A": result_a, "B": confirmation}, "B_confirmed"


def _initialize_cpu_worker(counter: Any) -> None:
    global _CPU_WORKER_ID
    with counter.get_lock():
        _CPU_WORKER_ID = int(counter.value)
        counter.value += 1
    base.jax.config.update("jax_platform_name", "cpu")


def _sample_cpu_worker(job: _CandidateJob) -> _CandidateResult:
    try:
        task = _generate_task(job)
        accepted, certification, reason = _certify_task(
            task,
            job.config,
            job_id=job.job_id,
        )
        if not accepted:
            return _CandidateResult(
                job_id=job.job_id,
                worker_id=max(_CPU_WORKER_ID, 0),
                worker_pid=os.getpid(),
                target=job.target,
                candidate=None,
                reason=reason,
            )

        descriptor = _zone_descriptor(task)
        confirmation = certification["B"]
        digest = base._task_hash(task)
        metadata = task.setdefault("metadata", {})
        metadata.update(
            {
                "validation_status": "confirmed",
                "validation_gate": {
                    "selected": True,
                    "reason": "B_confirmed",
                    "policy": "adaptive_A_then_frozen_original_flipped_B",
                },
                "fast_filter_eval": certification["A"],
                "confirmation": confirmation,
                "filter_eval": confirmation["original"],
                "evaluation_protocol": {
                    "epsilon": base._evaluation_epsilon(job.config),
                    "A_seed_stages": list(job.config.discovery_seed_stages),
                    "B_seed_stages": list(job.config.confirmation_seed_stages),
                    "task_frozen_before_B": True,
                    "combat_stat_repair": False,
                },
            }
        )
        candidate = _CertifiedCandidate(
            task=task,
            digest=digest,
            descriptor=descriptor,
            parameter_signature=_zone_parameter_signature(task),
            behavior_signature=_zone_behavior_signature(
                task,
                confirmation,
                job.config.filter_max_episode_steps,
            ),
            roster_key=_roster_key(task),
            family_id=_family_id(task, descriptor),
        )
        return _CandidateResult(
            job_id=job.job_id,
            worker_id=max(_CPU_WORKER_ID, 0),
            worker_pid=os.getpid(),
            target=job.target,
            candidate=candidate,
            reason="B_confirmed",
        )
    except Exception as exc:
        return _CandidateResult(
            job_id=job.job_id,
            worker_id=max(_CPU_WORKER_ID, 0),
            worker_pid=os.getpid(),
            target=job.target,
            candidate=None,
            reason=f"{type(exc).__name__}:{exc}",
        )


def _rms_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(first - second))))


def _nearest(
    signature: np.ndarray,
    candidates: Sequence[_CertifiedCandidate],
    *,
    behavior: bool,
) -> float:
    if not candidates:
        return math.inf
    return min(
        _rms_distance(
            signature,
            candidate.behavior_signature if behavior else candidate.parameter_signature,
        )
        for candidate in candidates
    )


def _zone_cell(candidate: _CertifiedCandidate) -> tuple[str, ...]:
    return tuple(candidate.descriptor[dimension] for dimension in COVERAGE_DIMENSIONS)


def _duplicate_reason(
    candidate: _CertifiedCandidate,
    pool: Sequence[_CertifiedCandidate],
    config: BridgeBalancedConfig,
) -> str | None:
    if any(candidate.digest == other.digest for other in pool):
        return "exact_duplicate"
    # A common aggregate outcome must not erase a genuinely different zone
    # structure.  Approximate rejection therefore requires the same semantic
    # cell and simultaneous parameter *and* outcome proximity.
    same_cell = [other for other in pool if _zone_cell(other) == _zone_cell(candidate)]
    if not same_cell:
        return None
    parameter = _nearest(candidate.parameter_signature, same_cell, behavior=False)
    behavior = _nearest(candidate.behavior_signature, same_cell, behavior=True)
    if (
        parameter < config.zone_parameter_distance_threshold
        and behavior < config.zone_behavior_distance_threshold
    ):
        return "conditional_zone_duplicate"
    return None


def _select_candidates(
    pool: Sequence[_CertifiedCandidate],
    config: BridgeBalancedConfig,
) -> list[_CertifiedCandidate]:
    remaining = list(pool)
    selected: list[_CertifiedCandidate] = []
    coverage = _CoverageState(config)
    cell_counts: Counter[tuple[str, ...]] = Counter()
    roster_counts: Counter[tuple[tuple[int, ...], tuple[int, ...]]] = Counter()
    family_counts: Counter[str] = Counter()
    cell_cap = max(1, int(math.ceil(config.n_tasks * config.max_zone_cell_fraction)))
    roster_cap = max(
        1,
        int(math.ceil(config.n_tasks * config.max_exact_roster_fraction)),
    )

    while remaining and len(selected) < config.n_tasks:
        best_index: int | None = None
        best_score = -math.inf
        for index, candidate in enumerate(remaining):
            cell = _zone_cell(candidate)
            if cell_counts[cell] >= cell_cap:
                continue
            if roster_counts[candidate.roster_key] >= roster_cap:
                continue
            if family_counts[candidate.family_id] >= config.max_family_tasks:
                continue

            parameter_distance = _nearest(
                candidate.parameter_signature,
                selected,
                behavior=False,
            )
            behavior_distance = _nearest(
                candidate.behavior_signature,
                selected,
                behavior=True,
            )
            parameter_novelty = (
                1.0
                if not math.isfinite(parameter_distance)
                else min(parameter_distance / config.parameter_novelty_scale, 1.0)
            )
            behavior_novelty = (
                1.0
                if not math.isfinite(behavior_distance)
                else min(behavior_distance / config.behavior_novelty_scale, 1.0)
            )
            score = (
                config.coverage_score_weight * coverage.gain(candidate.descriptor)
                + config.parameter_novelty_weight * parameter_novelty
                + config.behavior_novelty_weight * behavior_novelty
            )
            score += int(candidate.digest[:12], 16) / float(16**12) * 1e-12
            if score > best_score:
                best_score = score
                best_index = index

        if best_index is None:
            break
        candidate = remaining.pop(best_index)
        selected.append(candidate)
        coverage.add(candidate.descriptor)
        cell_counts[_zone_cell(candidate)] += 1
        roster_counts[candidate.roster_key] += 1
        family_counts[candidate.family_id] += 1
    return selected


def _multiprocessing_context() -> multiprocessing.context.BaseContext:
    method = "spawn" if os.name == "nt" else "forkserver"
    try:
        return multiprocessing.get_context(method)
    except ValueError:
        return multiprocessing.get_context("spawn")


def _worker_count(config: BridgeBalancedConfig, budget: int) -> int:
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return max(1, min(config.cpu_cores, available, budget))


def _coverage_from_candidates(
    candidates: Iterable[_CertifiedCandidate],
    config: BridgeBalancedConfig,
) -> _CoverageState:
    coverage = _CoverageState(config)
    for candidate in candidates:
        coverage.add(candidate.descriptor)
    return coverage


def _leased_coverage(
    pool: Sequence[_CertifiedCandidate],
    leases: Counter[tuple[str, ...]],
    config: BridgeBalancedConfig,
) -> _CoverageState:
    coverage = _coverage_from_candidates(pool, config)
    for values, count in leases.items():
        descriptor = dict(zip(COVERAGE_DIMENSIONS, values))
        for _ in range(count):
            coverage.add(descriptor)
    return coverage


def _validate_config(config: BridgeBalancedConfig) -> None:
    if config.n_tasks <= 0:
        raise ValueError("n_tasks must be positive.")
    if config.cpu_cores <= 0:
        raise ValueError("cpu_cores must be positive.")
    if config.max_n_zone != 10:
        raise ValueError("This sampler requires max_n_zone=10.")
    if config.open_ended_generation or not math.isclose(config.programmatic_ratio, 1.0):
        raise ValueError("This sampler only uses its structured programmatic generator.")
    if (
        config.static_balance_repair
        or config.simulation_balance_repair
        or config.simulation_repair_rounds
    ):
        raise ValueError("Combat-stat repair must remain disabled.")
    if not config.filter_by_win_rate or not config.confirmation_enabled:
        raise ValueError("Both A discovery and B confirmation must be enabled.")
    if len(config.zone_count_target_weights) != len(COUNT_BUCKET_RANGES):
        raise ValueError("zone_count_target_weights must match COUNT_BUCKET_RANGES.")
    if any(value < 0.0 for value in config.zone_count_target_weights):
        raise ValueError("zone_count_target_weights must be non-negative.")
    if not math.isclose(sum(config.zone_count_target_weights), 1.0, abs_tol=1e-9):
        raise ValueError("zone_count_target_weights must sum to one.")
    if config.target_draws < 1:
        raise ValueError("target_draws must be positive.")
    if config.candidate_budget_multiplier < 1:
        raise ValueError("candidate_budget_multiplier must be positive.")
    if 0 < config.candidate_budget < config.n_tasks:
        raise ValueError("candidate_budget must be zero or at least n_tasks.")
    if config.certified_pool_multiplier < 1.0:
        raise ValueError("certified_pool_multiplier must be at least one.")
    if config.max_in_flight < 0:
        raise ValueError("max_in_flight must be non-negative.")
    if config.collection_time_budget_seconds < 0:
        raise ValueError("collection_time_budget_seconds must be non-negative.")
    if not (
        0.0 < config.zone_coverage_fraction_min <= config.zone_coverage_fraction_max < 1.0
    ):
        raise ValueError("Invalid zone coverage fraction range.")
    unit_interval = (
        config.zone_marginal_fraction,
        config.zone_pair_fraction,
        config.coverage_score_weight,
        config.parameter_novelty_weight,
        config.behavior_novelty_weight,
        config.max_zone_cell_fraction,
        config.max_exact_roster_fraction,
    )
    if any(not 0.0 <= value <= 1.0 for value in unit_interval):
        raise ValueError("Coverage, selector, and cap fractions must lie in [0, 1].")
    if not math.isclose(
        config.coverage_score_weight
        + config.parameter_novelty_weight
        + config.behavior_novelty_weight,
        1.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Final selector weights must sum to one.")
    supported = (
        (set(config.composition_archetypes), set(COMPOSITION_ARCHETYPES), "composition"),
        (
            set(config.composition_match_modes),
            set(COMPOSITION_MATCH_MODES),
            "match mode",
        ),
        (set(config.layout_archetypes), set(LAYOUT_ARCHETYPES), "layout"),
        (set(config.distance_buckets), set(DISTANCE_BUCKETS), "distance"),
        (set(config.spread_buckets), set(SPREAD_BUCKETS), "spread"),
    )
    for selected, available, label in supported:
        unknown = selected.difference(available)
        if unknown:
            raise ValueError(f"Unsupported {label} values: {sorted(unknown)}.")


def _normalized_entropy(values: Sequence[str]) -> float:
    if not values:
        return 0.0
    counts = np.asarray(list(Counter(values).values()), dtype=float)
    probabilities = counts / counts.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(len(counts)) if len(counts) > 1 else 0.0


def _quota_audit(
    candidates: Sequence[_CertifiedCandidate],
    config: BridgeBalancedConfig,
) -> dict[str, Any]:
    coverage = _coverage_from_candidates(candidates, config)
    deficits = []
    for dimension in COVERAGE_DIMENSIONS:
        for value in ACTIVE_COVERAGE_VALUES[dimension]:
            target = coverage.marginal_target(dimension, value)
            actual = int(coverage.marginals[(dimension, value)])
            if target > actual:
                deficits.append(
                    {
                        "dimension": dimension,
                        "value": value,
                        "target": target,
                        "actual": actual,
                        "deficit": target - actual,
                    }
                )
    return {
        "marginal_targets_satisfied": not deficits,
        "marginal_deficits": deficits,
    }


def _feedback_report(feedback: _BucketFeedback) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dimension in COVERAGE_DIMENSIONS:
        rows = {}
        for value in ACTIVE_COVERAGE_VALUES[dimension]:
            key = (dimension, value)
            generated = int(feedback.generated[key])
            confirmed = int(feedback.confirmed[key])
            rows[value] = {
                "generated": generated,
                "confirmed": confirmed,
                "confirmed_rate": confirmed / generated if generated else 0.0,
            }
        result[dimension] = rows
    return result


def _collection_report(
    selected: Sequence[_CertifiedCandidate],
    pool: Sequence[_CertifiedCandidate],
    *,
    config: BridgeBalancedConfig,
    generation_seed: int,
    candidate_budget: int,
    submitted_jobs: int,
    reject_counts: Counter[str],
    feedback: _BucketFeedback,
    elapsed_seconds: float,
    worker_count: int,
    max_in_flight: int,
    start_method: str,
) -> dict[str, Any]:
    descriptor_distributions = {
        dimension: dict(
            Counter(candidate.descriptor[dimension] for candidate in selected)
        )
        for dimension in COVERAGE_DIMENSIONS
    }
    entropy = {
        dimension: _normalized_entropy(
            [candidate.descriptor[dimension] for candidate in selected]
        )
        for dimension in COVERAGE_DIMENSIONS
    }
    zone_type_counts: Counter[str] = Counter()
    effect_values: dict[str, list[float]] = {"lava": [], "swamp": []}
    total_zones = 0
    for candidate in selected:
        zones = candidate.task["zone_scenario"]
        types = np.asarray(zones["zone_type"], dtype=int).reshape(-1)
        effects = np.asarray(zones["effect_value"], dtype=float).reshape(-1)
        total_zones += len(types)
        for zone_type, effect in zip(types, effects):
            name = ZONE_TYPE_NAMES[int(zone_type)]
            zone_type_counts[name] += 1
            if name in effect_values:
                effect_values[name].append(float(effect))

    return {
        "strategy": "conditional_zone_quality_diversity",
        "task_profile": "balanced_zone_diverse",
        "complete": len(selected) == config.n_tasks,
        "requested_tasks": config.n_tasks,
        "selected_tasks": len(selected),
        "certified_pool": len(pool),
        "task_shortfall": max(0, config.n_tasks - len(selected)),
        "candidate_budget": candidate_budget,
        "submitted_jobs": submitted_jobs,
        "generation_seed": generation_seed,
        "evaluation_seed": config.seed,
        "elapsed_seconds": elapsed_seconds,
        "reject_counts": dict(reject_counts),
        "target_feedback": _feedback_report(feedback),
        "descriptor_distributions": descriptor_distributions,
        "normalized_entropy": entropy,
        "quota_audit": _quota_audit(selected, config),
        "zone_inventory": {
            "total_zones": total_zones,
            "mean_zones_per_task": total_zones / len(selected) if selected else 0.0,
            "type_counts": dict(zone_type_counts),
            "effect_ranges": {
                name: {
                    "min": min(values, default=None),
                    "mean": float(np.mean(values)) if values else None,
                    "max": max(values, default=None),
                }
                for name, values in effect_values.items()
            },
        },
        "parallel_sampling": {
            "execution_backend": "cpu",
            "workers": worker_count,
            "threads_per_worker": 1,
            "max_in_flight": max_in_flight,
            "scheduling": "adaptive_asynchronous_refill",
            "multiprocessing_start_method": start_method,
            "commit_policy": "confirmed_pool_then_constrained_greedy_selection",
        },
    }


def sample_tasks_with_report(
    config: BridgeBalancedConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect and select B-confirmed tasks using CPU worker processes."""

    _validate_config(config)
    generation_seed = (
        secrets.randbits(32)
        if config.generation_seed is None
        else int(config.generation_seed)
    )
    budget = (
        config.candidate_budget
        if config.candidate_budget > 0
        else config.n_tasks * config.candidate_budget_multiplier
    )
    pool_target = max(
        config.n_tasks,
        int(math.ceil(config.n_tasks * config.certified_pool_multiplier)),
    )
    worker_count = _worker_count(config, budget)
    max_in_flight = max(worker_count, config.max_in_flight or 2 * worker_count)
    pool: list[_CertifiedCandidate] = []
    feedback = _BucketFeedback()
    reject_counts: Counter[str] = Counter()
    target_leases: Counter[tuple[str, ...]] = Counter()
    next_job_id = 0
    started = time.monotonic()

    print(
        "Balanced zone-diverse CPU sampling: "
        f"workers={worker_count}, max_in_flight={max_in_flight}, "
        f"target={config.n_tasks}, certified_pool_target={pool_target}, "
        f"candidate_budget={budget}, generation_seed={generation_seed}, "
        f"evaluation_seed={config.seed}",
        flush=True,
    )

    def time_exhausted() -> bool:
        return bool(
            config.collection_time_budget_seconds > 0
            and time.monotonic() - started >= config.collection_time_budget_seconds
        )

    def make_job() -> _CandidateJob:
        nonlocal next_job_id
        job_id = next_job_id
        scheduler_rng = random.Random(
            _derive_seed(generation_seed, job_id, "schedule")
        )
        coverage = _leased_coverage(pool, target_leases, config)
        target = _choose_target(scheduler_rng, config, coverage, feedback)
        descriptor = _target_descriptor(target)
        lease = tuple(descriptor[dimension] for dimension in COVERAGE_DIMENSIONS)
        target_leases[lease] += 1
        feedback.record_generated(descriptor)
        next_job_id += 1
        return _CandidateJob(
            job_id=job_id,
            generation_seed=generation_seed,
            target=target,
            config=config,
        )

    def commit(result: _CandidateResult) -> None:
        target_descriptor = _target_descriptor(result.target)
        lease = tuple(
            target_descriptor[dimension] for dimension in COVERAGE_DIMENSIONS
        )
        target_leases[lease] -= 1
        if result.candidate is None:
            reject_counts[result.reason] += 1
            return
        feedback.record_confirmed(target_descriptor)
        duplicate = _duplicate_reason(result.candidate, pool, config)
        if duplicate is not None:
            reject_counts[duplicate] += 1
            return
        result.candidate.task.setdefault("metadata", {})["parallel_sampling"] = {
            "job_id": result.job_id,
            "worker_id": result.worker_id,
            "worker_pid": result.worker_pid,
            "execution_backend": "cpu",
            "commit_policy": "B_confirmed_pool",
        }
        pool.append(result.candidate)
        print(
            f"job={result.job_id + 1}/{budget} CONFIRMED "
            f"pool={len(pool)}/{pool_target} "
            f"zones={result.candidate.task['zone_scenario']['n_zone']} "
            f"cell={_zone_cell(result.candidate)} worker={result.worker_id}",
            flush=True,
        )

    executor: ProcessPoolExecutor | None = None
    start_method = "local"
    try:
        if worker_count == 1:
            while (
                next_job_id < budget
                and len(pool) < pool_target
                and not time_exhausted()
            ):
                commit(_sample_cpu_worker(make_job()))
        else:
            context = _multiprocessing_context()
            start_method = context.get_start_method()
            worker_counter = context.Value("i", 0)
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=context,
                initializer=_initialize_cpu_worker,
                initargs=(worker_counter,),
            )
            pending: dict[Any, _CandidateJob] = {}

            def refill() -> None:
                while (
                    len(pending) < max_in_flight
                    and next_job_id < budget
                    and len(pool) < pool_target
                    and not time_exhausted()
                ):
                    job = make_job()
                    pending[executor.submit(_sample_cpu_worker, job)] = job

            refill()
            while pending and len(pool) < pool_target and not time_exhausted():
                completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in sorted(completed, key=lambda item: pending[item].job_id):
                    job = pending.pop(future)
                    result = future.result()
                    if result.job_id != job.job_id:
                        raise RuntimeError("Worker returned a mismatched job_id.")
                    commit(result)
                    if len(pool) >= pool_target:
                        break
                refill()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    if not pool:
        raise RuntimeError(
            "No B-confirmed task was collected. "
            f"candidate_budget={budget}, rejections={dict(reject_counts)}"
        )
    selected = _select_candidates(pool, config)
    if len(selected) < config.n_tasks and not config.save_partial:
        raise RuntimeError(
            f"Only selected {len(selected)}/{config.n_tasks} tasks; "
            f"certified_pool={len(pool)}, rejections={dict(reject_counts)}"
        )
    if len(selected) < config.n_tasks:
        print(
            f"WARNING: returning partial result {len(selected)}/{config.n_tasks}; "
            f"certified_pool={len(pool)}",
            flush=True,
        )

    tasks: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected):
        task = candidate.task
        digest = base._task_hash(task)
        task["task_id"] = f"task_{index:06d}_{digest[:10]}"
        task.setdefault("metadata", {})["canonical_hash"] = digest
        task["metadata"]["selection"] = {
            "selected": True,
            "rank": index,
            "policy": "conditional_coverage_plus_zone_parameter_and_outcome_novelty",
        }
        tasks.append(task)

    report = _collection_report(
        selected,
        pool,
        config=config,
        generation_seed=generation_seed,
        candidate_budget=budget,
        submitted_jobs=next_job_id,
        reject_counts=reject_counts,
        feedback=feedback,
        elapsed_seconds=time.monotonic() - started,
        worker_count=worker_count,
        max_in_flight=max_in_flight,
        start_method=start_method,
    )
    return tasks, report


def save_task_bank(
    config: BridgeBalancedConfig,
    tasks: Sequence[dict[str, Any]],
    report: dict[str, Any],
) -> Path:
    """Save a standard TABX bank with a complete sampling audit."""

    filter_protocol = {
        "enabled": True,
        "policy": "adaptive_A_then_frozen_original_flipped_B",
        "task_profile": "balanced_zone_diverse",
        "evaluation_epsilon": base._evaluation_epsilon(config),
        "physics": config.physics,
        "heuristic": config.heuristic,
        "max_episode_steps": config.filter_max_episode_steps,
        "A": {
            "seed_stages": list(config.discovery_seed_stages),
            "minimum_accept_seeds": config.discovery_min_accept_seeds,
            "candidate_win_rate": [
                config.discovery_candidate_win_rate_min,
                config.discovery_candidate_win_rate_max,
            ],
            "target_win_rate": [config.win_rate_min, config.win_rate_max],
            "max_abs_hp_margin": config.discovery_max_abs_hp_margin,
        },
        "B": {
            "seed_stages": list(config.confirmation_seed_stages),
            "minimum_accept_seeds": config.confirmation_min_accept_seeds,
            "orientation_win_rate": [
                config.confirmation_win_rate_min,
                config.confirmation_win_rate_max,
            ],
            "team_invariant_win_rate": [
                config.confirmation_team_win_rate_min,
                config.confirmation_team_win_rate_max,
            ],
            "balance_probability": config.confirmation_balance_probability,
            "max_side_bias": config.confirmation_max_side_bias,
            "max_abs_hp_margin": config.confirmation_max_abs_hp_margin,
        },
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
        filter_protocol=filter_protocol,
    )
    bank = json.loads(output.read_text(encoding="utf-8"))
    bank["manifest"]["generator"] = "src.tabx.sample_task_bridge_balanced"
    bank["manifest"]["generator_protocol"] = {
        "strategy": "conditional_zone_quality_diversity",
        "task_profile": "balanced_zone_diverse",
        "balance_is_hard_constraint": True,
        "combat_stat_repair": False,
        "zone_descriptor_source": "recomputed_from_final_zone_scenario",
        "conditional_coverage": {
            "dimensions": list(COVERAGE_DIMENSIONS),
            "pair_dimensions": [list(pair) for pair in PAIR_DIMENSIONS],
            "count_bucket_ranges": {
                key: list(value) for key, value in COUNT_BUCKET_RANGES.items()
            },
            "count_target_weights": list(config.zone_count_target_weights),
            "inapplicable_values_excluded": sorted(INAPPLICABLE_VALUES),
        },
        "zone_generation": {
            "max_n_zone": config.max_n_zone,
            "topologies": list(TOPOLOGIES),
            "type_mixes": list(TYPE_MIXES),
            "intensity_bins": list(INTENSITY_BINS),
            "relations": list(RELATIONS),
            "effect_profiles": list(EFFECT_PROFILES),
            "spatial_roles": list(SPATIAL_ROLES),
            "per_zone_effects": True,
            "symmetry_aware_geometry": True,
            "lava_spawn_intersection_rejected": True,
        },
        "parallelism": {
            "backend": "cpu",
            "workers": config.cpu_cores,
            "threads_per_worker": 1,
        },
        "collection_report": report,
    }
    output.write_text(
        json.dumps(bank, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    config = tyro.cli(BridgeBalancedConfig)
    tasks, report = sample_tasks_with_report(config)
    output = save_task_bank(config, tasks, report)
    status = (
        "complete"
        if report["complete"]
        else f"partial={len(tasks)}/{config.n_tasks}"
    )
    print(
        f"Saved {len(tasks)} balanced zone-diverse tasks to {output}; "
        f"collection={status}, certified_pool={report['certified_pool']}, "
        f"mean_zones={report['zone_inventory']['mean_zones_per_task']:.3f}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
