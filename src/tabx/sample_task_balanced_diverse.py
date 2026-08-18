"""Balance- or hard-certified, diversity-directed TABX task collection.

This module deliberately leaves :mod:`src.tabx.sample_task` unchanged.  It
reuses that module's task schema, rollout evaluation, and independent A/B
validation protocol, but replaces its collection policy with a constrained
quality-diversity policy:

* static combat balance is a hard admission constraint;
* only geometry and one in-role roster replacement may repair a candidate;
* asymmetric combat-stat repair is never called;
* target buckets are scheduled from accepted-coverage deficits;
* B-confirmed candidates are buffered and selected by marginal coverage gain
  plus parameter/behaviour novelty.

The default ``hard_task_mode=False`` preserves the original balanced
confirmation protocol.  ``hard_task_mode=True`` keeps the same generator,
structural repair, de-duplication, and diversity selection pipeline, but
certifies an intentionally difficult team-0 matchup using complementary
original/flipped win-rate bands.

Run it with::

    python -m src.tabx.sample_task_balanced_diverse --n-tasks 100 \
        --cpu-cores 8 --output outputs/balanced_diverse_tasks.json
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
import secrets
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# The CLI is CPU-only.  Configure JAX before importing the legacy sampler,
# which imports JAX as part of its task-bank helpers.
if __name__ == "__main__":
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_SKIP_CUDA_CONSTRAINTS_CHECK"] = "1"

import numpy as np
import tyro

from src.tabx import sample_task as base
from src.tabx import task_generators as generators
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


SCENARIO_DIMENSIONS = (
    "composition",
    "layout",
    "distance",
    "spread",
    "zone",
    "zone_intensity",
    "zone_relation",
    "match_mode",
)

PROFILE_DIMENSIONS = (
    "team_size",
    "ranged_profile",
    "healer_profile",
)

BEHAVIOR_DIMENSIONS = (
    "first_contact",
    "duration",
    "healing",
)

ALL_DIVERSITY_DIMENSIONS = SCENARIO_DIMENSIONS + PROFILE_DIMENSIONS + BEHAVIOR_DIMENSIONS

PAIR_DIMENSIONS = (
    ("composition", "distance"),
    ("composition", "zone"),
    ("layout", "zone"),
    ("match_mode", "distance"),
    ("healer_profile", "duration"),
)

RANGED_UNIT_IDS = frozenset((4, 5, 6))
HEALER_UNIT_IDS = frozenset((7, 8))
_CPU_WORKER_ID = -1


@dataclass(frozen=True)
class BalancedDiverseConfig(base.SampleConfig):
    """Configuration for balanced/hard quality-diversity collection."""

    output: str = "balanced_diverse_tasks.json"
    n_tasks: int = 100
    cpu_cores: int = 4
    max_n_zone: int = 10
    # Backwards-compatible CLI name; under asynchronous scheduling this is the
    # maximum number of submitted/in-flight jobs.  Zero keeps one running and
    # one queued job per worker so a worker can pick up new work immediately.
    commit_batch_size: int = 0

    # ``seed`` is inherited from SampleConfig and remains the fixed A/B
    # evaluation seed.  Leaving this field unset draws a fresh generation seed
    # once per invocation; pass a value only when a sampling run must be replayed.
    generation_seed: int | None = None

    # Optional asymmetric hard-task profile.  False is deliberately the
    # backwards-compatible default.  Hard tasks keep the same static balance
    # screen, so their difficulty must come from tactical interactions rather
    # than a large raw combat-value gap.
    hard_task_mode: bool = False
    hard_win_rate_min: float = 0.20
    hard_win_rate_max: float = 0.40
    hard_discovery_candidate_min: float = 0.10
    hard_discovery_candidate_max: float = 0.50
    hard_discovery_seed_stages: tuple[int, ...] = (16, 32, 64, 128)
    hard_discovery_min_accept_seeds: int = 64
    hard_discovery_max_abs_hp_margin: float = 0.65
    hard_confirmation_seed_stages: tuple[int, ...] = (64, 128, 256)
    hard_confirmation_min_accept_seeds: int = 128
    hard_confirmation_band_probability: float = 0.85
    hard_max_side_bias: float = 0.10
    hard_max_truncation_rate: float = 0.15
    hard_max_abs_hp_margin: float = 0.65
    hard_max_hp_side_bias: float = 0.20
    hard_min_weak_damage_share: float = 0.15

    # Source mixture.  The remainder is target-directed programmatic sampling.
    open_ended_generation: bool = True
    programmatic_ratio: float = 0.75
    free_generation_ratio: float = 0.10
    parent_mutation_ratio: float = 0.10
    parent_crossover_ratio: float = 0.05

    # Unit templates remain semantically stable by default.  If callers expose
    # more values, the selected scale is still applied symmetrically to both teams.
    stat_scales: tuple[float, ...] = (1.0,)
    couple_stat_scales: bool = True

    # The old sampler's stat-repair paths must stay disabled in this module.
    static_balance_repair: bool = False
    simulation_balance_repair: bool = False
    simulation_repair_rounds: int = 0

    allow_geometry_repair: bool = True
    allow_roster_repair: bool = True
    geometry_repair_distance_factor: float = 0.85
    geometry_repair_map_factor: float = 0.95
    max_roster_repair_trials: int = 6
    static_balance_accept_gap: float = 0.10
    static_balance_resample_gap: float = 0.15

    # Candidate collection and deficit-directed scheduling.
    diversity_candidate_budget: int = 0
    diversity_candidate_budget_multiplier: int = 20
    # Keep only n_tasks certified candidates by default.  Values above 1.0
    # restore the old collect-more-then-select policy.  At 1.0, accepted tasks
    # are appended permanently; diversity is handled by deficit-directed
    # generation plus online cell/roster/family/repair admission limits.
    certified_pool_multiplier: float = 1.0
    target_draws: int = 64
    target_accept_rate_floor: float = 0.05
    target_weight_cap: float = 100.0
    target_exploration: float = 0.10

    # Marginal and selected-pair lower-bound targets.  Targets are deliberately
    # fractional so the selector retains room for novelty-driven samples.
    scenario_marginal_fraction: float = 0.65
    profile_marginal_fraction: float = 0.40
    behavior_marginal_fraction: float = 0.25
    pair_coverage_fraction: float = 0.25

    # Final constrained greedy selection.
    coverage_score_weight: float = 0.50
    parameter_novelty_weight: float = 0.28
    behavior_novelty_weight: float = 0.22
    parameter_novelty_scale: float = 0.20
    behavior_novelty_scale: float = 0.20
    max_map_cell_fraction: float = 0.02
    max_exact_roster_fraction: float = 0.01
    max_family_tasks: int = 2
    min_natural_fraction: float = 0.70
    max_geometry_repaired_fraction: float = 0.15
    max_roster_repaired_fraction: float = 0.15


@dataclass(eq=False)
class _CertifiedCandidate:
    task: dict[str, Any]
    digest: str
    parameter_signature: np.ndarray
    behavior_signature: np.ndarray
    descriptor: dict[str, str]
    map_cell: tuple[str, ...]
    roster_key: tuple[tuple[int, ...], tuple[int, ...]]
    family_id: str
    source_kind: str
    repair_types: tuple[str, ...]


@dataclass
class _CandidateOutcome:
    candidate: _CertifiedCandidate | None
    reason: str
    failure_type: str | None = None


@dataclass(frozen=True)
class _ParentPayload:
    task: dict[str, Any]
    digest: str


@dataclass(frozen=True)
class _ParallelJob:
    candidate_id: int
    generation_seed: int
    target: tuple[str, ...]
    source_kind: str
    parents: tuple[_ParentPayload, ...]
    config: BalancedDiverseConfig


@dataclass
class _ParallelResult:
    candidate_id: int
    target: tuple[str, ...]
    source_kind: str
    parent_ids: tuple[str, ...]
    worker_id: int
    worker_pid: int
    outcome: _CandidateOutcome


class _BucketFeedback:
    """Acceptance estimates indexed by individual target dimensions."""

    def __init__(self) -> None:
        self.generated: Counter[tuple[str, str]] = Counter()
        self.confirmed: Counter[tuple[str, str]] = Counter()

    def record_generated(self, target: tuple[str, ...]) -> None:
        for dimension, value in zip(SCENARIO_DIMENSIONS, target):
            self.generated[(dimension, value)] += 1

    def record_confirmed(self, target: tuple[str, ...]) -> None:
        for dimension, value in zip(SCENARIO_DIMENSIONS, target):
            self.confirmed[(dimension, value)] += 1

    def estimated_accept_rate(self, target: tuple[str, ...]) -> float:
        rates = []
        for dimension, value in zip(SCENARIO_DIMENSIONS, target):
            key = (dimension, value)
            rates.append((self.confirmed[key] + 1.0) / (self.generated[key] + 2.0))
        return float(np.mean(rates)) if rates else 0.5


class _CoverageState:
    """Hierarchical marginal/pair coverage without an eight-way Cartesian grid."""

    def __init__(self, config: BalancedDiverseConfig) -> None:
        self.config = config
        self.marginals: Counter[tuple[str, str]] = Counter()
        self.pairs: Counter[tuple[str, str, str, str]] = Counter()

    def copy(self) -> _CoverageState:
        result = _CoverageState(self.config)
        result.marginals.update(self.marginals)
        result.pairs.update(self.pairs)
        return result

    def add(self, descriptor: dict[str, str]) -> None:
        for dimension in ALL_DIVERSITY_DIMENSIONS:
            value = descriptor.get(dimension)
            if value is not None:
                self.marginals[(dimension, value)] += 1
        for first, second in PAIR_DIMENSIONS:
            first_value = descriptor.get(first)
            second_value = descriptor.get(second)
            if first_value is not None and second_value is not None:
                self.pairs[(first, first_value, second, second_value)] += 1

    def marginal_target(self, dimension: str, value: str) -> int:
        values = _dimension_values(self.config, dimension)
        if value not in values or value == "free":
            return 0
        if dimension in SCENARIO_DIMENSIONS:
            fraction = self.config.scenario_marginal_fraction
        elif dimension in PROFILE_DIMENSIONS:
            fraction = self.config.profile_marginal_fraction
        else:
            fraction = self.config.behavior_marginal_fraction
        return max(1, int(math.floor(self.config.n_tasks * fraction / len(values))))

    def pair_target(
        self,
        first: str,
        first_value: str,
        second: str,
        second_value: str,
    ) -> int:
        first_values = _dimension_values(self.config, first)
        second_values = _dimension_values(self.config, second)
        if first_value not in first_values or second_value not in second_values:
            return 0
        combinations = max(1, len(first_values) * len(second_values))
        raw = self.config.n_tasks * self.config.pair_coverage_fraction / combinations
        return max(1, int(math.floor(raw))) if raw >= 1.0 else 0

    def gain(self, descriptor: dict[str, str]) -> float:
        gains: list[float] = []
        for dimension in ALL_DIVERSITY_DIMENSIONS:
            value = descriptor.get(dimension)
            if value is None:
                continue
            target = self.marginal_target(dimension, value)
            if target > 0 and self.marginals[(dimension, value)] < target:
                gains.append(1.0 / target)
        pair_gains: list[float] = []
        for first, second in PAIR_DIMENSIONS:
            first_value = descriptor.get(first)
            second_value = descriptor.get(second)
            if first_value is None or second_value is None:
                continue
            target = self.pair_target(first, first_value, second, second_value)
            key = (first, first_value, second, second_value)
            if target > 0 and self.pairs[key] < target:
                pair_gains.append(1.0 / target)
        # This is the marginal progress toward all active lower bounds.  Do not
        # average it by the number of dimensions: while any hard quota is open,
        # coverage must dominate the soft novelty terms.
        marginal_gain = float(sum(gains))
        pair_gain = float(sum(pair_gains))
        return marginal_gain + pair_gain


def _validate_config(config: BalancedDiverseConfig) -> None:
    if config.n_tasks <= 0:
        raise ValueError("n_tasks must be positive.")
    if not config.filter_by_win_rate or not config.confirmation_enabled:
        raise ValueError("Collection requires both A filtering and B confirmation.")
    if config.static_balance_repair or config.simulation_balance_repair:
        raise ValueError(
            "Legacy combat-stat repairs must remain disabled; this sampler performs only "
            "structural repairs."
        )
    if config.simulation_repair_rounds != 0:
        raise ValueError("simulation_repair_rounds must be zero in this sampler.")
    source_ratios = (
        config.programmatic_ratio,
        config.free_generation_ratio,
        config.parent_mutation_ratio,
        config.parent_crossover_ratio,
    )
    if any(not 0.0 <= value <= 1.0 for value in source_ratios):
        raise ValueError("Generation source ratios must lie in [0, 1].")
    if config.open_ended_generation and not math.isclose(sum(source_ratios), 1.0):
        raise ValueError("Open-ended generation source ratios must sum to 1.")
    if config.cpu_cores < 1:
        raise ValueError("cpu_cores must be positive.")
    if config.generation_seed is not None and not 0 <= config.generation_seed <= 0xFFFFFFFF:
        raise ValueError("generation_seed must lie in [0, 2**32 - 1].")
    if config.commit_batch_size < 0:
        raise ValueError("commit_batch_size must be non-negative.")
    if not 0.0 < config.geometry_repair_distance_factor < 1.0:
        raise ValueError("geometry_repair_distance_factor must lie in (0, 1).")
    if not 0.0 < config.geometry_repair_map_factor <= 1.0:
        raise ValueError("geometry_repair_map_factor must lie in (0, 1].")
    if not (
        0.0
        <= config.static_balance_accept_gap
        <= config.static_balance_resample_gap
        <= 1.0
    ):
        raise ValueError("Require 0 <= static accept gap <= resample gap <= 1.")
    if config.max_roster_repair_trials < 1:
        raise ValueError("max_roster_repair_trials must be positive.")
    if config.diversity_candidate_budget_multiplier < 1:
        raise ValueError("diversity_candidate_budget_multiplier must be positive.")
    if 0 < config.diversity_candidate_budget < config.n_tasks:
        raise ValueError("An explicit diversity_candidate_budget must be at least n_tasks.")
    if config.certified_pool_multiplier < 1.0:
        raise ValueError("certified_pool_multiplier must be at least 1.")
    if config.target_draws < 1:
        raise ValueError("target_draws must be positive.")
    fractions = (
        config.scenario_marginal_fraction,
        config.profile_marginal_fraction,
        config.behavior_marginal_fraction,
        config.pair_coverage_fraction,
        config.max_map_cell_fraction,
        config.max_exact_roster_fraction,
        config.min_natural_fraction,
        config.max_geometry_repaired_fraction,
        config.max_roster_repaired_fraction,
    )
    if any(not 0.0 <= value <= 1.0 for value in fractions):
        raise ValueError("Coverage and quota fractions must lie in [0, 1].")
    repaired_cap = (
        config.max_geometry_repaired_fraction + config.max_roster_repaired_fraction
    )
    if repaired_cap > 1.0 - config.min_natural_fraction + 1e-12:
        raise ValueError(
            "Geometry and roster repair caps must guarantee min_natural_fraction."
        )
    if config.max_family_tasks < 1:
        raise ValueError("max_family_tasks must be positive.")
    if config.parameter_novelty_scale <= 0 or config.behavior_novelty_scale <= 0:
        raise ValueError("Novelty scales must be positive.")
    if not config.stat_scales or any(value <= 0 for value in config.stat_scales):
        raise ValueError("stat_scales must contain positive values.")
    if not config.couple_stat_scales:
        raise ValueError("Both teams must use the same generated stat scale.")
    if config.max_no_interaction_rate != 0.0:
        raise ValueError("Core balanced collection requires max_no_interaction_rate=0.")
    if config.hard_task_mode:
        if not 0.0 < config.hard_win_rate_min < config.hard_win_rate_max < 0.5:
            raise ValueError(
                "Hard-task target band must satisfy 0 < min < max < 0.5."
            )
        if not (
            0.0
            <= config.hard_discovery_candidate_min
            <= config.hard_win_rate_min
            <= config.hard_win_rate_max
            <= config.hard_discovery_candidate_max
            <= 1.0
        ):
            raise ValueError("Invalid hard-task discovery candidate/target bands.")
        discovery_stages = tuple(sorted(set(config.hard_discovery_seed_stages)))
        confirmation_stages = tuple(sorted(set(config.hard_confirmation_seed_stages)))
        if not discovery_stages or any(value <= 0 for value in discovery_stages):
            raise ValueError("hard_discovery_seed_stages must contain positive values.")
        if not confirmation_stages or any(value <= 0 for value in confirmation_stages):
            raise ValueError("hard_confirmation_seed_stages must contain positive values.")
        if config.hard_discovery_min_accept_seeds not in discovery_stages:
            raise ValueError(
                "hard_discovery_min_accept_seeds must be in hard_discovery_seed_stages."
            )
        if config.hard_confirmation_min_accept_seeds not in confirmation_stages:
            raise ValueError(
                "hard_confirmation_min_accept_seeds must be in "
                "hard_confirmation_seed_stages."
            )
        unit_interval_fields = {
            "hard_discovery_max_abs_hp_margin": config.hard_discovery_max_abs_hp_margin,
            "hard_confirmation_band_probability": config.hard_confirmation_band_probability,
            "hard_max_side_bias": config.hard_max_side_bias,
            "hard_max_truncation_rate": config.hard_max_truncation_rate,
            "hard_max_abs_hp_margin": config.hard_max_abs_hp_margin,
            "hard_max_hp_side_bias": config.hard_max_hp_side_bias,
            "hard_min_weak_damage_share": config.hard_min_weak_damage_share,
        }
        invalid = [
            name for name, value in unit_interval_fields.items() if not 0.0 <= value <= 1.0
        ]
        if invalid:
            raise ValueError(f"Hard-task thresholds must lie in [0, 1]: {invalid}.")
        if config.hard_max_truncation_rate > config.max_truncation_rate:
            raise ValueError(
                "hard_max_truncation_rate cannot exceed the shared max_truncation_rate."
            )


def _dimension_values(config: BalancedDiverseConfig, dimension: str) -> tuple[str, ...]:
    values: dict[str, tuple[str, ...]] = {
        "composition": tuple(config.composition_archetypes),
        "layout": tuple(config.layout_archetypes),
        "distance": tuple(config.distance_buckets),
        "spread": tuple(config.spread_buckets),
        "zone": tuple(config.zone_archetypes),
        "zone_intensity": tuple(config.zone_intensity_buckets),
        "zone_relation": tuple(config.zone_relation_buckets),
        "match_mode": tuple(config.composition_match_modes),
        "team_size": ("small", "medium", "large"),
        "ranged_profile": ("low", "mixed", "high"),
        "healer_profile": ("none", "some", "heavy"),
        "first_contact": ("early", "middle", "late"),
        "duration": ("short", "medium", "long"),
        "healing": ("none", "low", "high"),
    }
    try:
        return values[dimension]
    except KeyError as exc:
        raise ValueError(f"Unknown diversity dimension {dimension!r}.") from exc


def _derive_seed(global_seed: int, candidate_id: int, phase: str) -> int:
    payload = f"balanced-diverse:{global_seed}:{candidate_id}:{phase}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _draw_target(rng: random.Random, config: BalancedDiverseConfig) -> tuple[str, ...]:
    return (
        rng.choice(config.composition_archetypes),
        rng.choice(config.layout_archetypes),
        rng.choice(config.distance_buckets),
        rng.choice(config.spread_buckets),
        rng.choice(config.zone_archetypes),
        rng.choice(config.zone_intensity_buckets),
        rng.choice(config.zone_relation_buckets),
        rng.choice(config.composition_match_modes),
    )


def _target_descriptor(target: tuple[str, ...]) -> dict[str, str]:
    return dict(zip(SCENARIO_DIMENSIONS, target))


def _choose_target(
    rng: random.Random,
    config: BalancedDiverseConfig,
    coverage: _CoverageState,
    feedback: _BucketFeedback,
) -> tuple[str, ...]:
    best: tuple[str, ...] | None = None
    best_score = -math.inf
    for _ in range(config.target_draws):
        target = _draw_target(rng, config)
        deficit = coverage.gain(_target_descriptor(target))
        accept_rate = max(
            feedback.estimated_accept_rate(target), config.target_accept_rate_floor
        )
        directed = min(deficit / accept_rate, config.target_weight_cap)
        exploration = config.target_exploration * rng.random()
        score = directed + exploration
        if score > best_score:
            best = target
            best_score = score
    assert best is not None
    return best


def _sample_scale(
    rng: random.Random,
    values: Sequence[float],
    continuous: bool | None,
    fallback: bool,
) -> float:
    enabled = fallback if continuous is None else continuous
    return base._sample_scale(rng, values, enabled)


def _draw_source_kind(
    rng: random.Random,
    config: BalancedDiverseConfig,
    parents: Sequence[_CertifiedCandidate] | Sequence[_ParentPayload],
) -> str:
    if not config.open_ended_generation:
        return "programmatic"
    crossover = config.parent_crossover_ratio if len(parents) >= 2 else 0.0
    mutation = config.parent_mutation_ratio if parents else 0.0
    free = config.free_generation_ratio
    target = rng.random()
    if target < crossover:
        return "crossover"
    if target < crossover + mutation:
        return "mutation"
    if target < crossover + mutation + free:
        return "free"
    return "programmatic"


def _generate_raw_candidate(
    config: BalancedDiverseConfig,
    *,
    candidate_id: int,
    generation_seed: int,
    target: tuple[str, ...],
    parents: Sequence[_ParentPayload],
    forced_source_kind: str | None = None,
) -> tuple[dict[str, Any], str]:
    seed = _derive_seed(generation_seed, candidate_id, "generate")
    rng = random.Random(seed)
    source_kind = forced_source_kind or _draw_source_kind(rng, config, parents)
    map_scale = _sample_scale(
        rng,
        config.map_scales,
        config.continuous_map_sampling,
        config.continuous_sampling,
    )
    if source_kind == "crossover":
        first, second = rng.sample(list(parents), 2)
        first_task = copy.deepcopy(first.task)
        second_task = copy.deepcopy(second.task)
        first_task["task_id"] = first.digest
        second_task["task_id"] = second.digest
        task = crossover_parent_tasks(
            rng,
            first_task,
            second_task,
            max_n_ally=config.max_n_ally,
            max_n_enemy=config.max_n_enemy,
            max_n_zone=config.max_n_zone,
            map_scale=map_scale,
        )
    elif source_kind == "mutation":
        parent = rng.choice(list(parents))
        parent_task = copy.deepcopy(parent.task)
        parent_task["task_id"] = parent.digest
        task = mutate_parent_task(
            rng,
            parent_task,
            max_n_ally=config.max_n_ally,
            max_n_enemy=config.max_n_enemy,
            max_n_zone=config.max_n_zone,
            map_scale=map_scale,
        )
    elif source_kind == "free":
        task = generate_free_task(
            rng,
            max_n_ally=config.max_n_ally,
            max_n_enemy=config.max_n_enemy,
            max_n_zone=config.max_n_zone,
            map_scale=map_scale,
        )
    else:
        (
            composition,
            layout,
            distance,
            spread,
            zone,
            zone_intensity,
            zone_relation,
            match_mode,
        ) = target
        distance_scale = _sample_scale(
            rng,
            config.distance_scales,
            config.continuous_distance_sampling,
            config.continuous_sampling,
        )
        spread_scale = _sample_scale(
            rng,
            config.spread_scales,
            config.continuous_spread_sampling,
            config.continuous_sampling,
        )
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
            health_frac_buckets=base._health_fraction_buckets(rng, config),
            health_frac_min=config.health_frac_min,
            health_frac_max=config.health_frac_max,
            distance_scale=distance_scale,
            spread_scale=spread_scale,
        )

    transform = rng.choice(config.transforms)
    symmetric_stat_scale = _sample_scale(
        rng,
        config.stat_scales,
        config.continuous_stat_sampling,
        config.continuous_sampling,
    )
    zone_strength_scale = _sample_scale(
        rng,
        config.zone_strength_scales,
        config.continuous_zone_strength_sampling,
        config.continuous_sampling,
    )
    base._transform_task(task, transform)
    base._scale_team_stats(task, symmetric_stat_scale, symmetric_stat_scale)
    base._scale_zone_strength(task, zone_strength_scale)
    metadata = task.setdefault("metadata", {})
    metadata.update(
        {
            "transform": transform,
            "symmetric_stat_scale": symmetric_stat_scale,
            "zone_strength_scale": zone_strength_scale,
            "target_bucket": _target_descriptor(target),
            "candidate_id": candidate_id,
            "generation_seed": generation_seed,
        }
    )
    return task, source_kind


def _parallel_worker_count(config: BalancedDiverseConfig, candidate_budget: int) -> int:
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return max(1, min(config.cpu_cores, available, candidate_budget))


def _multiprocessing_context() -> multiprocessing.context.BaseContext:
    method = "spawn" if os.name == "nt" else "forkserver"
    try:
        return multiprocessing.get_context(method)
    except ValueError:
        return multiprocessing.get_context("spawn")


def _initialize_cpu_worker(counter: Any) -> None:
    global _CPU_WORKER_ID
    with counter.get_lock():
        _CPU_WORKER_ID = int(counter.value)
        counter.value += 1
    # The process inherits the CPU-only environment before importing this
    # module.  This update additionally protects library callers that imported
    # the module before invoking the sampler.
    base.jax.config.update("jax_platform_name", "cpu")


def _assign_parent_payloads(
    source_kind: str,
    pool: Sequence[_CertifiedCandidate],
    parent_leases: Counter[str],
    rng: random.Random,
) -> tuple[_ParentPayload, ...]:
    required = 2 if source_kind == "crossover" else 1 if source_kind == "mutation" else 0
    if required == 0:
        return ()
    if len(pool) < required:
        raise RuntimeError(f"Source {source_kind!r} requires {required} confirmed parents.")

    remaining = list(pool)
    selected: list[_CertifiedCandidate] = []
    for _ in range(required):
        least_leased = min(parent_leases[candidate.digest] for candidate in remaining)
        group = [
            candidate
            for candidate in remaining
            if parent_leases[candidate.digest] == least_leased
        ]
        chosen = rng.choice(group)
        selected.append(chosen)
        remaining = [candidate for candidate in remaining if candidate is not chosen]
    payloads = []
    for candidate in selected:
        parent_leases[candidate.digest] += 1
        payloads.append(
            _ParentPayload(task=copy.deepcopy(candidate.task), digest=candidate.digest)
        )
    return tuple(payloads)


def _leased_coverage(
    pool: Sequence[_CertifiedCandidate],
    target_leases: Counter[tuple[str, ...]],
    config: BalancedDiverseConfig,
) -> _CoverageState:
    coverage = _coverage_from_pool(pool, config)
    for target, count in target_leases.items():
        descriptor = _target_descriptor(target)
        for _ in range(max(0, count)):
            coverage.add(descriptor)
    return coverage


def _sample_candidate_cpu_worker(job: _ParallelJob) -> _ParallelResult:
    """Generate, structurally repair, and A/B-certify one immutable CPU job."""

    try:
        with redirect_stdout(io.StringIO()):
            raw_task, source_kind = _generate_raw_candidate(
                job.config,
                candidate_id=job.candidate_id,
                generation_seed=job.generation_seed,
                target=job.target,
                parents=job.parents,
                forced_source_kind=job.source_kind,
            )
            outcome = _certify_candidate(
                raw_task,
                source_kind,
                job.config,
                candidate_id=job.candidate_id,
            )
    except (RuntimeError, ValueError) as exc:
        outcome = _CandidateOutcome(
            None,
            f"worker:{type(exc).__name__}:{exc}",
            "worker_error",
        )
    return _ParallelResult(
        candidate_id=job.candidate_id,
        target=job.target,
        source_kind=job.source_kind,
        parent_ids=tuple(parent.digest for parent in job.parents),
        worker_id=0 if _CPU_WORKER_ID < 0 else _CPU_WORKER_ID,
        worker_pid=os.getpid(),
        outcome=outcome,
    )


def _static_screen(
    task: dict[str, Any], config: BalancedDiverseConfig
) -> tuple[str | None, dict[str, Any], dict[str, Any]]:
    base.validate_task(task)
    values = base.estimate_dynamic_team_values(task)
    engagement = base.estimate_static_engagement(task, dt=config.static_engagement_dt)
    if not engagement["reachable"]:
        return "no_interaction", values, engagement
    contact = engagement["estimated_first_contact_steps"]
    max_contact = config.filter_max_episode_steps * config.static_engagement_max_step_fraction
    if contact is not None and float(contact) > max_contact:
        return "low_interaction_truncation", values, engagement
    earliest = engagement["estimated_earliest_elimination_steps"]
    ttk_ratio = engagement["estimated_ttk_ratio"]
    if (
        config.static_stomp_filter
        and earliest is not None
        and ttk_ratio is not None
        and float(earliest)
        < config.filter_max_episode_steps * config.static_stomp_min_episode_fraction
        and float(ttk_ratio) > config.static_stomp_min_ttk_ratio
    ):
        return "fast_stomp", values, engagement
    gap = float(values["relative_gap"])
    if gap > config.static_balance_resample_gap:
        return "static_imbalance_hard", values, engagement
    if gap > config.static_balance_accept_gap:
        return "static_imbalance_soft", values, engagement
    return None, values, engagement


def _geometry_repair(
    task: dict[str, Any], config: BalancedDiverseConfig
) -> dict[str, Any] | None:
    trial = copy.deepcopy(task)
    combat_before = _combat_fingerprint(trial)
    base._move_teams_closer(trial, config.geometry_repair_distance_factor)
    scaled = copy.deepcopy(trial)
    base._scale_map(scaled, config.geometry_repair_map_factor)
    try:
        base.validate_task(scaled)
        trial = scaled
    except ValueError:
        try:
            base.validate_task(trial)
        except ValueError:
            return None
    if _combat_fingerprint(trial) != combat_before:
        raise RuntimeError("Geometry repair unexpectedly changed combat attributes.")
    return trial


def _combat_fingerprint(task: dict[str, Any]) -> str:
    scenario = task["scenario"]
    payload = {
        field: scenario[field]
        for field in (
            "unit_ids",
            "healths",
            "attack_damages",
            "attack_ranges",
            "attack_cooldowns",
            "speeds",
        )
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _unit_specs_numpy() -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value, dtype=float).reshape(-1)
        for key, value in get_all_unit_spec().items()
    }


def _replace_unit_template(
    task: dict[str, Any], index: int, replacement: int
) -> dict[str, Any]:
    trial = copy.deepcopy(task)
    scenario = trial["scenario"]
    specs = _unit_specs_numpy()
    unit_ids = np.asarray(scenario["unit_ids"], dtype=int).reshape(-1)
    old = int(unit_ids[index])
    if old == replacement:
        return trial

    old_health = float(np.asarray(scenario["healths"], dtype=float).reshape(-1)[index])
    health_ratio = old_health / max(float(specs["healths"][old]), 1e-8)
    old_damage = float(
        np.asarray(scenario["attack_damages"], dtype=float).reshape(-1)[index]
    )
    base_damage = float(specs["attack_damages"][old])
    damage_scale = old_damage / base_damage if abs(base_damage) > 1e-8 else 1.0
    old_speed = float(np.asarray(scenario["speeds"], dtype=float).reshape(-1)[index])
    speed_scale = old_speed / max(float(specs["speeds"][old]), 1e-8)

    def set_scalar(field: str, value: float | int) -> None:
        values = np.asarray(scenario[field]).copy()
        values.reshape(-1)[index] = value
        scenario[field] = values.tolist()

    set_scalar("unit_ids", replacement)
    set_scalar("body_weights", float(specs["body_weights"][replacement]))
    set_scalar("body_radiuss", float(specs["body_radiuses"][replacement]))
    if "body_radii" in scenario:
        set_scalar("body_radii", float(specs["body_radiuses"][replacement]))
    set_scalar("healths", float(specs["healths"][replacement]) * health_ratio)
    replacement_damage = float(specs["attack_damages"][replacement]) * damage_scale
    set_scalar("attack_damages", replacement_damage)
    set_scalar("attack_ranges", float(specs["attack_ranges"][replacement]))
    set_scalar("attack_cooldowns", float(specs["attack_cooldown"][replacement]))
    set_scalar("sight_angles", float(specs["sight_angles"][replacement]))
    set_scalar("speeds", float(specs["speeds"][replacement]) * speed_scale)
    set_scalar("attack_types", int(replacement_damage < 0.0))

    radius = float(specs["body_radiuses"][replacement])
    width = float(trial.get("grid_info", {}).get("max_field_width", 121.0))
    height = float(trial.get("grid_info", {}).get("max_field_height", 78.0))
    pos_max = np.asarray(scenario["pos_max"], dtype=float).copy()
    pos_min = np.asarray(scenario["pos_min"], dtype=float).copy()
    pos_max[index] = (width / 2.0 - radius, height / 2.0 - radius)
    pos_min[index] = -pos_max[index]
    scenario["pos_max"] = pos_max.tolist()
    scenario["pos_min"] = pos_min.tolist()
    _refresh_composition_metadata(trial)
    return trial


def _refresh_composition_metadata(task: dict[str, Any]) -> None:
    scenario = task["scenario"]
    specs = _unit_specs_numpy()
    teams = np.asarray(scenario["teams"], dtype=int).reshape(-1)
    unit_ids = np.asarray(scenario["unit_ids"], dtype=int).reshape(-1)
    health = np.asarray(scenario["healths"], dtype=float).reshape(-1)
    rosters = [unit_ids[teams == team].tolist() for team in (0, 1)]
    fractions = [
        [float(health[index] / max(specs["healths"][unit_ids[index]], 1e-8)) for index in indexes]
        for indexes in (np.flatnonzero(teams == 0), np.flatnonzero(teams == 1))
    ]
    metadata = task.setdefault("metadata", {})
    metadata["composition_features"] = generators.composition_features(
        rosters[0], rosters[1], fractions[0], fractions[1]
    )
    match = metadata.get("composition_match")
    if isinstance(match, dict):
        ally_price = generators.team_price(rosters[0])
        enemy_price = generators.team_price(rosters[1])
        ally_effective = generators.team_effective_value(rosters[0], fractions[0])
        enemy_effective = generators.team_effective_value(rosters[1], fractions[1])
        match.update(
            {
                "ally_price": ally_price,
                "enemy_price": enemy_price,
                "price_rel_diff": abs(ally_price - enemy_price)
                / max(ally_price, enemy_price, 1.0),
                "ally_effective": ally_effective,
                "enemy_effective": enemy_effective,
                "effective_rel_diff": abs(ally_effective - enemy_effective)
                / max(ally_effective, enemy_effective, 1.0),
                "ally_health_fracs": fractions[0],
                "enemy_health_fracs": fractions[1],
            }
        )


def _roster_repair_trials(
    task: dict[str, Any],
    config: BalancedDiverseConfig,
    *,
    weak_team: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    scenario = task["scenario"]
    specs = _unit_specs_numpy()
    prices = specs["prices"]
    teams = np.asarray(scenario["teams"], dtype=int).reshape(-1)
    unit_ids = np.asarray(scenario["unit_ids"], dtype=int).reshape(-1)
    before_gap = float(base.estimate_dynamic_team_values(task)["relative_gap"])
    proposals: list[tuple[float, dict[str, Any], dict[str, Any]]] = []

    for team, direction in ((weak_team, 1), (1 - weak_team, -1)):
        for index in np.flatnonzero(teams == team):
            old = int(unit_ids[index])
            role_pool = generators._role_pool_for(old)
            alternatives = [
                other
                for other in role_pool
                if other != old
                and (
                    (direction > 0 and prices[other] > prices[old])
                    or (direction < 0 and prices[other] < prices[old])
                )
                and abs(float(prices[other] - prices[old])) / max(float(prices[old]), 1.0)
                <= 0.50
            ]
            for replacement in alternatives:
                trial = _replace_unit_template(task, int(index), int(replacement))
                try:
                    base.validate_task(trial)
                    values = base.estimate_dynamic_team_values(trial)
                except ValueError:
                    continue
                gap = float(values["relative_gap"])
                if gap + 1e-9 >= before_gap or gap > config.static_balance_resample_gap:
                    continue
                action = {
                    "kind": "roster",
                    "team": int(team),
                    "index": int(index),
                    "old_unit_id": old,
                    "new_unit_id": int(replacement),
                    "before_gap": before_gap,
                    "after_gap": gap,
                }
                proposals.append((gap, trial, action))
    proposals.sort(key=lambda item: (item[0], json.dumps(item[2], sort_keys=True)))
    return [
        (trial, action)
        for _, trial, action in proposals[: config.max_roster_repair_trials]
    ]


def _evaluate_a(
    task: dict[str, Any], config: BalancedDiverseConfig, *, seed: int
) -> tuple[bool, dict[str, Any]]:
    if config.hard_task_mode:
        accepted, result = base._passes_win_rate_filter(
            task,
            physics=config.physics,
            heuristic=config.heuristic,
            num_seeds=max(config.hard_discovery_seed_stages),
            seed=seed,
            epsilon=base._evaluation_epsilon(config),
            max_episode_steps=config.filter_max_episode_steps,
            win_rate_min=config.hard_win_rate_min,
            win_rate_max=config.hard_win_rate_max,
            reject_all_truncated=config.filter_reject_all_truncated,
            max_truncation_rate=config.hard_max_truncation_rate,
            max_no_interaction_rate=config.max_no_interaction_rate,
            max_n_ally=config.max_n_ally,
            max_n_enemy=config.max_n_enemy,
            max_n_zone=config.max_n_zone,
            seed_stages=config.hard_discovery_seed_stages,
            min_accept_seeds=config.hard_discovery_min_accept_seeds,
            candidate_win_rate_min=config.hard_discovery_candidate_min,
            candidate_win_rate_max=config.hard_discovery_candidate_max,
            max_abs_hp_margin=config.hard_discovery_max_abs_hp_margin,
        )
        # The base A-stage deliberately returns broad-band candidates at its
        # final stage.  For hard mode, convert that provisional result into a
        # miss so the existing single in-role roster repair can move it toward
        # the actual hard band before expensive B confirmation.
        win_rate = float(result["win_rate"])
        if accepted and not config.hard_win_rate_min <= win_rate <= config.hard_win_rate_max:
            result["reject_reason"] = "hard_target_band"
            return False, result
        return accepted, result
    return base._passes_win_rate_filter(
        task,
        physics=config.physics,
        heuristic=config.heuristic,
        num_seeds=base._discovery_num_seeds(config),
        seed=seed,
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


def _classify_a_failure(
    result: dict[str, Any], config: BalancedDiverseConfig
) -> str:
    """Classify A failures without interpreting hard-band misses as balance targets."""

    failure = base._classify_simulation_failure(result, config)
    if not config.hard_task_mode:
        return failure
    if failure in {
        "no_interaction",
        "low_interaction_truncation",
        "high_interaction_truncation",
        "fast_stomp",
    }:
        return failure
    win_rate = float(result["win_rate"])
    if win_rate < config.hard_win_rate_min or win_rate > config.hard_win_rate_max:
        return "hard_band_miss"
    return "hard_quality"


def _hard_confirmation_bands(
    config: BalancedDiverseConfig,
) -> tuple[tuple[float, float], tuple[float, float]]:
    original = (config.hard_win_rate_min, config.hard_win_rate_max)
    flipped = (1.0 - config.hard_win_rate_max, 1.0 - config.hard_win_rate_min)
    return original, flipped


def _run_hard_confirmation(
    task: dict[str, Any], config: BalancedDiverseConfig, *, seed: int
) -> tuple[bool, dict[str, Any]]:
    """Certify a frozen hard task on paired original/flipped seed prefixes."""

    from src.tabx.eval_task import aggregate_episode_samples, evaluate_tasks

    team_max = max(config.max_n_ally, config.max_n_enemy)
    stages = sorted(set(int(value) for value in config.hard_confirmation_seed_stages))
    original_band, flipped_band = _hard_confirmation_bands(config)
    states = {
        "original": {"task": task, "samples": None, "completed": 0},
        "flipped": {
            "task": base._flip_task_for_validation(task),
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
            epsilon_override=base._evaluation_epsilon(config),
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
        original_win_rate = float(original["win_rate"])
        flipped_win_rate = float(flipped["win_rate"])
        original_wins = int(round(original_win_rate * target))
        flipped_wins = int(round(flipped_win_rate * target))
        original_probability = base._balance_posterior_probability(
            original_wins, target, *original_band
        )
        flipped_probability = base._balance_posterior_probability(
            flipped_wins, target, *flipped_band
        )
        team_invariant_win_rate = (
            original_win_rate + 1.0 - flipped_win_rate
        ) * 0.5
        side_bias = original_win_rate + flipped_win_rate - 1.0
        original_hp = float(original["hp_margin"]["mean"])
        flipped_hp = float(flipped["hp_margin"]["mean"])
        hp_side_bias = original_hp + flipped_hp
        original_damage = np.asarray(original.get("damage_share", (0.0, 0.0)), dtype=float)
        flipped_damage = np.asarray(flipped.get("damage_share", (0.0, 0.0)), dtype=float)
        original_weak_damage = float(original_damage[0]) if original_damage.size > 0 else 0.0
        flipped_weak_damage = float(flipped_damage[1]) if flipped_damage.size > 1 else 0.0
        weak_damage_share = min(original_weak_damage, flipped_weak_damage)

        reasons = []
        if not original_band[0] <= original_win_rate <= original_band[1]:
            reasons.append("original_win_rate")
        if not flipped_band[0] <= flipped_win_rate <= flipped_band[1]:
            reasons.append("flipped_win_rate")
        if original_probability < config.hard_confirmation_band_probability:
            reasons.append("original_band_probability")
        if flipped_probability < config.hard_confirmation_band_probability:
            reasons.append("flipped_band_probability")
        if not original_band[0] <= team_invariant_win_rate <= original_band[1]:
            reasons.append("team_invariant_win_rate")
        if abs(side_bias) > config.hard_max_side_bias:
            reasons.append("side_bias")
        if abs(original_hp) > config.hard_max_abs_hp_margin:
            reasons.append("original_hp_margin")
        if abs(flipped_hp) > config.hard_max_abs_hp_margin:
            reasons.append("flipped_hp_margin")
        if abs(hp_side_bias) > config.hard_max_hp_side_bias:
            reasons.append("hp_side_bias")
        if weak_damage_share < config.hard_min_weak_damage_share:
            reasons.append("weak_damage_share")

        for label, result in (("original", original), ("flipped", flipped)):
            truncation_rate = float(result["truncation_rate"])
            truncations = int(round(truncation_rate * target))
            if (
                base._wilson_upper(truncations, target)
                > config.hard_max_truncation_rate
                and truncation_rate > 0.0
            ):
                reasons.append(f"{label}_truncation")
            no_interaction_rate = float(
                result["quality_flags"]["no_interaction_rate"]
            )
            if no_interaction_rate > config.max_no_interaction_rate:
                reasons.append(f"{label}_no_interaction")

        stage_record = {
            "n_seeds_per_orientation": target,
            "accepted": (
                not reasons and target >= config.hard_confirmation_min_accept_seeds
            ),
            "original_win_rate": original_win_rate,
            "flipped_win_rate": flipped_win_rate,
            "original_band_probability": original_probability,
            "flipped_band_probability": flipped_probability,
            "team_invariant_win_rate": team_invariant_win_rate,
            "side_bias": side_bias,
            "hp_side_bias": hp_side_bias,
            "weak_damage_share": weak_damage_share,
            "pending_reasons": reasons,
        }
        stage_history.append(stage_record)
        last = {
            "original": original,
            "flipped": flipped,
            "original_probability": original_probability,
            "flipped_probability": flipped_probability,
            "team_invariant_win_rate": team_invariant_win_rate,
            "side_bias": side_bias,
            "hp_side_bias": hp_side_bias,
            "weak_damage_share": weak_damage_share,
            "target": target,
        }
        if target >= config.hard_confirmation_min_accept_seeds and not reasons:
            accepted = True
            break
        clearly_outside = (
            target >= stages[0]
            and (
                original_probability < config.confirmation_early_reject_probability
                or flipped_probability < config.confirmation_early_reject_probability
            )
            and (
                not original_band[0] <= original_win_rate <= original_band[1]
                or not flipped_band[0] <= flipped_win_rate <= flipped_band[1]
            )
        )
        if clearly_outside:
            reasons.append("posterior_early_reject")
            break
        if any(reason.endswith("_no_interaction") for reason in reasons):
            reasons.append("no_interaction_early_reject")
            break

    payload = {
        "phase": "B",
        "profile": "hard",
        "accepted": accepted,
        "reject_reasons": [] if accepted else reasons,
        "seed": seed,
        "num_seeds": last["target"],
        "rollouts_consumed": last["target"] * 2,
        "epsilon": base._evaluation_epsilon(config),
        "original_band": list(original_band),
        "flipped_band": list(flipped_band),
        "original_band_probability": last["original_probability"],
        "flipped_band_probability": last["flipped_probability"],
        # Compatibility aliases used by the existing collection report.
        "original_balance_probability": last["original_probability"],
        "flipped_balance_probability": last["flipped_probability"],
        "team_invariant_win_rate": last["team_invariant_win_rate"],
        "side_bias": last["side_bias"],
        "hp_side_bias": last["hp_side_bias"],
        "weak_damage_share": last["weak_damage_share"],
        "stage_history": stage_history,
        "original": last["original"],
        "flipped": last["flipped"],
    }
    return accepted, payload


def _run_confirmation_for_mode(
    task: dict[str, Any], config: BalancedDiverseConfig, *, seed: int
) -> tuple[bool, dict[str, Any]]:
    if config.hard_task_mode:
        return _run_hard_confirmation(task, config, seed=seed)
    return base._run_confirmation(task, config, seed=seed)


def _best_roster_repair_by_a(
    task: dict[str, Any],
    config: BalancedDiverseConfig,
    *,
    weak_team: int,
    discovery_seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    for trial, action in _roster_repair_trials(task, config, weak_team=weak_team):
        accepted, result = _evaluate_a(trial, config, seed=discovery_seed)
        if accepted:
            return trial, result, action
    return None


def _certify_candidate(
    raw_task: dict[str, Any],
    source_kind: str,
    config: BalancedDiverseConfig,
    *,
    candidate_id: int,
) -> _CandidateOutcome:
    task = copy.deepcopy(raw_task)
    before_hash = base._task_hash(task)
    source = task.get("metadata", {}).get("source", {})
    if source_kind == "mutation" and source.get("parent"):
        family_id = f"mutation:{source['parent']}"
    elif source_kind == "crossover" and source.get("parents"):
        parent_ids = sorted(str(value) for value in source["parents"])
        family_id = "crossover:" + ":".join(parent_ids)
    else:
        family_id = before_hash
    discovery_seed = _derive_seed(config.seed, candidate_id, "A")
    repairs: list[dict[str, Any]] = []

    try:
        static_reason, static_values, engagement = _static_screen(task, config)
    except ValueError as exc:
        return _CandidateOutcome(None, f"static_invalid:{exc}", "static_invalid")

    task.setdefault("metadata", {})["static_diagnostics_before_repair"] = {
        "team_values": static_values,
        "engagement": engagement,
        "failure_type": static_reason,
    }

    geometry_used = False
    roster_used = False
    if static_reason in {"no_interaction", "low_interaction_truncation"}:
        if not config.allow_geometry_repair:
            return _CandidateOutcome(None, static_reason, static_reason)
        repaired = _geometry_repair(task, config)
        if repaired is None:
            return _CandidateOutcome(None, "geometry_repair_invalid", static_reason)
        task = repaired
        geometry_used = True
        repairs.append(
            {
                "kind": "geometry",
                "reason": static_reason,
                "distance_factor": config.geometry_repair_distance_factor,
                "map_factor": config.geometry_repair_map_factor,
            }
        )
        try:
            static_reason, static_values, engagement = _static_screen(task, config)
        except ValueError as exc:
            return _CandidateOutcome(None, f"geometry_repair_invalid:{exc}", static_reason)

    if static_reason in {"fast_stomp", "static_imbalance_hard"}:
        return _CandidateOutcome(None, static_reason, static_reason)

    if static_reason in {"no_interaction", "low_interaction_truncation"}:
        return _CandidateOutcome(
            None, f"post_geometry:{static_reason}", static_reason
        )

    if static_reason == "static_imbalance_soft":
        if not config.allow_roster_repair:
            return _CandidateOutcome(None, static_reason, static_reason)
        weak_team = 0 if float(static_values["ally"]) < float(static_values["enemy"]) else 1
        trials = _roster_repair_trials(task, config, weak_team=weak_team)
        if not trials:
            return _CandidateOutcome(None, "roster_resample_required", static_reason)
        task, action = trials[0]
        roster_used = True
        repairs.append(action)
        try:
            static_reason, static_values, engagement = _static_screen(task, config)
        except ValueError as exc:
            return _CandidateOutcome(None, f"roster_repair_invalid:{exc}", static_reason)
        if static_reason is not None:
            return _CandidateOutcome(None, f"post_roster:{static_reason}", static_reason)

    accepted_a, result_a = _evaluate_a(task, config, seed=discovery_seed)
    if not accepted_a:
        failure_type = _classify_a_failure(result_a, config)
        if (
            failure_type in {"no_interaction", "low_interaction_truncation"}
            and config.allow_geometry_repair
            and not geometry_used
        ):
            repaired = _geometry_repair(task, config)
            if repaired is not None:
                repaired_accepted, repaired_result = _evaluate_a(
                    repaired, config, seed=discovery_seed
                )
                if repaired_accepted:
                    task = repaired
                    result_a = repaired_result
                    accepted_a = True
                    geometry_used = True
                    repairs.append(
                        {
                            "kind": "geometry",
                            "reason": failure_type,
                            "distance_factor": config.geometry_repair_distance_factor,
                            "map_factor": config.geometry_repair_map_factor,
                        }
                    )
        elif (
            failure_type in {"combat_imbalance", "hard_band_miss"}
            and config.allow_roster_repair
            and not roster_used
        ):
            if config.hard_task_mode:
                # Move toward the asymmetric target band: strengthen team 0
                # when the task is too hard, otherwise strengthen team 1.
                weak_team = (
                    0
                    if float(result_a["win_rate"]) < config.hard_win_rate_min
                    else 1
                )
            else:
                weak_team = 1 if float(result_a["win_rate"]) > 0.5 else 0
            repaired = _best_roster_repair_by_a(
                task,
                config,
                weak_team=weak_team,
                discovery_seed=discovery_seed,
            )
            if repaired is not None:
                task, result_a, action = repaired
                accepted_a = True
                roster_used = True
                repairs.append(action)
        if not accepted_a:
            return _CandidateOutcome(None, f"A_rejected:{failure_type}", failure_type)

    # Freeze the physical task before B.  Only metadata may change afterward.
    try:
        static_reason, static_values, engagement = _static_screen(task, config)
    except ValueError as exc:
        return _CandidateOutcome(None, f"post_A_static_invalid:{exc}", "static_invalid")
    if static_reason is not None:
        return _CandidateOutcome(None, f"post_A_static:{static_reason}", static_reason)
    frozen_hash = base._task_hash(task)
    confirmation_seed = (
        base._phase_seed_base(task, config.seed) + config.confirmation_seed_offset
    )
    accepted_b, confirmation = _run_confirmation_for_mode(
        task, config, seed=confirmation_seed
    )
    if not accepted_b:
        reasons = ",".join(confirmation.get("reject_reasons", ()))
        return _CandidateOutcome(None, f"B_rejected:{reasons}", "confirmation")
    if base._task_hash(task) != frozen_hash:
        raise RuntimeError("B confirmation mutated the frozen task.")

    repair_types = tuple(action["kind"] for action in repairs)
    if geometry_used and roster_used:
        balance_origin = "geometry_roster_repaired"
    elif geometry_used:
        balance_origin = "geometry_repaired"
    elif roster_used:
        balance_origin = "roster_repaired"
    else:
        balance_origin = "natural"

    metadata = task.setdefault("metadata", {})
    task_profile = "hard" if config.hard_task_mode else "balanced"
    validation_reason = "B_confirmed_hard" if config.hard_task_mode else "B_confirmed"
    metadata.update(
        {
            "task_profile": task_profile,
            "balance_origin": balance_origin,
            "core_eligible": True,
            "frozen_task_hash": frozen_hash,
            "static_diagnostics_after_repair": {
                "team_values": static_values,
                "engagement": engagement,
                "failure_type": None,
            },
            "repair_summary": {
                "applied": bool(repairs),
                "repair_types": list(repair_types),
                "rounds": len(repairs),
                "cumulative_combat_scale": 1.0,
                "units_replaced": int(roster_used),
                "geometry_changed": geometry_used,
                "before_hash": before_hash,
                "after_hash": frozen_hash,
            },
            "structural_balance_repair": {"history": repairs},
            "validation_status": "confirmed",
            "validation_gate": {
                "selected": True,
                "reason": validation_reason,
                "policy": "structural_A_then_frozen_B",
                "profile": task_profile,
            },
            "fast_filter_eval": result_a,
            "confirmation": confirmation,
            "filter_eval": confirmation["original"],
            "evaluation_protocol": {
                "epsilon": base._evaluation_epsilon(config),
                "discovery_seed": discovery_seed,
                "discovery_seed_stages": list(
                    config.hard_discovery_seed_stages
                    if config.hard_task_mode
                    else config.discovery_seed_stages
                ),
                "confirmation_seed": confirmation_seed,
                "confirmation_seed_stages": list(
                    config.hard_confirmation_seed_stages
                    if config.hard_task_mode
                    else config.confirmation_seed_stages
                ),
                "task_frozen_before_B": True,
            },
        }
    )
    if config.hard_task_mode:
        metadata["hard_task_metrics"] = {
            "original_win_rate": float(confirmation["original"]["win_rate"]),
            "flipped_win_rate": float(confirmation["flipped"]["win_rate"]),
            "team_invariant_win_rate": float(
                confirmation["team_invariant_win_rate"]
            ),
            "side_bias": float(confirmation["side_bias"]),
            "hp_side_bias": float(confirmation["hp_side_bias"]),
            "weak_damage_share": float(confirmation["weak_damage_share"]),
        }

    parameter_signature = _canonical_parameter_signature(task)
    behavior_signature = _balanced_behavior_signature(
        confirmation, config.filter_max_episode_steps
    )
    descriptor = _diversity_descriptor(
        task, confirmation, config.filter_max_episode_steps
    )
    map_cell = _map_cell(descriptor)
    roster_key = _roster_key(task)
    digest = base._task_hash(task)
    candidate = _CertifiedCandidate(
        task=task,
        digest=digest,
        parameter_signature=parameter_signature,
        behavior_signature=behavior_signature,
        descriptor=descriptor,
        map_cell=map_cell,
        roster_key=roster_key,
        family_id=family_id,
        source_kind=source_kind,
        repair_types=repair_types,
    )
    return _CandidateOutcome(candidate, validation_reason)


def _canonical_parameter_signature(task: dict[str, Any]) -> np.ndarray:
    """Make parameter novelty invariant to team labels and map symmetries."""

    variants: list[np.ndarray] = []
    for transform in base.SUPPORTED_TRANSFORMS:
        transformed = copy.deepcopy(task)
        base._transform_task(transformed, transform)
        variants.append(base._parameter_signature(transformed))
        variants.append(base._parameter_signature(base._flip_task_for_validation(transformed)))
    return min(variants, key=lambda value: tuple(np.round(value, 10).tolist()))


def _parameter_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Group-weighted distance so 18 roster slots cannot swamp terrain/geometry."""

    groups = (
        (np.r_[0:9, 16:25], 0.35),
        (np.r_[9:13, 25:29, 32], 0.30),
        (np.r_[13:16, 29:32], 0.10),
        (np.r_[33:38], 0.25),
    )
    total = 0.0
    for indexes, weight in groups:
        delta = first[indexes] - second[indexes]
        total += weight * float(np.sqrt(np.mean(np.square(delta))))
    return total


def _balanced_behavior_signature(
    confirmation: dict[str, Any], max_episode_steps: int
) -> np.ndarray:
    """Orientation-invariant dynamics descriptor with balance metrics removed."""

    original = confirmation["original"]
    flipped = confirmation["flipped"]

    def mean_scalar(path: tuple[str, ...], default: float = 0.0) -> float:
        values = []
        for result in (original, flipped):
            current: Any = result
            for key in path:
                if not isinstance(current, dict) or key not in current:
                    current = default
                    break
                current = current[key]
            values.append(float(current if current is not None else default))
        return float(np.mean(values))

    def mean_team_metric(name: str) -> float:
        values = []
        for result in (original, flipped):
            metric = np.asarray(result.get(name, (0.0, 0.0)), dtype=float).reshape(-1)
            values.append(float(np.mean(metric)) if len(metric) else 0.0)
        return float(np.mean(values))

    duration = mean_scalar(("episode_length", "mean")) / max(max_episode_steps, 1)
    first_contact = mean_scalar(
        ("first_interaction_step", "mean"), float(max_episode_steps)
    ) / max(max_episode_steps, 1)
    interaction_rate = mean_scalar(("first_interaction_step", "rate"))
    damage = math.log1p(max(mean_team_metric("damage_mean"), 0.0)) / math.log1p(10_000.0)
    healing = math.log1p(max(mean_team_metric("healing_mean"), 0.0)) / math.log1p(10_000.0)
    success = mean_team_metric("attack_success_rate")
    return np.asarray(
        [duration, first_contact, interaction_rate, damage, healing, success], dtype=float
    )


def _behavior_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(first - second))))


def _actual_distance_bucket(task: dict[str, Any]) -> str:
    scenario = task["scenario"]
    teams = np.asarray(scenario["teams"], dtype=int).reshape(-1)
    positions = np.asarray(scenario["positions"], dtype=float)
    centers = [positions[teams == team].mean(axis=0) for team in (0, 1)]
    width = float(task.get("grid_info", {}).get("max_field_width", 121.0))
    ratio = float(np.linalg.norm(centers[0] - centers[1])) / max(width, 1.0)
    if ratio < 0.20:
        return "close"
    if ratio < 0.45:
        return "medium"
    return "far"


def _profile(value: float, low: float, high: float, labels: tuple[str, str, str]) -> str:
    if value < low:
        return labels[0]
    if value < high:
        return labels[1]
    return labels[2]


def _mean_confirmation_value(
    confirmation: dict[str, Any], path: tuple[str, ...], default: float
) -> float:
    values = []
    for label in ("original", "flipped"):
        current: Any = confirmation[label]
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = default
                break
            current = current[key]
        values.append(float(current if current is not None else default))
    return float(np.mean(values))


def _diversity_descriptor(
    task: dict[str, Any], confirmation: dict[str, Any], max_episode_steps: int
) -> dict[str, str]:
    metadata = task.get("metadata", {})
    match = metadata.get("composition_match") or {}
    teams = np.asarray(task["scenario"]["teams"], dtype=int).reshape(-1)
    unit_ids = np.asarray(task["scenario"]["unit_ids"], dtype=int).reshape(-1)
    counts = [int(np.count_nonzero(teams == team)) for team in (0, 1)]
    ranged = float(np.mean(np.isin(unit_ids, tuple(RANGED_UNIT_IDS))))
    healers = float(np.mean(np.isin(unit_ids, tuple(HEALER_UNIT_IDS))))
    first_contact = _mean_confirmation_value(
        confirmation,
        ("first_interaction_step", "mean"),
        float(max_episode_steps),
    ) / max(max_episode_steps, 1)
    duration = _mean_confirmation_value(
        confirmation, ("episode_length", "mean"), float(max_episode_steps)
    ) / max(max_episode_steps, 1)
    behavior_signature = _balanced_behavior_signature(confirmation, max_episode_steps)
    healing_signal = float(behavior_signature[4])
    return {
        "composition": str(metadata.get("composition_archetype", "free")),
        "layout": str(metadata.get("layout_archetype", "free")),
        "distance": _actual_distance_bucket(task),
        "spread": str(metadata.get("spread_bucket", "free")),
        "zone": str(metadata.get("zone_archetype", "free")),
        "zone_intensity": str(metadata.get("zone_intensity", "free")),
        "zone_relation": str(metadata.get("zone_relation", "free")),
        "match_mode": str(match.get("match_mode", "free")),
        "team_size": _profile(float(max(counts)), 4.0, 7.0, ("small", "medium", "large")),
        "ranged_profile": _profile(ranged, 0.25, 0.60, ("low", "mixed", "high")),
        "healer_profile": _profile(healers, 0.01, 0.30, ("none", "some", "heavy")),
        "first_contact": _profile(
            first_contact, 0.12, 0.32, ("early", "middle", "late")
        ),
        "duration": _profile(duration, 0.25, 0.60, ("short", "medium", "long")),
        "healing": _profile(healing_signal, 0.001, 0.08, ("none", "low", "high")),
    }


def _map_cell(descriptor: dict[str, str]) -> tuple[str, ...]:
    return tuple(
        descriptor[name]
        for name in (
            "team_size",
            "ranged_profile",
            "healer_profile",
            "distance",
            "zone",
            "first_contact",
            "duration",
        )
    )


def _roster_key(task: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    teams = np.asarray(task["scenario"]["teams"], dtype=int).reshape(-1)
    unit_ids = np.asarray(task["scenario"]["unit_ids"], dtype=int).reshape(-1)
    rosters = [tuple(sorted(unit_ids[teams == team].tolist())) for team in (0, 1)]
    return tuple(sorted(rosters))  # type: ignore[return-value]


def _nearest_parameter(
    signature: np.ndarray, candidates: Sequence[_CertifiedCandidate]
) -> float:
    if not candidates:
        return math.inf
    return min(_parameter_distance(signature, other.parameter_signature) for other in candidates)


def _nearest_behavior(
    signature: np.ndarray, candidates: Sequence[_CertifiedCandidate]
) -> float:
    if not candidates:
        return math.inf
    return min(_behavior_distance(signature, other.behavior_signature) for other in candidates)


def _is_pool_duplicate(
    candidate: _CertifiedCandidate,
    pool: Sequence[_CertifiedCandidate],
    config: BalancedDiverseConfig,
) -> str | None:
    if any(candidate.digest == other.digest for other in pool):
        return "exact_duplicate"
    parameter_distance = _nearest_parameter(candidate.parameter_signature, pool)
    if parameter_distance < config.parameter_distance_threshold:
        return "parameter_duplicate"
    behavior_distance = _nearest_behavior(candidate.behavior_signature, pool)
    if behavior_distance < config.behavior_distance_threshold:
        return "behavior_duplicate"
    return None


def _repair_quota_allows(
    candidate: _CertifiedCandidate,
    repair_counts: Counter[str],
    config: BalancedDiverseConfig,
) -> bool:
    geometry_cap = int(math.floor(config.n_tasks * config.max_geometry_repaired_fraction))
    roster_cap = int(math.floor(config.n_tasks * config.max_roster_repaired_fraction))
    if "geometry" in candidate.repair_types and repair_counts["geometry"] >= geometry_cap:
        return False
    if "roster" in candidate.repair_types and repair_counts["roster"] >= roster_cap:
        return False
    return True


def _online_append_allows(
    candidate: _CertifiedCandidate,
    pool: Sequence[_CertifiedCandidate],
    config: BalancedDiverseConfig,
) -> bool:
    """Apply final hard caps before append, without replacing retained tasks."""

    cell_cap = max(1, int(math.ceil(config.n_tasks * config.max_map_cell_fraction)))
    roster_cap = max(
        1, int(math.ceil(config.n_tasks * config.max_exact_roster_fraction))
    )
    if sum(item.map_cell == candidate.map_cell for item in pool) >= cell_cap:
        return False
    if sum(item.roster_key == candidate.roster_key for item in pool) >= roster_cap:
        return False
    if sum(item.family_id == candidate.family_id for item in pool) >= config.max_family_tasks:
        return False
    repair_counts = Counter(
        repair_type for item in pool for repair_type in item.repair_types
    )
    return _repair_quota_allows(candidate, repair_counts, config)


def _select_diverse_candidates(
    pool: Sequence[_CertifiedCandidate], config: BalancedDiverseConfig
) -> list[_CertifiedCandidate]:
    remaining = list(pool)
    selected: list[_CertifiedCandidate] = []
    coverage = _CoverageState(config)
    cell_counts: Counter[tuple[str, ...]] = Counter()
    roster_counts: Counter[tuple[tuple[int, ...], tuple[int, ...]]] = Counter()
    family_counts: Counter[str] = Counter()
    repair_counts: Counter[str] = Counter()
    cell_cap = max(1, int(math.ceil(config.n_tasks * config.max_map_cell_fraction)))
    roster_cap = max(1, int(math.ceil(config.n_tasks * config.max_exact_roster_fraction)))

    while remaining and len(selected) < config.n_tasks:
        best: _CertifiedCandidate | None = None
        best_index: int | None = None
        best_score = -math.inf
        for candidate_index, candidate in enumerate(remaining):
            if cell_counts[candidate.map_cell] >= cell_cap:
                continue
            if roster_counts[candidate.roster_key] >= roster_cap:
                continue
            if family_counts[candidate.family_id] >= config.max_family_tasks:
                continue
            if not _repair_quota_allows(candidate, repair_counts, config):
                continue
            parameter_distance = _nearest_parameter(candidate.parameter_signature, selected)
            behavior_distance = _nearest_behavior(candidate.behavior_signature, selected)
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
            tie_break = int(candidate.digest[:12], 16) / float(16**12)
            score += tie_break * 1e-12
            if score > best_score:
                best = candidate
                best_index = candidate_index
                best_score = score
        if best is None or best_index is None:
            break
        selected.append(best)
        remaining.pop(best_index)
        coverage.add(best.descriptor)
        cell_counts[best.map_cell] += 1
        roster_counts[best.roster_key] += 1
        family_counts[best.family_id] += 1
        for repair_type in best.repair_types:
            repair_counts[repair_type] += 1
        best.task.setdefault("metadata", {})["diversity_selection"] = {
            "score_at_selection": best_score,
            "descriptor": best.descriptor,
            "map_cell": list(best.map_cell),
            "family_id": best.family_id,
            "source_kind": best.source_kind,
            "scope": "B_confirmed_candidate_pool",
        }
    return selected


def _collection_budget(config: BalancedDiverseConfig) -> int:
    if config.diversity_candidate_budget > 0:
        return config.diversity_candidate_budget
    return config.n_tasks * config.diversity_candidate_budget_multiplier


def _coverage_from_pool(
    candidates: Iterable[_CertifiedCandidate], config: BalancedDiverseConfig
) -> _CoverageState:
    coverage = _CoverageState(config)
    for candidate in candidates:
        coverage.add(candidate.descriptor)
    return coverage


def sample_tasks_with_report(
    config: BalancedDiverseConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect B-confirmed candidates and commit a quota-constrained diverse subset."""

    _validate_config(config)
    budget = _collection_budget(config)
    generation_seed = (
        secrets.randbits(32)
        if config.generation_seed is None
        else int(config.generation_seed)
    )
    generation_seed_policy = (
        "random_per_run" if config.generation_seed is None else "explicit"
    )
    pool_target = max(
        config.n_tasks,
        int(math.ceil(config.n_tasks * config.certified_pool_multiplier)),
    )
    direct_append_pool = pool_target == config.n_tasks
    pool: list[_CertifiedCandidate] = []
    feedback = _BucketFeedback()
    reject_counts: Counter[str] = Counter()
    source_generated: Counter[str] = Counter()
    source_confirmed: Counter[str] = Counter()
    parent_leases: Counter[str] = Counter()
    target_leases: Counter[tuple[str, ...]] = Counter()
    started = time.monotonic()
    worker_count = _parallel_worker_count(config, budget)
    max_in_flight = max(
        worker_count,
        config.commit_batch_size or 2 * worker_count,
    )
    next_candidate_id = 0
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
    previous_environment = {
        name: os.environ.get(name) for name in managed_environment
    }
    os.environ.update(managed_environment)
    task_profile = "hard" if config.hard_task_mode else "balanced"
    print(
        f"{task_profile.capitalize()}-diverse CPU sampling: "
        f"workers={worker_count}, max_in_flight={max_in_flight}, "
        f"target={config.n_tasks}, candidate_budget={budget}, "
        f"pool_policy={'direct_append' if direct_append_pool else 'oversample'}, "
        f"generation_seed={generation_seed}, evaluation_seed={config.seed}",
        flush=True,
    )

    def build_job() -> _ParallelJob:
        nonlocal next_candidate_id

        candidate_id = next_candidate_id
        scheduler_rng = random.Random(
            _derive_seed(generation_seed, candidate_id, "target")
        )
        coverage = _leased_coverage(pool, target_leases, config)
        target = _choose_target(scheduler_rng, config, coverage, feedback)
        source_kind = _draw_source_kind(scheduler_rng, config, pool)
        parents = _assign_parent_payloads(
            source_kind,
            pool,
            parent_leases,
            scheduler_rng,
        )
        job = _ParallelJob(
            candidate_id=candidate_id,
            generation_seed=generation_seed,
            target=target,
            source_kind=source_kind,
            parents=parents,
            config=config,
        )
        source_generated[source_kind] += 1
        if source_kind == "programmatic":
            target_leases[target] += 1
            feedback.record_generated(target)
        next_candidate_id += 1
        return job

    def commit_result(result: _ParallelResult) -> None:
        for parent_id in result.parent_ids:
            parent_leases[parent_id] -= 1
        if result.source_kind == "programmatic":
            target_leases[result.target] -= 1
        if direct_append_pool and len(pool) >= pool_target:
            reject_counts["completed_after_target"] += 1
            return
        outcome = result.outcome
        if outcome.candidate is None:
            reject_counts[outcome.reason] += 1
            return
        duplicate_reason = _is_pool_duplicate(outcome.candidate, pool, config)
        if duplicate_reason is not None:
            reject_counts[duplicate_reason] += 1
            return
        candidate = outcome.candidate
        candidate.task.setdefault("metadata", {})["parallel_sampling"] = {
            "candidate_id": result.candidate_id,
            "worker_id": result.worker_id,
            "worker_pid": result.worker_pid,
            "parent_ids": list(result.parent_ids),
            "commit_policy": "asynchronous_completion_order",
        }
        if direct_append_pool and not _online_append_allows(candidate, pool, config):
            reject_counts["online_diversity_rejected"] += 1
            return
        pool.append(candidate)
        if result.source_kind == "programmatic":
            feedback.record_confirmed(result.target)
        source_confirmed[result.source_kind] += 1
        print(
            f"candidate={result.candidate_id + 1}/{budget} B_CONFIRMED "
            f"pool={len(pool)}/{pool_target} source={result.source_kind} "
            f"worker={result.worker_id} "
            f"origin={candidate.task['metadata']['balance_origin']} "
            "pool_action=added",
            flush=True,
        )

    def collection_complete() -> bool:
        if len(pool) < pool_target:
            return False
        if direct_append_pool:
            return True
        selected = _select_diverse_candidates(pool, config)
        audit = _quota_audit(selected[: config.n_tasks], config)
        return bool(
            len(selected) >= config.n_tasks
            and audit["marginal_targets_satisfied"]
            and audit["pair_targets_satisfied"]
        )

    def time_budget_exhausted() -> bool:
        return bool(
            config.collection_time_budget_seconds > 0
            and time.monotonic() - started >= config.collection_time_budget_seconds
        )

    executor: ProcessPoolExecutor | None = None
    try:
        if worker_count == 1:
            while next_candidate_id < budget and not collection_complete():
                if time_budget_exhausted():
                    reject_counts["time_budget_exhausted"] += 1
                    break
                job = build_job()
                commit_result(_sample_candidate_cpu_worker(job))
        else:
            context = _multiprocessing_context()
            worker_counter = context.Value("i", 0)
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=context,
                initializer=_initialize_cpu_worker,
                initargs=(worker_counter,),
            )
            pending: dict[Any, _ParallelJob] = {}

            def refill_workers() -> None:
                while (
                    len(pending) < max_in_flight
                    and next_candidate_id < budget
                    and not collection_complete()
                    and not time_budget_exhausted()
                ):
                    job = build_job()
                    future = executor.submit(_sample_candidate_cpu_worker, job)
                    pending[future] = job

            refill_workers()
            while pending and not collection_complete():
                if time_budget_exhausted():
                    reject_counts["time_budget_exhausted"] += 1
                    break
                completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in sorted(
                    completed, key=lambda item: pending[item].candidate_id
                ):
                    job = pending.pop(future)
                    result = future.result()
                    if result.candidate_id != job.candidate_id:
                        raise RuntimeError("Worker returned a mismatched candidate_id.")
                    commit_result(result)
                refill_workers()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        for name, previous in previous_environment.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous

    selected = (
        list(pool[: config.n_tasks])
        if direct_append_pool
        else _select_diverse_candidates(pool, config)
    )
    if direct_append_pool:
        for candidate in selected:
            candidate.task.setdefault("metadata", {})["diversity_selection"] = {
                "score_at_selection": None,
                "descriptor": candidate.descriptor,
                "map_cell": list(candidate.map_cell),
                "family_id": candidate.family_id,
                "source_kind": candidate.source_kind,
                "scope": "deficit_directed_direct_append",
            }
    audit = _quota_audit(selected[: config.n_tasks], config)
    quotas_satisfied = bool(
        audit["marginal_targets_satisfied"] and audit["pair_targets_satisfied"]
    )
    if not selected:
        raise RuntimeError(
            "Could not collect any certified task to save: "
            f"selected=0/{config.n_tasks}, certified_pool={len(pool)}, "
            f"candidate_budget={budget}, rejections={dict(reject_counts)}. "
            f"marginal_deficits={audit['marginal_deficits'][:10]}, "
            f"pair_deficits={audit['pair_deficits'][:10]}."
        )
    if len(selected) < config.n_tasks:
        print(
            "WARNING: collection ended before reaching the requested task count; "
            f"saving a partial bank with {len(selected)}/{config.n_tasks} tasks. "
            f"certified_pool={len(pool)}, candidate_budget={budget}, "
            f"rejections={dict(reject_counts)}.",
            flush=True,
        )
    elif not direct_append_pool and not quotas_satisfied:
        raise RuntimeError(
            "Could not satisfy the hard balance and diversity quotas: "
            f"selected={len(selected)}/{config.n_tasks}, certified_pool={len(pool)}, "
            f"candidate_budget={budget}, rejections={dict(reject_counts)}. "
            f"marginal_deficits={audit['marginal_deficits'][:10]}, "
            f"pair_deficits={audit['pair_deficits'][:10]}. "
            "Increase diversity_candidate_budget; do not relax balance thresholds."
        )

    tasks = [candidate.task for candidate in selected[: config.n_tasks]]
    for index, (task, candidate) in enumerate(zip(tasks, selected)):
        task["task_id"] = f"task_{index:06d}_{candidate.digest[:10]}"
        task["metadata"]["canonical_hash"] = candidate.digest

    report = _build_report(
        selected[: config.n_tasks],
        pool,
        reject_counts,
        source_generated,
        source_confirmed,
        feedback,
        config,
        budget,
        time.monotonic() - started,
    )
    report["generation_seed"] = generation_seed
    report["generation_seed_policy"] = generation_seed_policy
    report["evaluation_seed"] = config.seed
    saved_tasks = len(tasks)
    collection_complete = saved_tasks == config.n_tasks
    if collection_complete:
        termination_reason = "target_reached"
    elif reject_counts.get("time_budget_exhausted", 0) > 0:
        termination_reason = "time_budget_exhausted"
    elif next_candidate_id >= budget:
        termination_reason = "candidate_budget_exhausted"
    else:
        termination_reason = "collection_stopped"
    report.update(
        {
            "complete": collection_complete,
            "partial_result": not collection_complete,
            "requested_tasks": config.n_tasks,
            "saved_tasks": saved_tasks,
            "task_shortfall": max(0, config.n_tasks - saved_tasks),
            "termination_reason": termination_reason,
        }
    )
    report["parallel_sampling"] = {
        "execution_backend": "cpu",
        "workers": worker_count,
        "threads_per_worker": 1,
        "max_in_flight": max_in_flight,
        "scheduling": "asynchronous_refill",
        "pool_policy": "direct_append" if direct_append_pool else "oversample",
        "certified_pool_capacity": pool_target,
        "coverage_quota_enforcement": (
            "report_only" if direct_append_pool else "hard_gate"
        ),
        "multiprocessing_start_method": (
            "local" if worker_count == 1 else _multiprocessing_context().get_start_method()
        ),
        "scheduler": "central_target_and_parent_leases",
        "commit_policy": "asynchronous_completion_order",
    }
    return tasks, report


def sample_tasks(config: BalancedDiverseConfig) -> list[dict[str, Any]]:
    """Compatibility wrapper returning only the selected task list."""

    tasks, _ = sample_tasks_with_report(config)
    return tasks


def _distribution(
    candidates: Sequence[_CertifiedCandidate], dimension: str
) -> dict[str, int]:
    counts = Counter(candidate.descriptor[dimension] for candidate in candidates)
    return dict(sorted(counts.items()))


def _nearest_distribution(
    candidates: Sequence[_CertifiedCandidate], *, behavior: bool
) -> list[float]:
    values = []
    for index, candidate in enumerate(candidates):
        others = list(candidates[:index]) + list(candidates[index + 1 :])
        distance = (
            _nearest_behavior(candidate.behavior_signature, others)
            if behavior
            else _nearest_parameter(candidate.parameter_signature, others)
        )
        if math.isfinite(distance):
            values.append(float(distance))
    return values


def _percentiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"p10": None, "p50": None, "p90": None}
    array = np.asarray(values, dtype=float)
    return {
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
    }


def _build_report(
    selected: Sequence[_CertifiedCandidate],
    pool: Sequence[_CertifiedCandidate],
    reject_counts: Counter[str],
    source_generated: Counter[str],
    source_confirmed: Counter[str],
    feedback: _BucketFeedback,
    config: BalancedDiverseConfig,
    candidate_budget: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    origins = Counter(
        str(candidate.task["metadata"]["balance_origin"]) for candidate in selected
    )
    cells = Counter(candidate.map_cell for candidate in selected)
    rosters = Counter(candidate.roster_key for candidate in selected)
    confirmations = [candidate.task["metadata"]["confirmation"] for candidate in selected]
    report = {
        "strategy": (
            "constrained_quality_diversity_hard"
            if config.hard_task_mode
            else "constrained_quality_diversity"
        ),
        "task_profile": "hard" if config.hard_task_mode else "balanced",
        "balance_hard_gate": not config.hard_task_mode,
        "structural_balance_hard_gate": True,
        "selected_tasks": len(selected),
        "certified_pool": len(pool),
        "candidate_budget": candidate_budget,
        "elapsed_seconds": elapsed_seconds,
        "reject_counts": dict(reject_counts),
        "source_generated": dict(source_generated),
        "source_confirmed": dict(source_confirmed),
        "target_feedback": _target_feedback_report(feedback),
        "balance_origins": dict(origins),
        "combat_stat_repaired": 0,
        "all_B_confirmed": all(item.get("accepted") is True for item in confirmations),
        "min_original_balance_probability": min(
            float(item["original_balance_probability"]) for item in confirmations
        ),
        "min_flipped_balance_probability": min(
            float(item["flipped_balance_probability"]) for item in confirmations
        ),
        "max_abs_side_bias": max(abs(float(item["side_bias"])) for item in confirmations),
        "max_map_cell_count": max(cells.values(), default=0),
        "occupied_map_cells": len(cells),
        "max_exact_roster_count": max(rosters.values(), default=0),
        "unique_exact_rosters": len(rosters),
        "marginal_distributions": {
            dimension: _distribution(selected, dimension)
            for dimension in ALL_DIVERSITY_DIMENSIONS
        },
        "quota_audit": _quota_audit(selected, config),
        "parameter_nearest_distance": _percentiles(
            _nearest_distribution(selected, behavior=False)
        ),
        "behavior_nearest_distance": _percentiles(
            _nearest_distribution(selected, behavior=True)
        ),
    }
    if config.hard_task_mode:
        original_band, flipped_band = _hard_confirmation_bands(config)
        report.update(
            {
                "hard_original_win_rate_range": list(original_band),
                "hard_flipped_win_rate_range": list(flipped_band),
                "min_original_band_probability": min(
                    float(item["original_band_probability"])
                    for item in confirmations
                ),
                "min_flipped_band_probability": min(
                    float(item["flipped_band_probability"])
                    for item in confirmations
                ),
                "max_abs_hp_side_bias": max(
                    abs(float(item["hp_side_bias"])) for item in confirmations
                ),
                "min_weak_damage_share": min(
                    float(item["weak_damage_share"]) for item in confirmations
                ),
            }
        )
    return report


def _target_feedback_report(feedback: _BucketFeedback) -> dict[str, Any]:
    rows: dict[str, dict[str, dict[str, float | int]]] = {}
    keys = sorted(set(feedback.generated) | set(feedback.confirmed))
    for dimension, value in keys:
        generated = int(feedback.generated[(dimension, value)])
        confirmed = int(feedback.confirmed[(dimension, value)])
        rows.setdefault(dimension, {})[value] = {
            "generated": generated,
            "confirmed": confirmed,
            "confirmed_rate": confirmed / generated if generated else 0.0,
        }
    return rows


def _quota_audit(
    selected: Sequence[_CertifiedCandidate], config: BalancedDiverseConfig
) -> dict[str, Any]:
    coverage = _coverage_from_pool(selected, config)
    marginal_deficits: list[dict[str, Any]] = []
    for dimension in ALL_DIVERSITY_DIMENSIONS:
        for value in _dimension_values(config, dimension):
            target = coverage.marginal_target(dimension, value)
            actual = int(coverage.marginals[(dimension, value)])
            if target > actual:
                marginal_deficits.append(
                    {
                        "dimension": dimension,
                        "value": value,
                        "target": target,
                        "actual": actual,
                        "deficit": target - actual,
                    }
                )

    pair_deficits: list[dict[str, Any]] = []
    for first, second in PAIR_DIMENSIONS:
        for first_value in _dimension_values(config, first):
            for second_value in _dimension_values(config, second):
                target = coverage.pair_target(first, first_value, second, second_value)
                key = (first, first_value, second, second_value)
                actual = int(coverage.pairs[key])
                if target > actual:
                    pair_deficits.append(
                        {
                            "dimensions": [first, second],
                            "values": [first_value, second_value],
                            "target": target,
                            "actual": actual,
                            "deficit": target - actual,
                        }
                    )
    return {
        "marginal_targets_satisfied": not marginal_deficits,
        "pair_targets_satisfied": not pair_deficits,
        "marginal_deficits": marginal_deficits,
        "pair_deficits": pair_deficits,
    }


def _filter_protocol(config: BalancedDiverseConfig) -> dict[str, Any]:
    if config.hard_task_mode:
        original_band, flipped_band = _hard_confirmation_bands(config)
        discovery_stages = config.hard_discovery_seed_stages
        discovery_min_accept = config.hard_discovery_min_accept_seeds
        discovery_candidate_band = [
            config.hard_discovery_candidate_min,
            config.hard_discovery_candidate_max,
        ]
        discovery_hp_margin = config.hard_discovery_max_abs_hp_margin
        confirmation = {
            "seed_stages": list(config.hard_confirmation_seed_stages),
            "minimum_accept_seeds": config.hard_confirmation_min_accept_seeds,
            "original_win_rate": list(original_band),
            "flipped_win_rate": list(flipped_band),
            "team_invariant_win_rate": list(original_band),
            "band_probability": config.hard_confirmation_band_probability,
            "max_side_bias": config.hard_max_side_bias,
            "max_truncation_rate": config.hard_max_truncation_rate,
            "max_no_interaction_rate": config.max_no_interaction_rate,
            "max_abs_hp_margin": config.hard_max_abs_hp_margin,
            "max_abs_hp_side_bias": config.hard_max_hp_side_bias,
            "min_weak_damage_share": config.hard_min_weak_damage_share,
        }
    else:
        discovery_stages = config.discovery_seed_stages
        discovery_min_accept = config.discovery_min_accept_seeds
        discovery_candidate_band = [
            config.discovery_candidate_win_rate_min,
            config.discovery_candidate_win_rate_max,
        ]
        discovery_hp_margin = config.discovery_max_abs_hp_margin
        confirmation = {
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
        }
    return {
        "enabled": True,
        "policy": "structural_A_then_frozen_B",
        "task_profile": "hard" if config.hard_task_mode else "balanced",
        "evaluation_epsilon": base._evaluation_epsilon(config),
        "physics": config.physics,
        "heuristic": config.heuristic,
        "max_episode_steps": config.filter_max_episode_steps,
        "max_truncation_rate": config.max_truncation_rate,
        "max_no_interaction_rate": config.max_no_interaction_rate,
        "A": {
            "seed_stages": list(discovery_stages),
            "minimum_accept_seeds": discovery_min_accept,
            "candidate_win_rate": discovery_candidate_band,
            "target_win_rate": (
                [config.hard_win_rate_min, config.hard_win_rate_max]
                if config.hard_task_mode
                else [config.win_rate_min, config.win_rate_max]
            ),
            "max_abs_hp_margin": discovery_hp_margin,
        },
        "B": confirmation,
    }


def save_task_bank(
    config: BalancedDiverseConfig,
    tasks: Sequence[dict[str, Any]],
    report: dict[str, Any],
) -> Path:
    """Save through the stable bank writer and identify this independent generator."""

    task_profile = "hard" if config.hard_task_mode else "balanced"
    generator_protocol = {
        "strategy": (
            "constrained_quality_diversity_hard"
            if config.hard_task_mode
            else "constrained_quality_diversity"
        ),
        "task_profile": task_profile,
        "generation_seed": report["generation_seed"],
        "generation_seed_policy": report["generation_seed_policy"],
        "evaluation_seed": config.seed,
        "balance_is_hard_constraint": not config.hard_task_mode,
        "structural_balance_is_hard_constraint": True,
        "legacy_combat_stat_repair": False,
        "allowed_repairs": {
            "geometry": config.allow_geometry_repair,
            "roster_single_in_role_replacement": config.allow_roster_repair,
            "initial_health": False,
            "combat_stats": False,
        },
        "repair_caps": {
            "minimum_natural_fraction": config.min_natural_fraction,
            "maximum_geometry_repaired_fraction": config.max_geometry_repaired_fraction,
            "maximum_roster_repaired_fraction": config.max_roster_repaired_fraction,
        },
        "diversity": {
            "dimensions": list(ALL_DIVERSITY_DIMENSIONS),
            "pair_dimensions": [list(pair) for pair in PAIR_DIMENSIONS],
            "symmetry_invariant_parameter_signature": True,
            "behavior_signature_excludes_balance_metrics": True,
            "max_map_cell_fraction": config.max_map_cell_fraction,
            "max_exact_roster_fraction": config.max_exact_roster_fraction,
            "max_family_tasks": config.max_family_tasks,
        },
        "collection_report": report,
    }
    output_name = config.output
    if config.hard_task_mode and output_name == "balanced_diverse_tasks.json":
        output_name = "hard_diverse_tasks.json"
    output = base.save_task_bank(
        output_name,
        tasks,
        seed=config.seed,
        physics=config.physics,
        heuristic=config.heuristic,
        max_n_ally=config.max_n_ally,
        max_n_enemy=config.max_n_enemy,
        max_n_zone=config.max_n_zone,
        filter_protocol=_filter_protocol(config),
        generator_protocol=generator_protocol,
    )
    bank = json.loads(output.read_text(encoding="utf-8"))
    bank["manifest"]["generator"] = "src.tabx.sample_task_balanced_diverse"
    output.write_text(
        json.dumps(bank, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    config = tyro.cli(BalancedDiverseConfig)
    tasks, report = sample_tasks_with_report(config)
    output = save_task_bank(config, tasks, report)
    task_profile = "hard" if config.hard_task_mode else "balanced"
    completion = (
        "complete"
        if report["complete"]
        else f"partial={len(tasks)}/{report['requested_tasks']}"
    )
    print(
        f"Saved {len(tasks)} {task_profile} diverse tasks to {output}; "
        f"collection={completion}, "
        f"occupied_cells={report['occupied_map_cells']}, "
        f"origins={report['balance_origins']}."
    )


if __name__ == "__main__":
    main()
