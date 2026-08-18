"""Anchor-aware, CPU-parallel quality-diversity task sampling for TABX.

``sampleV1`` extends the balance-certified sampler with three changes:

* candidates are compared with a fixed reference task bank before any rollout;
* global constructive restarts and radical mutations can leave the local
  neighbourhood of the existing task archive;
* cheap validity, static-quality, and parameter-distance checks run in the
  coordinator before expensive A/B rollout certification is submitted to CPU
  workers.

The physical task is still frozen before confirmation and the existing
orientation-paired A/B evaluation is reused.  Run with::

    python -m src.tabx.sampleV1 \
        --reference-task-file task_files/balanced_tasks_v1.json \
        --output outputs/sampleV1_tasks.json \
        --n-tasks 100 \
        --cpu-cores 8
"""

from __future__ import annotations

import copy
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

# Protect the command-line entry point from initializing a CUDA backend in the
# coordinator or in spawned Windows workers.
if __name__ == "__main__":
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
from src.tabx import sample_task_balanced_diverse as qd
from src.tabx import task_generators as generators


PROPOSAL_KINDS = (
    "global_constructive",
    "programmatic",
    "radical_mutation",
    "distant_crossover",
    "local_mutation",
)

_DISTANCE_GROUPS = (
    (np.r_[0:9, 16:25], 0.35),
    (np.r_[9:13, 25:29, 32], 0.30),
    (np.r_[13:16, 29:32], 0.10),
    (np.r_[33:38], 0.25),
)


@dataclass(frozen=True)
class SampleV1Config(qd.BalancedDiverseConfig):
    """Configuration for anchor-aware global quality-diversity collection."""

    output: str = "outputs/sampleV1_tasks.json"
    n_tasks: int = 100
    cpu_cores: int = 8

    # Static JAX shapes for the generated task bank.
    max_n_ally: int = 20
    max_n_enemy: int = 20
    max_n_zone: int = 20

    # Wider balance band requested for sampleV1.
    win_rate_min: float = 0.30
    win_rate_max: float = 0.70
    discovery_candidate_win_rate_min: float = 0.20
    discovery_candidate_win_rate_max: float = 0.80
    confirmation_win_rate_min: float = 0.30
    confirmation_win_rate_max: float = 0.70
    confirmation_team_win_rate_min: float = 0.30
    confirmation_team_win_rate_max: float = 0.70

    # Keep every incremental rollout chunk at one static seed-batch shape.
    # JAX otherwise recompiles run_episode for every distinct chunk length.
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

    # One coordinator process compiles the required rollout shapes into this
    # persistent cache before CPU workers start.  Workers then load the same
    # executable instead of producing a cold-compilation storm.
    jax_persistent_cache: bool = True
    jax_cache_dir: str = ".jax_cache/sampleV1_cpu"
    jax_cache_prewarm: bool = True

    # Proposal mixture.  ``free_generation_ratio`` is implemented as a global
    # constructive restart; mutation is split into radical/local proposals.
    open_ended_generation: bool = True
    programmatic_ratio: float = 0.30
    free_generation_ratio: float = 0.40
    parent_mutation_ratio: float = 0.20
    parent_crossover_ratio: float = 0.10
    radical_mutation_fraction: float = 0.75

    # A fixed task bank defines the distribution that new candidates should
    # leave.  Zero anchor_min_distance derives the gate from leave-one-out
    # nearest-neighbour distances in that bank.
    reference_task_file: str = "task_files/balanced_tasks_v1.json"
    anchor_min_distance: float = 0.0
    anchor_nn_quantile: float = 0.75
    programmatic_anchor_scale: float = 0.65
    local_mutation_anchor_scale: float = 0.50

    # Raw proposal budget.  Most rejected proposals are removed before rollout,
    # so this can be larger than the legacy rollout-candidate budget.
    diversity_candidate_budget_multiplier: int = 40
    certified_pool_multiplier: float = 1.0
    collection_time_budget_seconds: float = 0.0

    # Cost-aware adaptive proposal scheduling with an exploration floor.
    source_adaptation_rate: float = 0.25
    source_exploration_floor: float = 0.15

    # Global restart count buckets: 1-3, 4-6, 7-10, 11-15, 16-20.
    team_size_bucket_weights: tuple[float, ...] = (0.12, 0.18, 0.25, 0.27, 0.18)
    global_map_scale_max: float = 2.25
    radical_replace_fraction_min: float = 0.35
    radical_replace_fraction_max: float = 0.75


@dataclass(frozen=True)
class _V1ParallelJob:
    candidate_id: int
    target: tuple[str, ...]
    proposal_kind: str
    certification_source_kind: str
    parent_ids: tuple[str, ...]
    raw_task: dict[str, Any]
    config: SampleV1Config


@dataclass
class _V1ParallelResult:
    candidate_id: int
    target: tuple[str, ...]
    proposal_kind: str
    parent_ids: tuple[str, ...]
    worker_id: int
    worker_pid: int
    outcome: qd._CandidateOutcome


def _weighted_choice_index(rng: random.Random, weights: Sequence[float]) -> int:
    total = float(sum(max(float(value), 0.0) for value in weights))
    if total <= 0:
        raise ValueError("At least one sampling weight must be positive.")
    target = rng.random() * total
    cumulative = 0.0
    for index, value in enumerate(weights):
        cumulative += max(float(value), 0.0)
        if target <= cumulative:
            return index
    return len(weights) - 1


def _sample_team_count(
    rng: random.Random, capacity: int, weights: Sequence[float]
) -> int:
    ranges = ((1, 3), (4, 6), (7, 10), (11, 15), (16, 20))
    available: list[tuple[int, int, float]] = []
    for (lower, upper), weight in zip(ranges, weights):
        clipped_upper = min(upper, capacity)
        if clipped_upper >= lower:
            available.append((lower, clipped_upper, float(weight)))
    if not available:
        raise ValueError("Team capacity must be at least one.")
    chosen = available[
        _weighted_choice_index(rng, [item[2] for item in available])
    ]
    return rng.randint(chosen[0], chosen[1])


def _sample_palette_roster(rng: random.Random, count: int) -> list[int]:
    if count < 1:
        raise ValueError("A roster must contain at least one unit.")
    palette_size = min(count, rng.randint(2, 6)) if count > 1 else 1
    palette = rng.sample(range(9), k=palette_size)
    # Random positive weights create role-heavy, mixed, and approximately
    # uniform teams without tying the roster to an existing scenario.
    weights = [0.15 + rng.random() ** 2 for _ in palette]
    return list(rng.choices(palette, weights=weights, k=count))


def _team_price(units: Sequence[int]) -> float:
    return float(generators.team_price(units))


def _effective_value(units: Sequence[int], health: Sequence[float]) -> float:
    return float(generators.team_effective_value(units, health))


def _match_global_opponent(
    rng: random.Random,
    ally: Sequence[int],
    ally_health: Sequence[float],
    capacity: int,
) -> tuple[list[int], list[float], dict[str, Any]]:
    target_price = _team_price(ally)
    target_effective = _effective_value(ally, ally_health)
    draw = rng.random()

    if draw < 0.45 and len(ally) <= capacity:
        enemy = list(ally)
        enemy_health = list(ally_health)
        mode = "mirror"
    elif draw < 0.75 and len(ally) <= capacity:
        enemy = generators.unit_swap_team(
            rng,
            ally,
            capacity,
            n_replace=max(1, min(len(ally), int(math.ceil(len(ally) * 0.25)))),
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
        count = min(
            capacity,
            max(1, len(ally) + rng.choice((-2, -1, 0, 0, 0, 1, 2))),
        )
        best: list[int] | None = None
        best_diff = math.inf
        for _ in range(96):
            candidate = _sample_palette_roster(rng, count)
            diff = abs(_team_price(candidate) - target_price) / max(target_price, 1.0)
            if diff < best_diff:
                best = candidate
                best_diff = diff
            if diff <= 0.08:
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

    enemy_price = _team_price(enemy)
    enemy_effective = _effective_value(enemy, enemy_health)
    match_info = {
        "match_mode": mode,
        "ally_price": target_price,
        "enemy_price": enemy_price,
        "price_rel_diff": abs(target_price - enemy_price) / max(target_price, 1.0),
        "ally_effective": target_effective,
        "enemy_effective": enemy_effective,
        "effective_rel_diff": abs(target_effective - enemy_effective)
        / max(target_effective, 1.0),
    }
    return enemy, enemy_health, match_info


def _resize_and_radically_mutate_roster(
    rng: random.Random,
    roster: Sequence[int],
    capacity: int,
    config: SampleV1Config,
) -> tuple[list[int], dict[str, Any]]:
    target_count = _sample_team_count(rng, capacity, config.team_size_bucket_weights)
    values = list(roster)
    rng.shuffle(values)
    if len(values) > target_count:
        values = values[:target_count]
    while len(values) < target_count:
        values.append(rng.randrange(9))

    fraction = rng.uniform(
        config.radical_replace_fraction_min,
        config.radical_replace_fraction_max,
    )
    replace_count = min(
        len(values),
        max(1, int(math.ceil(len(values) * fraction))),
    )
    for index in rng.sample(range(len(values)), k=replace_count):
        values[index] = rng.randrange(9)
    return values, {
        "target_count": target_count,
        "replace_count": replace_count,
        "replace_fraction": fraction,
    }


def _custom_task_metadata(
    *,
    source: dict[str, Any],
    ally: Sequence[int],
    enemy: Sequence[int],
    ally_health: Sequence[float],
    enemy_health: Sequence[float],
    match_info: dict[str, Any],
    target: tuple[str, ...],
    map_scale: float,
    distance_scale: float,
    spread_scale: float,
) -> dict[str, Any]:
    target_descriptor = qd._target_descriptor(target)
    return {
        "source": source,
        "composition_archetype": "global",
        "composition_match": match_info,
        "composition_features": generators.composition_features(
            ally,
            enemy,
            ally_health,
            enemy_health,
        ),
        "layout_archetype": target_descriptor["layout"],
        "distance_bucket": target_descriptor["distance"],
        "distance_scale": distance_scale,
        "spread_bucket": target_descriptor["spread"],
        "spread_scale": spread_scale,
        "map_bucket": f"scale_{map_scale:g}",
        "map_scale": map_scale,
        "zone_archetype": "free",
        "zone_intensity": "free",
        "zone_relation": "free",
        "scenario_bucket": {
            **target_descriptor,
            "composition": "global",
            "zone": "free",
            "zone_intensity": "free",
            "zone_relation": "free",
            "match_mode": match_info["match_mode"],
        },
    }


def _finalize_custom_task(
    rng: random.Random,
    task: dict[str, Any],
    config: SampleV1Config,
    *,
    target: tuple[str, ...],
    candidate_id: int,
    generation_seed: int,
) -> dict[str, Any]:
    transform = rng.choice(config.transforms)
    symmetric_stat_scale = qd._sample_scale(
        rng,
        config.stat_scales,
        config.continuous_stat_sampling,
        config.continuous_sampling,
    )
    zone_strength_scale = qd._sample_scale(
        rng,
        config.zone_strength_scales,
        config.continuous_zone_strength_sampling,
        config.continuous_sampling,
    )
    base._transform_task(task, transform)
    base._scale_team_stats(task, symmetric_stat_scale, symmetric_stat_scale)
    base._scale_zone_strength(task, zone_strength_scale)
    task.setdefault("metadata", {}).update(
        {
            "transform": transform,
            "symmetric_stat_scale": symmetric_stat_scale,
            "zone_strength_scale": zone_strength_scale,
            "target_bucket": qd._target_descriptor(target),
            "candidate_id": candidate_id,
            "generation_seed": generation_seed,
        }
    )
    return task


def _generate_constructive_task(
    config: SampleV1Config,
    *,
    candidate_id: int,
    generation_seed: int,
    target: tuple[str, ...],
    seed_roster: Sequence[int] | None = None,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seed = qd._derive_seed(generation_seed, candidate_id, "sampleV1-constructive")
    rng = random.Random(seed)
    descriptor = qd._target_descriptor(target)

    if seed_roster is None:
        ally_count = _sample_team_count(
            rng,
            config.max_n_ally,
            config.team_size_bucket_weights,
        )
        ally = _sample_palette_roster(rng, ally_count)
        radical_info: dict[str, Any] | None = None
    else:
        ally, radical_info = _resize_and_radically_mutate_roster(
            rng,
            seed_roster,
            config.max_n_ally,
            config,
        )

    ally_health = [
        rng.uniform(config.health_frac_min, config.health_frac_max)
        for _ in ally
    ]
    enemy, enemy_health, match_info = _match_global_opponent(
        rng,
        ally,
        ally_health,
        config.max_n_enemy,
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
    zone_scenario = generators.build_free_zones(
        rng,
        scenario,
        grid_info,
        config.max_n_zone,
    )
    source_payload = source or {
        "kind": "global",
        "subtype": "constructive_restart",
    }
    if radical_info is not None:
        source_payload = {**source_payload, "radical_mutation": radical_info}
    task = {
        "grid_info": grid_info,
        "scenario": scenario,
        "zone_scenario": zone_scenario,
        "metadata": _custom_task_metadata(
            source=source_payload,
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
    return _finalize_custom_task(
        rng,
        task,
        config,
        target=target,
        candidate_id=candidate_id,
        generation_seed=generation_seed,
    )


def _parent_roster(task: dict[str, Any], team: int) -> list[int]:
    teams = np.asarray(task["scenario"]["teams"], dtype=int).reshape(-1)
    unit_ids = np.asarray(task["scenario"]["unit_ids"], dtype=int).reshape(-1)
    return unit_ids[teams == team].tolist()


def _generate_proposal(
    config: SampleV1Config,
    *,
    candidate_id: int,
    generation_seed: int,
    target: tuple[str, ...],
    proposal_kind: str,
    parents: Sequence[qd._ParentPayload],
) -> tuple[dict[str, Any], str]:
    if proposal_kind == "global_constructive":
        return (
            _generate_constructive_task(
                config,
                candidate_id=candidate_id,
                generation_seed=generation_seed,
                target=target,
            ),
            "global_constructive",
        )
    if proposal_kind == "radical_mutation":
        if not parents:
            raise ValueError("Radical mutation requires one parent.")
        parent = parents[0]
        seed = qd._derive_seed(generation_seed, candidate_id, "sampleV1-radical-parent")
        team = random.Random(seed).choice((0, 1))
        task = _generate_constructive_task(
            config,
            candidate_id=candidate_id,
            generation_seed=generation_seed,
            target=target,
            seed_roster=_parent_roster(parent.task, team),
            source={
                "kind": "mutation",
                "subtype": "radical_restart",
                "parent": parent.digest,
                "parent_team": team,
            },
        )
        return task, "mutation"

    forced = {
        "programmatic": "programmatic",
        "distant_crossover": "crossover",
        "local_mutation": "mutation",
    }[proposal_kind]
    task, source_kind = qd._generate_raw_candidate(
        config,
        candidate_id=candidate_id,
        generation_seed=generation_seed,
        target=target,
        parents=parents,
        forced_source_kind=forced,
    )
    task.setdefault("metadata", {}).setdefault("source", {})[
        "sample_v1_proposal"
    ] = proposal_kind
    return task, source_kind


def _parameter_distances(
    signature: np.ndarray,
    references: Sequence[np.ndarray],
) -> np.ndarray:
    if not references:
        return np.empty(0, dtype=float)
    matrix = np.asarray(references, dtype=float)
    result = np.zeros(len(matrix), dtype=float)
    for indexes, weight in _DISTANCE_GROUPS:
        delta = matrix[:, indexes] - signature[indexes]
        result += weight * np.sqrt(np.mean(np.square(delta), axis=1))
    return result


def _nearest_parameter_distance(
    signature: np.ndarray,
    references: Sequence[np.ndarray],
) -> float:
    distances = _parameter_distances(signature, references)
    return math.inf if not len(distances) else float(np.min(distances))


def _stage_deltas(stages: Sequence[int]) -> tuple[int, ...]:
    completed = 0
    deltas = []
    for target in sorted(set(int(value) for value in stages)):
        if target <= 0:
            continue
        deltas.append(target - completed)
        completed = target
    return tuple(deltas)


def _rollout_batch_sizes(config: SampleV1Config) -> tuple[int, ...]:
    """Return every seed-batch shape that can trigger a JAX compilation."""

    return tuple(
        sorted(
            set(_stage_deltas(config.discovery_seed_stages))
            | set(_stage_deltas(config.confirmation_seed_stages))
        )
    )


def _configure_jax_persistent_cache(config: SampleV1Config) -> Path | None:
    if not config.jax_persistent_cache:
        return None
    cache_dir = Path(config.jax_cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    base.jax.config.update("jax_enable_compilation_cache", True)
    base.jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    base.jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    base.jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    return cache_dir


def _prewarm_rollout_cache(
    config: SampleV1Config,
    reference_task: dict[str, Any],
) -> dict[str, Any]:
    """Compile each rollout batch shape once before worker processes start."""

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
            "sampleV1 JAX cache warm-up: "
            f"shape={index}/{len(batch_sizes)}, seed_batch={batch_size}. "
            "A cold cache may take several minutes once; please wait.",
            flush=True,
        )
        seeds = np.arange(batch_size, dtype=np.uint32)
        keys = base.jax.vmap(base.jax.random.key)(base.jnp.asarray(seeds))
        executable = runner.lower(keys, task_params).compile()
        compile_seconds[str(batch_size)] = time.monotonic() - shape_started
        del executable

    elapsed = time.monotonic() - started
    print(
        "sampleV1 JAX cache warm-up complete: "
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
    config: SampleV1Config,
    reference_task: dict[str, Any],
    worker_count: int,
) -> dict[str, Any]:
    """Warm in a disposable process so its large compiler state is released."""

    if (
        worker_count <= 1
        or not config.jax_persistent_cache
        or not config.jax_cache_prewarm
    ):
        return _prewarm_rollout_cache(config, reference_task)

    context = qd._multiprocessing_context()
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as warmup_executor:
        return warmup_executor.submit(
            _prewarm_rollout_cache,
            config,
            reference_task,
        ).result()


def _load_reference_archive(
    config: SampleV1Config,
) -> tuple[list[np.ndarray], float, dict[str, Any], dict[str, Any]]:
    reference_path = Path(config.reference_task_file)
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Reference task bank does not exist: {reference_path.resolve()}"
        )
    bank = base.load_task_bank(reference_path)
    tasks = list(bank["tasks"])
    if not tasks:
        raise ValueError("Reference task bank must contain at least one task.")
    signatures = [qd._canonical_parameter_signature(task) for task in tasks]

    if config.anchor_min_distance > 0:
        threshold = float(config.anchor_min_distance)
        threshold_policy = "explicit"
        nearest_values: list[float] = []
    elif len(signatures) == 1:
        threshold = float(config.parameter_distance_threshold)
        threshold_policy = "single_reference_fallback"
        nearest_values = []
    else:
        nearest_values = []
        for index, signature in enumerate(signatures):
            distances = _parameter_distances(signature, signatures)
            distances[index] = math.inf
            nearest_values.append(float(np.min(distances)))
        threshold = max(
            float(
                np.quantile(
                    np.asarray(nearest_values, dtype=float),
                    config.anchor_nn_quantile,
                )
            ),
            float(config.parameter_distance_threshold),
        )
        threshold_policy = "reference_leave_one_out_quantile"

    report = {
        "file": str(reference_path),
        "n_tasks": len(tasks),
        "threshold": threshold,
        "threshold_policy": threshold_policy,
        "nearest_neighbor_quantile": config.anchor_nn_quantile,
        "reference_nn_min": min(nearest_values) if nearest_values else None,
        "reference_nn_median": (
            float(np.median(nearest_values)) if nearest_values else None
        ),
        "reference_nn_max": max(nearest_values) if nearest_values else None,
    }
    return signatures, threshold, report, copy.deepcopy(tasks[0])


def _anchor_scale(config: SampleV1Config, proposal_kind: str) -> float:
    if proposal_kind == "programmatic":
        return float(config.programmatic_anchor_scale)
    if proposal_kind == "local_mutation":
        return float(config.local_mutation_anchor_scale)
    return 1.0


def _prefilter_candidate(
    task: dict[str, Any],
    proposal_kind: str,
    config: SampleV1Config,
    *,
    anchor_signatures: Sequence[np.ndarray],
    anchor_threshold: float,
    pool: Sequence[qd._CertifiedCandidate],
) -> tuple[np.ndarray | None, str | None]:
    try:
        base.validate_task(task)
        signature = qd._canonical_parameter_signature(task)
    except ValueError as exc:
        return None, f"prefilter_invalid:{exc}"

    anchor_distance = _nearest_parameter_distance(signature, anchor_signatures)
    required_anchor_distance = anchor_threshold * _anchor_scale(
        config, proposal_kind
    )
    if anchor_distance < required_anchor_distance:
        return None, "prefilter_anchor_near"

    pool_distance = _nearest_parameter_distance(
        signature,
        [candidate.parameter_signature for candidate in pool],
    )
    if pool_distance < config.parameter_distance_threshold:
        return None, "prefilter_pool_near"

    try:
        static_reason, static_values, engagement = qd._static_screen(task, config)
    except ValueError as exc:
        return None, f"prefilter_static_invalid:{exc}"
    if static_reason in {"fast_stomp", "static_imbalance_hard"}:
        return None, f"prefilter_{static_reason}"

    task.setdefault("metadata", {})["sample_v1_prefilter"] = {
        "proposal_kind": proposal_kind,
        "anchor_distance": anchor_distance,
        "required_anchor_distance": required_anchor_distance,
        "pool_distance": None if not math.isfinite(pool_distance) else pool_distance,
        "static_reason": static_reason,
        "static_team_values": static_values,
        "static_engagement": engagement,
        "rollout_skipped": False,
    }
    return signature, None


def _v1_count_bucket(value: int) -> str:
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


def _extend_v1_candidate_cell(candidate: qd._CertifiedCandidate) -> None:
    teams = np.asarray(candidate.task["scenario"]["teams"], dtype=int).reshape(-1)
    team_size = max(int(np.count_nonzero(teams == team)) for team in (0, 1))
    zone_count = int(candidate.task["zone_scenario"]["n_zone"])
    team_bucket = _v1_count_bucket(team_size)
    zone_bucket = _v1_count_bucket(zone_count)
    candidate.map_cell = (*candidate.map_cell, team_bucket, zone_bucket)
    candidate.task.setdefault("metadata", {}).setdefault("sample_v1", {}).update(
        {
            "team_size_bucket": team_bucket,
            "zone_count_bucket": zone_bucket,
        }
    )


def _sample_candidate_cpu_worker(job: _V1ParallelJob) -> _V1ParallelResult:
    """A/B-certify one coordinator-prefiltered candidate on one CPU worker."""

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
            _extend_v1_candidate_cell(outcome.candidate)
    except (RuntimeError, ValueError) as exc:
        outcome = qd._CandidateOutcome(
            None,
            f"worker:{type(exc).__name__}:{exc}",
            "worker_error",
        )
    return _V1ParallelResult(
        candidate_id=job.candidate_id,
        target=job.target,
        proposal_kind=job.proposal_kind,
        parent_ids=job.parent_ids,
        worker_id=0 if qd._CPU_WORKER_ID < 0 else qd._CPU_WORKER_ID,
        worker_pid=os.getpid(),
        outcome=outcome,
    )


def _proposal_base_weights(config: SampleV1Config) -> dict[str, float]:
    radical = config.parent_mutation_ratio * config.radical_mutation_fraction
    local = config.parent_mutation_ratio - radical
    return {
        "global_constructive": config.free_generation_ratio,
        "programmatic": config.programmatic_ratio,
        "radical_mutation": radical,
        "distant_crossover": config.parent_crossover_ratio,
        "local_mutation": local,
    }


def _draw_proposal_kind(
    rng: random.Random,
    config: SampleV1Config,
    pool: Sequence[qd._CertifiedCandidate],
    generated: Counter[str],
    confirmed: Counter[str],
    anchor_distance_sums: Counter[str],
    anchor_threshold: float,
) -> str:
    weights = _proposal_base_weights(config)
    if not pool:
        weights["radical_mutation"] = 0.0
        weights["local_mutation"] = 0.0
    if len(pool) < 2:
        weights["distant_crossover"] = 0.0

    adapted: list[tuple[str, float]] = []
    for kind in PROPOSAL_KINDS:
        base_weight = weights[kind]
        if base_weight <= 0:
            adapted.append((kind, 0.0))
            continue
        success_rate = (confirmed[kind] + 1.0) / (generated[kind] + 2.0)
        mean_anchor = (
            anchor_distance_sums[kind] / confirmed[kind]
            if confirmed[kind] > 0
            else anchor_threshold
        )
        novelty_yield = min(mean_anchor / max(anchor_threshold, 1e-9), 2.0)
        utility = success_rate * novelty_yield
        multiplier = (
            1.0
            - config.source_adaptation_rate
            + config.source_adaptation_rate * 2.0 * utility
        )
        adapted.append(
            (
                kind,
                base_weight
                * max(multiplier, config.source_exploration_floor),
            )
        )

    index = _weighted_choice_index(rng, [weight for _, weight in adapted])
    return adapted[index][0]


def _choose_target(
    rng: random.Random,
    config: SampleV1Config,
    coverage: qd._CoverageState,
    feedback: qd._BucketFeedback,
) -> tuple[str, ...]:
    """Choose a coverage deficit while accounting for empirical feasibility."""

    best: tuple[str, ...] | None = None
    best_score = -math.inf
    for _ in range(config.target_draws):
        target = qd._draw_target(rng, config)
        deficit = coverage.gain(qd._target_descriptor(target))
        accept_rate = max(
            feedback.estimated_accept_rate(target),
            config.target_accept_rate_floor,
        )
        # Expected coverage gain avoids spending most of the budget on cells
        # whose acceptance probability has collapsed, while the exploration
        # term and the scheduler's source floor still revisit uncertain cells.
        score = deficit * math.sqrt(accept_rate)
        score += config.target_exploration * rng.random()
        if score > best_score:
            best = target
            best_score = score
    assert best is not None
    return best


def _assign_parents(
    proposal_kind: str,
    pool: Sequence[qd._CertifiedCandidate],
    parent_leases: Counter[str],
    rng: random.Random,
) -> tuple[qd._ParentPayload, ...]:
    required = 2 if proposal_kind == "distant_crossover" else (
        1 if proposal_kind in {"radical_mutation", "local_mutation"} else 0
    )
    if required == 0:
        return ()
    if len(pool) < required:
        raise RuntimeError(
            f"Proposal {proposal_kind!r} requires {required} confirmed parents."
        )

    least = min(parent_leases[candidate.digest] for candidate in pool)
    first_group = [
        candidate
        for candidate in pool
        if parent_leases[candidate.digest] == least
    ]
    first = rng.choice(first_group)
    selected = [first]
    if required == 2:
        remaining = [candidate for candidate in pool if candidate is not first]
        minimum_lease = min(parent_leases[candidate.digest] for candidate in remaining)
        lease_group = [
            candidate
            for candidate in remaining
            if parent_leases[candidate.digest] <= minimum_lease + 1
        ]
        second = max(
            lease_group,
            key=lambda candidate: qd._parameter_distance(
                first.parameter_signature,
                candidate.parameter_signature,
            ),
        )
        selected.append(second)

    payloads = []
    for candidate in selected:
        parent_leases[candidate.digest] += 1
        payloads.append(
            qd._ParentPayload(
                task=copy.deepcopy(candidate.task),
                digest=candidate.digest,
            )
        )
    return tuple(payloads)


def _validate_config(config: SampleV1Config) -> None:
    qd._validate_config(config)
    if config.hard_task_mode:
        raise ValueError(
            "sampleV1 is the balanced 0.3-0.7 sampler; hard_task_mode is unsupported."
        )
    if config.max_n_ally < 1 or config.max_n_enemy < 1:
        raise ValueError("max_n_ally and max_n_enemy must be positive.")
    if config.max_n_zone < 0:
        raise ValueError("max_n_zone must be non-negative.")
    if max(config.max_n_ally, config.max_n_enemy, config.max_n_zone) > 20:
        raise ValueError("sampleV1 supports at most 20 units per team and 20 zones.")
    if not (
        0.0
        <= config.discovery_candidate_win_rate_min
        <= config.win_rate_min
        < config.win_rate_max
        <= config.discovery_candidate_win_rate_max
        <= 1.0
    ):
        raise ValueError(
            "Discovery candidate range must contain the ordered target win-rate range."
        )
    if not (
        config.confirmation_win_rate_min
        == config.confirmation_team_win_rate_min
        == config.win_rate_min
        and config.confirmation_win_rate_max
        == config.confirmation_team_win_rate_max
        == config.win_rate_max
    ):
        raise ValueError(
            "Discovery target and both confirmation win-rate ranges must match."
        )
    if len(config.team_size_bucket_weights) != 5:
        raise ValueError("team_size_bucket_weights must contain five values.")
    if any(value < 0 for value in config.team_size_bucket_weights):
        raise ValueError("team_size_bucket_weights must be non-negative.")
    if sum(config.team_size_bucket_weights) <= 0:
        raise ValueError("At least one team_size_bucket_weight must be positive.")
    if config.jax_persistent_cache and not config.jax_cache_dir.strip():
        raise ValueError("jax_cache_dir must not be empty when caching is enabled.")
    if any(value <= 0 for value in _rollout_batch_sizes(config)):
        raise ValueError("Rollout stage increments must be positive.")
    if not 0.0 <= config.anchor_nn_quantile <= 1.0:
        raise ValueError("anchor_nn_quantile must lie in [0, 1].")
    if config.anchor_min_distance < 0:
        raise ValueError("anchor_min_distance must be non-negative.")
    if config.programmatic_anchor_scale < 0 or config.local_mutation_anchor_scale < 0:
        raise ValueError("Anchor source scales must be non-negative.")
    if not 0.0 <= config.radical_mutation_fraction <= 1.0:
        raise ValueError("radical_mutation_fraction must lie in [0, 1].")
    if not (
        0.0
        <= config.radical_replace_fraction_min
        <= config.radical_replace_fraction_max
        <= 1.0
    ):
        raise ValueError("Invalid radical replacement fraction range.")
    if not 0.0 <= config.source_adaptation_rate <= 1.0:
        raise ValueError("source_adaptation_rate must lie in [0, 1].")
    if not 0.0 <= config.source_exploration_floor <= 1.0:
        raise ValueError("source_exploration_floor must lie in [0, 1].")


def _coverage_from_pool(
    candidates: Iterable[qd._CertifiedCandidate],
    config: SampleV1Config,
) -> qd._CoverageState:
    return qd._coverage_from_pool(candidates, config)


def sample_tasks_with_report(
    config: SampleV1Config,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect anchor-distant, B-confirmed tasks with CPU-parallel rollout."""

    _validate_config(config)
    (
        anchor_signatures,
        anchor_threshold,
        anchor_report,
        reference_task,
    ) = _load_reference_archive(config)
    budget = qd._collection_budget(config)
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

    pool: list[qd._CertifiedCandidate] = []
    feedback = qd._BucketFeedback()
    reject_counts: Counter[str] = Counter()
    prefilter_reject_counts: Counter[str] = Counter()
    source_generated: Counter[str] = Counter()
    source_submitted: Counter[str] = Counter()
    source_confirmed: Counter[str] = Counter()
    source_anchor_distance_sums: Counter[str] = Counter()
    parent_leases: Counter[str] = Counter()
    target_leases: Counter[tuple[str, ...]] = Counter()
    started = 0.0
    worker_count = qd._parallel_worker_count(config, budget)
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
        "sampleV1 anchor-aware CPU sampling: "
        f"workers={worker_count}, max_in_flight={max_in_flight}, "
        f"target={config.n_tasks}, raw_budget={budget}, "
        f"reference_tasks={anchor_report['n_tasks']}, "
        f"anchor_threshold={anchor_threshold:.4f}, "
        f"win_rate_band=[{config.win_rate_min:.2f}, {config.win_rate_max:.2f}], "
        f"limits=({config.max_n_ally}, {config.max_n_enemy}, {config.max_n_zone}), "
        f"generation_seed={generation_seed}",
        flush=True,
    )

    def time_budget_exhausted() -> bool:
        return bool(
            config.collection_time_budget_seconds > 0
            and time.monotonic() - started
            >= config.collection_time_budget_seconds
        )

    def collection_complete() -> bool:
        return len(pool) >= pool_target

    def release_parents(parents: Sequence[qd._ParentPayload]) -> None:
        for parent in parents:
            parent_leases[parent.digest] -= 1

    def build_job() -> _V1ParallelJob | None:
        nonlocal next_candidate_id

        while next_candidate_id < budget and not time_budget_exhausted():
            candidate_id = next_candidate_id
            next_candidate_id += 1
            scheduler_rng = random.Random(
                qd._derive_seed(generation_seed, candidate_id, "sampleV1-scheduler")
            )
            coverage = qd._leased_coverage(pool, target_leases, config)
            target = _choose_target(scheduler_rng, config, coverage, feedback)
            proposal_kind = _draw_proposal_kind(
                scheduler_rng,
                config,
                pool,
                source_generated,
                source_confirmed,
                source_anchor_distance_sums,
                anchor_threshold,
            )
            parents = _assign_parents(
                proposal_kind,
                pool,
                parent_leases,
                scheduler_rng,
            )
            source_generated[proposal_kind] += 1
            if proposal_kind == "programmatic":
                feedback.record_generated(target)

            try:
                raw_task, certification_source_kind = _generate_proposal(
                    config,
                    candidate_id=candidate_id,
                    generation_seed=generation_seed,
                    target=target,
                    proposal_kind=proposal_kind,
                    parents=parents,
                )
                _, rejection = _prefilter_candidate(
                    raw_task,
                    proposal_kind,
                    config,
                    anchor_signatures=anchor_signatures,
                    anchor_threshold=anchor_threshold,
                    pool=pool,
                )
            except (RuntimeError, ValueError) as exc:
                rejection = f"prefilter_generation:{type(exc).__name__}:{exc}"
                raw_task = {}
                certification_source_kind = proposal_kind

            if rejection is not None:
                release_parents(parents)
                reject_counts[rejection] += 1
                prefilter_reject_counts[rejection] += 1
                continue

            if proposal_kind == "programmatic":
                target_leases[target] += 1
            source_submitted[proposal_kind] += 1
            return _V1ParallelJob(
                candidate_id=candidate_id,
                target=target,
                proposal_kind=proposal_kind,
                certification_source_kind=certification_source_kind,
                parent_ids=tuple(parent.digest for parent in parents),
                raw_task=raw_task,
                config=config,
            )
        return None

    def commit_result(result: _V1ParallelResult) -> None:
        for parent_id in result.parent_ids:
            parent_leases[parent_id] -= 1
        if result.proposal_kind == "programmatic":
            target_leases[result.target] -= 1
        if direct_append_pool and len(pool) >= pool_target:
            reject_counts["completed_after_target"] += 1
            return

        outcome = result.outcome
        if outcome.candidate is None:
            reject_counts[outcome.reason] += 1
            return
        candidate = outcome.candidate
        final_anchor_distance = _nearest_parameter_distance(
            candidate.parameter_signature,
            anchor_signatures,
        )
        final_required = anchor_threshold * _anchor_scale(
            config,
            result.proposal_kind,
        )
        if final_anchor_distance < final_required:
            reject_counts["post_certification_anchor_near"] += 1
            return

        duplicate_reason = qd._is_pool_duplicate(candidate, pool, config)
        if duplicate_reason is not None:
            reject_counts[duplicate_reason] += 1
            return
        if direct_append_pool and not qd._online_append_allows(
            candidate,
            pool,
            config,
        ):
            reject_counts["online_diversity_rejected"] += 1
            return

        sample_v1_metadata = candidate.task.setdefault("metadata", {}).setdefault(
            "sample_v1",
            {},
        )
        sample_v1_metadata.update(
            {
                "proposal_kind": result.proposal_kind,
                "post_certification_anchor_distance": final_anchor_distance,
                "required_anchor_distance": final_required,
                "reference_task_file": config.reference_task_file,
            }
        )
        candidate.task["metadata"]["parallel_sampling"] = {
            "candidate_id": result.candidate_id,
            "worker_id": result.worker_id,
            "worker_pid": result.worker_pid,
            "parent_ids": list(result.parent_ids),
            "commit_policy": "asynchronous_completion_order",
            "prefilter_location": "coordinator_before_worker_submission",
        }
        pool.append(candidate)
        if result.proposal_kind == "programmatic":
            feedback.record_confirmed(result.target)
        source_confirmed[result.proposal_kind] += 1
        source_anchor_distance_sums[result.proposal_kind] += final_anchor_distance
        print(
            f"candidate={result.candidate_id + 1}/{budget} B_CONFIRMED "
            f"pool={len(pool)}/{pool_target} "
            f"source={result.proposal_kind} "
            f"anchor_distance={final_anchor_distance:.4f} "
            f"worker={result.worker_id}",
            flush=True,
        )

    executor: ProcessPoolExecutor | None = None
    prewarm_report: dict[str, Any] = {
        "enabled": False,
        "batch_sizes": list(_rollout_batch_sizes(config)),
        "elapsed_seconds": 0.0,
    }
    try:
        cache_dir = _configure_jax_persistent_cache(config)
        prewarm_report = _prepare_rollout_cache(
            config,
            reference_task,
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
            pending: dict[Any, _V1ParallelJob] = {}

            def refill_workers() -> None:
                while (
                    len(pending) < max_in_flight
                    and not collection_complete()
                    and not time_budget_exhausted()
                ):
                    job = build_job()
                    if job is None:
                        break
                    future = executor.submit(_sample_candidate_cpu_worker, job)
                    pending[future] = job

            refill_workers()
            while pending and not collection_complete():
                if time_budget_exhausted():
                    reject_counts["time_budget_exhausted"] += 1
                    break
                completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
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

    selected = (
        list(pool[: config.n_tasks])
        if direct_append_pool
        else qd._select_diverse_candidates(pool, config)
    )
    if not selected:
        raise RuntimeError(
            "sampleV1 could not collect any certified task: "
            f"raw_budget={budget}, rejections={dict(reject_counts)}."
        )
    if len(selected) < config.n_tasks:
        print(
            "WARNING: sampleV1 exhausted its budget before reaching the target; "
            f"saving {len(selected)}/{config.n_tasks} tasks.",
            flush=True,
        )

    tasks = [candidate.task for candidate in selected[: config.n_tasks]]
    for index, (task, candidate) in enumerate(zip(tasks, selected)):
        task["task_id"] = f"task_{index:06d}_{candidate.digest[:10]}"
        task["metadata"]["canonical_hash"] = candidate.digest

    report = qd._build_report(
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
                "target_reached"
                if len(tasks) == config.n_tasks
                else (
                    "time_budget_exhausted"
                    if time_budget_exhausted()
                    else "candidate_budget_exhausted"
                )
            ),
            "reference_archive": anchor_report,
            "prefilter_rejections": dict(prefilter_reject_counts),
            "proposal_sources": {
                "generated": dict(source_generated),
                "submitted_to_rollout": dict(source_submitted),
                "confirmed": dict(source_confirmed),
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
                "directory": None if cache_dir is None else str(cache_dir),
                "prewarm": prewarm_report,
                "rollout_batch_sizes": list(_rollout_batch_sizes(config)),
            },
        }
    )
    return tasks, report


def sample_tasks(config: SampleV1Config) -> list[dict[str, Any]]:
    """Compatibility wrapper returning only the sampled task list."""

    tasks, _ = sample_tasks_with_report(config)
    return tasks


def save_task_bank(
    config: SampleV1Config,
    tasks: Sequence[dict[str, Any]],
    report: dict[str, Any],
) -> Path:
    """Save a sampleV1 task bank through the stable task-bank writer."""

    generator_protocol = {
        "strategy": "anchor_aware_global_quality_diversity",
        "task_profile": "balanced_wide_band",
        "generation_seed": report["generation_seed"],
        "generation_seed_policy": report["generation_seed_policy"],
        "evaluation_seed": config.seed,
        "reference_archive": report["reference_archive"],
        "proposal_mix": _proposal_base_weights(config),
        "global_constructive_generation": True,
        "radical_mutation": {
            "enabled": True,
            "fraction_of_mutation": config.radical_mutation_fraction,
            "replacement_fraction": [
                config.radical_replace_fraction_min,
                config.radical_replace_fraction_max,
            ],
        },
        "distant_parent_crossover": True,
        "rollout_prefilter": {
            "anchor_distance_before_rollout": True,
            "accepted_pool_distance_before_rollout": True,
            "static_screen_before_rollout": True,
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
    bank["manifest"]["generator"] = "src.tabx.sampleV1"
    output.write_text(
        json.dumps(bank, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    config = tyro.cli(SampleV1Config)
    tasks, report = sample_tasks_with_report(config)
    output = save_task_bank(config, tasks, report)
    completion = (
        "complete"
        if report["complete"]
        else f"partial={len(tasks)}/{report['requested_tasks']}"
    )
    print(
        f"Saved {len(tasks)} sampleV1 tasks to {output}; "
        f"collection={completion}, "
        f"prefilter_rejections={sum(report['prefilter_rejections'].values())}, "
        f"sources={report['proposal_sources']['confirmed']}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
