"""Evaluate a fixed TABX task bank with heuristic agents on both teams."""

from __future__ import annotations

import copy
import io
import json
import math
import multiprocessing
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple, Sequence


def _early_compute_backend(argv: Sequence[str]) -> str:
    """Read the backend early enough to configure JAX before importing it."""

    for index, argument in enumerate(argv):
        if argument.startswith("--compute-backend="):
            return argument.split("=", 1)[1]
        if argument == "--compute-backend" and index + 1 < len(argv):
            return argv[index + 1]
    return "cpu"

if __name__ == "__main__":
    _CLI_COMPUTE_BACKEND = _early_compute_backend(sys.argv[1:])
    if _CLI_COMPUTE_BACKEND == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_SKIP_CUDA_CONSTRAINTS_CHECK"] = "1"
    elif _CLI_COMPUTE_BACKEND == "gpu":
        os.environ["JAX_PLATFORMS"] = "cuda"

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from src.tabx.config import TABXConfig
from src.tabx.heuristic_policy import LastVisibleTarget, heuristic_policy
from src.tabx.heuristic_policy.params import TABXHeuristicParam
from src.tabx.heuristic_policy.utils import load_heuristic_params_from_json
from src.tabx.sample_task import (
    build_batched_env_params_from_tasks,
    infer_task_limits,
    load_task_bank,
    task_ids,
)
from src.tabx.tabx import TABX

# Difficulty ladder: weaken the stronger side until win rate enters the band.
# expert vs expert balanced → equal
# ally wins too often → downgrade ally (advanced → easy, medium → very_easy)
# ally loses too often → downgrade enemy (advanced → hard, medium → very_hard)
DIFFICULTY_EQUAL = "equal"
DIFFICULTY_EASY = "easy"
DIFFICULTY_VERY_EASY = "very_easy"
DIFFICULTY_HARD = "hard"
DIFFICULTY_VERY_HARD = "very_hard"
DIFFICULTY_INVALID = "invalid"

VALID_DIFFICULTIES = (
    DIFFICULTY_EQUAL,
    DIFFICULTY_EASY,
    DIFFICULTY_VERY_EASY,
    DIFFICULTY_HARD,
    DIFFICULTY_VERY_HARD,
)


@dataclass(frozen=True)
class EvalConfig:
    """Command-line configuration for fixed-task heuristic evaluation."""

    task_file: str
    output: str = "task_eval.json"
    heuristic: str | None = None
    physics: str | None = None
    num_seeds: int = 32
    seed: int = 0
    # Match the sampler's discovery/confirmation/certification protocol by default.
    epsilon_override: float | None = 0.05
    max_episode_steps: int = 512
    evaluate_flipped: bool = False
    # ``require_consistent`` re-runs the selected difficulty matchup after
    # swapping team identities and invalidates classifications that are not
    # balanced in both directions. ``ignore`` preserves the legacy protocol.
    difficulty_flip_mode: str = "require_consistent"
    difficulty_max_abs_side_bias: float = 0.20
    # When set, write a task-bank JSON containing only tasks that remain
    # balanced in both the original and flipped evaluations.
    filtered_task_output: str | None = None
    filter_max_abs_side_bias: float = 0.20
    # Evaluate every task for this many fresh, disjoint seed rounds, aggregate
    # all episode samples, then apply the balance filter exactly once.
    filter_rounds: int = 3
    # Ladder classification against a balanced win-rate band.
    classify_difficulty: bool = True
    win_rate_min: float = 0.4
    win_rate_max: float = 0.6
    baseline_heuristic: str = "expert"
    # Applied to the stronger side, in order from milder to stronger downgrade.
    downgrade_heuristics: tuple[str, ...] = ("advanced", "medium")
    compute_backend: str = "cpu"
    cpu_cores: int = 1


class EpisodeMetrics(NamedTuple):
    wins: jax.Array
    episode_length: jax.Array
    truncation: jax.Array
    hp_margin: jax.Array
    # One-hot/multi-hot team of the first casualty: [ally, enemy]. A
    # simultaneous first casualty is [1, 1], and no casualty is [0, 0].
    first_casualty_team: jax.Array
    attack_attempts: jax.Array
    attack_successes: jax.Array
    damage: jax.Array
    healing: jax.Array
    first_interaction_step: jax.Array


@dataclass(frozen=True)
class _EvalChunkJob:
    chunk_id: int
    tasks: tuple[dict[str, Any], ...]
    physics: str
    heuristic: str
    num_seeds: int
    seed: int
    epsilon_override: float | None
    max_episode_steps: int
    ally_heuristic: str | None
    enemy_heuristic: str | None
    max_n_ally: int
    max_n_enemy: int
    max_n_zone: int
    include_episode_samples: bool


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    """Return a two-sided Wilson score interval for a Bernoulli rate."""

    if total <= 0:
        return [math.nan, math.nan]
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(probability * (1.0 - probability) / total + z * z / (4.0 * total**2))
        / denominator
    )
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def _team_sum(values: jax.Array, n_ally: int) -> jax.Array:
    values = values.reshape(-1)
    return jnp.stack([values[:n_ally].sum(), values[n_ally:].sum()])


def _apply_epsilon_override(
    params: TABXHeuristicParam, epsilon_override: float | None
) -> TABXHeuristicParam:
    if epsilon_override is None:
        return params
    if not 0.0 <= epsilon_override <= 1.0:
        raise ValueError("epsilon_override must be in [0, 1].")
    return params.replace(epsilon=jnp.full_like(params.epsilon, epsilon_override))


def load_heuristic_params(
    name: str, *, epsilon_override: float | None = None
) -> TABXHeuristicParam:
    """Load one heuristic preset, optionally overriding epsilon."""

    return _apply_epsilon_override(
        load_heuristic_params_from_json(name), epsilon_override
    )


def _is_balanced(win_rate: float, win_rate_min: float, win_rate_max: float) -> bool:
    return win_rate_min <= win_rate <= win_rate_max


def _make_episode_runner(env: TABX):
    """Create one JAX-compatible episode function for a fixed environment schema."""

    ally_key_set = set(env.ally_keys)

    def run_episode(key: jax.Array, env_params: dict[str, Any]) -> EpisodeMetrics:
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key, env_params)
        targets = {unit: LastVisibleTarget() for unit in env.unit_keys}
        zero_team = jnp.zeros((env.max_team,), dtype=jnp.float32)
        carry = (
            key,
            obs,
            state,
            targets,
            jnp.array(False),
            jnp.array(0, dtype=jnp.int32),
            zero_team,
            jnp.array(False),
            zero_team,
            zero_team,
            zero_team,
            zero_team,
            zero_team,
            jnp.array(-1, dtype=jnp.int32),
        )

        def condition(loop_carry):
            done = loop_carry[4]
            steps = loop_carry[5]
            return (~done) & (steps < env.max_episode_steps)

        def body(loop_carry):
            (
                rng,
                current_obs,
                current_state,
                last_targets,
                _,
                steps,
                _,
                _,
                first_casualty_team,
                cumulative_attempts,
                cumulative_successes,
                cumulative_damage,
                cumulative_healing,
                first_interaction_step,
            ) = loop_carry
            split_keys = jax.random.split(rng, len(env.unit_keys) + 2)
            next_rng = split_keys[0]
            step_key = split_keys[1]
            shared = env_params["heuristic_params"]
            ally_heuristic = env_params.get("ally_heuristic_params", shared)
            enemy_heuristic = env_params.get("enemy_heuristic_params", shared)
            actions = {}
            next_targets = {}
            for index, unit in enumerate(env.unit_keys):
                heuristic_params = (
                    ally_heuristic if unit in ally_key_set else enemy_heuristic
                )
                actions[unit], next_targets[unit] = heuristic_policy(
                    split_keys[index + 2],
                    current_obs[unit],
                    last_targets[unit],
                    env.num_agents,
                    env.max_n_zone,
                    heuristic_params,
                    current_state["physics_params"],
                )

            next_obs, next_state, _, dones, info = env.step(step_key, current_state, actions)
            done = dones["__all__"]
            is_attacking = info["is_attacking"].astype(jnp.float32)
            damage_dealt = info["damage_dealt"]
            attempts = _team_sum(is_attacking, env.max_n_ally)
            successes = _team_sum(
                (jnp.abs(damage_dealt) > 1e-6).astype(jnp.float32), env.max_n_ally
            )
            damage = _team_sum(jnp.maximum(damage_dealt, 0.0), env.max_n_ally)
            healing = _team_sum(jnp.maximum(-damage_dealt, 0.0), env.max_n_ally)
            has_interaction = jnp.abs(damage_dealt).sum() > 1e-6
            first_interaction_step = jnp.where(
                (first_interaction_step < 0) & has_interaction,
                steps + 1,
                first_interaction_step,
            )

            ally_dead = jnp.stack(
                [
                    (~next_state["state"][unit].status.is_alive)
                    & (~next_state["state"][unit].status.is_disabled)
                    for unit in env.ally_keys
                ]
            ).sum()
            enemy_dead = jnp.stack(
                [
                    (~next_state["state"][unit].status.is_alive)
                    & (~next_state["state"][unit].status.is_disabled)
                    for unit in env.enemy_keys
                ]
            ).sum()
            dead_units = jnp.stack([ally_dead, enemy_dead]) > 0
            first_casualty_seen = first_casualty_team.sum() > 0
            new_first_casualty_team = dead_units.astype(jnp.float32)
            first_casualty_team = jnp.where(
                first_casualty_seen,
                first_casualty_team,
                new_first_casualty_team,
            )

            return (
                next_rng,
                next_obs,
                next_state,
                next_targets,
                done,
                steps + 1,
                info["is_win"],
                info["truncation"][0],
                first_casualty_team,
                cumulative_attempts + attempts,
                cumulative_successes + successes,
                cumulative_damage + damage,
                cumulative_healing + healing,
                first_interaction_step,
            )

        (
            _,
            _,
            final_state,
            _,
            _,
            steps,
            wins,
            truncation,
            first_casualty_team,
            attempts,
            successes,
            damage,
            healing,
            first_interaction_step,
        ) = jax.lax.while_loop(condition, body, carry)
        hp_ratio = final_state["state"]["game_manager"].team_hp_ratio
        return EpisodeMetrics(
            wins=wins,
            episode_length=steps,
            truncation=truncation,
            hp_margin=hp_ratio[0] - hp_ratio[1],
            first_casualty_team=first_casualty_team,
            attack_attempts=attempts,
            attack_successes=successes,
            damage=damage,
            healing=healing,
            first_interaction_step=first_interaction_step,
        )

    return jax.jit(jax.vmap(run_episode, in_axes=(0, None)))


@lru_cache(maxsize=16)
def _cached_episode_runner(
    max_n_ally: int,
    max_n_enemy: int,
    max_n_zone: int,
    max_episode_steps: int,
):
    """Reuse one compiled runner for every task sharing the same padded schema."""

    env = TABX(
        cfg=TABXConfig(
            max_n_ally=max_n_ally,
            max_n_enemy=max_n_enemy,
            max_n_zone=max_n_zone,
        ),
        max_episode_steps=max_episode_steps,
    )
    return _make_episode_runner(env)


def _aggregate_episode_metrics(metrics: EpisodeMetrics, max_episode_steps: int) -> dict[str, Any]:
    wins = np.asarray(metrics.wins)
    lengths = np.asarray(metrics.episode_length)
    truncations = np.asarray(metrics.truncation, dtype=float)
    hp_margins = np.asarray(metrics.hp_margin)
    first_casualties = np.asarray(metrics.first_casualty_team)
    attempts = np.asarray(metrics.attack_attempts)
    successes = np.asarray(metrics.attack_successes)
    damage = np.asarray(metrics.damage)
    healing = np.asarray(metrics.healing)
    first_interaction_steps = np.asarray(metrics.first_interaction_step)

    n_rollouts = len(lengths)
    ally_wins = int(np.rint(wins[:, 0].sum()))
    total_interaction = damage.sum(axis=1) + healing.sum(axis=1)
    no_interaction = total_interaction <= 1e-6
    damage_totals = damage.sum(axis=0)
    damage_share = np.divide(
        damage_totals,
        damage_totals.sum(),
        out=np.full(2, 0.5, dtype=float),
        where=damage_totals.sum() > 0,
    )
    observed_first_interaction = first_interaction_steps[first_interaction_steps >= 0]
    short_cutoff = max(5, int(max_episode_steps * 0.05))

    def team_rate(numerator: np.ndarray, denominator: np.ndarray) -> list[float]:
        total_denominator = denominator.sum(axis=0)
        return np.divide(
            numerator.sum(axis=0),
            total_denominator,
            out=np.zeros(2, dtype=float),
            where=total_denominator > 0,
        ).tolist()

    def team_mean_ci95(values: np.ndarray) -> list[list[float]]:
        means = values.mean(axis=0)
        if len(values) <= 1:
            return [[float(value), float(value)] for value in means]
        half_width = 1.959963984540054 * values.std(axis=0, ddof=1) / math.sqrt(len(values))
        return [
            [float(mean - half), float(mean + half)]
            for mean, half in zip(means, half_width)
        ]

    attempt_totals = attempts.sum(axis=0)
    success_totals = successes.sum(axis=0)
    attack_success_ci95 = [
        (
            wilson_interval(int(round(success)), int(round(total)))
            if total > 0
            else [0.0, 1.0]
        )
        for success, total in zip(success_totals, attempt_totals)
    ]
    hp_half_width = (
        1.959963984540054 * hp_margins.std(ddof=1) / math.sqrt(n_rollouts)
        if n_rollouts > 1
        else 0.0
    )

    return {
        "n_rollouts": n_rollouts,
        "win_rate": float(wins[:, 0].mean()),
        "win_rate_ci95": wilson_interval(ally_wins, n_rollouts),
        "episode_length": {
            "mean": float(lengths.mean()),
            "median": float(np.median(lengths)),
            "p25": float(np.quantile(lengths, 0.25)),
            "p75": float(np.quantile(lengths, 0.75)),
        },
        "truncation_rate": float(truncations.mean()),
        "hp_margin": {
            "mean": float(hp_margins.mean()),
            "std": float(hp_margins.std()),
            "ci95": [
                float(hp_margins.mean() - hp_half_width),
                float(hp_margins.mean() + hp_half_width),
            ],
        },
        "first_casualty_team_rate": first_casualties.mean(axis=0).tolist(),
        "simultaneous_first_casualty_rate": float(
            np.all(first_casualties > 0.5, axis=1).mean()
        ),
        "no_casualty_rate": float(np.all(first_casualties < 0.5, axis=1).mean()),
        # A kill is credited to the team opposite the first casualty. A
        # simultaneous casualty credits neither team instead of both teams.
        "first_kill_credit_rate": np.where(
            np.all(first_casualties > 0.5, axis=1, keepdims=True),
            0.0,
            first_casualties[:, ::-1],
        ).mean(axis=0).tolist(),
        # Deprecated compatibility alias. Consumers should migrate to the two
        # explicitly named metrics above.
        "first_kill_rate": np.where(
            np.all(first_casualties > 0.5, axis=1, keepdims=True),
            0.0,
            first_casualties[:, ::-1],
        ).mean(axis=0).tolist(),
        "attack_success_rate": team_rate(successes, attempts),
        "attack_success_rate_ci95": attack_success_ci95,
        "attack_attempts_mean": attempts.mean(axis=0).tolist(),
        "damage_mean": damage.mean(axis=0).tolist(),
        "damage_mean_ci95": team_mean_ci95(damage),
        "damage_share": damage_share.tolist(),
        "healing_mean": healing.mean(axis=0).tolist(),
        "healing_mean_ci95": team_mean_ci95(healing),
        "first_interaction_step": {
            "mean": (
                float(observed_first_interaction.mean())
                if len(observed_first_interaction)
                else None
            ),
            "rate": float((first_interaction_steps >= 0).mean()),
        },
        "seed_win_std": float(wins[:, 0].std()),
        "quality_flags": {
            "no_interaction_rate": float(no_interaction.mean()),
            "short_episode_rate": float((lengths <= short_cutoff).mean()),
            "all_truncated": bool(np.all(truncations > 0.5)),
            "all_one_sided": bool(np.all(wins[:, 0] == wins[0, 0])),
        },
    }


def episode_metrics_to_samples(metrics: EpisodeMetrics) -> dict[str, list[Any]]:
    """Convert rollout metrics to mergeable per-episode samples."""

    return {
        field: np.asarray(getattr(metrics, field)).tolist()
        for field in EpisodeMetrics._fields
    }


def merge_episode_samples(
    chunks: Sequence[dict[str, Sequence[Any]]],
) -> dict[str, list[Any]]:
    """Merge disjoint seed chunks without re-running earlier episodes."""

    merged = {field: [] for field in EpisodeMetrics._fields}
    for chunk in chunks:
        for field in EpisodeMetrics._fields:
            merged[field].extend(chunk[field])
    return merged


def aggregate_episode_samples(
    samples: dict[str, Sequence[Any]], max_episode_steps: int
) -> dict[str, Any]:
    """Aggregate samples produced by :func:`episode_metrics_to_samples`."""

    metrics = EpisodeMetrics(
        *(jnp.asarray(samples[field]) for field in EpisodeMetrics._fields)
    )
    return _aggregate_episode_metrics(metrics, max_episode_steps)


def _with_team_heuristics(
    task_params: dict[str, Any],
    ally_params: TABXHeuristicParam,
    enemy_params: TABXHeuristicParam,
) -> dict[str, Any]:
    updated = dict(task_params)
    updated["ally_heuristic_params"] = ally_params
    updated["enemy_heuristic_params"] = enemy_params
    # Keep shared field for compatibility with older callers / physics code paths.
    updated["heuristic_params"] = ally_params
    return updated


def _evaluate_tasks_serial(
    tasks: Sequence[dict[str, Any]],
    *,
    physics: str,
    heuristic: str,
    num_seeds: int,
    seed: int,
    epsilon_override: float | None,
    max_episode_steps: int,
    ally_heuristic: str | None = None,
    enemy_heuristic: str | None = None,
    max_n_ally: int | None = None,
    max_n_enemy: int | None = None,
    max_n_zone: int | None = None,
    include_episode_samples: bool = False,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate each task using common random seeds and both-team heuristics."""

    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive.")
    ally_name = ally_heuristic or heuristic
    enemy_name = enemy_heuristic or heuristic
    env_params, tabx_config = build_batched_env_params_from_tasks(
        tasks,
        physics=physics,
        heuristic=ally_name,
        max_n_ally=max_n_ally,
        max_n_enemy=max_n_enemy,
        max_n_zone=max_n_zone,
    )
    ally_params = load_heuristic_params(ally_name, epsilon_override=epsilon_override)
    enemy_params = load_heuristic_params(enemy_name, epsilon_override=epsilon_override)

    run_episodes = _cached_episode_runner(
        tabx_config.max_n_ally,
        tabx_config.max_n_enemy,
        tabx_config.max_n_zone,
        max_episode_steps,
    )
    seeds = np.arange(seed, seed + num_seeds, dtype=np.uint32)
    keys = jax.vmap(jax.random.key)(jnp.asarray(seeds))
    ids = task_ids(tasks)
    results = []
    for index, task_id in enumerate(ids):
        if show_progress:
            suffix = " (initial JIT compilation may take a while)" if index == 0 else ""
            print(
                f"[{index + 1}/{len(ids)}] evaluating {task_id}{suffix}",
                flush=True,
            )
        task_params = jax.tree.map(lambda value: value[index], env_params)
        task_params = _with_team_heuristics(task_params, ally_params, enemy_params)
        metrics = run_episodes(keys, task_params)
        result = {
            "task_id": task_id,
            "ally_heuristic": ally_name,
            "enemy_heuristic": enemy_name,
        }
        result.update(_aggregate_episode_metrics(metrics, max_episode_steps))
        if include_episode_samples:
            result["episode_samples"] = episode_metrics_to_samples(metrics)
        results.append(result)
    return results


def _evaluate_task_chunk_worker(job: _EvalChunkJob) -> tuple[int, list[dict[str, Any]]]:
    """Evaluate one task chunk in a CPU-only worker process."""

    jax.config.update("jax_platform_name", "cpu")
    with redirect_stdout(io.StringIO()):
        results = _evaluate_tasks_serial(
            job.tasks,
            physics=job.physics,
            heuristic=job.heuristic,
            num_seeds=job.num_seeds,
            seed=job.seed,
            epsilon_override=job.epsilon_override,
            max_episode_steps=job.max_episode_steps,
            ally_heuristic=job.ally_heuristic,
            enemy_heuristic=job.enemy_heuristic,
            max_n_ally=job.max_n_ally,
            max_n_enemy=job.max_n_enemy,
            max_n_zone=job.max_n_zone,
            include_episode_samples=job.include_episode_samples,
        )
    return job.chunk_id, results


def evaluate_tasks(
    tasks: Sequence[dict[str, Any]],
    *,
    physics: str,
    heuristic: str,
    num_seeds: int,
    seed: int,
    epsilon_override: float | None,
    max_episode_steps: int,
    ally_heuristic: str | None = None,
    enemy_heuristic: str | None = None,
    max_n_ally: int | None = None,
    max_n_enemy: int | None = None,
    max_n_zone: int | None = None,
    include_episode_samples: bool = False,
    show_progress: bool = False,
    cpu_cores: int = 1,
) -> list[dict[str, Any]]:
    """Evaluate tasks serially or in CPU worker chunks."""

    if cpu_cores <= 0:
        raise ValueError("cpu_cores must be positive.")
    if cpu_cores == 1 or len(tasks) <= 1:
        return _evaluate_tasks_serial(
            tasks,
            physics=physics,
            heuristic=heuristic,
            num_seeds=num_seeds,
            seed=seed,
            epsilon_override=epsilon_override,
            max_episode_steps=max_episode_steps,
            ally_heuristic=ally_heuristic,
            enemy_heuristic=enemy_heuristic,
            max_n_ally=max_n_ally,
            max_n_enemy=max_n_enemy,
            max_n_zone=max_n_zone,
            include_episode_samples=include_episode_samples,
            show_progress=show_progress,
        )

    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    if cpu_cores > available:
        raise ValueError(
            f"Requested {cpu_cores} CPU cores, but only {available} are available."
        )
    worker_count = min(cpu_cores, len(tasks))
    inferred_ally, inferred_enemy, inferred_zone = infer_task_limits(tasks)
    common_limits = (
        max_n_ally if max_n_ally is not None else inferred_ally,
        max_n_enemy if max_n_enemy is not None else inferred_enemy,
        max_n_zone if max_n_zone is not None else inferred_zone,
    )
    base_size, remainder = divmod(len(tasks), worker_count)
    jobs: list[_EvalChunkJob] = []
    start = 0
    for chunk_id in range(worker_count):
        size = base_size + int(chunk_id < remainder)
        chunk = tuple(tasks[start : start + size])
        start += size
        jobs.append(
            _EvalChunkJob(
                chunk_id=chunk_id,
                tasks=chunk,
                physics=physics,
                heuristic=heuristic,
                num_seeds=num_seeds,
                seed=seed,
                epsilon_override=epsilon_override,
                max_episode_steps=max_episode_steps,
                ally_heuristic=ally_heuristic,
                enemy_heuristic=enemy_heuristic,
                max_n_ally=common_limits[0],
                max_n_enemy=common_limits[1],
                max_n_zone=common_limits[2],
                include_episode_samples=include_episode_samples,
            )
        )

    managed_environment = {
        "JAX_PLATFORMS": "cpu",
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_SKIP_CUDA_CONSTRAINTS_CHECK": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    }
    previous_environment = {name: os.environ.get(name) for name in managed_environment}
    os.environ.update(managed_environment)
    print(
        f"Parallel evaluation: workers={worker_count}, "
        f"task_chunks={[len(job.tasks) for job in jobs]}",
        flush=True,
    )
    chunk_results: dict[int, list[dict[str, Any]]] = {}
    try:
        context = multiprocessing.get_context("forkserver")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
        ) as executor:
            future_to_job = {
                executor.submit(_evaluate_task_chunk_worker, job): job for job in jobs
            }
            completed_tasks = 0
            for future in as_completed(future_to_job):
                chunk_id, results = future.result()
                chunk_results[chunk_id] = results
                completed_tasks += len(results)
                if show_progress:
                    print(
                        f"evaluated={completed_tasks}/{len(tasks)} "
                        f"completed_chunks={len(chunk_results)}/{worker_count}",
                        flush=True,
                    )
    finally:
        for name, previous in previous_environment.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
    return [
        result
        for chunk_id in range(worker_count)
        for result in chunk_results[chunk_id]
    ]


def classify_task_difficulties(
    tasks: Sequence[dict[str, Any]],
    *,
    physics: str,
    num_seeds: int,
    seed: int,
    epsilon_override: float | None,
    max_episode_steps: int,
    win_rate_min: float = 0.4,
    win_rate_max: float = 0.6,
    baseline_heuristic: str = "expert",
    downgrade_heuristics: Sequence[str] = ("advanced", "medium"),
    flip_mode: str = "ignore",
    max_abs_side_bias: float = 0.20,
) -> list[dict[str, Any]]:
    """Classify each task by weakening the stronger side until win rate balances.

    Protocol (ally = A, enemy = B):
    1. Both ``baseline_heuristic`` (default expert). Balanced → ``equal``.
    2. If A win rate > max: downgrade A through ``downgrade_heuristics``.
       First level that balances → ``easy``; second → ``very_easy``.
    3. If A win rate < min: downgrade B through the same ladder.
       First level → ``hard``; second → ``very_hard``.
    4. Otherwise → ``invalid`` (not considered a usable task).
    """

    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive.")
    if not 0.0 <= win_rate_min <= win_rate_max <= 1.0:
        raise ValueError("Require 0 <= win_rate_min <= win_rate_max <= 1.")
    if not downgrade_heuristics:
        raise ValueError("downgrade_heuristics must not be empty.")
    if len(downgrade_heuristics) < 2:
        raise ValueError(
            "Need at least two downgrade heuristics "
            "(e.g. advanced, medium) for easy/very_easy and hard/very_hard."
        )
    if flip_mode not in {"ignore", "require_consistent"}:
        raise ValueError("flip_mode must be 'ignore' or 'require_consistent'.")
    if max_abs_side_bias < 0:
        raise ValueError("max_abs_side_bias must be non-negative.")

    mild_downgrade, strong_downgrade = downgrade_heuristics[0], downgrade_heuristics[1]
    easy_label_by_downgrade = {
        mild_downgrade: DIFFICULTY_EASY,
        strong_downgrade: DIFFICULTY_VERY_EASY,
    }
    hard_label_by_downgrade = {
        mild_downgrade: DIFFICULTY_HARD,
        strong_downgrade: DIFFICULTY_VERY_HARD,
    }

    env_params, tabx_config = build_batched_env_params_from_tasks(
        tasks, physics=physics, heuristic=baseline_heuristic
    )
    run_episodes = _cached_episode_runner(
        tabx_config.max_n_ally,
        tabx_config.max_n_enemy,
        tabx_config.max_n_zone,
        max_episode_steps,
    )
    seeds = np.arange(seed, seed + num_seeds, dtype=np.uint32)
    keys = jax.vmap(jax.random.key)(jnp.asarray(seeds))
    ids = task_ids(tasks)

    preset_cache: dict[str, TABXHeuristicParam] = {}

    def heuristic_params(name: str) -> TABXHeuristicParam:
        if name not in preset_cache:
            preset_cache[name] = load_heuristic_params(
                name, epsilon_override=epsilon_override
            )
        return preset_cache[name]

    def run_pair(
        index: int,
        task_id: str,
        ally_name: str,
        enemy_name: str,
    ) -> dict[str, Any]:
        task_params = jax.tree.map(lambda value: value[index], env_params)
        task_params = _with_team_heuristics(
            task_params,
            heuristic_params(ally_name),
            heuristic_params(enemy_name),
        )
        metrics = run_episodes(keys, task_params)
        result = {
            "task_id": task_id,
            "ally_heuristic": ally_name,
            "enemy_heuristic": enemy_name,
        }
        result.update(_aggregate_episode_metrics(metrics, max_episode_steps))
        return result

    results: list[dict[str, Any]] = []
    for index, task_id in enumerate(ids):
        ladder_steps: list[dict[str, Any]] = []
        print(
            f"[{index + 1}/{len(ids)}] {task_id}: "
            f"baseline {baseline_heuristic} vs {baseline_heuristic} ..."
        )
        baseline = run_pair(index, task_id, baseline_heuristic, baseline_heuristic)
        ladder_steps.append(
            {
                "ally_heuristic": baseline_heuristic,
                "enemy_heuristic": baseline_heuristic,
                "win_rate": baseline["win_rate"],
                "win_rate_ci95": baseline["win_rate_ci95"],
                "truncation_rate": baseline["truncation_rate"],
            }
        )
        win_rate = float(baseline["win_rate"])
        difficulty = DIFFICULTY_INVALID
        balancing = None
        valid = False

        if _is_balanced(win_rate, win_rate_min, win_rate_max):
            difficulty = DIFFICULTY_EQUAL
            balancing = {
                "ally_heuristic": baseline_heuristic,
                "enemy_heuristic": baseline_heuristic,
                "win_rate": win_rate,
            }
            valid = True
        elif win_rate > win_rate_max:
            # A too strong → weaken A.
            for downgrade in (mild_downgrade, strong_downgrade):
                print(
                    f"  ally wins ({win_rate:.3f}) > {win_rate_max:.2f}; "
                    f"try ally={downgrade} vs enemy={baseline_heuristic} ..."
                )
                trial = run_pair(index, task_id, downgrade, baseline_heuristic)
                trial_wr = float(trial["win_rate"])
                ladder_steps.append(
                    {
                        "ally_heuristic": downgrade,
                        "enemy_heuristic": baseline_heuristic,
                        "win_rate": trial_wr,
                        "win_rate_ci95": trial["win_rate_ci95"],
                        "truncation_rate": trial["truncation_rate"],
                    }
                )
                if _is_balanced(trial_wr, win_rate_min, win_rate_max):
                    difficulty = easy_label_by_downgrade[downgrade]
                    balancing = {
                        "ally_heuristic": downgrade,
                        "enemy_heuristic": baseline_heuristic,
                        "win_rate": trial_wr,
                    }
                    valid = True
                    baseline = trial
                    break
        else:
            # A too weak → weaken B.
            for downgrade in (mild_downgrade, strong_downgrade):
                print(
                    f"  ally loses ({win_rate:.3f}) < {win_rate_min:.2f}; "
                    f"try ally={baseline_heuristic} vs enemy={downgrade} ..."
                )
                trial = run_pair(index, task_id, baseline_heuristic, downgrade)
                trial_wr = float(trial["win_rate"])
                ladder_steps.append(
                    {
                        "ally_heuristic": baseline_heuristic,
                        "enemy_heuristic": downgrade,
                        "win_rate": trial_wr,
                        "win_rate_ci95": trial["win_rate_ci95"],
                        "truncation_rate": trial["truncation_rate"],
                    }
                )
                if _is_balanced(trial_wr, win_rate_min, win_rate_max):
                    difficulty = hard_label_by_downgrade[downgrade]
                    balancing = {
                        "ally_heuristic": baseline_heuristic,
                        "enemy_heuristic": downgrade,
                        "win_rate": trial_wr,
                    }
                    valid = True
                    baseline = trial
                    break

        proposed_difficulty = difficulty
        flip_validation = None
        invalid_reason = None
        if valid and flip_mode == "require_consistent" and balancing is not None:
            flipped_result = evaluate_tasks(
                [flip_task(tasks[index])],
                physics=physics,
                heuristic=baseline_heuristic,
                ally_heuristic=str(balancing["enemy_heuristic"]),
                enemy_heuristic=str(balancing["ally_heuristic"]),
                num_seeds=num_seeds,
                seed=seed,
                epsilon_override=epsilon_override,
                max_episode_steps=max_episode_steps,
                max_n_ally=max(tabx_config.max_n_ally, tabx_config.max_n_enemy),
                max_n_enemy=max(tabx_config.max_n_ally, tabx_config.max_n_enemy),
                max_n_zone=tabx_config.max_n_zone,
            )[0]
            original_wr = float(baseline["win_rate"])
            flipped_wr = float(flipped_result["win_rate"])
            side_bias = original_wr + flipped_wr - 1.0
            team_invariant_wr = (original_wr + (1.0 - flipped_wr)) * 0.5
            flip_balanced = _is_balanced(
                flipped_wr, win_rate_min, win_rate_max
            )
            side_consistent = abs(side_bias) <= max_abs_side_bias
            flip_validation = {
                "mode": flip_mode,
                "accepted": flip_balanced and side_consistent,
                "flipped": flipped_result,
                "side_bias": side_bias,
                "team_invariant_win_rate": team_invariant_wr,
                "flipped_balanced": flip_balanced,
                "side_consistent": side_consistent,
            }
            if not flip_validation["accepted"]:
                valid = False
                difficulty = DIFFICULTY_INVALID
                invalid_reason = (
                    "flipped_matchup_out_of_band"
                    if not flip_balanced
                    else "side_bias"
                )

        result = dict(baseline)
        result.update(
            {
                "task_id": task_id,
                "difficulty": difficulty,
                "valid": valid,
                "baseline_win_rate": float(ladder_steps[0]["win_rate"]),
                "balancing_matchup": balancing,
                "ladder": ladder_steps,
                "proposed_difficulty": proposed_difficulty,
                "flip_validation": flip_validation,
                "invalid_reason": invalid_reason,
            }
        )
        status = "VALID" if valid else "INVALID"
        bal_wr = balancing["win_rate"] if balancing else float("nan")
        print(
            f"  → {status} difficulty={difficulty} "
            f"baseline_wr={ladder_steps[0]['win_rate']:.3f} "
            f"balanced_wr={bal_wr:.3f}"
        )
        results.append(result)
    return results


def flip_task(task: dict[str, Any]) -> dict[str, Any]:
    """Swap team identities while preserving each unit's physical placement."""

    flipped = copy.deepcopy(task)
    scenario = flipped["scenario"]
    teams = np.asarray(scenario["teams"]).reshape(-1)
    order = np.concatenate([np.flatnonzero(teams == 1), np.flatnonzero(teams == 0)])
    for field in scenario:
        values = np.asarray(scenario[field])
        scenario[field] = values[order].tolist()
    scenario["teams"] = (1 - np.asarray(scenario["teams"], dtype=int)).tolist()
    flipped["task_id"] = f"{task.get('task_id', 'task')}__flipped"
    metadata = dict(flipped.get("metadata", {}))
    metadata["flipped_from"] = task.get("task_id")
    flipped["metadata"] = metadata
    return flipped


def _write_results(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    return output_path


def assess_balance_filter(
    result: dict[str, Any],
    *,
    win_rate_min: float,
    win_rate_max: float,
    max_abs_side_bias: float,
) -> dict[str, Any]:
    """Assess whether an original/flipped evaluation is safe to retain."""

    flipped = result.get("flipped")
    if not isinstance(flipped, dict):
        raise ValueError(
            f"Task {result.get('task_id', '<unknown>')} has no flipped evaluation."
        )

    original_win_rate = float(result["win_rate"])
    flipped_win_rate = float(flipped["win_rate"])
    side_bias = original_win_rate + flipped_win_rate - 1.0
    team_invariant_win_rate = (
        original_win_rate + (1.0 - flipped_win_rate)
    ) * 0.5
    reasons = []
    if not _is_balanced(original_win_rate, win_rate_min, win_rate_max):
        reasons.append("original_win_rate_out_of_band")
    if not _is_balanced(flipped_win_rate, win_rate_min, win_rate_max):
        reasons.append("flipped_win_rate_out_of_band")
    if abs(side_bias) > max_abs_side_bias:
        reasons.append("side_bias")

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "original_win_rate": original_win_rate,
        "flipped_win_rate": flipped_win_rate,
        "team_invariant_win_rate": team_invariant_win_rate,
        "side_bias": side_bias,
    }


def _build_task_bank_subset(
    bank: dict[str, Any],
    kept_tasks: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Copy a task bank and replace its tasks with a filtered subset."""

    filtered_bank = copy.deepcopy(bank)
    filtered_bank["tasks"] = copy.deepcopy(list(kept_tasks))
    filtered_manifest = dict(filtered_bank.get("manifest", {}))
    filtered_manifest["n_tasks"] = len(kept_tasks)
    filtered_manifest["evaluation_balance_filter"] = copy.deepcopy(summary)
    filtered_bank["manifest"] = filtered_manifest
    return filtered_bank


def build_filtered_task_bank(
    bank: dict[str, Any],
    results: Sequence[dict[str, Any]],
    *,
    win_rate_min: float,
    win_rate_max: float,
    max_abs_side_bias: float,
    evaluation_file: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a task bank with evaluations outside the balance band removed."""

    assessments: dict[str, dict[str, Any]] = {}
    for result in results:
        task_id = str(result["task_id"])
        if task_id in assessments:
            raise ValueError(f"Duplicate evaluation result for task {task_id}.")
        assessment = assess_balance_filter(
            result,
            win_rate_min=win_rate_min,
            win_rate_max=win_rate_max,
            max_abs_side_bias=max_abs_side_bias,
        )
        result["balance_filter"] = assessment
        assessments[task_id] = assessment

    tasks = bank["tasks"]
    task_id_list = [str(task["task_id"]) for task in tasks]
    missing = [task_id for task_id in task_id_list if task_id not in assessments]
    extra = sorted(set(assessments).difference(task_id_list))
    if missing or extra:
        raise ValueError(
            "Evaluation/task-bank mismatch while filtering: "
            f"missing_results={missing}, extra_results={extra}."
        )

    kept_tasks = [
        task
        for task in tasks
        if assessments[str(task["task_id"])]["accepted"]
    ]
    removed_task_ids = [
        task_id for task_id in task_id_list if not assessments[task_id]["accepted"]
    ]
    reason_counts = Counter(
        reason
        for task_id in removed_task_ids
        for reason in assessments[task_id]["reasons"]
    )
    summary = {
        "enabled": True,
        "criteria": {
            "win_rate_min": win_rate_min,
            "win_rate_max": win_rate_max,
            "max_abs_side_bias": max_abs_side_bias,
            "require_original_in_band": True,
            "require_flipped_in_band": True,
        },
        "evaluation_file": str(Path(evaluation_file)),
        "n_input_tasks": len(tasks),
        "n_kept_tasks": len(kept_tasks),
        "n_removed_tasks": len(removed_task_ids),
        "removed_task_ids": removed_task_ids,
        "reason_counts": dict(reason_counts),
    }

    filtered_bank = _build_task_bank_subset(bank, kept_tasks, summary)
    return filtered_bank, summary


def evaluate_fixed_rounds_then_filter(
    tasks: Sequence[dict[str, Any]],
    *,
    physics: str,
    heuristic: str,
    num_seeds: int,
    seed: int,
    epsilon_override: float | None,
    max_episode_steps: int,
    win_rate_min: float,
    win_rate_max: float,
    max_abs_side_bias: float,
    cpu_cores: int,
    rounds: int = 3,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Aggregate fixed fresh-seed rounds and apply one final balance filter."""

    if not tasks:
        raise ValueError("Cannot filter an empty task bank.")
    if rounds <= 0:
        raise ValueError("rounds must be positive.")

    evaluation_tasks = list(tasks)
    evaluation_tasks.extend(flip_task(task) for task in tasks)
    sample_chunks: dict[str, list[dict[str, Sequence[Any]]]] = {}
    round_history: list[dict[str, Any]] = []
    round_seed_ranges = []

    for round_index in range(1, rounds + 1):
        round_seed = seed + (round_index - 1) * num_seeds
        round_seed_ranges.append(
            {
                "round": round_index,
                "seed_start": round_seed,
                "seed_end_exclusive": round_seed + num_seeds,
            }
        )
        print(
            f"Balance-evaluation round {round_index}/{rounds}: evaluating "
            f"{len(tasks)} tasks with seeds "
            f"[{round_seed}, {round_seed + num_seeds}).",
            flush=True,
        )
        combined_results = evaluate_tasks(
            evaluation_tasks,
            physics=physics,
            heuristic=heuristic,
            num_seeds=num_seeds,
            seed=round_seed,
            epsilon_override=epsilon_override,
            max_episode_steps=max_episode_steps,
            include_episode_samples=True,
            show_progress=True,
            cpu_cores=cpu_cores,
        )
        round_results = combined_results[: len(tasks)]
        flipped_results = combined_results[len(tasks) :]
        for result in combined_results:
            task_id = str(result["task_id"])
            samples = result.pop("episode_samples")
            sample_chunks.setdefault(task_id, []).append(samples)

        round_out_of_band = 0
        for original, flipped in zip(round_results, flipped_results):
            original["flipped"] = flipped
            original["side_bias"] = (
                float(original["win_rate"]) + float(flipped["win_rate"]) - 1.0
            )
            preview = assess_balance_filter(
                original,
                win_rate_min=win_rate_min,
                win_rate_max=win_rate_max,
                max_abs_side_bias=max_abs_side_bias,
            )
            original["round_balance_preview"] = preview
            round_out_of_band += int(not preview["accepted"])

        round_record = {
            "round": round_index,
            "seed_start": round_seed,
            "seed_end_exclusive": round_seed + num_seeds,
            "n_tasks": len(tasks),
            "n_out_of_band_preview": round_out_of_band,
            "results": round_results,
        }
        round_history.append(round_record)
        print(
            f"Balance-evaluation round {round_index}/{rounds}: "
            f"out_of_band_preview={round_out_of_band}; no tasks removed.",
            flush=True,
        )

    cumulative_results: list[dict[str, Any]] = []
    kept_tasks: list[dict[str, Any]] = []
    removed_task_ids: list[str] = []
    reason_counts: Counter[str] = Counter()
    for task in tasks:
        task_id = str(task["task_id"])
        flipped_task_id = f"{task_id}__flipped"
        original = {
            "task_id": task_id,
            "ally_heuristic": heuristic,
            "enemy_heuristic": heuristic,
        }
        original.update(
            aggregate_episode_samples(
                merge_episode_samples(sample_chunks[task_id]),
                max_episode_steps,
            )
        )
        flipped = {
            "task_id": flipped_task_id,
            "ally_heuristic": heuristic,
            "enemy_heuristic": heuristic,
        }
        flipped.update(
            aggregate_episode_samples(
                merge_episode_samples(sample_chunks[flipped_task_id]),
                max_episode_steps,
            )
        )
        original["flipped"] = flipped
        original["side_bias"] = (
            float(original["win_rate"]) + float(flipped["win_rate"]) - 1.0
        )
        assessment = assess_balance_filter(
            original,
            win_rate_min=win_rate_min,
            win_rate_max=win_rate_max,
            max_abs_side_bias=max_abs_side_bias,
        )
        original["balance_filter"] = assessment
        cumulative_results.append(original)
        if assessment["accepted"]:
            kept_tasks.append(task)
        else:
            removed_task_ids.append(task_id)
            reason_counts.update(assessment["reasons"])

    summary = {
        "enabled": True,
        "mode": "fixed_rounds_cumulative_filter",
        "criteria": {
            "win_rate_min": win_rate_min,
            "win_rate_max": win_rate_max,
            "max_abs_side_bias": max_abs_side_bias,
            "require_original_in_band": True,
            "require_flipped_in_band": True,
        },
        "n_rounds": rounds,
        "num_seeds_per_round": num_seeds,
        "n_rollouts_per_orientation": rounds * num_seeds,
        "round_seed_ranges": round_seed_ranges,
        "n_input_tasks": len(tasks),
        "n_kept_tasks": len(kept_tasks),
        "n_removed_tasks": len(removed_task_ids),
        "removed_task_ids": removed_task_ids,
        "reason_counts": dict(reason_counts),
    }
    return kept_tasks, cumulative_results, round_history, summary


def main() -> None:
    config = tyro.cli(EvalConfig)
    if config.compute_backend not in {"cpu", "gpu"}:
        raise ValueError("compute_backend must be 'cpu' or 'gpu'.")
    if config.cpu_cores <= 0:
        raise ValueError("cpu_cores must be positive.")
    if config.compute_backend == "gpu" and config.cpu_cores != 1:
        raise ValueError("GPU evaluation requires cpu_cores=1.")
    if config.classify_difficulty and config.cpu_cores != 1:
        raise ValueError(
            "Parallel CPU evaluation currently requires --no-classify-difficulty."
        )
    if not 0.0 <= config.win_rate_min <= config.win_rate_max <= 1.0:
        raise ValueError("win_rate_min/max must satisfy 0 <= min <= max <= 1.")
    if config.filter_max_abs_side_bias < 0:
        raise ValueError("filter_max_abs_side_bias must be non-negative.")
    if config.filter_rounds <= 0:
        raise ValueError("filter_rounds must be positive.")
    if config.filtered_task_output is not None:
        if config.classify_difficulty:
            raise ValueError(
                "Filtered task output requires --no-classify-difficulty."
            )
        if not config.evaluate_flipped:
            raise ValueError(
                "Filtered task output requires --evaluate-flipped."
            )
    print(
        f"Evaluation backend: {jax.default_backend()} "
        f"({len(jax.devices())} visible device(s)).",
        flush=True,
    )
    bank = load_task_bank(config.task_file)
    manifest = bank["manifest"]
    physics = config.physics or manifest["physics"]
    heuristic = config.heuristic or manifest["heuristic"]
    tasks = bank["tasks"]
    flipped_already_evaluated = False
    filtered_tasks = None
    filter_round_history = None
    filter_summary = None

    if config.classify_difficulty:
        results = classify_task_difficulties(
            tasks,
            physics=physics,
            num_seeds=config.num_seeds,
            seed=config.seed,
            epsilon_override=config.epsilon_override,
            max_episode_steps=config.max_episode_steps,
            win_rate_min=config.win_rate_min,
            win_rate_max=config.win_rate_max,
            baseline_heuristic=config.baseline_heuristic,
            downgrade_heuristics=config.downgrade_heuristics,
            flip_mode=config.difficulty_flip_mode,
            max_abs_side_bias=config.difficulty_max_abs_side_bias,
        )
    elif config.filtered_task_output is not None:
        filtered_tasks, results, filter_round_history, filter_summary = (
            evaluate_fixed_rounds_then_filter(
                tasks,
                physics=physics,
                heuristic=heuristic,
                num_seeds=config.num_seeds,
                seed=config.seed,
                epsilon_override=config.epsilon_override,
                max_episode_steps=config.max_episode_steps,
                win_rate_min=config.win_rate_min,
                win_rate_max=config.win_rate_max,
                max_abs_side_bias=config.filter_max_abs_side_bias,
                cpu_cores=config.cpu_cores,
                rounds=config.filter_rounds,
            )
        )
        flipped_already_evaluated = True
    else:
        evaluation_tasks = list(tasks)
        if config.evaluate_flipped:
            evaluation_tasks.extend(flip_task(task) for task in tasks)
        combined_results = evaluate_tasks(
            evaluation_tasks,
            physics=physics,
            heuristic=heuristic,
            num_seeds=config.num_seeds,
            seed=config.seed,
            epsilon_override=config.epsilon_override,
            max_episode_steps=config.max_episode_steps,
            show_progress=True,
            cpu_cores=config.cpu_cores,
        )
        results = combined_results[: len(tasks)]
        if config.evaluate_flipped:
            flipped_results = combined_results[len(tasks) :]
            for original, flipped in zip(results, flipped_results):
                original["flipped"] = flipped
                original["side_bias"] = original["win_rate"] + flipped["win_rate"] - 1.0
            flipped_already_evaluated = True

    if config.evaluate_flipped and not flipped_already_evaluated:
        flipped_results = evaluate_tasks(
            [flip_task(task) for task in tasks],
            physics=physics,
            heuristic=heuristic,
            num_seeds=config.num_seeds,
            seed=config.seed,
            epsilon_override=config.epsilon_override,
            max_episode_steps=config.max_episode_steps,
            show_progress=True,
            cpu_cores=config.cpu_cores,
        )
        for original, flipped in zip(results, flipped_results):
            original["flipped"] = flipped
            original["side_bias"] = original["win_rate"] + flipped["win_rate"] - 1.0

    protocol: dict[str, Any] = {
        "physics": physics,
        "heuristic": heuristic,
        "epsilon_override": config.epsilon_override,
        "seed_start": config.seed,
        "num_seeds": config.num_seeds,
        "max_episode_steps": config.max_episode_steps,
        "evaluate_flipped": config.evaluate_flipped,
        "classify_difficulty": config.classify_difficulty,
        "compute_backend": config.compute_backend,
        "cpu_cores": config.cpu_cores,
        "difficulty_flip_mode": config.difficulty_flip_mode,
        "difficulty_max_abs_side_bias": config.difficulty_max_abs_side_bias,
        "filtered_task_output": config.filtered_task_output,
        "filter_max_abs_side_bias": config.filter_max_abs_side_bias,
        "filter_mode": (
            "fixed_rounds_cumulative_filter"
            if config.filtered_task_output is not None
            else None
        ),
        "filter_rounds": config.filter_rounds,
    }
    if config.classify_difficulty:
        protocol.update(
            {
                "win_rate_min": config.win_rate_min,
                "win_rate_max": config.win_rate_max,
                "baseline_heuristic": config.baseline_heuristic,
                "downgrade_heuristics": list(config.downgrade_heuristics),
                "difficulty_labels": {
                    "equal": f"both {config.baseline_heuristic}",
                    "easy": f"ally={config.downgrade_heuristics[0]}, enemy={config.baseline_heuristic}",
                    "very_easy": (
                        f"ally={config.downgrade_heuristics[1]}, "
                        f"enemy={config.baseline_heuristic}"
                    ),
                    "hard": f"ally={config.baseline_heuristic}, enemy={config.downgrade_heuristics[0]}",
                    "very_hard": (
                        f"ally={config.baseline_heuristic}, "
                        f"enemy={config.downgrade_heuristics[1]}"
                    ),
                    "invalid": "no balanced matchup on the ladder",
                },
            }
        )

    payload = {
        "schema_version": "1.1",
        "task_file": str(Path(config.task_file)),
        "protocol": protocol,
        "results": results,
    }
    filtered_bank = None
    if filtered_tasks is not None and filter_summary is not None:
        filter_summary["evaluation_file"] = str(Path(config.output))
        filtered_bank = _build_task_bank_subset(
            bank, filtered_tasks, filter_summary
        )
        payload["balance_filter_summary"] = filter_summary
        payload["balance_filter_rounds"] = filter_round_history
    if config.classify_difficulty:
        difficulty_counts = Counter(result["difficulty"] for result in results)
        payload["summary"] = {
            "n_tasks": len(results),
            "n_valid": sum(1 for result in results if result["valid"]),
            "n_invalid": sum(1 for result in results if not result["valid"]),
            "difficulty_counts": dict(difficulty_counts),
        }

    output = _write_results(config.output, payload)
    filtered_output = None
    if filtered_bank is not None:
        filtered_output = _write_results(config.filtered_task_output, filtered_bank)
    if config.classify_difficulty:
        summary = payload["summary"]
        print(
            f"Classified {summary['n_tasks']} tasks × {config.num_seeds} seeds; "
            f"valid={summary['n_valid']} invalid={summary['n_invalid']}; "
            f"counts={summary['difficulty_counts']}. Results: {output}"
        )
    else:
        mean_win_rate = float(np.mean([result["win_rate"] for result in results]))
        print(
            f"Evaluated {len(results)} tasks × {config.num_seeds} seeds; "
            f"mean ally win rate={mean_win_rate:.3f}. Results: {output}"
        )
    if filtered_output is not None:
        summary = payload["balance_filter_summary"]
        print(
            f"Balance filter kept {summary['n_kept_tasks']}/"
            f"{summary['n_input_tasks']} tasks and removed "
            f"{summary['n_removed_tasks']}. Filtered task bank: "
            f"{filtered_output}"
        )


if __name__ == "__main__":
    main()
