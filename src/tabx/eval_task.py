"""Evaluate a fixed TABX task bank with heuristic agents on both teams."""

from __future__ import annotations

import copy
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from src.tabx.heuristic_policy import LastVisibleTarget, heuristic_policy
from src.tabx.heuristic_policy.params import TABXHeuristicParam
from src.tabx.heuristic_policy.utils import load_heuristic_params_from_json
from src.tabx.sample_task import (
    build_batched_env_params_from_tasks,
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
    # None keeps each preset's native epsilon; set to force a shared value.
    epsilon_override: float | None = None
    max_episode_steps: int = 512
    evaluate_flipped: bool = False
    # Ladder classification against a balanced win-rate band.
    classify_difficulty: bool = True
    win_rate_min: float = 0.4
    win_rate_max: float = 0.6
    baseline_heuristic: str = "expert"
    # Applied to the stronger side, in order from milder to stronger downgrade.
    downgrade_heuristics: tuple[str, ...] = ("advanced", "medium")


class EpisodeMetrics(NamedTuple):
    wins: jax.Array
    episode_length: jax.Array
    truncation: jax.Array
    hp_margin: jax.Array
    first_kill: jax.Array
    attack_attempts: jax.Array
    attack_successes: jax.Array
    damage: jax.Array
    healing: jax.Array


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
                first_kill,
                cumulative_attempts,
                cumulative_successes,
                cumulative_damage,
                cumulative_healing,
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
            first_kill_seen = first_kill.sum() > 0
            new_first_kill = dead_units[::-1].astype(jnp.float32)
            first_kill = jnp.where(first_kill_seen, first_kill, new_first_kill)

            return (
                next_rng,
                next_obs,
                next_state,
                next_targets,
                done,
                steps + 1,
                info["is_win"],
                info["truncation"][0],
                first_kill,
                cumulative_attempts + attempts,
                cumulative_successes + successes,
                cumulative_damage + damage,
                cumulative_healing + healing,
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
            first_kill,
            attempts,
            successes,
            damage,
            healing,
        ) = jax.lax.while_loop(condition, body, carry)
        hp_ratio = final_state["state"]["game_manager"].team_hp_ratio
        return EpisodeMetrics(
            wins=wins,
            episode_length=steps,
            truncation=truncation,
            hp_margin=hp_ratio[0] - hp_ratio[1],
            first_kill=first_kill,
            attack_attempts=attempts,
            attack_successes=successes,
            damage=damage,
            healing=healing,
        )

    return jax.jit(jax.vmap(run_episode, in_axes=(0, None)))


def _aggregate_episode_metrics(metrics: EpisodeMetrics, max_episode_steps: int) -> dict[str, Any]:
    wins = np.asarray(metrics.wins)
    lengths = np.asarray(metrics.episode_length)
    truncations = np.asarray(metrics.truncation, dtype=float)
    hp_margins = np.asarray(metrics.hp_margin)
    first_kills = np.asarray(metrics.first_kill)
    attempts = np.asarray(metrics.attack_attempts)
    successes = np.asarray(metrics.attack_successes)
    damage = np.asarray(metrics.damage)
    healing = np.asarray(metrics.healing)

    n_rollouts = len(lengths)
    ally_wins = int(np.rint(wins[:, 0].sum()))
    total_interaction = damage.sum(axis=1) + healing.sum(axis=1)
    no_interaction = total_interaction <= 1e-6
    short_cutoff = max(5, int(max_episode_steps * 0.05))

    def team_rate(numerator: np.ndarray, denominator: np.ndarray) -> list[float]:
        total_denominator = denominator.sum(axis=0)
        return np.divide(
            numerator.sum(axis=0),
            total_denominator,
            out=np.zeros(2, dtype=float),
            where=total_denominator > 0,
        ).tolist()

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
        },
        "first_kill_rate": first_kills.mean(axis=0).tolist(),
        "attack_success_rate": team_rate(successes, attempts),
        "attack_attempts_mean": attempts.mean(axis=0).tolist(),
        "damage_mean": damage.mean(axis=0).tolist(),
        "healing_mean": healing.mean(axis=0).tolist(),
        "seed_win_std": float(wins[:, 0].std()),
        "quality_flags": {
            "no_interaction_rate": float(no_interaction.mean()),
            "short_episode_rate": float((lengths <= short_cutoff).mean()),
            "all_truncated": bool(np.all(truncations > 0.5)),
            "all_one_sided": bool(np.all(wins[:, 0] == wins[0, 0])),
        },
    }


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
) -> list[dict[str, Any]]:
    """Evaluate each task using common random seeds and both-team heuristics."""

    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive.")
    ally_name = ally_heuristic or heuristic
    enemy_name = enemy_heuristic or heuristic
    env_params, tabx_config = build_batched_env_params_from_tasks(
        tasks, physics=physics, heuristic=ally_name
    )
    ally_params = load_heuristic_params(ally_name, epsilon_override=epsilon_override)
    enemy_params = load_heuristic_params(enemy_name, epsilon_override=epsilon_override)

    env = TABX(cfg=tabx_config, max_episode_steps=max_episode_steps)
    run_episodes = _make_episode_runner(env)
    seeds = np.arange(seed, seed + num_seeds, dtype=np.uint32)
    keys = jax.vmap(jax.random.key)(jnp.asarray(seeds))
    ids = task_ids(tasks)
    results = []
    for index, task_id in enumerate(ids):
        task_params = jax.tree.map(lambda value: value[index], env_params)
        task_params = _with_team_heuristics(task_params, ally_params, enemy_params)
        metrics = run_episodes(keys, task_params)
        result = {
            "task_id": task_id,
            "ally_heuristic": ally_name,
            "enemy_heuristic": enemy_name,
        }
        result.update(_aggregate_episode_metrics(metrics, max_episode_steps))
        results.append(result)
    return results


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
    env = TABX(cfg=tabx_config, max_episode_steps=max_episode_steps)
    run_episodes = _make_episode_runner(env)
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

        result = dict(baseline)
        result.update(
            {
                "task_id": task_id,
                "difficulty": difficulty,
                "valid": valid,
                "baseline_win_rate": float(ladder_steps[0]["win_rate"]),
                "balancing_matchup": balancing,
                "ladder": ladder_steps,
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


def main() -> None:
    config = tyro.cli(EvalConfig)
    bank = load_task_bank(config.task_file)
    manifest = bank["manifest"]
    physics = config.physics or manifest["physics"]
    heuristic = config.heuristic or manifest["heuristic"]
    tasks = bank["tasks"]

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
        )
    else:
        results = evaluate_tasks(
            tasks,
            physics=physics,
            heuristic=heuristic,
            num_seeds=config.num_seeds,
            seed=config.seed,
            epsilon_override=config.epsilon_override,
            max_episode_steps=config.max_episode_steps,
        )

    if config.evaluate_flipped:
        flipped_results = evaluate_tasks(
            [flip_task(task) for task in tasks],
            physics=physics,
            heuristic=heuristic,
            num_seeds=config.num_seeds,
            seed=config.seed,
            epsilon_override=config.epsilon_override,
            max_episode_steps=config.max_episode_steps,
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
    if config.classify_difficulty:
        difficulty_counts = Counter(result["difficulty"] for result in results)
        payload["summary"] = {
            "n_tasks": len(results),
            "n_valid": sum(1 for result in results if result["valid"]),
            "n_invalid": sum(1 for result in results if not result["valid"]),
            "difficulty_counts": dict(difficulty_counts),
        }

    output = _write_results(config.output, payload)
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


if __name__ == "__main__":
    main()
