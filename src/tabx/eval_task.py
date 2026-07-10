"""Evaluate a fixed TABX task bank with heuristic agents on both teams."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from src.tabx.heuristic_policy import LastVisibleTarget, heuristic_policy
from src.tabx.sample_task import (
    build_batched_env_params_from_tasks,
    load_task_bank,
    task_ids,
)
from src.tabx.tabx import TABX


@dataclass(frozen=True)
class EvalConfig:
    """Command-line configuration for fixed-task heuristic evaluation."""

    task_file: str
    output: str = "task_eval.json"
    heuristic: str | None = None
    physics: str | None = None
    num_seeds: int = 32
    seed: int = 0
    epsilon_override: float | None = 0.0
    max_episode_steps: int = 512
    evaluate_flipped: bool = False


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


def _make_episode_runner(env: TABX):
    """Create one JAX-compatible episode function for a fixed environment schema."""

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
            actions = {}
            next_targets = {}
            for index, unit in enumerate(env.unit_keys):
                actions[unit], next_targets[unit] = heuristic_policy(
                    split_keys[index + 2],
                    current_obs[unit],
                    last_targets[unit],
                    env.num_agents,
                    env.max_n_zone,
                    env_params["heuristic_params"],
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


def evaluate_tasks(
    tasks: Sequence[dict[str, Any]],
    *,
    physics: str,
    heuristic: str,
    num_seeds: int,
    seed: int,
    epsilon_override: float | None,
    max_episode_steps: int,
) -> list[dict[str, Any]]:
    """Evaluate each task using common random seeds and both-team heuristics."""

    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive.")
    env_params, tabx_config = build_batched_env_params_from_tasks(
        tasks, physics=physics, heuristic=heuristic
    )
    if epsilon_override is not None:
        if not 0.0 <= epsilon_override <= 1.0:
            raise ValueError("epsilon_override must be in [0, 1].")
        env_params["heuristic_params"] = env_params["heuristic_params"].replace(
            epsilon=jnp.full_like(env_params["heuristic_params"].epsilon, epsilon_override)
        )

    env = TABX(cfg=tabx_config, max_episode_steps=max_episode_steps)
    run_episodes = _make_episode_runner(env)
    seeds = np.arange(seed, seed + num_seeds, dtype=np.uint32)
    keys = jax.vmap(jax.random.key)(jnp.asarray(seeds))
    ids = task_ids(tasks)
    results = []
    for index, task_id in enumerate(ids):
        task_params = jax.tree.map(lambda value: value[index], env_params)
        metrics = run_episodes(keys, task_params)
        result = {"task_id": task_id}
        result.update(_aggregate_episode_metrics(metrics, max_episode_steps))
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

    payload = {
        "schema_version": "1.0",
        "task_file": str(Path(config.task_file)),
        "protocol": {
            "physics": physics,
            "heuristic": heuristic,
            "epsilon_override": config.epsilon_override,
            "seed_start": config.seed,
            "num_seeds": config.num_seeds,
            "max_episode_steps": config.max_episode_steps,
            "evaluate_flipped": config.evaluate_flipped,
        },
        "results": results,
    }
    output = _write_results(config.output, payload)
    mean_win_rate = float(np.mean([result["win_rate"] for result in results]))
    print(
        f"Evaluated {len(results)} tasks × {config.num_seeds} seeds; "
        f"mean ally win rate={mean_win_rate:.3f}. Results: {output}"
    )


if __name__ == "__main__":
    main()
