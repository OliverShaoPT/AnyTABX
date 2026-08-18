"""Step 1: online env-centric record generation (behavior + reference)."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from generate.behavior_mix import (
    BehaviorSwitcher,
    DEFAULT_BEHAVIOR_MIX,
    BehaviorSpec,
    behavior_mix_from_config,
    policy_id_mapping_json,
    policy_tag_for_spec,
    policy_tag_legend_json,
)
from generate.dump_schema import ensure_dir, write_env_centric_record
from generate.oracle_loader import OracleCoach, load_oracle_coach
from generate.policies import (
    BEST_ADVANCED,
    BEST_ORACLE,
    SharedAllyPolicy,
    reference_labels,
    reference_labels_best_teacher,
)
from generate.task_package import TaskPackage, discover_task_packages, load_package_task_bank
from src.tabx.eval_task import load_heuristic_params
from src.tabx.heuristic_policy import LastVisibleTarget
from src.tabx import TABX
from src.tabx.sample_task import build_batched_env_params_from_tasks
from src.tabx.wrappers.individual_reward import IndividualRewardConfig, compute_shaped_rewards
from src.tabx.wrappers.wrappers import TABXEnemyHeuristicWrapper


@dataclass
class RecordGenContext:
    """Reusable env + oracle for many records of the same task (amortize JAX JIT)."""

    package: TaskPackage
    env: Any
    env_params: Any
    task: dict[str, Any]
    manifest: dict[str, Any]
    oracle: OracleCoach
    ally_keys: list[str]
    unit_keys: list[str]
    n_ally: int
    action_dim: int
    obs_dim: int
    n_units_total: int


def sample_record_seed(explicit: int | None = None, *, salt: int = 0) -> int:
    """Sample a per-record seed from wall-clock time and process id.

    Matches the requested formula::

        seed = int(time.time() * 1000) % (2**32 - 1) + pid

    ``salt`` (typically ``record_id``) avoids collisions when multiple records
    are generated in the same millisecond by one process. Pass ``explicit`` to
    override for reproducible debugging.
    """

    if explicit is not None:
        seed = int(explicit)
    else:
        pid = os.getpid()
        seed = int(time.time() * 1000) % (2**32 - 1) + pid + int(salt)
    # Keep Python/NumPy global RNGs aligned with the sampled seed.
    random.seed(seed)
    np.random.seed(seed % (2**32))
    return seed


def _to_numpy(x: Any) -> np.ndarray:
    return np.asarray(jax.device_get(x))


def _squeeze_env_params(env_params: dict[str, Any]) -> dict[str, Any]:
    """Take index 0 when params are batched with leading env dim."""

    def squeeze_leaf(value):
        arr = jnp.asarray(value)
        if arr.ndim == 0:
            return arr
        return arr[0]

    return jax.tree.map(squeeze_leaf, env_params)


def _extract_unit_arrays(state: dict[str, Any], unit_keys: list[str]) -> dict[str, np.ndarray]:
    positions = []
    rotations = []
    healths = []
    max_healths = []
    alives = []
    teams = []
    for key in unit_keys:
        unit = state[key]
        positions.append(_to_numpy(unit.transform.position).reshape(-1))
        rotations.append(float(_to_numpy(unit.transform.rotation).reshape(-1)[0]))
        healths.append(float(_to_numpy(unit.status.health).reshape(-1)[0]))
        max_healths.append(float(_to_numpy(unit.status.max_health).reshape(-1)[0]))
        alives.append(bool(_to_numpy(unit.status.is_alive).reshape(-1)[0]))
        teams.append(int(_to_numpy(unit.team).reshape(-1)[0]))
    return {
        "unit_position": np.stack(positions, axis=0).astype(np.float32),
        "unit_rotation": np.asarray(rotations, dtype=np.float32),
        "unit_health": np.asarray(healths, dtype=np.float32),
        "unit_max_health": np.asarray(max_healths, dtype=np.float32),
        "unit_is_alive": np.asarray(alives, dtype=np.uint8),
        "unit_team": np.asarray(teams, dtype=np.int32),
    }


def build_env_and_params(package: TaskPackage):
    bank = load_package_task_bank(package)
    task = bank["tasks"][0]
    manifest = bank["manifest"]
    schema = manifest["schema"]
    physics = str(manifest.get("physics", "default"))
    heuristic = str(manifest.get("heuristic", "medium"))
    env_params, tabx_config = build_batched_env_params_from_tasks(
        [task],
        physics=physics,
        heuristic=heuristic,
        max_n_ally=int(schema["max_n_ally"]),
        max_n_enemy=int(schema["max_n_enemy"]),
        max_n_zone=int(schema["max_n_zone"]),
    )
    env_params = _squeeze_env_params(env_params)
    env = TABX(cfg=tabx_config)
    env = TABXEnemyHeuristicWrapper(env)
    return env, env_params, task, manifest


def build_record_gen_context(package: TaskPackage) -> RecordGenContext:
    """Build env + oracle once; reuse across records for the same package."""

    env, env_params, task, manifest = build_env_and_params(package)
    ally_keys = list(env.agents)
    unit_keys = list(env.unit_keys)
    action_dim = int(env.action_space(ally_keys[0]).n)
    obs_dim = int(env.observation_space(ally_keys[0]).shape[0])
    oracle = load_oracle_coach(package.oracle_dir, obs_dim=obs_dim, action_dim=action_dim)
    return RecordGenContext(
        package=package,
        env=env,
        env_params=env_params,
        task=task,
        manifest=manifest,
        oracle=oracle,
        ally_keys=ally_keys,
        unit_keys=unit_keys,
        n_ally=len(ally_keys),
        action_dim=action_dim,
        obs_dim=obs_dim,
        n_units_total=len(unit_keys),
    )


def warmup_record_gen_context(
    ctx: RecordGenContext,
    *,
    behavior_mix: tuple[BehaviorSpec, ...] | None = None,
    steps: int = 2,
    seed: int = 0,
) -> float:
    """Force JAX/XLA compile for every behavior in the mix; return wall seconds.

    Walks **each** ``BehaviorSpec`` (not just one oracle + one heuristic) plus an
    explicit mid-episode ``reset``, so generate-time policy switches are less
    likely to trigger new JIT. Cannot literally compile every dynamic branch in
    TABX, but covers the paths used by record generation.
    """

    mix = behavior_mix or DEFAULT_BEHAVIOR_MIX
    t0 = time.perf_counter()
    env = ctx.env
    env_params = ctx.env_params
    ally_keys = ctx.ally_keys
    unit_keys = ctx.unit_keys
    oracle = ctx.oracle
    indiv_cfg = IndividualRewardConfig()
    key = jax.random.key(int(seed) % (2**32))
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key, env_params)
    n_steps = max(1, int(steps))

    for spec_index, spec in enumerate(mix):
        shared = SharedAllyPolicy.from_spec(
            spec,
            oracle=oracle,
            ally_keys=ally_keys,
            n_agents=ctx.n_units_total,
            max_n_zone=env.max_n_zone,
        )
        for _ in range(n_steps):
            avail = env.get_avail_actions(state)
            key, bkey, skey = jax.random.split(key, 3)
            behavior = shared.act(
                key=bkey,
                obs_by_agent=obs,
                avail_by_agent=avail,
                ally_keys=ally_keys,
                physics_params=state["physics_params"],
            )
            _ = reference_labels(
                oracle,
                obs_by_agent=obs,
                avail_by_agent=avail,
                ally_keys=ally_keys,
            )
            prev_state = state
            obs, state, rewards, dones, info = env.step(skey, state, dict(behavior))
            # Same reward shaping path as generate_one_record.
            team_by_agent = {agent: rewards[agent] for agent in ally_keys}
            damage_dealt = info.get("damage_dealt")
            if damage_dealt is None:
                damage_dealt = jnp.stack(
                    [state["state"][unit].damage_dealt for unit in unit_keys]
                )
            _ = compute_shaped_rewards(
                unit_keys=unit_keys,
                agent_keys=ally_keys,
                prev_state=prev_state["state"],
                next_state=state["state"],
                team_reward_by_agent=team_by_agent,
                damage_dealt=damage_dealt,
                config=indiv_cfg,
            )
            if bool(_to_numpy(dones["__all__"])):
                key, reset_key = jax.random.split(key)
                obs, state = env.reset(reset_key, env_params)

        # Explicit reset between policies (covers mid-run episode boundaries).
        if spec_index + 1 < len(mix):
            key, reset_key = jax.random.split(key)
            obs, state = env.reset(reset_key, env_params)

    # Ensure device work finished before stopping the timer.
    visible = state["state"]["game_manager"].visible_matrix
    if hasattr(visible, "block_until_ready"):
        visible.block_until_ready()
    return float(time.perf_counter() - t0)


def generate_one_record(
    package: TaskPackage,
    *,
    record_id: int,
    output_root: Path,
    total_timesteps: int,
    seed: int | None,
    min_behavior_steps: int,
    behavior_switch_prob: float,
    behavior_mix: tuple[BehaviorSpec, ...] | None = None,
    ctx: RecordGenContext | None = None,
    scan_rollout: bool = True,
    rollout_fn: Any | None = None,
    independent_ally_policies: bool = False,
    mid_episode_policy_switch: bool = True,
    best_teacher_reference: bool = False,
    best_policy: str = BEST_ORACLE,
) -> Path:
    """Generate one env-centric record.

    ``scan_rollout=True`` (default) runs a device-side ``lax.scan`` and syncs
    once at the end. Set ``False`` for the legacy per-step Python loop.
    """

    if scan_rollout:
        from generate.scan_rollout import generate_one_record_scan

        return generate_one_record_scan(
            package,
            record_id=record_id,
            output_root=output_root,
            total_timesteps=total_timesteps,
            seed=seed,
            min_behavior_steps=min_behavior_steps,
            behavior_switch_prob=behavior_switch_prob,
            behavior_mix=behavior_mix,
            ctx=ctx,
            rollout_fn=rollout_fn,
            independent_ally_policies=independent_ally_policies,
            mid_episode_policy_switch=mid_episode_policy_switch,
            best_teacher_reference=best_teacher_reference,
            best_policy=best_policy,
        )

    seed = sample_record_seed(seed, salt=record_id)
    mix = behavior_mix or DEFAULT_BEHAVIOR_MIX
    independent = bool(independent_ally_policies)
    mid_switch = bool(mid_episode_policy_switch)
    teacher = str(best_policy if best_teacher_reference else BEST_ORACLE)
    if ctx is None:
        ctx = build_record_gen_context(package)
    elif ctx.package.name != package.name:
        raise ValueError(
            f"RecordGenContext package mismatch: ctx={ctx.package.name} vs {package.name}"
        )
    env = ctx.env
    env_params = ctx.env_params
    task = ctx.task
    manifest = ctx.manifest
    ally_keys = ctx.ally_keys
    unit_keys = ctx.unit_keys
    n_ally = ctx.n_ally
    action_dim = ctx.action_dim
    obs_dim = ctx.obs_dim
    oracle = ctx.oracle
    n_units_total = ctx.n_units_total
    indiv_cfg = IndividualRewardConfig()
    # heuristic_policy parses obs with total unit count (allies + enemies).
    n_units_total = len(unit_keys)
    advanced_params = (
        load_heuristic_params("advanced") if teacher == BEST_ADVANCED else None
    )
    ref_last_visible = {agent: LastVisibleTarget() for agent in ally_keys}

    key = jax.random.key(seed % (2**32))
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key, env_params)

    if independent:
        switchers = [
            BehaviorSwitcher(
                mix=mix,
                min_behavior_steps=min_behavior_steps,
                behavior_switch_prob=behavior_switch_prob,
                mid_episode_policy_switch=mid_switch,
                rng=np.random.default_rng(seed + 17 * (i + 1)),
            )
            for i in range(n_ally)
        ]
        specs = [s.force_sample() for s in switchers]
        policies = [
            SharedAllyPolicy.from_spec(
                spec,
                oracle=oracle,
                ally_keys=[agent],
                n_agents=n_units_total,
                max_n_zone=env.max_n_zone,
            )
            for spec, agent in zip(specs, ally_keys)
        ]
        switcher = None
        shared = None
    else:
        switcher = BehaviorSwitcher(
            mix=mix,
            min_behavior_steps=min_behavior_steps,
            behavior_switch_prob=behavior_switch_prob,
            mid_episode_policy_switch=mid_switch,
            rng=np.random.default_rng(seed),
        )
        spec = switcher.force_sample()
        shared = SharedAllyPolicy.from_spec(
            spec,
            oracle=oracle,
            ally_keys=ally_keys,
            n_agents=n_units_total,
            max_n_zone=env.max_n_zone,
        )
        switchers = None
        policies = None

    buffers: dict[str, list] = {
        "actions_behavior": [],
        "actions_reference": [],
        "actions_reference_distribution": [],
        "reward_team": [],
        "reward_individual": [],
        "done": [],
        "truncation": [],
        "is_win": [],
        "reset": [],
        "episode_id": [],
        "behavior_policy_id": [],
        "policy_tag": [],
        "visible_matrix": [],
        "obs_flat": [],
        "unit_position": [],
        "unit_rotation": [],
        "unit_health": [],
        "unit_max_health": [],
        "unit_is_alive": [],
        "unit_team": [],
    }

    episode_id = 0
    is_reset_step = True
    steps = 0

    while steps < total_timesteps:
        avail = env.get_avail_actions(state)
        key, bkey, skey = jax.random.split(key, 3)
        if independent:
            assert switchers is not None and policies is not None
            behavior = {}
            policy_id_row = np.zeros((n_ally,), dtype=np.int32)
            policy_tag_row = np.zeros((n_ally,), dtype=np.int32)
            for i, agent in enumerate(ally_keys):
                spec_i = switchers[i].maybe_switch()
                if spec_i.policy_id != policies[i].spec.policy_id:
                    policies[i] = SharedAllyPolicy.from_spec(
                        spec_i,
                        oracle=oracle,
                        ally_keys=[agent],
                        n_agents=n_units_total,
                        max_n_zone=env.max_n_zone,
                    )
                key, sub = jax.random.split(key)
                act_i = policies[i].act(
                    key=sub,
                    obs_by_agent=obs,
                    avail_by_agent=avail,
                    ally_keys=[agent],
                    physics_params=state["physics_params"],
                )
                behavior[agent] = act_i[agent]
                policy_id_row[i] = int(spec_i.policy_id)
                policy_tag_row[i] = int(policy_tag_for_spec(spec_i))
        else:
            assert switcher is not None and shared is not None
            # Shared behavior for all allies (default).
            spec = switcher.maybe_switch()
            if spec.policy_id != shared.spec.policy_id:
                shared = SharedAllyPolicy.from_spec(
                    spec,
                    oracle=oracle,
                    ally_keys=ally_keys,
                    n_agents=n_units_total,
                    max_n_zone=env.max_n_zone,
                )
            behavior = shared.act(
                key=bkey,
                obs_by_agent=obs,
                avail_by_agent=avail,
                ally_keys=ally_keys,
                physics_params=state["physics_params"],
            )
            policy_id_row = np.full((n_ally,), int(shared.spec.policy_id), dtype=np.int32)
            policy_tag_row = np.full(
                (n_ally,), int(policy_tag_for_spec(shared.spec)), dtype=np.int32
            )
        key, rkey = jax.random.split(key)
        ref, ref_dist, ref_last_visible = reference_labels_best_teacher(
            oracle,
            key=rkey,
            obs_by_agent=obs,
            avail_by_agent=avail,
            ally_keys=ally_keys,
            best_policy=teacher,
            n_agents=n_units_total,
            max_n_zone=env.max_n_zone,
            physics_params=state["physics_params"],
            advanced_params=advanced_params,
            ref_last_visible=ref_last_visible,
        )

        unit_pack = _extract_unit_arrays(state["state"], unit_keys)
        visible = _to_numpy(state["state"]["game_manager"].visible_matrix).astype(np.uint8)

        prev_state = state
        obs_next, state_next, rewards, dones, info = env.step(skey, state, dict(behavior))

        team_by_agent = {agent: rewards[agent] for agent in ally_keys}
        damage_dealt = info.get("damage_dealt")
        if damage_dealt is None:
            damage_dealt = jnp.stack(
                [state_next["state"][unit].damage_dealt for unit in unit_keys]
            )
        shaped, _ = compute_shaped_rewards(
            unit_keys=unit_keys,
            agent_keys=ally_keys,
            prev_state=prev_state["state"],
            next_state=state_next["state"],
            team_reward_by_agent=team_by_agent,
            damage_dealt=damage_dealt,
            config=indiv_cfg,
        )

        ep_done = bool(_to_numpy(dones["__all__"]))
        # info["truncation"]: timed out at max_episode_steps
        # info["is_win"]: per-team; index 0 is ally (team 0); non-terminal steps are 0
        trunc = bool(_to_numpy(info["truncation"]).reshape(-1)[0])
        ally_win = bool(_to_numpy(info["is_win"]).reshape(-1)[0])
        buffers["actions_behavior"].append(
            np.asarray([int(behavior[a]) for a in ally_keys], dtype=np.int32)
        )
        buffers["actions_reference"].append(
            np.asarray([int(ref[a]) for a in ally_keys], dtype=np.int32)
        )
        buffers["actions_reference_distribution"].append(
            np.asarray(ref_dist, dtype=np.float32)
        )
        buffers["reward_team"].append(
            np.asarray([float(_to_numpy(team_by_agent[a])) for a in ally_keys], dtype=np.float32)
        )
        buffers["reward_individual"].append(
            np.asarray([float(_to_numpy(shaped[a])) for a in ally_keys], dtype=np.float32)
        )
        buffers["done"].append(np.uint8(ep_done))
        buffers["truncation"].append(np.uint8(trunc))
        buffers["is_win"].append(np.uint8(ally_win))
        buffers["reset"].append(np.uint8(is_reset_step))
        buffers["episode_id"].append(np.int32(episode_id))
        buffers["behavior_policy_id"].append(policy_id_row)
        buffers["policy_tag"].append(policy_tag_row)
        buffers["visible_matrix"].append(visible)
        buffers["obs_flat"].append(
            np.stack(
                [
                    np.nan_to_num(_to_numpy(obs[a]).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                    for a in ally_keys
                ],
                axis=0,
            )
        )
        for name, value in unit_pack.items():
            buffers[name].append(value)

        steps += 1
        is_reset_step = False
        obs, state = obs_next, state_next

        if ep_done and steps < total_timesteps:
            key, reset_key = jax.random.split(key)
            obs, state = env.reset(reset_key, env_params)
            episode_id += 1
            is_reset_step = True
            ref_last_visible = {agent: LastVisibleTarget() for agent in ally_keys}
            # Every episode reset resamples behavior (shared or per-agent).
            if independent:
                assert switchers is not None and policies is not None
                for i, agent in enumerate(ally_keys):
                    spec_i = switchers[i].force_sample()
                    policies[i] = SharedAllyPolicy.from_spec(
                        spec_i,
                        oracle=oracle,
                        ally_keys=[agent],
                        n_agents=n_units_total,
                        max_n_zone=env.max_n_zone,
                    )
            else:
                assert switcher is not None
                spec = switcher.force_sample()
                shared = SharedAllyPolicy.from_spec(
                    spec,
                    oracle=oracle,
                    ally_keys=ally_keys,
                    n_agents=n_units_total,
                    max_n_zone=env.max_n_zone,
                )

    arrays = {
        "actions_behavior": np.stack(buffers["actions_behavior"], axis=0),
        "actions_reference": np.stack(buffers["actions_reference"], axis=0),
        # Soft oracle policy π_ref(a|o); HVAC-style label_action_distribution.
        "actions_reference_distribution": np.stack(
            buffers["actions_reference_distribution"], axis=0
        ),
        "reward_team": np.stack(buffers["reward_team"], axis=0),
        "reward_individual": np.stack(buffers["reward_individual"], axis=0),
        "done": np.asarray(buffers["done"], dtype=np.uint8),
        "truncation": np.asarray(buffers["truncation"], dtype=np.uint8),
        "is_win": np.asarray(buffers["is_win"], dtype=np.uint8),
        "reset": np.asarray(buffers["reset"], dtype=np.uint8),
        "episode_id": np.asarray(buffers["episode_id"], dtype=np.int32),
        "behavior_policy_id": np.stack(buffers["behavior_policy_id"], axis=0),
        "policy_tag": np.stack(buffers["policy_tag"], axis=0),
        "visible_matrix": np.stack(buffers["visible_matrix"], axis=0),
        "obs_flat": np.stack(buffers["obs_flat"], axis=0),
        "unit_position": np.stack(buffers["unit_position"], axis=0),
        "unit_rotation": np.stack(buffers["unit_rotation"], axis=0),
        "unit_health": np.stack(buffers["unit_health"], axis=0),
        "unit_max_health": np.stack(buffers["unit_max_health"], axis=0),
        "unit_is_alive": np.stack(buffers["unit_is_alive"], axis=0),
        "unit_team": np.stack(buffers["unit_team"], axis=0),
    }

    safe_id = package.task_id.replace("/", "_")
    record_dir = (
        Path(output_root)
        / f"task_{package.task_index:05d}_{safe_id}"
        / f"record-{record_id:06d}"
    )
    meta = {
        "schema": "env_centric_v1",
        "task_index": package.task_index,
        "task_id": package.task_id,
        "record_id": record_id,
        "total_timesteps": total_timesteps,
        "n_ally": n_ally,
        "ally_keys": ally_keys,
        "unit_keys": unit_keys,
        "n_units": len(unit_keys),
        "max_n_zone": int(env.max_n_zone),
        "action_dim": action_dim,
        "obs_dim": obs_dim,
        "seed": seed,
        "min_behavior_steps": min_behavior_steps,
        "behavior_switch_prob": behavior_switch_prob,
        "mid_episode_policy_switch": mid_switch,
        "shared_ally_policy": not independent,
        "independent_ally_policies": independent,
        "best_teacher_reference": bool(best_teacher_reference),
        "best_policy": teacher,
        "individual_reward_config": asdict(indiv_cfg),
        "behavior_mix": [asdict(s) for s in mix],
        "policy_tag_legend": policy_tag_legend_json(),
        "policy_id_mapping": policy_id_mapping_json(mix),
        "oracle": {
            "algorithm": oracle.algorithm,
            "ckpt": str(package.oracle_ckpt),
            "config": str(package.oracle_config),
        },
        "physics": manifest.get("physics"),
        "enemy_heuristic": manifest.get("heuristic"),
        "n_episodes_seen": int(episode_id + 1),
        "scan_rollout": False,
        "switcher_rng": "numpy",
    }
    write_env_centric_record(record_dir, arrays, meta)
    return record_dir


def _worker(args: tuple) -> str:
    (
        packages_root,
        package_name,
        record_ids,
        output_root,
        total_timesteps,
        seed,
        min_behavior_steps,
        behavior_switch_prob,
        scan_rollout,
    ) = args
    # Prefer CPU in workers to avoid multi-process GPU contention.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    packages = {p.name: p for p in discover_task_packages(packages_root)}
    package = packages[package_name]
    paths = []
    for record_id in record_ids:
        path = generate_one_record(
            package,
            record_id=record_id,
            output_root=Path(output_root),
            total_timesteps=total_timesteps,
            seed=seed,
            min_behavior_steps=min_behavior_steps,
            behavior_switch_prob=behavior_switch_prob,
            scan_rollout=bool(scan_rollout),
        )
        paths.append(str(path))
        print(f"[env_centric] wrote {path}", flush=True)
    return ",".join(paths)


def _split_ids(start: int, count: int, workers: int) -> list[list[int]]:
    ids = list(range(start, start + count))
    if workers <= 1:
        return [ids]
    chunks: list[list[int]] = [[] for _ in range(workers)]
    for i, record_id in enumerate(ids):
        chunks[i % workers].append(record_id)
    return [c for c in chunks if c]


def run_parallel(
    *,
    task_packages_root: Path,
    output_root: Path,
    total_timesteps: int,
    records_per_task: int,
    start_index: int,
    workers: int,
    seed: int | None,
    min_behavior_steps: int,
    behavior_switch_prob: float,
    task_filter: list[int] | None = None,
    scan_rollout: bool = True,
) -> None:
    packages = discover_task_packages(task_packages_root)
    if task_filter is not None:
        allowed = set(task_filter)
        packages = [p for p in packages if p.task_index in allowed]
    ensure_dir(output_root)

    jobs = []
    for package in packages:
        for chunk in _split_ids(start_index, records_per_task, max(1, workers)):
            jobs.append(
                (
                    str(task_packages_root),
                    package.name,
                    chunk,
                    str(output_root),
                    total_timesteps,
                    seed,
                    min_behavior_steps,
                    behavior_switch_prob,
                    scan_rollout,
                )
            )

    if workers <= 1:
        for job in jobs:
            _worker(job)
        return

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        for _ in pool.imap_unordered(_worker, jobs):
            pass


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate env-centric TABX records")
    parser.add_argument(
        "--coach_root",
        type=str,
        default=None,
        help="Root of self-contained coach leaves (task.json + safetensors)",
    )
    parser.add_argument(
        "--task_packages_root",
        type=str,
        default=None,
        help="Alias for --coach_root (legacy name)",
    )
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--total_timesteps", type=int, default=512)
    parser.add_argument("--records_per_task", type=int, default=1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional fixed seed for reproducibility. Default: sample from time+pid per record.",
    )
    parser.add_argument("--min_behavior_steps", type=int, default=64)
    parser.add_argument("--behavior_switch_prob", type=float, default=0.2)
    parser.add_argument(
        "--scan_rollout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Device-side lax.scan per record (default). --no-scan_rollout for step loop.",
    )
    parser.add_argument(
        "--task_index",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of task indices",
    )
    args = parser.parse_args(argv)
    coach_root = args.coach_root or args.task_packages_root
    if not coach_root:
        parser.error("one of --coach_root / --task_packages_root is required")
    run_parallel(
        task_packages_root=Path(coach_root),
        output_root=Path(args.output_root),
        total_timesteps=args.total_timesteps,
        records_per_task=args.records_per_task,
        start_index=args.start_index,
        workers=args.workers,
        seed=args.seed,
        min_behavior_steps=args.min_behavior_steps,
        behavior_switch_prob=args.behavior_switch_prob,
        task_filter=args.task_index,
        scan_rollout=bool(args.scan_rollout),
    )


if __name__ == "__main__":
    main()
