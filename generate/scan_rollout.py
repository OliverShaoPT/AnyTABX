"""Device-side env-centric rollout via ``jax.lax.scan`` (one sync per record)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from generate.behavior_mix import (
    BehaviorSpec,
    DEFAULT_BEHAVIOR_MIX,
    maybe_switch_policy_id,
    maybe_switch_policy_ids,
    mix_cdf_jax,
    policy_id_mapping_json,
    policy_tag_legend_json,
    policy_tag_table,
    sample_policy_index,
    sample_policy_indices,
)
from generate.dump_schema import write_env_centric_record
from generate.env_centric import RecordGenContext, sample_record_seed
from generate.oracle_loader import OracleCoach
from generate.policies import (
    BEST_ADVANCED,
    BEST_ORACLE,
    act_shared_jax,
    reference_labels_best_teacher_jax,
)
from generate.task_package import TaskPackage
from src.tabx.eval_task import load_heuristic_params
from src.tabx.heuristic_policy import LastVisibleTarget
from src.tabx.wrappers.individual_reward import IndividualRewardConfig, compute_shaped_rewards


def _extract_attack_target_jax(state: dict[str, Any]) -> jax.Array:
    """``game_manager.attack_target`` at the same tick as ``visible_matrix``."""

    return jnp.asarray(state["game_manager"].attack_target, dtype=jnp.int32).reshape(-1)


def _extract_unit_arrays_jax(state: dict[str, Any], unit_keys: list[str]) -> dict[str, jax.Array]:
    positions = []
    rotations = []
    healths = []
    max_healths = []
    alives = []
    teams = []
    for key in unit_keys:
        unit = state[key]
        positions.append(jnp.asarray(unit.transform.position).reshape(-1))
        rotations.append(jnp.asarray(unit.transform.rotation).reshape(-1)[0])
        healths.append(jnp.asarray(unit.status.health).reshape(-1)[0])
        max_healths.append(jnp.asarray(unit.status.max_health).reshape(-1)[0])
        alives.append(jnp.asarray(unit.status.is_alive).reshape(-1)[0])
        teams.append(jnp.asarray(unit.team).reshape(-1)[0])
    return {
        "unit_position": jnp.stack(positions, axis=0).astype(jnp.float32),
        "unit_rotation": jnp.asarray(rotations, dtype=jnp.float32),
        "unit_health": jnp.asarray(healths, dtype=jnp.float32),
        "unit_max_health": jnp.asarray(max_healths, dtype=jnp.float32),
        "unit_is_alive": jnp.asarray(alives, dtype=jnp.uint8),
        "unit_team": jnp.asarray(teams, dtype=jnp.int32),
    }


def _init_last_visible(n_ally: int) -> list[LastVisibleTarget]:
    return [
        LastVisibleTarget(
            abs_position=jnp.zeros((2,), dtype=jnp.float32),
            ever_visible=jnp.asarray([False]),
        )
        for _ in range(n_ally)
    ]


def _build_policy_branches(
    mix: tuple[BehaviorSpec, ...],
    *,
    oracle: OracleCoach,
    ally_keys: list[str],
    n_agents: int,
    max_n_zone: int,
) -> list[Callable]:
    """One lax.switch branch per mix entry (policy_id order = mix order)."""

    branches: list[Callable] = []
    for spec in mix:
        heur = None
        if spec.kind == "heuristic":
            if not spec.heuristic:
                raise ValueError(f"heuristic spec missing preset: {spec}")
            heur = load_heuristic_params(spec.heuristic, epsilon_override=spec.epsilon)

        def branch(
            operands,
            *,
            _spec=spec,
            _heur=heur,
        ):
            key, obs, avail, physics_params, last_visible = operands
            actions, new_last, key = act_shared_jax(
                key=key,
                spec=_spec,
                oracle=oracle,
                obs_by_agent=obs,
                avail_by_agent=avail,
                ally_keys=ally_keys,
                n_agents=n_agents,
                max_n_zone=max_n_zone,
                physics_params=physics_params,
                heuristic_params=_heur,
                last_visible=last_visible,
            )
            return actions, new_last, key

        branches.append(branch)
    return branches


def _select_last_visible_by_mix(
    lasts_by_mix: list[list[LastVisibleTarget]],
    mix_indices: jax.Array,
    n_ally: int,
) -> list[LastVisibleTarget]:
    """Gather per-agent ``LastVisibleTarget`` from policy-indexed candidates."""

    pos = jnp.stack(
        [
            jnp.stack([lv.abs_position for lv in lasts], axis=0)
            for lasts in lasts_by_mix
        ],
        axis=0,
    )
    ever = jnp.stack(
        [
            jnp.stack([lv.ever_visible for lv in lasts], axis=0)
            for lasts in lasts_by_mix
        ],
        axis=0,
    )
    arange = jnp.arange(n_ally)
    sel_pos = pos[mix_indices, arange]
    sel_ever = ever[mix_indices, arange]
    return [
        LastVisibleTarget(abs_position=sel_pos[i], ever_visible=sel_ever[i])
        for i in range(n_ally)
    ]


def _act_independent_allies(
    *,
    key: jax.Array,
    mix_indices: jax.Array,
    branches: list[Callable],
    obs,
    avail,
    physics_params,
    last_visible: list[LastVisibleTarget],
    ally_keys: list[str],
    n_ally: int,
) -> tuple[dict[str, jax.Array], list[LastVisibleTarget], jax.Array]:
    """Run every mix policy, then pick each agent's action by its mix index."""

    actions_by_mix: list[jax.Array] = []
    lasts_by_mix: list[list[LastVisibleTarget]] = []
    key_work = key
    for branch in branches:
        key_work, sub = jax.random.split(key_work)
        acts, new_last, _ = branch(
            (sub, obs, avail, physics_params, last_visible)
        )
        actions_by_mix.append(
            jnp.stack(
                [jnp.asarray(acts[a], dtype=jnp.int32).reshape(()) for a in ally_keys]
            )
        )
        lasts_by_mix.append(new_last)
    stacked = jnp.stack(actions_by_mix, axis=0)  # (n_mix, n_ally)
    selected = stacked[mix_indices, jnp.arange(n_ally)]
    behavior = {
        agent: selected[i].astype(jnp.int32)
        for i, agent in enumerate(ally_keys)
    }
    new_last = _select_last_visible_by_mix(lasts_by_mix, mix_indices, n_ally)
    return behavior, new_last, key_work


def _make_single_env_rollout(
    ctx: RecordGenContext,
    *,
    mix: tuple[BehaviorSpec, ...],
    min_behavior_steps: int,
    behavior_switch_prob: float,
    total_timesteps: int,
    independent_ally_policies: bool = False,
    mid_episode_policy_switch: bool = True,
    best_teacher_policy: str = BEST_ORACLE,
    dump_attack_target: bool = False,
):
    """Unjitted ``(seed, cdf) -> traj`` for one env (vmappable over seed)."""

    env = ctx.env
    env_params = ctx.env_params
    ally_keys = list(ctx.ally_keys)
    unit_keys = list(ctx.unit_keys)
    n_ally = int(ctx.n_ally)
    n_agents = int(ctx.n_units_total)
    max_n_zone = int(env.max_n_zone)
    oracle = ctx.oracle
    indiv_cfg = IndividualRewardConfig()
    independent = bool(independent_ally_policies)
    allow_mid_switch = bool(mid_episode_policy_switch)
    best_policy = str(best_teacher_policy or BEST_ORACLE)
    include_attack_target = bool(dump_attack_target)
    advanced_params = (
        load_heuristic_params("advanced") if best_policy == BEST_ADVANCED else None
    )
    # Map mix position -> policy_id / quality tag (usually identity, but honor config).
    policy_ids = jnp.asarray([int(s.policy_id) for s in mix], dtype=jnp.int32)
    policy_tags = jnp.asarray(policy_tag_table(mix), dtype=jnp.int32)
    branches = _build_policy_branches(
        mix,
        oracle=oracle,
        ally_keys=ally_keys,
        n_agents=n_agents,
        max_n_zone=max_n_zone,
    )

    def rollout(seed: jax.Array, cdf: jax.Array):
        """``seed`` + runtime mix ``cdf`` (weights change without recompile)."""

        # Match env_centric: jax.random.key(seed % 2**32).
        key = jax.random.key(jnp.asarray(seed, dtype=jnp.uint32))
        key, reset_key, init_policy_key = jax.random.split(key, 3)
        obs, state = env.reset(reset_key, env_params)
        # Carry mix index (0..n_mix-1); dump BehaviorSpec.policy_id separately.
        if independent:
            mix_index = sample_policy_indices(init_policy_key, cdf, n_ally)
            steps0 = jnp.zeros((n_ally,), dtype=jnp.int32)
        else:
            mix_index = sample_policy_index(init_policy_key, cdf).astype(jnp.int32)
            steps0 = jnp.int32(0)
        last_visible = _init_last_visible(n_ally)
        ref_last_visible = _init_last_visible(n_ally)

        carry0 = (
            obs,
            state,
            key,
            mix_index,
            steps0,  # steps_since_switch
            jnp.int32(0),  # episode_id
            jnp.bool_(True),  # is_reset_step
            last_visible,
            ref_last_visible,
        )

        def body(carry, t):
            (
                obs,
                state,
                key,
                mix_index,
                steps_since_switch,
                episode_id,
                is_reset_step,
                last_visible,
                ref_last_visible,
            ) = carry

            if allow_mid_switch:
                if independent:
                    key, mix_index, steps_since_switch = maybe_switch_policy_ids(
                        key,
                        policy_ids=mix_index,
                        steps_since_switch=steps_since_switch,
                        cdf=cdf,
                        min_behavior_steps=min_behavior_steps,
                        behavior_switch_prob=behavior_switch_prob,
                    )
                else:
                    key, mix_index, steps_since_switch = maybe_switch_policy_id(
                        key,
                        policy_id=mix_index,
                        steps_since_switch=steps_since_switch,
                        cdf=cdf,
                        min_behavior_steps=min_behavior_steps,
                        behavior_switch_prob=behavior_switch_prob,
                    )
            else:
                # Still advance the cooldown counter for meta consistency.
                steps_since_switch = steps_since_switch + jnp.int32(1)

            if independent:
                policy_id_agents = policy_ids[mix_index]
                policy_tag_agents = policy_tags[mix_index]
            else:
                policy_id = policy_ids[mix_index]
                policy_tag = policy_tags[mix_index]
                policy_id_agents = jnp.full((n_ally,), policy_id, dtype=jnp.int32)
                policy_tag_agents = jnp.full((n_ally,), policy_tag, dtype=jnp.int32)

            avail = env.get_avail_actions(state)
            # Same split pattern as env_centric (key, bkey, skey[, rkey]).
            key, bkey, skey, rkey = jax.random.split(key, 4)
            if independent:
                behavior, last_visible, _bkey_out = _act_independent_allies(
                    key=bkey,
                    mix_indices=mix_index,
                    branches=branches,
                    obs=obs,
                    avail=avail,
                    physics_params=state["physics_params"],
                    last_visible=last_visible,
                    ally_keys=ally_keys,
                    n_ally=n_ally,
                )
            else:
                behavior, last_visible, _bkey_out = jax.lax.switch(
                    mix_index,
                    branches,
                    (bkey, obs, avail, state["physics_params"], last_visible),
                )

            ref, ref_dist, ref_last_visible, _rkey = reference_labels_best_teacher_jax(
                oracle,
                key=rkey,
                obs_by_agent=obs,
                avail_by_agent=avail,
                ally_keys=ally_keys,
                best_policy=best_policy,
                n_agents=n_agents,
                max_n_zone=max_n_zone,
                physics_params=state["physics_params"],
                advanced_params=advanced_params,
                ref_last_visible=ref_last_visible,
            )

            unit_pack = _extract_unit_arrays_jax(state["state"], unit_keys)
            if include_attack_target:
                unit_pack = {
                    **unit_pack,
                    "attack_target": _extract_attack_target_jax(state["state"]),
                }
            visible = jnp.asarray(
                state["state"]["game_manager"].visible_matrix, dtype=jnp.uint8
            )
            obs_flat = jnp.stack(
                [
                    jnp.nan_to_num(
                        jnp.asarray(obs[a], dtype=jnp.float32),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    for a in ally_keys
                ],
                axis=0,
            )

            prev_state = state
            obs_next, state_next, rewards, dones, info = env.step(
                skey, state, dict(behavior)
            )

            team_by_agent = {agent: rewards[agent] for agent in ally_keys}
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

            ep_done = jnp.asarray(dones["__all__"]).reshape(()).astype(jnp.bool_)
            trunc = jnp.asarray(info["truncation"]).reshape(-1)[0].astype(jnp.bool_)
            ally_win = jnp.asarray(info["is_win"]).reshape(-1)[0].astype(jnp.bool_)

            actions_behavior = jnp.stack(
                [jnp.asarray(behavior[a], dtype=jnp.int32).reshape(()) for a in ally_keys]
            )
            actions_reference = jnp.stack(
                [jnp.asarray(ref[a], dtype=jnp.int32).reshape(()) for a in ally_keys]
            )
            reward_team = jnp.stack(
                [jnp.asarray(team_by_agent[a], dtype=jnp.float32).reshape(()) for a in ally_keys]
            )
            reward_individual = jnp.stack(
                [jnp.asarray(shaped[a], dtype=jnp.float32).reshape(()) for a in ally_keys]
            )

            out = {
                "actions_behavior": actions_behavior,
                "actions_reference": actions_reference,
                "actions_reference_distribution": ref_dist.astype(jnp.float32),
                "reward_team": reward_team,
                "reward_individual": reward_individual,
                "done": ep_done.astype(jnp.uint8),
                "truncation": trunc.astype(jnp.uint8),
                "is_win": ally_win.astype(jnp.uint8),
                "reset": is_reset_step.astype(jnp.uint8),
                "episode_id": episode_id.astype(jnp.int32),
                "behavior_policy_id": policy_id_agents,
                "policy_tag": policy_tag_agents,
                "visible_matrix": visible,
                "obs_flat": obs_flat,
                **unit_pack,
            }

            do_reset = ep_done & (t + jnp.int32(1) < jnp.int32(total_timesteps))

            def _reset_branch(operands):
                (
                    key_in,
                    _obs,
                    _state,
                    _mix_index,
                    _steps,
                    episode_in,
                    _last,
                    _ref_last,
                ) = operands
                key_in, reset_key, sample_key = jax.random.split(key_in, 3)
                obs_r, state_r = env.reset(reset_key, env_params)
                if independent:
                    idx = sample_policy_indices(sample_key, cdf, n_ally)
                    steps_r = jnp.zeros((n_ally,), dtype=jnp.int32)
                else:
                    idx = sample_policy_index(sample_key, cdf).astype(jnp.int32)
                    steps_r = jnp.int32(0)
                last_r = _init_last_visible(n_ally)
                ref_last_r = _init_last_visible(n_ally)
                return (
                    obs_r,
                    state_r,
                    key_in,
                    idx,
                    steps_r,
                    (episode_in + jnp.int32(1)).astype(jnp.int32),
                    jnp.bool_(True),
                    last_r,
                    ref_last_r,
                )

            def _continue_branch(operands):
                (
                    key_in,
                    obs_in,
                    state_in,
                    mix_in,
                    steps_in,
                    episode_in,
                    last_in,
                    ref_last_in,
                ) = operands
                return (
                    obs_in,
                    state_in,
                    key_in,
                    mix_in,
                    steps_in,
                    episode_in,
                    jnp.bool_(False),
                    last_in,
                    ref_last_in,
                )

            new_carry = jax.lax.cond(
                do_reset,
                _reset_branch,
                _continue_branch,
                (
                    key,
                    obs_next,
                    state_next,
                    mix_index,
                    steps_since_switch,
                    episode_id,
                    last_visible,
                    ref_last_visible,
                ),
            )
            return new_carry, out

        _carry, traj = jax.lax.scan(
            body, carry0, jnp.arange(total_timesteps, dtype=jnp.int32)
        )
        return traj

    return rollout


def build_scan_rollout_fn(
    ctx: RecordGenContext,
    *,
    mix: tuple[BehaviorSpec, ...],
    min_behavior_steps: int,
    behavior_switch_prob: float,
    total_timesteps: int,
    parallel_envs: int = 1,
    independent_ally_policies: bool = False,
    mid_episode_policy_switch: bool = True,
    best_teacher_policy: str = BEST_ORACLE,
    dump_attack_target: bool = False,
):
    """Return jitted rollout.

    - ``parallel_envs <= 1``: ``(seed, cdf) -> traj`` with leaves ``(T, ...)``
    - ``parallel_envs > 1``: ``(seeds[B], cdf) -> traj`` with leaves ``(B, T, ...)``
      (vmap over seeds only; cdf broadcast). Fixed B; pad incomplete batches.
    """

    raw = _make_single_env_rollout(
        ctx,
        mix=mix,
        min_behavior_steps=min_behavior_steps,
        behavior_switch_prob=behavior_switch_prob,
        total_timesteps=total_timesteps,
        independent_ally_policies=independent_ally_policies,
        mid_episode_policy_switch=mid_episode_policy_switch,
        best_teacher_policy=best_teacher_policy,
        dump_attack_target=dump_attack_target,
    )
    b = max(1, int(parallel_envs))
    if b <= 1:
        return jax.jit(raw)
    return jax.jit(jax.vmap(raw, in_axes=(0, None)))


def warmup_scan_rollout(
    rollout_fn,
    *,
    mix: tuple[BehaviorSpec, ...] | None = None,
    cdf: jax.Array | None = None,
    seed: int = 0,
    parallel_envs: int = 1,
) -> float:
    import time

    if cdf is None:
        if mix is None:
            mix = DEFAULT_BEHAVIOR_MIX
        cdf = mix_cdf_jax(mix)
    t0 = time.perf_counter()
    b = max(1, int(parallel_envs))
    if b <= 1:
        traj = rollout_fn(jnp.asarray(seed, dtype=jnp.uint32), cdf)
    else:
        seeds = jnp.arange(b, dtype=jnp.uint32) + jnp.uint32(int(seed) % (2**32))
        traj = rollout_fn(seeds, cdf)
    # Touch a leaf so device work finishes.
    leaf = traj["done"]
    if hasattr(leaf, "block_until_ready"):
        leaf.block_until_ready()
    return float(time.perf_counter() - t0)


def _cast_array(name: str, arr: np.ndarray) -> np.ndarray:
    if name in {"done", "truncation", "is_win", "reset", "unit_is_alive"}:
        return arr.astype(np.uint8)
    if name in {
        "episode_id",
        "behavior_policy_id",
        "policy_tag",
        "unit_team",
        "actions_behavior",
        "actions_reference",
        "attack_target",
    }:
        return arr.astype(np.int32)
    return arr


def trajectory_to_numpy(traj: dict[str, Any]) -> dict[str, np.ndarray]:
    """Single-env traj leaves ``(T, ...)`` -> numpy."""

    host = jax.device_get(traj)
    return {name: _cast_array(name, np.asarray(value)) for name, value in host.items()}


def batch_trajectory_to_numpy(
    traj: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Batched traj leaves ``(B, T, ...)`` -> numpy (keep batch axis)."""

    host = jax.device_get(traj)
    return {name: _cast_array(name, np.asarray(value)) for name, value in host.items()}


def _record_meta(
    *,
    package: TaskPackage,
    ctx: RecordGenContext,
    record_id: int,
    total_timesteps: int,
    seed_i: int,
    min_behavior_steps: int,
    behavior_switch_prob: float,
    mix: tuple[BehaviorSpec, ...],
    arrays: dict[str, np.ndarray],
    parallel_envs: int,
    adapt: dict[str, Any] | None = None,
    independent_ally_policies: bool = False,
    mid_episode_policy_switch: bool = True,
    best_teacher_reference: bool = False,
    best_policy: str = BEST_ORACLE,
) -> dict[str, Any]:
    indiv_cfg = IndividualRewardConfig()
    independent = bool(independent_ally_policies)
    mid_switch = bool(mid_episode_policy_switch)
    meta: dict[str, Any] = {
        "schema": "env_centric_v1",
        "task_index": package.task_index,
        "task_id": package.task_id,
        "record_id": record_id,
        "total_timesteps": total_timesteps,
        "n_ally": ctx.n_ally,
        "ally_keys": ctx.ally_keys,
        "unit_keys": ctx.unit_keys,
        "n_units": len(ctx.unit_keys),
        "max_n_zone": int(ctx.env.max_n_zone),
        "action_dim": ctx.action_dim,
        "obs_dim": ctx.obs_dim,
        "seed": seed_i,
        "min_behavior_steps": min_behavior_steps,
        "behavior_switch_prob": behavior_switch_prob,
        "mid_episode_policy_switch": mid_switch,
        "shared_ally_policy": not independent,
        "independent_ally_policies": independent,
        "best_teacher_reference": bool(best_teacher_reference),
        "best_policy": str(best_policy),
        "individual_reward_config": asdict(indiv_cfg),
        "behavior_mix": [asdict(s) for s in mix],
        "policy_tag_legend": policy_tag_legend_json(),
        "policy_id_mapping": policy_id_mapping_json(mix),
        "oracle": {
            "algorithm": ctx.oracle.algorithm,
            "ckpt": str(package.oracle_ckpt),
            "config": str(package.oracle_config),
        },
        "physics": ctx.manifest.get("physics"),
        "enemy_heuristic": ctx.manifest.get("heuristic"),
        "n_episodes_seen": int(arrays["episode_id"][-1]) + 1
        if len(arrays["episode_id"])
        else 0,
        "scan_rollout": True,
        "switcher_rng": "jax",
        "parallel_envs": int(parallel_envs),
    }
    if "attack_target" in arrays:
        meta["schema"] = "env_centric_v1_comm"
        meta["has_attack_target"] = True
    if adapt is not None:
        meta["adapt"] = adapt
    return meta


def _write_one_from_arrays(
    package: TaskPackage,
    *,
    ctx: RecordGenContext,
    record_id: int,
    output_root: Path,
    total_timesteps: int,
    seed_i: int,
    min_behavior_steps: int,
    behavior_switch_prob: float,
    mix: tuple[BehaviorSpec, ...],
    arrays: dict[str, np.ndarray],
    parallel_envs: int,
    adapt: dict[str, Any] | None = None,
    independent_ally_policies: bool = False,
    mid_episode_policy_switch: bool = True,
    best_teacher_reference: bool = False,
    best_policy: str = BEST_ORACLE,
) -> Path:
    safe_id = package.task_id.replace("/", "_")
    record_dir = (
        Path(output_root)
        / f"task_{package.task_index:05d}_{safe_id}"
        / f"record-{record_id:06d}"
    )
    meta = _record_meta(
        package=package,
        ctx=ctx,
        record_id=record_id,
        total_timesteps=total_timesteps,
        seed_i=seed_i,
        min_behavior_steps=min_behavior_steps,
        behavior_switch_prob=behavior_switch_prob,
        mix=mix,
        arrays=arrays,
        parallel_envs=parallel_envs,
        adapt=adapt,
        independent_ally_policies=independent_ally_policies,
        mid_episode_policy_switch=mid_episode_policy_switch,
        best_teacher_reference=best_teacher_reference,
        best_policy=best_policy,
    )
    write_env_centric_record(record_dir, arrays, meta)
    return record_dir


def generate_one_record_scan(
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
    rollout_fn=None,
    parallel_envs: int = 1,
    adapt: dict[str, Any] | None = None,
    independent_ally_policies: bool = False,
    mid_episode_policy_switch: bool = True,
    best_teacher_reference: bool = False,
    best_policy: str = BEST_ORACLE,
    dump_attack_target: bool = False,
) -> Path:
    """Generate one record (single-env scan). For B>1 use ``generate_records_scan_batch``."""

    from generate.env_centric import build_record_gen_context

    seed_i = sample_record_seed(seed, salt=record_id)
    mix = behavior_mix or DEFAULT_BEHAVIOR_MIX
    cdf = mix_cdf_jax(mix)
    if ctx is None:
        ctx = build_record_gen_context(package)
    elif ctx.package.name != package.name:
        raise ValueError(
            f"RecordGenContext package mismatch: ctx={ctx.package.name} vs {package.name}"
        )

    teacher = str(best_policy if best_teacher_reference else BEST_ORACLE)
    if rollout_fn is None:
        rollout_fn = build_scan_rollout_fn(
            ctx,
            mix=mix,
            min_behavior_steps=min_behavior_steps,
            behavior_switch_prob=behavior_switch_prob,
            total_timesteps=total_timesteps,
            parallel_envs=1,
            independent_ally_policies=independent_ally_policies,
            mid_episode_policy_switch=mid_episode_policy_switch,
            best_teacher_policy=teacher,
            dump_attack_target=dump_attack_target,
        )

    traj = rollout_fn(jnp.asarray(seed_i % (2**32), dtype=jnp.uint32), cdf)
    arrays = trajectory_to_numpy(traj)
    return _write_one_from_arrays(
        package,
        ctx=ctx,
        record_id=record_id,
        output_root=output_root,
        total_timesteps=total_timesteps,
        seed_i=seed_i,
        min_behavior_steps=min_behavior_steps,
        behavior_switch_prob=behavior_switch_prob,
        mix=mix,
        arrays=arrays,
        parallel_envs=max(1, int(parallel_envs)),
        adapt=adapt,
        independent_ally_policies=independent_ally_policies,
        mid_episode_policy_switch=mid_episode_policy_switch,
        best_teacher_reference=best_teacher_reference,
        best_policy=teacher,
    )


def generate_records_scan_batch(
    package: TaskPackage,
    *,
    record_ids: list[int],
    output_root: Path,
    total_timesteps: int,
    seed: int | None,
    min_behavior_steps: int,
    behavior_switch_prob: float,
    behavior_mix: tuple[BehaviorSpec, ...] | None = None,
    ctx: RecordGenContext | None = None,
    rollout_fn=None,
    parallel_envs: int = 1,
    adapt: dict[str, Any] | None = None,
    independent_ally_policies: bool = False,
    mid_episode_policy_switch: bool = True,
    best_teacher_reference: bool = False,
    best_policy: str = BEST_ORACLE,
    dump_attack_target: bool = False,
) -> list[str]:
    """Generate ``len(record_ids)`` records using fixed-B vmap batches + padding."""

    from generate.env_centric import build_record_gen_context

    mix = behavior_mix or DEFAULT_BEHAVIOR_MIX
    cdf = mix_cdf_jax(mix)
    if ctx is None:
        ctx = build_record_gen_context(package)
    elif ctx.package.name != package.name:
        raise ValueError(
            f"RecordGenContext package mismatch: ctx={ctx.package.name} vs {package.name}"
        )

    teacher = str(best_policy if best_teacher_reference else BEST_ORACLE)
    b = max(1, int(parallel_envs))
    if rollout_fn is None:
        rollout_fn = build_scan_rollout_fn(
            ctx,
            mix=mix,
            min_behavior_steps=min_behavior_steps,
            behavior_switch_prob=behavior_switch_prob,
            total_timesteps=total_timesteps,
            parallel_envs=b,
            independent_ally_policies=independent_ally_policies,
            mid_episode_policy_switch=mid_episode_policy_switch,
            best_teacher_policy=teacher,
            dump_attack_target=dump_attack_target,
        )

    paths: list[str] = []
    if b <= 1:
        for record_id in record_ids:
            path = generate_one_record_scan(
                package,
                record_id=int(record_id),
                output_root=output_root,
                total_timesteps=total_timesteps,
                seed=seed,
                min_behavior_steps=min_behavior_steps,
                behavior_switch_prob=behavior_switch_prob,
                behavior_mix=mix,
                ctx=ctx,
                rollout_fn=rollout_fn,
                parallel_envs=1,
                adapt=adapt,
                independent_ally_policies=independent_ally_policies,
                mid_episode_policy_switch=mid_episode_policy_switch,
                best_teacher_reference=best_teacher_reference,
                best_policy=teacher,
                dump_attack_target=dump_attack_target,
            )
            paths.append(str(path))
        return paths

    ids = [int(r) for r in record_ids]
    for start in range(0, len(ids), b):
        chunk = ids[start : start + b]
        seeds_list: list[int] = []
        for rid in chunk:
            seeds_list.append(int(sample_record_seed(seed, salt=rid) % (2**32)))
        valid = len(chunk)
        while len(seeds_list) < b:
            seeds_list.append(0)

        traj = rollout_fn(jnp.asarray(seeds_list, dtype=jnp.uint32), cdf)
        batch = batch_trajectory_to_numpy(traj)
        for i in range(valid):
            rid = chunk[i]
            arrays = {name: value[i] for name, value in batch.items()}
            path = _write_one_from_arrays(
                package,
                ctx=ctx,
                record_id=rid,
                output_root=output_root,
                total_timesteps=total_timesteps,
                seed_i=seeds_list[i],
                min_behavior_steps=min_behavior_steps,
                behavior_switch_prob=behavior_switch_prob,
                mix=mix,
                arrays=arrays,
                parallel_envs=b,
                adapt=adapt,
                independent_ally_policies=independent_ally_policies,
                mid_episode_policy_switch=mid_episode_policy_switch,
                best_teacher_reference=best_teacher_reference,
                best_policy=teacher,
            )
            paths.append(str(path))
        # Padded slots (valid:b) are discarded; keeps a single JIT for fixed B.
    return paths

