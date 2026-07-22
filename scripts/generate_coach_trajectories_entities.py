from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from eval_trained_coach import load_config_from_ckpt, attach_checkpoint, TRAINERS
from src.baseline.utils import load_params


def unwrap_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def to_env_agent(x):
    # trainer._agents returns [A,E,...], convert to [E,A,...]
    if x.ndim == 2:
        return jnp.transpose(x, (1, 0))
    if x.ndim == 3:
        return jnp.transpose(x, (1, 0, 2))
    return x


def generate(trainer, session, seed: int, max_steps: int, deterministic: bool):
    num_envs = trainer.config.NUM_ENVS
    n_agents = trainer.n_agents
    base_env = unwrap_env(trainer.env)
    zone_keys = list(getattr(base_env, "zone_keys", []))
    has_world_state = "world_state" in session.obs

    def rollout(actor_params, rng):
        rng, reset_key = jax.random.split(rng)
        obs, env_state = jax.vmap(trainer.env.reset, in_axes=(0, 0))(
            jax.random.split(reset_key, num_envs),
            trainer.env_params,
        )

        def step(carry, _):
            obs, env_state, rng = carry
            rng, action_key, step_key = jax.random.split(rng, 3)

            avail = trainer._avail(env_state)

            agent_obs = to_env_agent(trainer._agents(obs))                   # [E,A,obs_dim]
            avail_arr = to_env_agent(trainer._agents(avail).astype(bool))    # [E,A,action_dim]

            flat_obs = trainer._flat_agents(obs)
            flat_avail = trainer._flat_agents(avail).astype(bool)
            logits = session.actor.apply_fn(actor_params, flat_obs, flat_avail)

            if deterministic:
                action = jnp.argmax(logits, axis=-1)
            else:
                action = jax.random.categorical(action_key, logits)

            action_arr = jnp.transpose(
                action.reshape(n_agents, num_envs),
                (1, 0),
            )  # [E,A]

            state = env_state["state"]
            gm = state["game_manager"]
            ps = gm.parsed_state

            if has_world_state:
                world_state = jnp.nan_to_num(obs["world_state"])             # [E,world_dim]
            else:
                world_state = jnp.zeros((num_envs, 0), dtype=jnp.float32)

            # Raw entity-level state, used later for KNN/radius local observation.
            unit_positions = ps.positions                                    # [E,N,2]
            unit_healths = ps.healths.squeeze(-1)                            # [E,N]
            unit_max_healths = ps.max_healths.squeeze(-1)                    # [E,N]
            unit_is_alives = ps.is_alives.squeeze(-1)                        # [E,N]
            unit_teams = ps.teams.squeeze(-1)                                # [E,N]
            unit_speeds = ps.speeds.squeeze(-1)                              # [E,N]
            unit_attack_ranges = ps.attack_ranges.squeeze(-1)                # [E,N]
            unit_attack_damages = ps.attack_damages.squeeze(-1)              # [E,N]
            unit_cooldowns = ps.cooldowns.squeeze(-1)                        # [E,N]
            unit_body_radiuss = ps.body_radiuss.squeeze(-1)                  # [E,N]

            visible_matrix = gm.visible_matrix                               # [E,N,N]
            attackable_matrix = gm.attackable_matrix                         # [E,N,N]
            attack_matrix = gm.attack_matrix                                 # [E,N,N]
            distance_matrix = gm.distance_matrix                             # [E,N,N,2]

            if len(zone_keys) > 0:
                zone_types = jnp.stack(
                    [state[z].zone_type.squeeze(-1) for z in zone_keys],
                    axis=1,
                )                                                            # [E,Z]
                zone_positions = jnp.stack(
                    [state[z].ellipse.position for z in zone_keys],
                    axis=1,
                )                                                            # [E,Z,2]
                zone_axes = jnp.stack(
                    [state[z].ellipse.axes for z in zone_keys],
                    axis=1,
                )                                                            # [E,Z,2]
                zone_effect_values = jnp.stack(
                    [state[z].effect_value.squeeze(-1) for z in zone_keys],
                    axis=1,
                )                                                            # [E,Z]
            else:
                zone_types = jnp.zeros((num_envs, 0), dtype=jnp.float32)
                zone_positions = jnp.zeros((num_envs, 0, 2), dtype=jnp.float32)
                zone_axes = jnp.zeros((num_envs, 0, 2), dtype=jnp.float32)
                zone_effect_values = jnp.zeros((num_envs, 0), dtype=jnp.float32)

            next_obs, next_state, reward, done, info = jax.vmap(
                trainer.env.step,
                in_axes=(0, 0, 0, 0),
            )(
                jax.random.split(step_key, num_envs),
                env_state,
                trainer._actions_dict(action),
                trainer.env_params,
            )

            reward_arr = to_env_agent(trainer._agents(reward))               # [E,A]
            done_agent_arr = to_env_agent(trainer._agents(done))             # [E,A]
            done_all = done["__all__"]                                       # [E]

            transition = {
                "world_state": world_state,
                "agent_obs": agent_obs,
                "available_actions": avail_arr,
                "actions": action_arr,
                "rewards": reward_arr,
                "done_agents": done_agent_arr,
                "done_all": done_all,

                "unit_positions": unit_positions,
                "unit_healths": unit_healths,
                "unit_max_healths": unit_max_healths,
                "unit_is_alives": unit_is_alives,
                "unit_teams": unit_teams,
                "unit_speeds": unit_speeds,
                "unit_attack_ranges": unit_attack_ranges,
                "unit_attack_damages": unit_attack_damages,
                "unit_cooldowns": unit_cooldowns,
                "unit_body_radiuss": unit_body_radiuss,

                "visible_matrix": visible_matrix,
                "attackable_matrix": attackable_matrix,
                "attack_matrix": attack_matrix,
                "distance_matrix": distance_matrix,

                "zone_types": zone_types,
                "zone_positions": zone_positions,
                "zone_axes": zone_axes,
                "zone_effect_values": zone_effect_values,
            }

            return (next_obs, next_state, rng), transition

        carry = (obs, env_state, rng)
        carry, traj = jax.lax.scan(step, carry, None, length=max_steps)
        return traj

    traj = jax.jit(rollout)(session.actor.params, jax.random.key(seed))
    traj = jax.tree.map(lambda x: np.asarray(jax.device_get(x)), traj)
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-index", type=int, required=True)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_config_from_ckpt(
        ckpt=ckpt,
        task_index=args.task_index,
        num_envs=args.num_envs,
        max_steps=args.max_steps,
    )

    if cfg.algorithm not in TRAINERS:
        raise ValueError(f"Unsupported algorithm: {cfg.algorithm}")

    trainer = TRAINERS[cfg.algorithm](cfg)
    session = trainer.initialize(jax.random.key(args.seed))
    params = load_params(ckpt)
    session = attach_checkpoint(session, params)

    traj = generate(
        trainer=trainer,
        session=session,
        seed=args.seed,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
    )

    base_env = unwrap_env(trainer.env)

    metadata = {
        "task_index": args.task_index,
        "algorithm": cfg.algorithm,
        "checkpoint": str(ckpt),
        "config_path": str(ckpt.parent / "config.json"),
        "num_envs": args.num_envs,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "deterministic": not args.stochastic,
        "n_agents": trainer.n_agents,
        "obs_dim": trainer.obs_dim,
        "world_dim": trainer.world_dim,
        "action_dim": trainer.action_dim,
        "agents": list(trainer.env.agents),
        "unit_keys": list(getattr(base_env, "unit_keys", [])),
        "ally_keys": list(getattr(base_env, "ally_keys", [])),
        "enemy_keys": list(getattr(base_env, "enemy_keys", [])),
        "zone_keys": list(getattr(base_env, "zone_keys", [])),
        "format": {
            "world_state": "[T,E,world_dim]",
            "agent_obs": "[T,E,A,obs_dim]",
            "available_actions": "[T,E,A,action_dim]",
            "actions": "[T,E,A]",
            "rewards": "[T,E,A]",
            "done_agents": "[T,E,A]",
            "done_all": "[T,E]",
            "unit_positions": "[T,E,N,2]",
            "unit_healths": "[T,E,N]",
            "visible_matrix": "[T,E,N,N]",
            "distance_matrix": "[T,E,N,N,2]",
            "zone_positions": "[T,E,Z,2]",
        },
    }

    np.savez_compressed(
        out,
        metadata=json.dumps(metadata),

        world_state=traj["world_state"],
        agent_obs=traj["agent_obs"],
        available_actions=traj["available_actions"],
        actions=traj["actions"],
        rewards=traj["rewards"],
        done_agents=traj["done_agents"],
        done_all=traj["done_all"],

        unit_positions=traj["unit_positions"],
        unit_healths=traj["unit_healths"],
        unit_max_healths=traj["unit_max_healths"],
        unit_is_alives=traj["unit_is_alives"],
        unit_teams=traj["unit_teams"],
        unit_speeds=traj["unit_speeds"],
        unit_attack_ranges=traj["unit_attack_ranges"],
        unit_attack_damages=traj["unit_attack_damages"],
        unit_cooldowns=traj["unit_cooldowns"],
        unit_body_radiuss=traj["unit_body_radiuss"],

        visible_matrix=traj["visible_matrix"],
        attackable_matrix=traj["attackable_matrix"],
        attack_matrix=traj["attack_matrix"],
        distance_matrix=traj["distance_matrix"],

        zone_types=traj["zone_types"],
        zone_positions=traj["zone_positions"],
        zone_axes=traj["zone_axes"],
        zone_effect_values=traj["zone_effect_values"],
    )

    print("saved:", out)
    for k, v in traj.items():
        print(k, v.shape)


if __name__ == "__main__":
    main()
