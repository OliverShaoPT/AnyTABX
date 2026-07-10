import jax
import jax.numpy as jnp

from src.tabx import TABX, build_batched_env_params_and_config
from src.tabx.wrappers.wrappers import (
    TABXEnemyHeuristicWrapper,
    TABXLogWrapper,
    TABXBalanceWrapper,
)

from pathlib import Path
import numpy as np


def keep_finished_envs(old_tree, new_tree, finished):
    if old_tree is None:
        return new_tree

    def keep_old_when_finished(old_value, new_value):
        mask = finished.reshape((finished.shape[0],) + (1,) * (new_value.ndim - 1))
        return jnp.where(mask, old_value, new_value)

    return jax.tree.map(keep_old_when_finished, old_tree, new_tree)


def mark_finished_dones(dones, finished):
    def mark_done(done):
        mask = finished.reshape((finished.shape[0],) + (1,) * (done.ndim - 1))
        return done | mask

    return jax.tree.map(mark_done, dones)

if __name__ == "__main__":
    n_envs = 100
    max_steps = 512
    scenario_name = "elbow"

    env_params, tabx_config = build_batched_env_params_and_config(
        scenario_names=scenario_name, n_repeat=n_envs
    )
    env = TABX(cfg=tabx_config)
    env = TABXLogWrapper(env)
    env = TABXBalanceWrapper(env)

    v_reset = jax.vmap(env.reset, in_axes=(0, 0))
    v_step = jax.vmap(env.step, in_axes=(0, 0, 0))

    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng)

    obs, env_state = v_reset(jax.random.split(_rng, n_envs), env_params)
    finished = jnp.zeros((n_envs,), dtype=bool)

    def sample_actions(rng, finished):
        rng, action_rng = jax.random.split(rng)
        env_actions = {}
        for agent in env.ally_keys:
            action_rngs = jax.random.split(action_rng, n_envs + 1)
            action_rng = action_rngs[0]
            sampled_actions = jax.vmap(env.action_spaces[env.unit_keys[0]].sample, in_axes=(0))(
                action_rngs[1:]
            )
            env_actions[agent] = jnp.where(finished, 0, sampled_actions)
        return rng, env_actions

    def step_once(obs, env_state, rng, finished, rewards, infos):
        finished_before_step = finished
        rng, env_actions = sample_actions(rng, finished_before_step)

        rng, _rng = jax.random.split(rng)
        next_obs, next_state, next_rewards, next_dones, next_infos = v_step(
            jax.random.split(_rng, n_envs), env_state, env_actions
        )

        obs = keep_finished_envs(obs, next_obs, finished_before_step)
        env_state = keep_finished_envs(env_state, next_state, finished_before_step)
        rewards = keep_finished_envs(rewards, next_rewards, finished_before_step)
        dones = mark_finished_dones(next_dones, finished_before_step)
        infos = keep_finished_envs(infos, next_infos, finished_before_step)

        finished = finished | dones["__all__"].reshape(finished.shape)
        transition = {
            "done": dones,
            "action": env_actions,
            "reward": rewards,
            "obs": obs,
            "info": infos,
        }
        return obs, env_state, rng, finished, rewards, infos, transition

    obs, env_state, rng, finished, rewards, infos, first_transition = step_once(
        obs, env_state, rng, finished, None, None
    )

    def _run(carry, _):
        obs, env_state, rng, finished, rewards, infos = carry
        obs, env_state, rng, finished, rewards, infos, transition = step_once(
            obs, env_state, rng, finished, rewards, infos
        )
        return (obs, env_state, rng, finished, rewards, infos), transition

    # 存储数据
    def flatten_tree(tree, prefix=""):
        flat = {}
        if isinstance(tree, dict):
            for key, value in tree.items():
                name = f"{prefix}/{key}" if prefix else key
                flat.update(flatten_tree(value, name))
        else:
            flat[prefix] = np.asarray(jax.device_get(tree))
        return flat


    (obs, env_state, rng, finished, rewards, infos), rest_trajs = jax.lax.scan(
        _run, (obs, env_state, rng, finished, rewards, infos), None, max_steps - 1
    )
    trajs = jax.tree.map(
        lambda first, rest: jnp.concatenate([first[None], rest], axis=0),
        first_transition,
        rest_trajs,
    )
    final_info = jax.tree.map(lambda x: x[-1], trajs["info"])

    # print("done:", jax.device_get(finished))
    # print("trajs:", jax.device_get(trajs))
    # print("win:", jax.device_get(final_info["returned_episode_wins"]))
    # print("return:", jax.device_get(final_info["returned_episode_returns"]))
    # print("length:", jax.device_get(final_info["returned_episode_lengths"]))



    # 存储数据
    output_path = Path("test2_trajs.npz")
    np.savez_compressed(output_path, **flatten_tree(trajs))
    print("saved trajs to:", output_path.resolve())
    # 计算胜率
    wins = jax.device_get(final_info["returned_episode_wins"])
    print("mean ally win rate:", wins[:, 0].mean())
