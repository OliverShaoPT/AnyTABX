from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange

from src.tabx import TABX, build_batched_env_params_and_config
from src.tabx.visualize import Visualizer
from src.tabx.wrappers.wrappers import TABXBalanceWrapper, TABXLogWrapper


# ===================== User config =====================
TRAJ_PATH = Path(r"E:\Project\AIRS_ICRA\test2_trajs.npz")
OUTPUT_DIR = Path("visualized_trajs")

n_envs = 100
ENV_INDICES = [3, 60, 78]

# 1 means ENV_INDICES=[3] renders the 3rd environment.
# Change to 0 if you want Python-style indexing, where [3] renders env index 3.
ENV_INDEX_BASE = 1

SCENARIO_NAME = "elbow"
SEED = 0
MAX_STEPS = None  # None uses all steps from TRAJ_PATH.
FRAME_STRIDE = 2  # Render every N steps to keep GIF size reasonable.
INTERVAL_MS = 80
# =======================================================


def keep_finished_envs(old_tree, new_tree, finished):
    def keep_old_when_finished(old_value, new_value):
        mask = finished.reshape((finished.shape[0],) + (1,) * (new_value.ndim - 1))
        return jnp.where(mask, old_value, new_value)

    return jax.tree.map(keep_old_when_finished, old_tree, new_tree)


def normalize_env_indices(env_indices, index_base, num_envs):
    normalized = [idx - index_base for idx in env_indices]
    invalid = [idx for idx in normalized if idx < 0 or idx >= num_envs]
    if invalid:
        original_invalid = [idx + index_base for idx in invalid]
        raise ValueError(
            f"Invalid ENV_INDICES {original_invalid}; valid range is "
            f"{index_base}..{num_envs - 1 + index_base}."
        )
    return normalized


def sample_actions(env, rng, finished, num_envs):
    rng, action_rng = jax.random.split(rng)
    env_actions = {}
    for agent in env.ally_keys:
        action_rngs = jax.random.split(action_rng, num_envs + 1)
        action_rng = action_rngs[0]
        sampled_actions = jax.vmap(env.action_spaces[env.unit_keys[0]].sample, in_axes=0)(
            action_rngs[1:]
        )
        env_actions[agent] = jnp.where(finished, 0, sampled_actions)
    return rng, env_actions


def main():
    trajs = np.load(TRAJ_PATH)
    saved_steps = trajs["done/__all__"].shape[0]
    saved_n_envs = trajs["done/__all__"].shape[1]

    if n_envs != saved_n_envs:
        raise ValueError(f"n_envs={n_envs}, but {TRAJ_PATH} contains {saved_n_envs} envs.")

    max_steps = saved_steps if MAX_STEPS is None else min(MAX_STEPS, saved_steps)
    env_indices = normalize_env_indices(ENV_INDICES, ENV_INDEX_BASE, n_envs)

    env_params, tabx_config = build_batched_env_params_and_config(
        scenario_names=SCENARIO_NAME,
        n_repeat=n_envs,
    )
    env = TABX(cfg=tabx_config)
    env = TABXLogWrapper(env)
    env = TABXBalanceWrapper(env)

    v_reset = jax.vmap(env.reset, in_axes=(0, 0))
    v_step = jax.vmap(env.step, in_axes=(0, 0, 0))

    rng = jax.random.PRNGKey(SEED)
    rng, reset_rng = jax.random.split(rng)
    _, env_state = v_reset(jax.random.split(reset_rng, n_envs), env_params)

    finished = jnp.zeros((n_envs,), dtype=bool)
    selected_state_seq = {env_idx: [] for env_idx in env_indices}

    for step in trange(max_steps, desc="Replaying trajectories"):
        finished_before_step = finished
        rng, env_actions = sample_actions(env, rng, finished_before_step, n_envs)

        rng, step_rng = jax.random.split(rng)
        _, next_state, _, dones, _ = v_step(
            jax.random.split(step_rng, n_envs),
            env_state,
            env_actions,
        )

        env_state = keep_finished_envs(env_state, next_state, finished_before_step)
        finished = finished | dones["__all__"].reshape(finished.shape)

        if step % FRAME_STRIDE == 0:
            for env_idx in env_indices:
                selected_state_seq[env_idx].append(jax.tree.map(lambda x: x[env_idx], env_state))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for env_idx in env_indices:
        display_idx = env_idx + ENV_INDEX_BASE
        output_path = OUTPUT_DIR / f"{SCENARIO_NAME}_env_{display_idx}.gif"
        visualizer = Visualizer(
            env,
            selected_state_seq[env_idx],
            interval=INTERVAL_MS,
        )
        visualizer.animate(save_fname=str(output_path), view=False)
        print(f"saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
