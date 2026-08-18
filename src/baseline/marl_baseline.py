"""Standalone, chunkable trainers for fixed TABX tasks.

All runner state (environment, optimizer, parameters, RNG and update counter)
is passed from one update to the next.  Host chunks are synchronization and
logging boundaries only: no prefix is replayed and every update runs once.
"""

from __future__ import annotations

import csv
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import wandb
from flax import struct
from flax.training.train_state import TrainState

from src.baseline.utils import save_params
from src.tabx import TABX
from src.tabx.sample_task import (
    build_batched_env_params_from_tasks,
    load_task_bank,
    save_task_bank,
)
from src.tabx.wrappers.wrappers import (
    TABXAutoResetWrapper,
    TABXEnemyAllyFlipWrapper,
    TABXEnemyHeuristicWrapper,
    TABXLogWrapper,
)

Algorithm = Literal["ippo", "mappo", "mappo_rnd", "iql", "vdn", "qmix"]
SUPPORTED_ALGORITHMS = ("ippo", "mappo", "mappo_rnd", "iql", "vdn", "qmix")


@dataclass
class MARLConfig:
    algorithm: Algorithm = "mappo"
    task_file_path: str = "task_outputs/tasks_20.json"
    task_index: int = 0
    save_path: str = "./ckpt/marl_baseline"
    seed: int = 0
    NUM_ENVS: int = 16
    NUM_STEPS: int = 128
    TOTAL_TIMESTEPS: int = 2_000_000
    host_chunk_updates: int = 10
    early_stop_enabled: bool = True
    early_stop_debug_mode: bool = False
    early_stop_window: int = 10
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.0
    early_stop_warmup: int = 10
    wandb_mode: Literal["online", "offline", "disabled"] = "disabled"
    wandb_run_name: str | None = None
    wandb_project: str = "tabx_marl_baseline"
    jit: bool = True
    PHYSICS: str | None = None
    HEURISTIC: str | None = None
    WORLD_STATE_TYPE: Literal["concat", "global"] = "global"
    POSITION_PERMUTATION: bool = False
    FLIP: bool = False
    # Post-train oracle_pure vs heuristic_advanced eval → task.json metadata.coach_eval
    COACH_EVAL: bool = True
    COACH_EVAL_EPISODES: int = 64
    COACH_EVAL_TIE_EPS: float = 0.02
    COACH_EVAL_MAX_EPISODE_STEPS: int = 512
    # vmap batch width for coach_eval episodes (capped by COACH_EVAL_EPISODES).
    COACH_EVAL_PARALLEL_ENVS: int = 32
    # Shared optimization
    LR: float = 4e-4
    GAMMA: float = 0.99
    MAX_GRAD_NORM: float = 0.5
    HIDDEN_SIZE: int = 128
    ANNEAL_LR: bool = True
    # PPO
    UPDATE_EPOCHS: int = 4
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.1
    ENT_COEF: float = 0.01
    VF_COEF: float = 0.5
    # RND
    RND_HIDDEN_DIM: int = 128
    RND_OUTPUT_DIM: int = 128
    RND_LR: float = 3e-4
    RND_REWARD_COEF: float = 0.5
    RND_LOSS_COEF: float = 0.01
    # Value based
    EPS_START: float = 1.0
    EPS_FINISH: float = 0.05
    EPS_DECAY: float = 0.1
    TARGET_UPDATE_INTERVAL: int = 10
    TAU: float = 1.0
    NUM_EPOCHS: int = 2
    REW_SCALE: float = 10.0
    MIXER_EMBEDDING_DIM: int = 64

    def validate(self) -> None:
        if self.algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm {self.algorithm!r}.")
        values = {
            "NUM_ENVS": self.NUM_ENVS,
            "NUM_STEPS": self.NUM_STEPS,
            "TOTAL_TIMESTEPS": self.TOTAL_TIMESTEPS,
            "host_chunk_updates": self.host_chunk_updates,
            "early_stop_window": self.early_stop_window,
            "early_stop_patience": self.early_stop_patience,
            "REW_SCALE": self.REW_SCALE,
        }
        invalid = [name for name, value in values.items() if value <= 0]
        if invalid:
            raise ValueError(f"These values must be positive: {invalid}.")
        if self.task_index < 0 or self.early_stop_warmup < 0:
            raise ValueError("task_index and early_stop_warmup must be non-negative.")
        if self.TOTAL_TIMESTEPS < self.NUM_ENVS * self.NUM_STEPS:
            raise ValueError("TOTAL_TIMESTEPS must contain at least one update.")
        if not 0.0 <= self.EPS_DECAY <= 1.0:
            raise ValueError("EPS_DECAY must be in [0, 1].")


@dataclass(frozen=True)
class TrainingSummary:
    algorithm: str
    task_index: int
    task_id: str
    seed: int
    completed_updates: int
    completed_timesteps: int
    best_score: float
    stopped_early: bool
    early_stop_trigger_count: int
    output_dir: str
    metrics_path: str
    early_stop_events_path: str
    best_checkpoint: str
    final_checkpoint: str


@dataclass(frozen=True)
class EarlyStopStatus:
    score: float
    best: float
    improved: bool
    ready: bool
    bad_updates: int
    triggered: bool
    would_stop: bool


class EarlyStopper:
    """Evaluate one rolling window per update."""

    def __init__(self, config: MARLConfig):
        self.enabled = config.early_stop_enabled
        self.window = config.early_stop_window
        self.patience = config.early_stop_patience
        self.min_delta = config.early_stop_min_delta
        self.warmup = config.early_stop_warmup
        self.returns: list[float] = []
        self.best = -float("inf")
        self.bad_updates = 0
        self.trigger_active = False
        self.trigger_count = 0

    def update(self, episode_return: float, update: int) -> EarlyStopStatus:
        self.returns.append(float(episode_return))
        score = float(np.mean(self.returns[-self.window :]))
        ready = update >= self.warmup and len(self.returns) >= self.window
        if not ready:
            return EarlyStopStatus(
                score, self.best, False, False, self.bad_updates, False, False
            )
        improved = score > self.best + self.min_delta
        if improved:
            self.best = score
            self.bad_updates = 0
            self.trigger_active = False
        else:
            self.bad_updates += 1
        would_stop = self.enabled and self.bad_updates >= self.patience
        triggered = would_stop and not self.trigger_active
        if triggered:
            self.trigger_active = True
            self.trigger_count += 1
        return EarlyStopStatus(
            score,
            self.best,
            improved,
            True,
            self.bad_updates,
            triggered,
            would_stop,
        )


class MetricRecorder:
    """Append and flush one CSV record per update for live inspection."""

    def __init__(self, path: Path):
        self.path = path
        self.stream = path.open("w", encoding="utf-8", newline="")
        self.writer: csv.DictWriter[str] | None = None

    def write(self, record: dict[str, Any]) -> None:
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(record))
            self.writer.writeheader()
        self.writer.writerow(record)
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()


class PolicyNetwork(nn.Module):
    action_dim: int
    hidden: int

    @nn.compact
    def __call__(self, obs: jax.Array, avail: jax.Array) -> jax.Array:
        x = nn.relu(nn.Dense(self.hidden)(obs))
        x = nn.relu(nn.Dense(self.hidden)(x))
        logits = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(x)
        return jnp.where(avail, logits, -1e10)


class ValueNetwork(nn.Module):
    hidden: int

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        x = nn.relu(nn.Dense(self.hidden)(obs))
        x = nn.relu(nn.Dense(self.hidden)(x))
        return nn.Dense(1)(x).squeeze(-1)


class QNetwork(nn.Module):
    action_dim: int
    hidden: int

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        x = nn.relu(nn.Dense(self.hidden)(obs))
        x = nn.relu(nn.Dense(self.hidden)(x))
        return nn.Dense(self.action_dim)(x)


class RNDNetwork(nn.Module):
    hidden: int
    output: int

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        x = nn.relu(nn.Dense(self.hidden)(obs))
        x = nn.relu(nn.Dense(self.hidden)(x))
        return nn.Dense(self.output)(x)


class MonotonicMixer(nn.Module):
    n_agents: int
    embedding: int

    @nn.compact
    def __call__(self, agent_q: jax.Array, state: jax.Array) -> jax.Array:
        state = nn.LayerNorm()(state)
        small = nn.initializers.orthogonal(0.001)
        weights = jnp.abs(nn.Dense(self.n_agents, kernel_init=small)(state))
        bias = nn.Dense(self.embedding, kernel_init=small)(state)
        bias = nn.relu(bias)
        bias = nn.Dense(1, kernel_init=small)(bias).squeeze(-1)
        return jnp.sum(agent_q * weights, axis=-1) + bias


class PPOTrajectory(NamedTuple):
    obs: jax.Array
    critic_obs: jax.Array
    avail: jax.Array
    action: jax.Array
    old_log_prob: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array


class QTrajectory(NamedTuple):
    obs: jax.Array
    next_obs: jax.Array
    world: jax.Array
    next_world: jax.Array
    avail: jax.Array
    next_avail: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array


@struct.dataclass
class PPOSession:
    actor: TrainState
    critic: TrainState
    env_state: Any
    obs: Any
    dones: jax.Array
    rng: jax.Array
    update: jax.Array
    rnd_predictor: TrainState | None = None
    rnd_target: Any = None


@struct.dataclass
class QSession:
    learner: TrainState
    target_params: Any
    env_state: Any
    obs: Any
    dones: Any
    rng: jax.Array
    update: jax.Array


def _tree_ready(tree: Any) -> Any:
    return jax.tree.map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        tree,
    )


class BaseMARLTrainer(ABC):
    """Owns fixed-task environment setup and the real host update loop."""

    def __init__(self, config: MARLConfig):
        config.validate()
        self.config = config
        bank = load_task_bank(config.task_file_path)
        if config.task_index >= len(bank["tasks"]):
            raise IndexError(
                f"task_index={config.task_index}, but the bank has {len(bank['tasks'])} tasks."
            )
        self.task = bank["tasks"][config.task_index]
        self.task_id = str(self.task.get("task_id", f"task_{config.task_index:06d}"))
        manifest = bank["manifest"]
        schema = manifest["schema"]
        self.physics = config.PHYSICS or manifest["physics"]
        self.heuristic = config.HEURISTIC or manifest["heuristic"]
        self.env_params, tabx_config = build_batched_env_params_from_tasks(
            [self.task] * config.NUM_ENVS,
            physics=self.physics,
            heuristic=self.heuristic,
            max_n_ally=int(schema["max_n_ally"]),
            max_n_enemy=int(schema["max_n_enemy"]),
            max_n_zone=int(schema["max_n_zone"]),
        )
        env = TABX(
            cfg=tabx_config,
            world_state_type=config.WORLD_STATE_TYPE,
            position_permutation=config.POSITION_PERMUTATION,
        )
        if config.FLIP:
            env = TABXEnemyAllyFlipWrapper(env)
        env = TABXLogWrapper(env)
        env = TABXEnemyHeuristicWrapper(env)
        self.env = TABXAutoResetWrapper(env)
        self.n_agents = self.env.num_agents
        self.n_actors = self.n_agents * config.NUM_ENVS
        self.action_dim = self.env.action_space(self.env.agents[0]).n
        self.obs_dim = self.env.observation_space(self.env.agents[0]).shape[0]
        self.world_dim = self.env.world_state_size()
        self.total_updates = (
            config.TOTAL_TIMESTEPS // config.NUM_ENVS // config.NUM_STEPS
        )
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.task_id)
        self.output = (
            Path(config.save_path)
            / config.algorithm
            / f"task-{config.task_index:06d}-{safe_id}"
            / f"seed-{config.seed}"
        )
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "config.json").write_text(
            json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
        )
        # Self-contained package leaf for generate-record: task.json next to weights.
        save_task_bank(
            self.output / "task.json",
            [self.task],
            seed=int(manifest.get("seed", config.seed)),
            physics=self.physics,
            heuristic=self.heuristic,
            max_n_ally=int(schema["max_n_ally"]),
            max_n_enemy=int(schema["max_n_enemy"]),
            max_n_zone=int(schema["max_n_zone"]),
            filter_protocol=manifest.get("filter_protocol"),
        )
        (self.output / "meta.json").write_text(
            json.dumps(
                {
                    "task_index": config.task_index,
                    "task_id": self.task_id,
                    "seed": config.seed,
                    "algorithm": config.algorithm,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.best_path = self.output / "best.safetensors"
        self.final_path = self.output / "final.safetensors"
        self.metrics_path = self.output / "training_metrics.csv"
        self.early_stop_events_path = self.output / "early_stop_events.jsonl"
        self._compiled_update = jax.jit(self.update_once) if config.jit else self.update_once

    def _agents(self, values: dict[str, jax.Array]) -> jax.Array:
        return jnp.nan_to_num(
            jnp.stack([values[agent] for agent in self.env.agents])
        )

    def _flat_agents(self, values: dict[str, jax.Array]) -> jax.Array:
        array = self._agents(values)
        return array.reshape((self.n_actors, *array.shape[2:]))

    def _actions_dict(self, actions: jax.Array) -> dict[str, jax.Array]:
        actions = actions.reshape(self.n_agents, self.config.NUM_ENVS)
        return {agent: actions[index] for index, agent in enumerate(self.env.agents)}

    def _avail(self, env_state: Any) -> dict[str, jax.Array]:
        return jax.vmap(self.env.get_avail_actions)(env_state)

    def _lr(self, steps_per_update: int = 1):
        if not self.config.ANNEAL_LR:
            return self.config.LR
        return optax.linear_schedule(
            self.config.LR,
            0.0,
            max(1, self.total_updates * steps_per_update),
        )

    def _tx(self, learning_rate: Any) -> optax.GradientTransformation:
        return optax.chain(
            optax.clip_by_global_norm(self.config.MAX_GRAD_NORM),
            optax.adam(learning_rate),
        )

    @abstractmethod
    def initialize(self, rng: jax.Array) -> Any:
        raise NotImplementedError

    @abstractmethod
    def update_once(self, session: Any) -> tuple[Any, dict[str, jax.Array]]:
        raise NotImplementedError

    @abstractmethod
    def checkpoint_params(self, session: Any) -> Any:
        raise NotImplementedError

    def train(self) -> TrainingSummary:
        run = wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            mode=self.config.wandb_mode,
            config=asdict(self.config),
        )
        stopper = EarlyStopper(self.config)
        recorder = MetricRecorder(self.metrics_path)
        self.early_stop_events_path.write_text("", encoding="utf-8")
        session = self.initialize(jax.random.key(self.config.seed))
        stopped = False
        completed = 0
        started_at = time.monotonic()
        try:
            while completed < self.total_updates and not stopped:
                chunk_end = min(
                    completed + self.config.host_chunk_updates, self.total_updates
                )
                while completed < chunk_end:
                    session, metrics = self._compiled_update(session)
                    session, metrics = _tree_ready((session, metrics))
                    completed += 1
                    episode_return = float(np.asarray(metrics["episode_returns"]))
                    status = stopper.update(episode_return, completed)
                    stopped = status.triggered and not self.config.early_stop_debug_mode
                    host_metrics = {
                        key: float(np.asarray(value)) for key, value in metrics.items()
                    }
                    host_metrics.update(
                        {
                            "early_stop/window_return": status.score,
                            "early_stop/best_return": status.best,
                            "early_stop/bad_updates": status.bad_updates,
                            "early_stop/triggered": int(status.triggered),
                            "early_stop/would_stop": int(status.would_stop),
                            "early_stop/trigger_count": stopper.trigger_count,
                            "update_steps": completed,
                            "env_steps": (
                                completed
                                * self.config.NUM_ENVS
                                * self.config.NUM_STEPS
                            ),
                        }
                    )
                    wandb.log(host_metrics)
                    recorder.write(
                        {
                            "elapsed_seconds": time.monotonic() - started_at,
                            **host_metrics,
                            "early_stop/enabled": int(self.config.early_stop_enabled),
                            "early_stop/debug_mode": int(
                                self.config.early_stop_debug_mode
                            ),
                            "early_stop/window": self.config.early_stop_window,
                            "early_stop/patience": self.config.early_stop_patience,
                            "early_stop/min_delta": self.config.early_stop_min_delta,
                            "early_stop/warmup": self.config.early_stop_warmup,
                        }
                    )
                    if status.triggered:
                        event = {
                            "update_steps": completed,
                            "env_steps": host_metrics["env_steps"],
                            "episode_returns": episode_return,
                            "rolling_return": status.score,
                            "best_return": status.best,
                            "bad_updates": status.bad_updates,
                            "trigger_count": stopper.trigger_count,
                            "debug_mode": self.config.early_stop_debug_mode,
                            "continued_training": self.config.early_stop_debug_mode,
                            "window": self.config.early_stop_window,
                            "patience": self.config.early_stop_patience,
                            "min_delta": self.config.early_stop_min_delta,
                            "warmup": self.config.early_stop_warmup,
                        }
                        with self.early_stop_events_path.open(
                            "a", encoding="utf-8"
                        ) as stream:
                            stream.write(json.dumps(event, sort_keys=True) + "\n")
                        wandb.log(
                            {
                                f"early_stop_event/{key}": value
                                for key, value in event.items()
                                if isinstance(value, (bool, int, float))
                            }
                        )
                    if status.improved:
                        save_params(self.checkpoint_params(session), self.best_path)
                    if stopped:
                        break
            save_params(self.checkpoint_params(session), self.final_path)
            if not self.best_path.exists():
                save_params(self.checkpoint_params(session), self.best_path)
                stopper.best = float(np.mean(stopper.returns[-stopper.window :]))
            if self.config.COACH_EVAL:
                try:
                    from generate.coach_eval import evaluate_and_write_coach_leaf

                    eval_result = evaluate_and_write_coach_leaf(
                        self.output,
                        num_episodes=int(self.config.COACH_EVAL_EPISODES),
                        seed=int(self.config.seed),
                        max_episode_steps=int(
                            self.config.COACH_EVAL_MAX_EPISODE_STEPS
                        ),
                        tie_eps=float(self.config.COACH_EVAL_TIE_EPS),
                        parallel_envs=int(self.config.COACH_EVAL_PARALLEL_ENVS),
                    )
                    print(
                        f"[coach_eval] wrote task.json metadata.coach_eval "
                        f"best_policy={eval_result.get('best_policy')} "
                        f"oracle={eval_result['oracle_pure']['win_rate']:.3f} "
                        f"advanced={eval_result['heuristic_advanced']['win_rate']:.3f}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[coach_eval] skipped after train failure: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
        finally:
            recorder.close()
            run.finish()
        return TrainingSummary(
            algorithm=self.config.algorithm,
            task_index=self.config.task_index,
            task_id=self.task_id,
            seed=self.config.seed,
            completed_updates=completed,
            completed_timesteps=(
                completed * self.config.NUM_ENVS * self.config.NUM_STEPS
            ),
            best_score=stopper.best,
            stopped_early=stopped,
            early_stop_trigger_count=stopper.trigger_count,
            output_dir=str(self.output),
            metrics_path=str(self.metrics_path),
            early_stop_events_path=str(self.early_stop_events_path),
            best_checkpoint=str(self.best_path),
            final_checkpoint=str(self.final_path),
        )


class PPOTrainer(BaseMARLTrainer):
    centralized_critic = False
    use_rnd = False

    def initialize(self, rng: jax.Array) -> PPOSession:
        rng, reset_key, actor_key, critic_key, rnd_key, target_key = jax.random.split(rng, 6)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, 0))(
            jax.random.split(reset_key, self.config.NUM_ENVS), self.env_params
        )
        actor_model = PolicyNetwork(self.action_dim, self.config.HIDDEN_SIZE)
        critic_model = ValueNetwork(self.config.HIDDEN_SIZE)
        actor_params = actor_model.init(
            actor_key,
            jnp.zeros((1, self.obs_dim)),
            jnp.ones((1, self.action_dim), dtype=bool),
        )
        critic_dim = self.world_dim if self.centralized_critic else self.obs_dim
        critic_params = critic_model.init(critic_key, jnp.zeros((1, critic_dim)))
        actor = TrainState.create(
            apply_fn=actor_model.apply,
            params=actor_params,
            tx=self._tx(self._lr(self.config.UPDATE_EPOCHS)),
        )
        critic = TrainState.create(
            apply_fn=critic_model.apply,
            params=critic_params,
            tx=self._tx(self._lr(self.config.UPDATE_EPOCHS)),
        )
        predictor = None
        target = None
        if self.use_rnd:
            rnd = RNDNetwork(self.config.RND_HIDDEN_DIM, self.config.RND_OUTPUT_DIM)
            dummy = jnp.zeros((1, self.world_dim))
            predictor = TrainState.create(
                apply_fn=rnd.apply,
                params=rnd.init(rnd_key, dummy),
                tx=self._tx(self.config.RND_LR),
            )
            target = rnd.init(target_key, dummy)
        return PPOSession(
            actor=actor,
            critic=critic,
            env_state=env_state,
            obs=obs,
            dones=jnp.zeros(self.n_actors, dtype=bool),
            rng=rng,
            update=jnp.asarray(0),
            rnd_predictor=predictor,
            rnd_target=target,
        )

    def _critic_obs(self, obs: dict[str, jax.Array]) -> jax.Array:
        if self.centralized_critic:
            return jnp.tile(jnp.nan_to_num(obs["world_state"]), (self.n_agents, 1))
        return self._flat_agents(obs)

    def update_once(self, session: PPOSession):
        def env_step(carry, _):
            actor, critic, env_state, obs, dones, rng = carry
            rng, action_key, step_key = jax.random.split(rng, 3)
            flat_obs = self._flat_agents(obs)
            avail = self._flat_agents(self._avail(env_state)).astype(bool)
            critic_obs = self._critic_obs(obs)
            logits = actor.apply_fn(actor.params, flat_obs, avail)
            action = jax.random.categorical(action_key, logits)
            log_prob = jnp.take_along_axis(
                jax.nn.log_softmax(logits), action[:, None], axis=-1
            ).squeeze(-1)
            value = critic.apply_fn(critic.params, critic_obs)
            next_obs, next_state, reward, done, _ = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, 0)
            )(
                jax.random.split(step_key, self.config.NUM_ENVS),
                env_state,
                self._actions_dict(action),
                self.env_params,
            )
            rewards = self._flat_agents(reward).squeeze()
            global_done = jnp.tile(done["__all__"], self.n_agents)
            trajectory = PPOTrajectory(
                flat_obs,
                critic_obs,
                avail,
                action,
                log_prob,
                value,
                rewards,
                global_done,
            )
            return (
                actor,
                critic,
                next_state,
                next_obs,
                self._flat_agents(done).squeeze(),
                rng,
            ), trajectory

        carry = (
            session.actor,
            session.critic,
            session.env_state,
            session.obs,
            session.dones,
            session.rng,
        )
        carry, trajectory = jax.lax.scan(
            env_step, carry, None, self.config.NUM_STEPS
        )
        actor, critic, env_state, obs, dones, rng = carry
        extrinsic_reward = trajectory.reward
        reward = extrinsic_reward
        rnd_predictor = session.rnd_predictor
        rnd_loss = jnp.asarray(0.0)
        if self.use_rnd:
            target_features = rnd_predictor.apply_fn(
                session.rnd_target, trajectory.critic_obs
            )
            predicted_features = rnd_predictor.apply_fn(
                rnd_predictor.params, trajectory.critic_obs
            )
            intrinsic = jnp.mean((predicted_features - target_features) ** 2, axis=-1)
            reward = reward + self.config.RND_REWARD_COEF * jax.lax.stop_gradient(
                intrinsic
            )

            def rnd_objective(params):
                prediction = rnd_predictor.apply_fn(params, trajectory.critic_obs)
                return self.config.RND_LOSS_COEF * jnp.mean(
                    (prediction - jax.lax.stop_gradient(target_features)) ** 2
                )

            rnd_loss, rnd_grads = jax.value_and_grad(rnd_objective)(
                rnd_predictor.params
            )
            rnd_predictor = rnd_predictor.apply_gradients(grads=rnd_grads)
        last_value = critic.apply_fn(critic.params, self._critic_obs(obs))

        def gae_step(carry, transition):
            gae, next_value = carry
            value, rew, done = transition
            delta = rew + self.config.GAMMA * next_value * (1 - done) - value
            gae = (
                delta
                + self.config.GAMMA
                * self.config.GAE_LAMBDA
                * (1 - done)
                * gae
            )
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            gae_step,
            (jnp.zeros_like(last_value), last_value),
            (trajectory.value, reward, trajectory.done),
            reverse=True,
        )
        targets = advantages + trajectory.value
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        def epoch_step(states, _):
            actor_state, critic_state = states

            def actor_loss_fn(params):
                logits = actor_state.apply_fn(
                    params, trajectory.obs, trajectory.avail
                )
                all_log_probs = jax.nn.log_softmax(logits)
                log_prob = jnp.take_along_axis(
                    all_log_probs, trajectory.action[..., None], axis=-1
                ).squeeze(-1)
                ratio = jnp.exp(log_prob - trajectory.old_log_prob)
                clipped = jnp.clip(
                    ratio, 1 - self.config.CLIP_EPS, 1 + self.config.CLIP_EPS
                )
                actor_loss = -jnp.mean(
                    jnp.minimum(ratio * advantages, clipped * advantages)
                )
                entropy = -jnp.mean(jnp.sum(jnp.exp(all_log_probs) * all_log_probs, axis=-1))
                return actor_loss - self.config.ENT_COEF * entropy, (
                    actor_loss,
                    entropy,
                )

            def critic_loss_fn(params):
                values = critic_state.apply_fn(params, trajectory.critic_obs)
                return self.config.VF_COEF * jnp.mean((values - targets) ** 2)

            (actor_total, actor_aux), actor_grads = jax.value_and_grad(
                actor_loss_fn, has_aux=True
            )(actor_state.params)
            critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(
                critic_state.params
            )
            actor_state = actor_state.apply_gradients(grads=actor_grads)
            critic_state = critic_state.apply_gradients(grads=critic_grads)
            return (actor_state, critic_state), (
                actor_total,
                actor_aux[0],
                actor_aux[1],
                critic_loss,
            )

        (actor, critic), losses = jax.lax.scan(
            epoch_step,
            (actor, critic),
            None,
            self.config.UPDATE_EPOCHS,
        )
        episode_return = env_state["log_state"].returned_episode_returns[:, 0].mean()
        new_session = session.replace(
            actor=actor,
            critic=critic,
            env_state=env_state,
            obs=obs,
            dones=dones,
            rng=rng,
            update=session.update + 1,
            rnd_predictor=rnd_predictor,
        )
        return new_session, {
            "episode_returns": episode_return,
            "rollout_reward_mean": extrinsic_reward.mean(),
            "loss/actor": losses[1].mean(),
            "loss/critic": losses[3].mean(),
            "loss/entropy": losses[2].mean(),
            "loss/rnd": rnd_loss,
        }

    def checkpoint_params(self, session: PPOSession) -> Any:
        params = {"actor": session.actor.params, "critic": session.critic.params}
        if session.rnd_predictor is not None:
            params["rnd_predictor"] = session.rnd_predictor.params
            params["rnd_target"] = session.rnd_target
        return params


class IPPOTrainer(PPOTrainer):
    """Independent value estimates from each agent's local observation."""


class MAPPOTrainer(PPOTrainer):
    """Shared actor with centralized world-state critic."""

    centralized_critic = True


class MAPPORNDTrainer(MAPPOTrainer):
    """MAPPO plus a trainable RND predictor and fixed target."""

    use_rnd = True


class ValueBasedTrainer(BaseMARLTrainer):
    mixing: Literal["individual", "sum", "qmix"] = "individual"

    def initialize(self, rng: jax.Array) -> QSession:
        rng, reset_key, network_key, mixer_key = jax.random.split(rng, 4)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, 0))(
            jax.random.split(reset_key, self.config.NUM_ENVS), self.env_params
        )
        network = QNetwork(self.action_dim, self.config.HIDDEN_SIZE)
        params: dict[str, Any] = {
            "agent": network.init(network_key, jnp.zeros((1, self.obs_dim)))
        }
        if self.mixing == "qmix":
            mixer = MonotonicMixer(self.n_agents, self.config.MIXER_EMBEDDING_DIM)
            params["mixer"] = mixer.init(
                mixer_key,
                jnp.zeros((1, self.n_agents)),
                jnp.zeros((1, self.world_dim)),
            )
        learner = TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=self._tx(self._lr(self.config.NUM_EPOCHS)),
        )
        dones = {
            agent: jnp.zeros(self.config.NUM_ENVS, dtype=bool)
            for agent in self.env.agents + ["__all__"]
        }
        return QSession(
            learner=learner,
            target_params=params,
            env_state=env_state,
            obs=obs,
            dones=dones,
            rng=rng,
            update=jnp.asarray(0),
        )

    def _epsilon(self, update: jax.Array) -> jax.Array:
        decay_updates = max(1, int(self.total_updates * self.config.EPS_DECAY))
        fraction = jnp.minimum(update / decay_updates, 1.0)
        return self.config.EPS_START + fraction * (
            self.config.EPS_FINISH - self.config.EPS_START
        )

    def update_once(self, session: QSession):
        network = QNetwork(self.action_dim, self.config.HIDDEN_SIZE)

        def env_step(carry, _):
            env_state, obs, dones, rng = carry
            rng, explore_key, random_key, step_key = jax.random.split(rng, 4)
            flat_obs = self._flat_agents(obs)
            avail_dict = self._avail(env_state)
            avail = self._agents(avail_dict).astype(bool)
            q_values = network.apply(session.learner.params["agent"], flat_obs)
            q_values = q_values.reshape(
                self.n_agents, self.config.NUM_ENVS, self.action_dim
            )
            greedy = jnp.argmax(jnp.where(avail, q_values, -1e10), axis=-1)
            random_logits = jnp.where(avail, 0.0, -1e10)
            random_actions = jax.random.categorical(random_key, random_logits)
            explore = jax.random.uniform(explore_key, greedy.shape) < self._epsilon(
                session.update
            )
            actions = jnp.where(explore, random_actions, greedy)
            action_dict = {
                agent: actions[index] for index, agent in enumerate(self.env.agents)
            }
            next_obs, next_state, rewards, next_dones, _ = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, 0)
            )(
                jax.random.split(step_key, self.config.NUM_ENVS),
                env_state,
                action_dict,
                self.env_params,
            )
            next_avail = self._agents(self._avail(next_state)).astype(bool)
            transition = QTrajectory(
                self._agents(obs),
                self._agents(next_obs),
                jnp.nan_to_num(obs["world_state"]),
                jnp.nan_to_num(next_obs["world_state"]),
                avail,
                next_avail,
                actions,
                self._agents(rewards) * self.config.REW_SCALE,
                next_dones["__all__"],
            )
            return (next_state, next_obs, next_dones, rng), transition

        (env_state, obs, dones, rng), trajectory = jax.lax.scan(
            env_step,
            (session.env_state, session.obs, session.dones, session.rng),
            None,
            self.config.NUM_STEPS,
        )
        mixer = MonotonicMixer(self.n_agents, self.config.MIXER_EMBEDDING_DIM)

        def loss_fn(params):
            q = network.apply(params["agent"], trajectory.obs)
            chosen = jnp.take_along_axis(
                q, trajectory.action[..., None], axis=-1
            ).squeeze(-1)
            active = trajectory.avail.any(axis=-1)
            chosen = jnp.where(active, chosen, 0.0)
            next_q = network.apply(
                session.target_params["agent"], trajectory.next_obs
            )
            next_q = jnp.max(
                jnp.where(trajectory.next_avail, next_q, -1e10), axis=-1
            )
            next_q = jnp.where(trajectory.next_avail.any(axis=-1), next_q, 0.0)
            not_done = 1.0 - trajectory.done
            if self.mixing == "individual":
                target = trajectory.reward + (
                    self.config.GAMMA * not_done[:, None, :] * next_q
                )
                td = (chosen - jax.lax.stop_gradient(target)) * active
            else:
                active_count = jnp.maximum(active.sum(axis=1), 1)
                team_reward = (trajectory.reward * active).sum(axis=1) / active_count
                if self.mixing == "sum":
                    chosen_total = chosen.sum(axis=1)
                    next_total = next_q.sum(axis=1)
                else:
                    chosen_total = mixer.apply(
                        params["mixer"],
                        jnp.swapaxes(chosen, 1, 2),
                        trajectory.world,
                    )
                    next_total = mixer.apply(
                        session.target_params["mixer"],
                        jnp.swapaxes(next_q, 1, 2),
                        trajectory.next_world,
                    )
                target = team_reward + self.config.GAMMA * not_done * next_total
                td = chosen_total - jax.lax.stop_gradient(target)
            return jnp.mean(td**2), jnp.mean(chosen)

        def epoch_step(learner, _):
            (loss, q_mean), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                learner.params
            )
            return learner.apply_gradients(grads=grads), (loss, q_mean)

        learner, losses = jax.lax.scan(
            epoch_step,
            session.learner,
            None,
            self.config.NUM_EPOCHS,
        )
        next_update = session.update + 1
        target_params = jax.lax.cond(
            next_update % self.config.TARGET_UPDATE_INTERVAL == 0,
            lambda _: optax.incremental_update(
                learner.params, session.target_params, self.config.TAU
            ),
            lambda _: session.target_params,
            operand=None,
        )
        episode_return = env_state["log_state"].returned_episode_returns[:, 0].mean()
        new_session = session.replace(
            learner=learner,
            target_params=target_params,
            env_state=env_state,
            obs=obs,
            dones=dones,
            rng=rng,
            update=next_update,
        )
        return new_session, {
            "episode_returns": episode_return,
            "rollout_reward_mean": trajectory.reward.mean() / self.config.REW_SCALE,
            "loss/td": losses[0].mean(),
            "q/mean": losses[1].mean(),
            "epsilon": self._epsilon(next_update),
        }

    def checkpoint_params(self, session: QSession) -> Any:
        return session.learner.params


class IQLTrainer(ValueBasedTrainer):
    mixing = "individual"


class VDNTrainer(ValueBasedTrainer):
    mixing = "sum"


class QMIXTrainer(ValueBasedTrainer):
    mixing = "qmix"


TRAINERS: dict[str, type[BaseMARLTrainer]] = {
    "ippo": IPPOTrainer,
    "mappo": MAPPOTrainer,
    "mappo_rnd": MAPPORNDTrainer,
    "iql": IQLTrainer,
    "vdn": VDNTrainer,
    "qmix": QMIXTrainer,
}


def create_trainer(config: MARLConfig) -> BaseMARLTrainer:
    config.validate()
    return TRAINERS[config.algorithm](config)


def main() -> None:
    summary = create_trainer(tyro.cli(MARLConfig)).train()
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
