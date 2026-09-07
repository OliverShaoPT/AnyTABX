"""Load a trained MARL coach checkpoint and produce greedy / train-like actions."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from src.baseline.marl_baseline import PolicyNetwork, QNetwork
from src.baseline.utils import load_params

AlgorithmFamily = Literal["ppo", "q"]
CHECKPOINT_META_NAME = "checkpoint_meta.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def total_updates_from_config(config: dict[str, Any]) -> int:
    """Match ``BaseMARLTrainer.total_updates``."""

    return max(
        1,
        int(config["TOTAL_TIMESTEPS"])
        // int(config["NUM_ENVS"])
        // int(config["NUM_STEPS"]),
    )


def q_epsilon_at_update(
    update: int | float,
    *,
    total_updates: int,
    eps_start: float = 1.0,
    eps_finish: float = 0.05,
    eps_decay: float = 0.1,
) -> float:
    """Linear ε schedule used by ``marl_baseline`` Q trainers (and RNN Q)."""

    decay_updates = max(1, int(total_updates * float(eps_decay)))
    fraction = min(max(float(update), 0.0) / float(decay_updates), 1.0)
    return float(eps_start) + fraction * (float(eps_finish) - float(eps_start))


def q_epsilon_from_config(config: dict[str, Any], update: int | float) -> float:
    return q_epsilon_at_update(
        update,
        total_updates=total_updates_from_config(config),
        eps_start=float(config.get("EPS_START", 1.0)),
        eps_finish=float(config.get("EPS_FINISH", 0.05)),
        eps_decay=float(config.get("EPS_DECAY", 0.1)),
    )


def _ckpt_kind(ckpt_path: Path) -> str:
    name = ckpt_path.name.lower()
    if name.startswith("best"):
        return "best"
    if name.startswith("final"):
        return "final"
    return "other"


def _update_from_metrics_csv(path: Path, *, prefer_best: bool) -> int | None:
    if not path.is_file():
        return None
    last: int | None = None
    last_at_best: int | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw = row.get("update_steps")
            if raw is None or raw == "":
                continue
            update = int(float(raw))
            last = update
            window = row.get("early_stop/window_return")
            best = row.get("early_stop/best_return")
            if window and best:
                try:
                    if abs(float(window) - float(best)) <= 1e-6:
                        last_at_best = update
                except ValueError:
                    pass
    if prefer_best:
        return last_at_best if last_at_best is not None else last
    return last


def resolve_checkpoint_update(
    oracle_dir: str | Path,
    ckpt_path: str | Path,
    config: dict[str, Any] | None = None,
) -> tuple[int, str]:
    """Recover the training update of a saved ckpt.

    Preference: ``checkpoint_meta.json`` → last matching ``training_metrics.csv``
    row → scheduled ``total_updates`` (end of ε decay).
    """

    oracle_dir = Path(oracle_dir)
    ckpt_path = Path(ckpt_path)
    kind = _ckpt_kind(ckpt_path)
    meta_path = oracle_dir / CHECKPOINT_META_NAME
    if meta_path.is_file():
        meta = _read_json(meta_path)
        key = "best_update_steps" if kind == "best" else "final_update_steps"
        if kind == "other":
            key = (
                "best_update_steps"
                if meta.get("best_update_steps") is not None
                else "final_update_steps"
            )
        raw = meta.get(key)
        if raw is None and kind != "best":
            raw = meta.get("best_update_steps")
        if raw is not None:
            return int(raw), f"checkpoint_meta.{key}"

    csv_update = _update_from_metrics_csv(
        oracle_dir / "training_metrics.csv",
        prefer_best=(kind == "best"),
    )
    if csv_update is not None:
        return csv_update, "training_metrics.csv"

    cfg = config or {}
    try:
        if all(key in cfg for key in ("TOTAL_TIMESTEPS", "NUM_ENVS", "NUM_STEPS")):
            return total_updates_from_config(cfg), "config.total_updates"
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        pass
    return 0, "fallback_zero"


def algorithm_family(algorithm: str) -> AlgorithmFamily:
    algo = algorithm.lower()
    if algo in {"ippo", "mappo", "mappo_rnd"}:
        return "ppo"
    if algo in {"iql", "vdn", "qmix"}:
        return "q"
    raise ValueError(f"Unsupported oracle algorithm: {algorithm!r}")


@dataclass
class OracleCoach:
    """Frozen coach used for reference labels and oracle behavior."""

    algorithm: str
    family: AlgorithmFamily
    hidden_size: int
    action_dim: int
    params: Any
    apply_fn: Any
    train_like_sample: bool = False
    train_epsilon: float = 0.0
    train_update_steps: int | None = None
    train_update_source: str = ""

    def _masked_scores(
        self,
        obs: np.ndarray | jax.Array,
        avail: np.ndarray | jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Return (masked logits-or-Q, avail) with batch dim. Shape (A, action_dim)."""

        obs_j = jnp.asarray(obs, dtype=jnp.float32)
        avail_j = jnp.asarray(avail, dtype=bool)
        if obs_j.ndim == 1:
            obs_j = obs_j[None]
            avail_j = avail_j[None]

        if self.family == "ppo":
            # PolicyNetwork already masks unavailable actions with -1e10.
            scores = self.apply_fn(self.params, obs_j, avail_j)
        else:
            q_values = self.apply_fn(self.params, obs_j)
            scores = jnp.where(avail_j, q_values, -1e10)
        return scores, avail_j

    def action_distribution_jax(
        self,
        obs: np.ndarray | jax.Array,
        avail: np.ndarray | jax.Array,
        *,
        temperature: float = 1.0,
    ) -> jax.Array:
        """Soft action probs on-device. Shape (A, action_dim)."""

        scores, _ = self._masked_scores(obs, avail)
        temp = jnp.maximum(jnp.asarray(temperature, dtype=jnp.float32), 1e-6)
        return jax.nn.softmax(scores / temp, axis=-1)

    def action_distribution(
        self,
        obs: np.ndarray | jax.Array,
        avail: np.ndarray | jax.Array,
        *,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Soft action probs for KL targets. Shape (A, action_dim), sums to 1 on avail."""

        return np.asarray(
            self.action_distribution_jax(obs, avail, temperature=temperature),
            dtype=np.float32,
        )

    def act_jax(
        self,
        obs: np.ndarray | jax.Array,
        avail: np.ndarray | jax.Array,
        *,
        key: jax.Array,
        epsilon: float = 0.0,
        train_like: bool | None = None,
    ) -> jax.Array:
        """On-device actions for a batch of agents. obs/avail: (A, ...).

        ``train_like=True`` matches training collection:
          PPO → categorical sample from logits
          Q   → ε-greedy with ``epsilon`` (reconstructed from saved update)
        Default is greedy argmax (or ε-greedy when ``epsilon>0``).
        """

        scores, avail_j = self._masked_scores(obs, avail)
        use_train = self.train_like_sample if train_like is None else bool(train_like)
        if use_train and self.family == "ppo":
            return jax.random.categorical(key, scores).astype(jnp.int32)

        greedy = jnp.argmax(scores, axis=-1)
        eps_value = float(epsilon)
        if use_train and self.family == "q" and eps_value <= 0.0:
            eps_value = float(self.train_epsilon)
        eps = jnp.asarray(eps_value, dtype=jnp.float32)

        def _explore(operands):
            greedy_a, avail_a, rng = operands
            rng, eps_key, rand_key = jax.random.split(rng, 3)
            explore = jax.random.bernoulli(eps_key, eps, shape=greedy_a.shape)
            noise = jax.random.uniform(rand_key, shape=avail_a.shape)
            masked = jnp.where(avail_a, noise, -1.0)
            random_action = jnp.argmax(masked, axis=-1)
            return jnp.where(explore, random_action, greedy_a).astype(jnp.int32)

        def _greedy(operands):
            greedy_a, _avail_a, _rng = operands
            return greedy_a.astype(jnp.int32)

        return jax.lax.cond(eps > 0.0, _explore, _greedy, (greedy, avail_j, key))

    def act_behavior_jax(
        self,
        obs: np.ndarray | jax.Array,
        avail: np.ndarray | jax.Array,
        *,
        key: jax.Array,
        spec_kind: str,
        spec_epsilon: float = 0.0,
    ) -> jax.Array:
        """Behavior-mix action: ``oracle_eps`` keeps its ε; else honor train-like."""

        if spec_kind == "oracle_eps":
            return self.act_jax(
                obs, avail, key=key, epsilon=float(spec_epsilon), train_like=False
            )
        if self.train_like_sample:
            return self.act_jax(
                obs, avail, key=key, epsilon=float(self.train_epsilon), train_like=True
            )
        return self.act_jax(obs, avail, key=key, epsilon=0.0, train_like=False)

    def act(
        self,
        obs: np.ndarray | jax.Array,
        avail: np.ndarray | jax.Array,
        *,
        key: jax.Array | None = None,
        epsilon: float = 0.0,
        train_like: bool | None = None,
    ) -> np.ndarray:
        """Return actions for a batch of agents. obs/avail: (A, ...)."""

        use_train = self.train_like_sample if train_like is None else bool(train_like)
        if key is None:
            scores, _ = self._masked_scores(obs, avail)
            return np.asarray(jnp.argmax(scores, axis=-1), dtype=np.int32)
        return np.asarray(
            self.act_jax(
                obs, avail, key=key, epsilon=epsilon, train_like=use_train
            ),
            dtype=np.int32,
        )


def load_oracle_coach(
    oracle_dir: str | Path,
    *,
    obs_dim: int,
    action_dim: int,
    train_like_sample: bool = False,
) -> OracleCoach:
    oracle_dir = Path(oracle_dir)
    config_path = oracle_dir / "config.json"
    config = _read_json(config_path)
    algorithm = str(config.get("algorithm", "mappo"))
    hidden = int(config.get("HIDDEN_SIZE", config.get("hidden_size", 128)))
    family = algorithm_family(algorithm)

    ckpt_path = oracle_dir / "best.safetensors"
    if not ckpt_path.exists():
        ckpt_path = oracle_dir / "final.safetensors"
    if not ckpt_path.exists():
        matches = sorted(oracle_dir.glob("*.safetensors"))
        if not matches:
            raise FileNotFoundError(f"No checkpoint in {oracle_dir}")
        ckpt_path = matches[0]

    raw = load_params(ckpt_path)
    if family == "ppo":
        if "actor" not in raw:
            raise KeyError(f"PPO checkpoint missing 'actor' in {ckpt_path}")
        params = raw["actor"]
        model = PolicyNetwork(action_dim, hidden)
        # Touch apply graph with dummy shapes for clarity.
        _ = model.init(jax.random.key(0), jnp.zeros((1, obs_dim)), jnp.ones((1, action_dim), dtype=bool))
        apply_fn = model.apply
        # MAPPO/IPPO train-like is categorical(logits). Old leaves have no
        # checkpoint_meta / ε schedule; missing update info must not block load.
        update_steps, update_source = _safe_checkpoint_update(
            oracle_dir, ckpt_path, config
        )
        train_epsilon = 0.0
    else:
        params = raw["agent"] if isinstance(raw, dict) and "agent" in raw else raw
        model = QNetwork(action_dim, hidden)
        _ = model.init(jax.random.key(0), jnp.zeros((1, obs_dim)))
        apply_fn = model.apply
        update_steps, update_source = _safe_checkpoint_update(
            oracle_dir, ckpt_path, config
        )
        try:
            train_epsilon = q_epsilon_from_config(config, update_steps)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            train_epsilon = float(config.get("EPS_FINISH", 0.05))
            update_source = f"{update_source}+eps_finish_fallback"

    return OracleCoach(
        algorithm=algorithm,
        family=family,
        hidden_size=hidden,
        action_dim=action_dim,
        params=params,
        apply_fn=apply_fn,
        train_like_sample=bool(train_like_sample),
        train_epsilon=float(train_epsilon),
        train_update_steps=int(update_steps),
        train_update_source=str(update_source),
    )


def _safe_checkpoint_update(
    oracle_dir: Path,
    ckpt_path: Path,
    config: dict[str, Any],
) -> tuple[int, str]:
    """Never raise: old coach leaves may lack meta / metrics / schedule keys."""

    try:
        return resolve_checkpoint_update(oracle_dir, ckpt_path, config)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OSError):
        return 0, "missing_update_meta"
