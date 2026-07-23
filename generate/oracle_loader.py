"""Load a trained MARL coach checkpoint and produce greedy / ε-greedy actions."""

from __future__ import annotations

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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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

    def act(
        self,
        obs: np.ndarray | jax.Array,
        avail: np.ndarray | jax.Array,
        *,
        key: jax.Array | None = None,
        epsilon: float = 0.0,
    ) -> np.ndarray:
        """Return actions for a batch of agents. obs/avail: (A, ...)."""

        obs_j = jnp.asarray(obs, dtype=jnp.float32)
        avail_j = jnp.asarray(avail, dtype=bool)
        if obs_j.ndim == 1:
            obs_j = obs_j[None]
            avail_j = avail_j[None]

        if self.family == "ppo":
            logits = self.apply_fn(self.params, obs_j, avail_j)
            greedy = jnp.argmax(logits, axis=-1)
        else:
            q_values = self.apply_fn(self.params, obs_j)
            q_values = jnp.where(avail_j, q_values, -1e10)
            greedy = jnp.argmax(q_values, axis=-1)

        if epsilon <= 0.0 or key is None:
            return np.asarray(greedy, dtype=np.int32)

        key, eps_key, rand_key = jax.random.split(key, 3)
        explore = jax.random.bernoulli(eps_key, epsilon, shape=greedy.shape)
        # Sample uniformly among available actions.
        noise = jax.random.uniform(rand_key, shape=avail_j.shape)
        masked = jnp.where(avail_j, noise, -1.0)
        random_action = jnp.argmax(masked, axis=-1)
        action = jnp.where(explore, random_action, greedy)
        return np.asarray(action, dtype=np.int32)


def load_oracle_coach(
    oracle_dir: str | Path,
    *,
    obs_dim: int,
    action_dim: int,
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
    else:
        params = raw["agent"] if isinstance(raw, dict) and "agent" in raw else raw
        model = QNetwork(action_dim, hidden)
        _ = model.init(jax.random.key(0), jnp.zeros((1, obs_dim)))
        apply_fn = model.apply

    return OracleCoach(
        algorithm=algorithm,
        family=family,
        hidden_size=hidden,
        action_dim=action_dim,
        params=params,
        apply_fn=apply_fn,
    )
