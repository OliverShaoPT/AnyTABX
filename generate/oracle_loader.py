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
    ) -> jax.Array:
        """On-device actions for a batch of agents. obs/avail: (A, ...)."""

        scores, avail_j = self._masked_scores(obs, avail)
        greedy = jnp.argmax(scores, axis=-1)
        eps = jnp.asarray(epsilon, dtype=jnp.float32)

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

    def act(
        self,
        obs: np.ndarray | jax.Array,
        avail: np.ndarray | jax.Array,
        *,
        key: jax.Array | None = None,
        epsilon: float = 0.0,
    ) -> np.ndarray:
        """Return actions for a batch of agents. obs/avail: (A, ...)."""

        if epsilon <= 0.0 or key is None:
            scores, _ = self._masked_scores(obs, avail)
            return np.asarray(jnp.argmax(scores, axis=-1), dtype=np.int32)
        return np.asarray(
            self.act_jax(obs, avail, key=key, epsilon=epsilon), dtype=np.int32
        )


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
