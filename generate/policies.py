"""Shared (all-ally) behavior policies for record generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from generate.behavior_mix import BehaviorSpec
from generate.oracle_loader import OracleCoach
from src.tabx.eval_task import load_heuristic_params
from src.tabx.heuristic_policy import LastVisibleTarget, heuristic_policy
from src.tabx.heuristic_policy.params import TABXHeuristicParam


@dataclass
class SharedAllyPolicy:
    """One policy instance shared by all allies at a given time."""

    spec: BehaviorSpec
    oracle: OracleCoach
    n_agents: int
    max_n_zone: int
    heuristic_params: TABXHeuristicParam | None = None
    last_visible: dict[str, LastVisibleTarget] = field(default_factory=dict)

    @classmethod
    def from_spec(
        cls,
        spec: BehaviorSpec,
        *,
        oracle: OracleCoach,
        ally_keys: list[str],
        n_agents: int,
        max_n_zone: int,
    ) -> "SharedAllyPolicy":
        heur = None
        if spec.kind == "heuristic":
            if not spec.heuristic:
                raise ValueError(f"heuristic spec missing preset: {spec}")
            heur = load_heuristic_params(spec.heuristic, epsilon_override=spec.epsilon)
        return cls(
            spec=spec,
            oracle=oracle,
            n_agents=n_agents,
            max_n_zone=max_n_zone,
            heuristic_params=heur,
            last_visible={agent: LastVisibleTarget() for agent in ally_keys},
        )

    def act(
        self,
        *,
        key: jax.Array,
        obs_by_agent: dict[str, Any],
        avail_by_agent: dict[str, Any],
        ally_keys: list[str],
        physics_params: Any,
    ) -> dict[str, np.ndarray]:
        """Return int actions for every ally under the shared policy."""

        if self.spec.kind == "heuristic":
            assert self.heuristic_params is not None
            actions: dict[str, np.ndarray] = {}
            for agent in ally_keys:
                key, sub = jax.random.split(key)
                action, self.last_visible[agent] = heuristic_policy(
                    sub,
                    obs_by_agent[agent],
                    self.last_visible[agent],
                    self.n_agents,
                    self.max_n_zone,
                    self.heuristic_params,
                    physics_params,
                )
                # TABX Unit.act calls action.reshape() — needs JAX array, not np.int32.
                actions[agent] = jnp.asarray(action, dtype=jnp.int32).reshape(())
            return actions

        # oracle / oracle_eps
        obs = np.stack([np.asarray(obs_by_agent[a], dtype=np.float32) for a in ally_keys], axis=0)
        avail = np.stack([np.asarray(avail_by_agent[a], dtype=bool) for a in ally_keys], axis=0)
        eps = float(self.spec.epsilon or 0.0) if self.spec.kind == "oracle_eps" else 0.0
        key, sub = jax.random.split(key)
        flat = self.oracle.act(obs, avail, key=sub, epsilon=eps)
        return {
            agent: jnp.asarray(flat[i], dtype=jnp.int32).reshape(())
            for i, agent in enumerate(ally_keys)
        }


def reference_actions(
    oracle: OracleCoach,
    *,
    key: jax.Array,
    obs_by_agent: dict[str, Any],
    avail_by_agent: dict[str, Any],
    ally_keys: list[str],
) -> dict[str, jnp.ndarray]:
    obs = np.stack([np.asarray(obs_by_agent[a], dtype=np.float32) for a in ally_keys], axis=0)
    avail = np.stack([np.asarray(avail_by_agent[a], dtype=bool) for a in ally_keys], axis=0)
    flat = oracle.act(obs, avail, key=key, epsilon=0.0)
    return {
        agent: jnp.asarray(flat[i], dtype=jnp.int32).reshape(())
        for i, agent in enumerate(ally_keys)
    }
