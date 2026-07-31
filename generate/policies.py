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

# Re-export for scan_rollout consumers.
__all__ = [
    "SharedAllyPolicy",
    "reference_labels",
    "reference_labels_jax",
    "reference_actions",
    "reference_action_distribution",
    "act_shared_jax",
    "stack_agent_obs",
    "stack_agent_avail",
]


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

        # oracle / oracle_eps (prefer on-device act_jax; host cast only at boundary)
        obs = stack_agent_obs(obs_by_agent, ally_keys)
        avail = stack_agent_avail(avail_by_agent, ally_keys)
        eps = float(self.spec.epsilon or 0.0) if self.spec.kind == "oracle_eps" else 0.0
        key, sub = jax.random.split(key)
        flat = self.oracle.act_jax(obs, avail, key=sub, epsilon=eps)
        return {
            agent: jnp.asarray(flat[i], dtype=jnp.int32).reshape(())
            for i, agent in enumerate(ally_keys)
        }


def stack_agent_obs(
    obs_by_agent: dict[str, Any], ally_keys: list[str]
) -> jax.Array:
    return jnp.stack(
        [jnp.asarray(obs_by_agent[a], dtype=jnp.float32) for a in ally_keys], axis=0
    )


def stack_agent_avail(
    avail_by_agent: dict[str, Any], ally_keys: list[str]
) -> jax.Array:
    return jnp.stack(
        [jnp.asarray(avail_by_agent[a], dtype=bool) for a in ally_keys], axis=0
    )


def reference_labels_jax(
    oracle: OracleCoach,
    *,
    obs_by_agent: dict[str, Any],
    avail_by_agent: dict[str, Any],
    ally_keys: list[str],
) -> tuple[dict[str, jax.Array], jax.Array]:
    """Hard + soft oracle labels on-device (no host sync)."""

    obs = stack_agent_obs(obs_by_agent, ally_keys)
    avail = stack_agent_avail(avail_by_agent, ally_keys)
    dist = oracle.action_distribution_jax(obs, avail)
    flat = jnp.argmax(dist, axis=-1).astype(jnp.int32)
    actions = {
        agent: flat[i].reshape(()).astype(jnp.int32)
        for i, agent in enumerate(ally_keys)
    }
    return actions, dist


def reference_labels(
    oracle: OracleCoach,
    *,
    obs_by_agent: dict[str, Any],
    avail_by_agent: dict[str, Any],
    ally_keys: list[str],
) -> tuple[dict[str, jnp.ndarray], np.ndarray]:
    """Hard + soft oracle labels from one forward pass.

    Returns:
      actions: dict agent -> int32 scalar
      distribution: (n_ally, action_dim) float32, HVAC-style soft KL target
    """

    actions, dist = reference_labels_jax(
        oracle,
        obs_by_agent=obs_by_agent,
        avail_by_agent=avail_by_agent,
        ally_keys=ally_keys,
    )
    return actions, np.asarray(dist, dtype=np.float32)


def act_shared_jax(
    *,
    key: jax.Array,
    spec: BehaviorSpec,
    oracle: OracleCoach,
    obs_by_agent: dict[str, Any],
    avail_by_agent: dict[str, Any],
    ally_keys: list[str],
    n_agents: int,
    max_n_zone: int,
    physics_params: Any,
    heuristic_params: TABXHeuristicParam | None,
    last_visible: list[LastVisibleTarget],
) -> tuple[dict[str, jax.Array], list[LastVisibleTarget], jax.Array]:
    """On-device shared ally act; returns actions, updated last_visible, key."""

    if spec.kind == "heuristic":
        assert heuristic_params is not None
        actions: dict[str, jax.Array] = {}
        new_last: list[LastVisibleTarget] = []
        for i, agent in enumerate(ally_keys):
            key, sub = jax.random.split(key)
            action, lv = heuristic_policy(
                sub,
                obs_by_agent[agent],
                last_visible[i],
                n_agents,
                max_n_zone,
                heuristic_params,
                physics_params,
            )
            actions[agent] = jnp.asarray(action, dtype=jnp.int32).reshape(())
            new_last.append(lv)
        return actions, new_last, key

    obs = stack_agent_obs(obs_by_agent, ally_keys)
    avail = stack_agent_avail(avail_by_agent, ally_keys)
    eps = float(spec.epsilon or 0.0) if spec.kind == "oracle_eps" else 0.0
    key, sub = jax.random.split(key)
    flat = oracle.act_jax(obs, avail, key=sub, epsilon=eps)
    actions = {
        agent: flat[i].reshape(()).astype(jnp.int32)
        for i, agent in enumerate(ally_keys)
    }
    return actions, last_visible, key


def reference_actions(
    oracle: OracleCoach,
    *,
    key: jax.Array,
    obs_by_agent: dict[str, Any],
    avail_by_agent: dict[str, Any],
    ally_keys: list[str],
) -> dict[str, jnp.ndarray]:
    del key  # greedy reference; kept for call-site compatibility
    actions, _ = reference_labels(
        oracle,
        obs_by_agent=obs_by_agent,
        avail_by_agent=avail_by_agent,
        ally_keys=ally_keys,
    )
    return actions


def reference_action_distribution(
    oracle: OracleCoach,
    *,
    obs_by_agent: dict[str, Any],
    avail_by_agent: dict[str, Any],
    ally_keys: list[str],
) -> np.ndarray:
    """Oracle soft labels for KL. Shape (n_ally, action_dim)."""

    _, dist = reference_labels(
        oracle,
        obs_by_agent=obs_by_agent,
        avail_by_agent=avail_by_agent,
        ally_keys=ally_keys,
    )
    return dist
