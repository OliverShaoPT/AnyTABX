"""Shared (all-ally) behavior policies for record generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from generate.coach_eval import (
    BEST_ADVANCED,
    BEST_ORACLE,
    BEST_ORACLE_ARGMAX,
    BEST_ORACLE_SAMPLE,
    apply_oracle_act_mode,
    normalize_best_policy,
    train_like_from_best_policy,
)
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
    "reference_labels_best_teacher",
    "reference_labels_best_teacher_jax",
    "reference_rl_pair_jax",
    "select_rl_hard_actions",
    "stack_actions",
    "reference_actions",
    "reference_action_distribution",
    "act_shared_jax",
    "stack_agent_obs",
    "stack_agent_avail",
    "is_rl_teacher",
    "reference_hard_kind",
    "normalize_best_policy",
    "train_like_from_best_policy",
    "apply_oracle_act_mode",
    "BEST_ORACLE",
    "BEST_ORACLE_ARGMAX",
    "BEST_ORACLE_SAMPLE",
    "BEST_ADVANCED",
]


def is_rl_teacher(best_policy: str) -> bool:
    return normalize_best_policy(best_policy) != BEST_ADVANCED


def reference_hard_kind(oracle: OracleCoach, best_policy: str) -> str:
    if not is_rl_teacher(best_policy):
        return "heuristic"
    return "sample" if oracle.train_like_sample else "argmax"


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
        flat = self.oracle.act_behavior_jax(
            obs,
            avail,
            key=sub,
            spec_kind=self.spec.kind,
            spec_epsilon=eps,
        )
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


def _actions_from_flat(flat: jax.Array, ally_keys: list[str]) -> dict[str, jax.Array]:
    return {
        agent: flat[i].reshape(()).astype(jnp.int32)
        for i, agent in enumerate(ally_keys)
    }


def stack_actions(actions: dict[str, jax.Array], ally_keys: list[str]) -> jax.Array:
    """``(n_ally,)`` int32 from an agent -> action dict."""

    return jnp.stack(
        [jnp.asarray(actions[agent], dtype=jnp.int32).reshape(()) for agent in ally_keys]
    )


def reference_rl_pair_jax(
    oracle: OracleCoach,
    *,
    key: jax.Array,
    obs_by_agent: dict[str, Any],
    avail_by_agent: dict[str, Any],
    ally_keys: list[str],
    oracle_params: Any = None,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array], jax.Array]:
    """Sample and argmax hard actions plus softmax, from one RL forward."""

    obs = stack_agent_obs(obs_by_agent, ally_keys)
    avail = stack_agent_avail(avail_by_agent, ally_keys)
    sample, argmax, dist = oracle.hard_pair_jax(
        obs, avail, key=key, params=oracle_params
    )
    return (
        _actions_from_flat(sample, ally_keys),
        _actions_from_flat(argmax, ally_keys),
        dist,
    )


def select_rl_hard_actions(
    oracle: OracleCoach,
    sample_actions: dict[str, jax.Array],
    argmax_actions: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    """Which RL hard action steps the env when behavior kind is ``oracle``.

    ``oracle.train_like_sample`` true → train-like sample; false → argmax.
    ``oracle_eps`` does not use this.
    """

    if oracle.train_like_sample:
        return sample_actions
    return argmax_actions


def reference_labels_jax(
    oracle: OracleCoach,
    *,
    obs_by_agent: dict[str, Any],
    avail_by_agent: dict[str, Any],
    ally_keys: list[str],
    key: jax.Array | None = None,
) -> tuple[dict[str, jax.Array], jax.Array]:
    """Hard + soft oracle labels on-device (no host sync).

    Soft ``dist`` is always RL softmax (KL target). Hard actions follow
    ``oracle.train_like_sample``: categorical / train-ε when true, else argmax.
    With a ``key``, sample and argmax come from one forward.
    """

    if key is None:
        if oracle.train_like_sample:
            raise ValueError("key is required when oracle.train_like_sample is true")
        obs = stack_agent_obs(obs_by_agent, ally_keys)
        avail = stack_agent_avail(avail_by_agent, ally_keys)
        dist = oracle.action_distribution_jax(obs, avail)
        flat = jnp.argmax(dist, axis=-1).astype(jnp.int32)
        return _actions_from_flat(flat, ally_keys), dist
    sample_actions, argmax_actions, dist = reference_rl_pair_jax(
        oracle,
        key=key,
        obs_by_agent=obs_by_agent,
        avail_by_agent=avail_by_agent,
        ally_keys=ally_keys,
    )
    actions = select_rl_hard_actions(oracle, sample_actions, argmax_actions)
    return actions, dist


def reference_labels(
    oracle: OracleCoach,
    *,
    obs_by_agent: dict[str, Any],
    avail_by_agent: dict[str, Any],
    ally_keys: list[str],
    key: jax.Array | None = None,
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
        key=key,
    )
    return actions, np.asarray(dist, dtype=np.float32)


def _advanced_hard_actions_jax(
    *,
    key: jax.Array,
    obs_by_agent: dict[str, Any],
    ally_keys: list[str],
    n_agents: int,
    max_n_zone: int,
    physics_params: Any,
    heuristic_params: TABXHeuristicParam,
    last_visible: list[LastVisibleTarget],
) -> tuple[dict[str, jax.Array], list[LastVisibleTarget], jax.Array]:
    """Greedy advanced heuristic hard actions; updates per-ally last_visible."""

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


def reference_labels_best_teacher_jax(
    oracle: OracleCoach,
    *,
    key: jax.Array,
    obs_by_agent: dict[str, Any],
    avail_by_agent: dict[str, Any],
    ally_keys: list[str],
    best_policy: str,
    n_agents: int,
    max_n_zone: int,
    physics_params: Any,
    advanced_params: TABXHeuristicParam | None,
    ref_last_visible: list[LastVisibleTarget],
    oracle_params: Any = None,
) -> tuple[
    dict[str, jax.Array],
    jax.Array,
    list[LastVisibleTarget],
    jax.Array,
    dict[str, jax.Array],
    dict[str, jax.Array],
]:
    """Hard labels from best teacher; sample and argmax from one RL forward.

    Returns:
      actions, distribution (n_ally, A), updated ref_last_visible, key,
      sample_actions, argmax_actions.

    ``actions`` is the teacher hard label (RL selection or heuristic).
    ``sample_actions`` / ``argmax_actions`` are always the RL pair.
    ``select_rl_hard_actions`` picks which of those two steps the env
    when the behavior kind is ``oracle``.
    """

    key, ref_key = jax.random.split(key)
    sample_actions, argmax_actions, dist = reference_rl_pair_jax(
        oracle,
        key=ref_key,
        obs_by_agent=obs_by_agent,
        avail_by_agent=avail_by_agent,
        ally_keys=ally_keys,
        oracle_params=oracle_params,
    )
    if is_rl_teacher(best_policy):
        actions = select_rl_hard_actions(oracle, sample_actions, argmax_actions)
        return actions, dist, ref_last_visible, key, sample_actions, argmax_actions
    if advanced_params is None:
        raise ValueError("advanced_params required when best_policy is heuristic_advanced")
    adv_actions, new_last, key = _advanced_hard_actions_jax(
        key=key,
        obs_by_agent=obs_by_agent,
        ally_keys=ally_keys,
        n_agents=n_agents,
        max_n_zone=max_n_zone,
        physics_params=physics_params,
        heuristic_params=advanced_params,
        last_visible=ref_last_visible,
    )
    return adv_actions, dist, new_last, key, sample_actions, argmax_actions


def reference_labels_best_teacher(
    oracle: OracleCoach,
    *,
    key: jax.Array,
    obs_by_agent: dict[str, Any],
    avail_by_agent: dict[str, Any],
    ally_keys: list[str],
    best_policy: str,
    n_agents: int,
    max_n_zone: int,
    physics_params: Any,
    advanced_params: TABXHeuristicParam | None,
    ref_last_visible: dict[str, LastVisibleTarget] | list[LastVisibleTarget],
) -> tuple[
    dict[str, jnp.ndarray],
    np.ndarray,
    dict[str, LastVisibleTarget],
    dict[str, jax.Array],
    dict[str, jax.Array],
]:
    """Host-friendly best-teacher reference labels.

    Returns actions, distribution, last_visible, sample_actions, argmax_actions.
    """

    if isinstance(ref_last_visible, dict):
        last_list = [ref_last_visible[a] for a in ally_keys]
    else:
        last_list = list(ref_last_visible)
    actions, dist, new_last, _key, sample_actions, argmax_actions = (
        reference_labels_best_teacher_jax(
            oracle,
            key=key,
            obs_by_agent=obs_by_agent,
            avail_by_agent=avail_by_agent,
            ally_keys=ally_keys,
            best_policy=best_policy,
            n_agents=n_agents,
            max_n_zone=max_n_zone,
            physics_params=physics_params,
            advanced_params=advanced_params,
            ref_last_visible=last_list,
        )
    )
    return (
        actions,
        np.asarray(dist, dtype=np.float32),
        {agent: new_last[i] for i, agent in enumerate(ally_keys)},
        sample_actions,
        argmax_actions,
    )


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
    oracle_params: Any = None,
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
    flat = oracle.act_behavior_jax(
        obs,
        avail,
        key=sub,
        spec_kind=spec.kind,
        spec_epsilon=eps,
        params=oracle_params,
    )
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
    actions, _ = reference_labels(
        oracle,
        key=key,
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
