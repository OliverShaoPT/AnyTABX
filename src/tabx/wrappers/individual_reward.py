"""Per-agent reward shaping wrapper for TABX (additive A/B layer).

Keeps the original team reward and adds a small individual term:

    r_i = team_coef * r_team
        + damage_coef * norm_damage_i
        + heal_coef * norm_heal_i
        - damage_taken_coef * norm_damage_taken_i
        - death_coef * death_i

Design notes
------------
- Being healed by an ally increases HP, so ``norm_damage_taken`` is 0 and the
  receiver gets no positive shaping. The healer is credited via ``norm_heal``.
- Kill / assist are logged in ``info`` only; they do not enter the reward.
- This module does not modify existing TABX / baseline files. Insert it outside
  ``TABXEnemyHeuristicWrapper`` (after team reward has been broadcast):

      env = TABX(...)
      env = TABXLogWrapper(env)
      env = TABXEnemyHeuristicWrapper(env)
      env = TABXIndividualRewardWrapper(env)  # optional
      env = TABXAutoResetWrapper(env)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp

from src.tabx.wrappers.wrappers import BaseWrapper


@dataclass(frozen=True)
class IndividualRewardConfig:
    """Balanced defaults: team signal dominates, shaping stays supportive."""

    enabled: bool = True
    team_coef: float = 1.0
    damage_coef: float = 0.05
    heal_coef: float = 0.05
    damage_taken_coef: float = 0.03
    death_coef: float = 0.5
    # Floor used when normalizing by target / own max HP.
    eps: float = 1e-6


def _stack_unit_field(state: dict[str, Any], unit_keys: list[str], getter) -> jax.Array:
    return jnp.stack([getter(state[unit]) for unit in unit_keys]).reshape(-1)


def compute_shaped_rewards(
    *,
    unit_keys: list[str],
    agent_keys: list[str],
    prev_state: dict[str, Any],
    next_state: dict[str, Any],
    team_reward_by_agent: dict[str, jax.Array],
    damage_dealt: jax.Array,
    config: IndividualRewardConfig,
) -> tuple[dict[str, jax.Array], dict[str, Any]]:
    """Compute hybrid rewards and diagnostic terms (JAX-friendly)."""

    prev_hp = _stack_unit_field(prev_state, unit_keys, lambda unit: unit.status.health)
    next_hp = _stack_unit_field(next_state, unit_keys, lambda unit: unit.status.health)
    max_hp = _stack_unit_field(prev_state, unit_keys, lambda unit: unit.status.max_health)
    prev_alive = _stack_unit_field(
        prev_state, unit_keys, lambda unit: unit.status.is_alive
    ).astype(jnp.bool_)
    next_alive = _stack_unit_field(
        next_state, unit_keys, lambda unit: unit.status.is_alive
    ).astype(jnp.bool_)
    is_disabled = _stack_unit_field(
        prev_state, unit_keys, lambda unit: unit.status.is_disabled
    ).astype(jnp.bool_)

    damage_dealt = jnp.asarray(damage_dealt).reshape(-1)
    attack_target = jnp.asarray(prev_state["game_manager"].attack_target).reshape(-1)
    target_max_hp = max_hp[attack_target]
    scale = jnp.maximum(target_max_hp, config.eps)

    # Outgoing combat: healer attack_damage < 0 ⇒ negative damage_dealt.
    raw_damage = jnp.maximum(damage_dealt, 0.0)
    raw_heal = jnp.maximum(-damage_dealt, 0.0)
    norm_damage = raw_damage / scale
    norm_heal = raw_heal / scale

    # Incoming: only HP loss counts. Receiving heal raises HP ⇒ no taken penalty
    # and no bonus for the receiver (healer is credited above).
    own_scale = jnp.maximum(max_hp, config.eps)
    norm_damage_taken = jnp.maximum(prev_hp - next_hp, 0.0) / own_scale
    death = (prev_alive & (~next_alive)).astype(jnp.float32)

    active = (~is_disabled) & prev_alive
    norm_damage = norm_damage * active
    norm_heal = norm_heal * active
    norm_damage_taken = norm_damage_taken * active
    death = death * (~is_disabled).astype(jnp.float32)

    # Kill / assist logging only (same-step attribution).
    target_died = prev_alive[attack_target] & (~next_alive[attack_target])
    participated = (raw_damage > config.eps) & target_died & active
    # Prefer the attacker whose selected target died; if several succeed, all
    # get a kill flag and none get assist. Same-step damage to a dying unit that
    # is not our selected target is treated as assist.
    kill = participated
    damaged_dying = jnp.zeros_like(participated)
    # Assist placeholder: reserved for richer multi-hit attribution later.
    assist = damaged_dying

    key_to_index = {unit: index for index, unit in enumerate(unit_keys)}
    shaped: dict[str, jax.Array] = {}
    terms: dict[str, dict[str, jax.Array]] = {}
    for agent in agent_keys:
        index = key_to_index[agent]
        team_term = config.team_coef * team_reward_by_agent[agent]
        damage_term = config.damage_coef * norm_damage[index]
        heal_term = config.heal_coef * norm_heal[index]
        taken_term = config.damage_taken_coef * norm_damage_taken[index]
        death_term = config.death_coef * death[index]
        total = team_term + damage_term + heal_term - taken_term - death_term
        shaped[agent] = total
        terms[agent] = {
            "team": team_term,
            "norm_damage": norm_damage[index],
            "norm_heal": norm_heal[index],
            "norm_damage_taken": norm_damage_taken[index],
            "death": death[index],
            "damage_term": damage_term,
            "heal_term": heal_term,
            "taken_term": taken_term,
            "death_term": death_term,
            "total": total,
        }

    diagnostics = {
        "individual_reward_terms": terms,
        "kill_log": kill.astype(jnp.float32),
        "assist_log": assist.astype(jnp.float32),
        "norm_damage": norm_damage,
        "norm_heal": norm_heal,
        "norm_damage_taken": norm_damage_taken,
        "death_log": death,
    }
    return shaped, diagnostics


class TABXIndividualRewardWrapper(BaseWrapper):
    """Wrap an ally-facing TABX env and replace rewards with hybrid scores."""

    def __init__(self, env, config: IndividualRewardConfig | None = None):
        super().__init__(env)
        self.config = config or IndividualRewardConfig()

    def reset(self, key, env_params):
        return self.env.reset(key, env_params)

    def step(self, key, state, action):
        obs, next_state, reward, done, info = self.env.step(key, state, action)
        if not self.config.enabled:
            return obs, next_state, reward, done, info

        if not isinstance(reward, dict):
            raise TypeError(
                "TABXIndividualRewardWrapper expects a per-agent reward dict. "
                "Place it after TABXEnemyHeuristicWrapper (or an equivalent)."
            )

        agent_keys = list(self.env.agents)
        team_reward_by_agent = {agent: reward[agent] for agent in agent_keys}
        damage_dealt = info.get("damage_dealt")
        if damage_dealt is None:
            damage_dealt = jnp.stack(
                [next_state["state"][unit].damage_dealt for unit in self.env.unit_keys]
            )

        shaped, diagnostics = compute_shaped_rewards(
            unit_keys=list(self.env.unit_keys),
            agent_keys=agent_keys,
            prev_state=state["state"],
            next_state=next_state["state"],
            team_reward_by_agent=team_reward_by_agent,
            damage_dealt=damage_dealt,
            config=self.config,
        )
        shaped["__all__"] = reward.get("__all__", team_reward_by_agent[agent_keys[0]])
        info = dict(info)
        info["individual_reward"] = diagnostics
        info["individual_reward_config"] = asdict(self.config)
        return obs, next_state, shaped, done, info
