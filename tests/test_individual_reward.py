from __future__ import annotations

import unittest
from types import SimpleNamespace

import jax.numpy as jnp

from src.tabx.wrappers.individual_reward import (
    IndividualRewardConfig,
    compute_shaped_rewards,
)


def _unit(health, max_health, *, alive=True, disabled=False):
    return SimpleNamespace(
        status=SimpleNamespace(
            health=jnp.asarray([health], dtype=jnp.float32),
            max_health=jnp.asarray([max_health], dtype=jnp.float32),
            is_alive=jnp.asarray([alive], dtype=jnp.bool_),
            is_disabled=jnp.asarray([disabled], dtype=jnp.bool_),
        )
    )


class IndividualRewardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.unit_keys = ["ally_0", "ally_1", "enemy_0"]
        self.agent_keys = ["ally_0", "ally_1"]
        self.config = IndividualRewardConfig(
            team_coef=1.0,
            damage_coef=0.05,
            heal_coef=0.05,
            damage_taken_coef=0.03,
            death_coef=0.5,
        )

    def _states(self, prev_units, next_units, attack_target):
        prev = {
            **prev_units,
            "game_manager": SimpleNamespace(attack_target=jnp.asarray(attack_target)),
        }
        nxt = {
            **next_units,
            "game_manager": SimpleNamespace(attack_target=jnp.asarray(attack_target)),
        }
        return prev, nxt

    def test_damage_credit_and_team_term(self) -> None:
        prev, nxt = self._states(
            {
                "ally_0": _unit(50, 100),
                "ally_1": _unit(80, 100),
                "enemy_0": _unit(40, 100),
            },
            {
                "ally_0": _unit(50, 100),
                "ally_1": _unit(80, 100),
                "enemy_0": _unit(20, 100),
            },
            attack_target=[2, 2, 0],
        )
        # ally_0 hits enemy for 20
        damage_dealt = jnp.asarray([20.0, 0.0, 0.0])
        team = {agent: jnp.asarray(0.1) for agent in self.agent_keys}

        shaped, diag = compute_shaped_rewards(
            unit_keys=self.unit_keys,
            agent_keys=self.agent_keys,
            prev_state=prev,
            next_state=nxt,
            team_reward_by_agent=team,
            damage_dealt=damage_dealt,
            config=self.config,
        )

        # 0.1 + 0.05 * (20/100) = 0.11
        self.assertAlmostEqual(float(shaped["ally_0"]), 0.11, places=5)
        self.assertAlmostEqual(float(shaped["ally_1"]), 0.1, places=5)
        self.assertAlmostEqual(float(diag["norm_damage"][0]), 0.2, places=5)

    def test_heal_credits_healer_not_receiver(self) -> None:
        prev, nxt = self._states(
            {
                "ally_0": _unit(50, 100),  # receiver
                "ally_1": _unit(80, 100),  # healer
                "enemy_0": _unit(40, 100),
            },
            {
                "ally_0": _unit(70, 100),  # received +20 heal
                "ally_1": _unit(80, 100),
                "enemy_0": _unit(40, 100),
            },
            attack_target=[0, 0, 0],
        )
        # healer deals -20 (TABX convention)
        damage_dealt = jnp.asarray([0.0, -20.0, 0.0])
        team = {agent: jnp.asarray(0.0) for agent in self.agent_keys}

        shaped, diag = compute_shaped_rewards(
            unit_keys=self.unit_keys,
            agent_keys=self.agent_keys,
            prev_state=prev,
            next_state=nxt,
            team_reward_by_agent=team,
            damage_dealt=damage_dealt,
            config=self.config,
        )

        # Receiver: HP up ⇒ no damage_taken, no heal credit.
        self.assertAlmostEqual(float(shaped["ally_0"]), 0.0, places=5)
        self.assertAlmostEqual(float(diag["norm_damage_taken"][0]), 0.0, places=5)
        # Healer: 0.05 * (20/100) = 0.01
        self.assertAlmostEqual(float(shaped["ally_1"]), 0.01, places=5)
        self.assertAlmostEqual(float(diag["norm_heal"][1]), 0.2, places=5)

    def test_damage_taken_and_death_penalties(self) -> None:
        prev, nxt = self._states(
            {
                "ally_0": _unit(10, 100),
                "ally_1": _unit(80, 100),
                "enemy_0": _unit(40, 100),
            },
            {
                "ally_0": _unit(0, 100, alive=False),
                "ally_1": _unit(80, 100),
                "enemy_0": _unit(40, 100),
            },
            attack_target=[2, 2, 0],
        )
        damage_dealt = jnp.asarray([0.0, 0.0, 10.0])
        team = {agent: jnp.asarray(0.0) for agent in self.agent_keys}

        shaped, diag = compute_shaped_rewards(
            unit_keys=self.unit_keys,
            agent_keys=self.agent_keys,
            prev_state=prev,
            next_state=nxt,
            team_reward_by_agent=team,
            damage_dealt=damage_dealt,
            config=self.config,
        )

        # taken: 0.03 * 0.1 = 0.003; death: 0.5 → -0.503
        self.assertAlmostEqual(float(diag["norm_damage_taken"][0]), 0.1, places=5)
        self.assertAlmostEqual(float(shaped["ally_0"]), -0.503, places=5)
        self.assertEqual(float(diag["death_log"][0]), 1.0)

    def test_kill_logged_but_not_in_reward(self) -> None:
        prev, nxt = self._states(
            {
                "ally_0": _unit(50, 100),
                "ally_1": _unit(80, 100),
                "enemy_0": _unit(5, 100),
            },
            {
                "ally_0": _unit(50, 100),
                "ally_1": _unit(80, 100),
                "enemy_0": _unit(0, 100, alive=False),
            },
            attack_target=[2, 2, 0],
        )
        damage_dealt = jnp.asarray([10.0, 0.0, 0.0])
        team = {agent: jnp.asarray(0.0) for agent in self.agent_keys}
        config = IndividualRewardConfig(
            damage_coef=0.0,
            heal_coef=0.0,
            damage_taken_coef=0.0,
            death_coef=0.0,
            team_coef=0.0,
        )

        shaped, diag = compute_shaped_rewards(
            unit_keys=self.unit_keys,
            agent_keys=self.agent_keys,
            prev_state=prev,
            next_state=nxt,
            team_reward_by_agent=team,
            damage_dealt=damage_dealt,
            config=config,
        )

        self.assertEqual(float(diag["kill_log"][0]), 1.0)
        self.assertEqual(float(diag["kill_log"][1]), 0.0)
        self.assertAlmostEqual(float(shaped["ally_0"]), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
