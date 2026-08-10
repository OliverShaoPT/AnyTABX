"""Unit tests for coach_eval best-policy resolution and task.json writeback."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from generate.coach_eval import (
    BEST_ADVANCED,
    BEST_ORACLE,
    coach_eval_for_task_metadata,
    resolve_best_policy,
    write_coach_eval_to_task_json,
    read_best_policy,
    read_coach_eval,
)


class ResolveBestPolicyTest(unittest.TestCase):
    def test_oracle_wins(self) -> None:
        self.assertEqual(
            resolve_best_policy(oracle_wr=0.6, advanced_wr=0.4, tie_eps=0.02),
            BEST_ORACLE,
        )

    def test_advanced_wins(self) -> None:
        self.assertEqual(
            resolve_best_policy(oracle_wr=0.3, advanced_wr=0.55, tie_eps=0.02),
            BEST_ADVANCED,
        )

    def test_tie_prefers_oracle(self) -> None:
        self.assertEqual(
            resolve_best_policy(oracle_wr=0.50, advanced_wr=0.50, tie_eps=0.02),
            BEST_ORACLE,
        )
        self.assertEqual(
            resolve_best_policy(oracle_wr=0.51, advanced_wr=0.50, tie_eps=0.02),
            BEST_ORACLE,
        )
        self.assertEqual(
            resolve_best_policy(oracle_wr=0.49, advanced_wr=0.50, tie_eps=0.02),
            BEST_ORACLE,
        )

    def test_just_outside_tie_eps(self) -> None:
        self.assertEqual(
            resolve_best_policy(oracle_wr=0.47, advanced_wr=0.50, tie_eps=0.02),
            BEST_ADVANCED,
        )


class WriteCoachEvalTest(unittest.TestCase):
    def test_write_and_read_roundtrip(self) -> None:
        bank = {
            "schema_version": "1.0",
            "manifest": {"schema": {"max_n_ally": 2, "max_n_enemy": 2, "max_n_zone": 0}},
            "tasks": [
                {
                    "task_id": "t0",
                    "metadata": {"composition_archetype": "balanced"},
                    "scenario": {},
                    "zone_scenario": {"n_zone": 0},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.json"
            path.write_text(json.dumps(bank), encoding="utf-8")
            eval_result = {
                "num_episodes": 8,
                "max_episode_steps": 64,
                "parallel_envs": 8,
                "seed": 1,
                "tie_eps": 0.02,
                "oracle_pure": {
                    "win": 3,
                    "draw": 0,
                    "loss": 5,
                    "episodes": 8,
                    "truncated": 0,
                    "win_rate": 0.375,
                },
                "heuristic_advanced": {
                    "win": 6,
                    "draw": 0,
                    "loss": 2,
                    "episodes": 8,
                    "truncated": 0,
                    "win_rate": 0.75,
                },
                "delta_oracle_minus_advanced": 0.375 - 0.75,
                "best_policy": BEST_ADVANCED,
            }
            write_coach_eval_to_task_json(
                path, coach_eval_for_task_metadata(eval_result)
            )
            block = read_coach_eval(path)
            self.assertIsNotNone(block)
            assert block is not None
            self.assertEqual(block["best_policy"], BEST_ADVANCED)
            self.assertAlmostEqual(block["oracle_pure"]["win_rate"], 0.375)
            self.assertEqual(read_best_policy(path), BEST_ADVANCED)
            # Other metadata preserved.
            reloaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                reloaded["tasks"][0]["metadata"]["composition_archetype"], "balanced"
            )


if __name__ == "__main__":
    unittest.main()
