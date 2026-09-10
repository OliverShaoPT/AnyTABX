"""Unit tests for coach_eval best-policy resolution and task.json writeback."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from generate.coach_eval import (
    BEST_ADVANCED,
    BEST_ORACLE,
    BEST_ORACLE_ARGMAX,
    BEST_ORACLE_SAMPLE,
    coach_eval_for_task_metadata,
    normalize_best_policy,
    resolve_best_policy,
    write_coach_eval_to_task_json,
    read_best_policy,
    read_coach_eval,
)


class ResolveBestPolicyTest(unittest.TestCase):
    def test_sample_wins(self) -> None:
        self.assertEqual(
            resolve_best_policy(
                argmax_wr=0.4, sample_wr=0.6, advanced_wr=0.4, tie_eps=0.02
            ),
            BEST_ORACLE_SAMPLE,
        )

    def test_argmax_wins(self) -> None:
        self.assertEqual(
            resolve_best_policy(
                argmax_wr=0.7, sample_wr=0.4, advanced_wr=0.4, tie_eps=0.02
            ),
            BEST_ORACLE_ARGMAX,
        )

    def test_advanced_wins(self) -> None:
        self.assertEqual(
            resolve_best_policy(
                argmax_wr=0.3, sample_wr=0.3, advanced_wr=0.55, tie_eps=0.02
            ),
            BEST_ADVANCED,
        )

    def test_tie_prefers_sample_over_argmax_and_advanced(self) -> None:
        self.assertEqual(
            resolve_best_policy(
                argmax_wr=0.50, sample_wr=0.50, advanced_wr=0.50, tie_eps=0.02
            ),
            BEST_ORACLE_SAMPLE,
        )
        self.assertEqual(
            resolve_best_policy(
                argmax_wr=0.51, sample_wr=0.50, advanced_wr=0.50, tie_eps=0.02
            ),
            BEST_ORACLE_SAMPLE,
        )
        self.assertEqual(
            resolve_best_policy(
                argmax_wr=0.50, sample_wr=0.49, advanced_wr=0.50, tie_eps=0.02
            ),
            BEST_ORACLE_SAMPLE,
        )

    def test_just_outside_tie_eps_picks_clear_leader(self) -> None:
        self.assertEqual(
            resolve_best_policy(
                argmax_wr=0.47, sample_wr=0.47, advanced_wr=0.50, tie_eps=0.02
            ),
            BEST_ADVANCED,
        )
        self.assertEqual(
            resolve_best_policy(
                argmax_wr=0.53, sample_wr=0.50, advanced_wr=0.50, tie_eps=0.02
            ),
            BEST_ORACLE_ARGMAX,
        )


class NormalizeBestPolicyTest(unittest.TestCase):
    def test_passthrough(self) -> None:
        self.assertEqual(normalize_best_policy(BEST_ORACLE_SAMPLE), BEST_ORACLE_SAMPLE)
        self.assertEqual(normalize_best_policy(BEST_ORACLE_ARGMAX), BEST_ORACLE_ARGMAX)
        self.assertEqual(normalize_best_policy(BEST_ADVANCED), BEST_ADVANCED)

    def test_legacy_oracle_pure(self) -> None:
        self.assertEqual(
            normalize_best_policy(BEST_ORACLE, oracle_act="greedy"),
            BEST_ORACLE_ARGMAX,
        )
        self.assertEqual(
            normalize_best_policy(BEST_ORACLE, oracle_act="argmax"),
            BEST_ORACLE_ARGMAX,
        )
        self.assertEqual(
            normalize_best_policy(BEST_ORACLE, oracle_act="train_like"),
            BEST_ORACLE_SAMPLE,
        )
        self.assertEqual(
            normalize_best_policy(BEST_ORACLE),
            BEST_ORACLE_SAMPLE,
        )


class TrainLikeFromBestPolicyTest(unittest.TestCase):
    def test_modes(self) -> None:
        from generate.coach_eval import train_like_from_best_policy

        self.assertTrue(train_like_from_best_policy(BEST_ORACLE_SAMPLE))
        self.assertFalse(train_like_from_best_policy(BEST_ORACLE_ARGMAX))
        self.assertIsNone(train_like_from_best_policy(BEST_ADVANCED))
        self.assertTrue(train_like_from_best_policy(BEST_ORACLE))
        self.assertFalse(
            train_like_from_best_policy(
                normalize_best_policy(BEST_ORACLE, oracle_act="greedy")
            )
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
                "oracle_argmax": {
                    "win": 3,
                    "draw": 0,
                    "loss": 5,
                    "episodes": 8,
                    "truncated": 0,
                    "win_rate": 0.375,
                    "act": "argmax",
                },
                "oracle_sample": {
                    "win": 2,
                    "draw": 0,
                    "loss": 6,
                    "episodes": 8,
                    "truncated": 0,
                    "win_rate": 0.25,
                    "act": "sample",
                },
                "oracle_pure": {
                    "win": 3,
                    "draw": 0,
                    "loss": 5,
                    "episodes": 8,
                    "truncated": 0,
                    "win_rate": 0.375,
                    "act": "argmax",
                },
                "heuristic_advanced": {
                    "win": 6,
                    "draw": 0,
                    "loss": 2,
                    "episodes": 8,
                    "truncated": 0,
                    "win_rate": 0.75,
                    "act": "heuristic",
                },
                "delta_oracle_minus_advanced": 0.375 - 0.75,
                "best_policy": BEST_ADVANCED,
                "oracle_act": "heuristic",
            }
            write_coach_eval_to_task_json(
                path, coach_eval_for_task_metadata(eval_result)
            )
            block = read_coach_eval(path)
            self.assertIsNotNone(block)
            assert block is not None
            self.assertEqual(block["best_policy"], BEST_ADVANCED)
            self.assertAlmostEqual(block["oracle_argmax"]["win_rate"], 0.375)
            self.assertAlmostEqual(block["oracle_sample"]["win_rate"], 0.25)
            self.assertEqual(read_best_policy(path), BEST_ADVANCED)
            reloaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                reloaded["tasks"][0]["metadata"]["composition_archetype"], "balanced"
            )

    def test_legacy_oracle_pure_greedy_reads_as_argmax(self) -> None:
        bank = {
            "schema_version": "1.0",
            "manifest": {},
            "tasks": [
                {
                    "task_id": "t0",
                    "metadata": {
                        "coach_eval": {
                            "best_policy": BEST_ORACLE,
                            "oracle_act": "greedy",
                            "oracle_pure": {"win_rate": 0.6},
                            "heuristic_advanced": {"win_rate": 0.4},
                        }
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.json"
            path.write_text(json.dumps(bank), encoding="utf-8")
            self.assertEqual(read_best_policy(path), BEST_ORACLE_ARGMAX)


class EnsureCoachEvalTest(unittest.TestCase):
    def test_reuse_existing_without_rewrite(self) -> None:
        from generate.coach_eval import ensure_coach_eval
        from generate.task_package import TaskPackage

        bank = {
            "schema_version": "1.0",
            "manifest": {},
            "tasks": [
                {
                    "task_id": "t0",
                    "metadata": {
                        "coach_eval": {
                            "best_policy": BEST_ADVANCED,
                            "num_episodes": 4,
                            "max_episode_steps": 64,
                            "parallel_envs": 4,
                            "seed": 0,
                            "tie_eps": 0.02,
                            "oracle_pure": {"win_rate": 0.2},
                            "heuristic_advanced": {"win_rate": 0.8},
                            "delta_oracle_minus_advanced": -0.6,
                        }
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            leaf = root / "leaf"
            leaf.mkdir()
            path = leaf / "task.json"
            path.write_text(json.dumps(bank), encoding="utf-8")
            package = TaskPackage(
                path=leaf,
                root=root,
                task_index=0,
                task_id="t0",
                task_json=path,
                oracle_dir=leaf,
                oracle_config=leaf / "config.json",
                oracle_ckpt=leaf / "best.safetensors",
            )
            out = ensure_coach_eval(package)
            self.assertFalse(out["wrote"])
            self.assertEqual(out["best_policy"], BEST_ADVANCED)
            self.assertIsNone(out["eval_result"])


if __name__ == "__main__":
    unittest.main()
