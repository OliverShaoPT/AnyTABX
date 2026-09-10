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
    present_three_way_blocks,
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


class PresentThreeWayBlocksTest(unittest.TestCase):
    def test_legacy_greedy_fills_argmax_keeps_advanced(self) -> None:
        keep = present_three_way_blocks(
            {
                "oracle_act": "greedy",
                "oracle_pure": {"win_rate": 0.6},
                "heuristic_advanced": {"win_rate": 0.4},
            }
        )
        self.assertEqual(set(keep), {BEST_ORACLE_ARGMAX, BEST_ADVANCED})
        self.assertAlmostEqual(keep[BEST_ORACLE_ARGMAX]["win_rate"], 0.6)
        self.assertNotIn(BEST_ORACLE_SAMPLE, keep)

    def test_legacy_train_like_fills_sample(self) -> None:
        keep = present_three_way_blocks(
            {
                "oracle_act": "train_like",
                "oracle_pure": {"win_rate": 0.5},
                "heuristic_advanced": {"win_rate": 0.4},
            }
        )
        self.assertEqual(set(keep), {BEST_ORACLE_SAMPLE, BEST_ADVANCED})
        self.assertNotIn(BEST_ORACLE_ARGMAX, keep)

    def test_explicit_zero_win_rate_counts_as_present(self) -> None:
        keep = present_three_way_blocks(
            {
                "oracle_argmax": {"win_rate": 0.0},
                "oracle_sample": {"win_rate": 0.2},
                "heuristic_advanced": {"win_rate": 0.1},
            }
        )
        self.assertEqual(
            set(keep),
            {BEST_ORACLE_ARGMAX, BEST_ORACLE_SAMPLE, BEST_ADVANCED},
        )

    def test_empty_block_without_win_rate_is_missing(self) -> None:
        keep = present_three_way_blocks(
            {
                "oracle_argmax": {},
                "heuristic_advanced": {"win_rate": 0.5},
            }
        )
        self.assertNotIn(BEST_ORACLE_ARGMAX, keep)
        self.assertNotIn(BEST_ORACLE_SAMPLE, keep)
        self.assertIn(BEST_ADVANCED, keep)


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
    @staticmethod
    def _package(tmp: str, bank: dict):
        from generate.task_package import TaskPackage

        root = Path(tmp)
        leaf = root / "leaf"
        leaf.mkdir()
        path = leaf / "task.json"
        path.write_text(json.dumps(bank), encoding="utf-8")
        return TaskPackage(
            path=leaf,
            root=root,
            task_index=0,
            task_id="t0",
            task_json=path,
            oracle_dir=leaf,
            oracle_config=leaf / "config.json",
            oracle_ckpt=leaf / "best.safetensors",
        )

    def test_reuse_complete_three_way_without_rewrite(self) -> None:
        from unittest.mock import patch

        from generate.coach_eval import ensure_coach_eval

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
                            "oracle_argmax": {"win_rate": 0.2},
                            "oracle_sample": {"win_rate": 0.1},
                            "heuristic_advanced": {"win_rate": 0.8},
                            "delta_oracle_minus_advanced": -0.6,
                        }
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            package = self._package(tmp, bank)
            with patch(
                "generate.coach_eval.evaluate_package_oracle_vs_advanced",
                side_effect=AssertionError("should not eval"),
            ):
                out = ensure_coach_eval(package)
            self.assertFalse(out["wrote"])
            self.assertEqual(out["best_policy"], BEST_ADVANCED)
            self.assertIsNone(out["eval_result"])

    def test_fills_only_missing_sample(self) -> None:
        from unittest.mock import patch

        from generate.coach_eval import ensure_coach_eval

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
                            "oracle_pure": {"win_rate": 0.6, "win": 6, "loss": 4},
                            "heuristic_advanced": {"win_rate": 0.4, "win": 4, "loss": 6},
                        }
                    },
                }
            ],
        }
        filled = {
            "num_episodes": 8,
            "max_episode_steps": 64,
            "parallel_envs": 8,
            "seed": 0,
            "tie_eps": 0.02,
            "oracle_argmax": {"win_rate": 0.6, "act": "argmax"},
            "oracle_sample": {
                "win": 2,
                "draw": 0,
                "loss": 6,
                "episodes": 8,
                "truncated": 0,
                "win_rate": 0.25,
                "act": "sample",
            },
            "oracle_pure": {"win_rate": 0.6, "act": "argmax"},
            "heuristic_advanced": {"win_rate": 0.4, "act": "heuristic"},
            "delta_oracle_minus_advanced": 0.2,
            "best_policy": BEST_ORACLE_ARGMAX,
            "filled_policies": [BEST_ORACLE_SAMPLE],
            "oracle_act": "argmax",
        }
        with tempfile.TemporaryDirectory() as tmp:
            package = self._package(tmp, bank)
            with patch(
                "generate.coach_eval.evaluate_package_oracle_vs_advanced",
                return_value=filled,
            ) as mock_eval:
                out = ensure_coach_eval(package)
            self.assertTrue(out["wrote"])
            self.assertEqual(out["best_policy"], BEST_ORACLE_ARGMAX)
            keep = mock_eval.call_args.kwargs["keep"]
            self.assertEqual(set(keep), {BEST_ORACLE_ARGMAX, BEST_ADVANCED})
            self.assertNotIn(BEST_ORACLE_SAMPLE, keep)
            reloaded = read_coach_eval(package)
            assert reloaded is not None
            self.assertAlmostEqual(reloaded["oracle_sample"]["win_rate"], 0.25)
            self.assertEqual(reloaded["filled_policies"], [BEST_ORACLE_SAMPLE])


if __name__ == "__main__":
    unittest.main()
