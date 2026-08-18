from __future__ import annotations

import copy
import unittest
from dataclasses import replace
from unittest.mock import patch

from src.tabx import sample_diverse


def _evaluation(win_rate: float = 0.5) -> dict[str, object]:
    return {
        "win_rate": win_rate,
        "win_rate_ci95": [0.38, 0.62],
        "episode_length": {"mean": 120.0},
        "truncation_rate": 0.0,
        "hp_margin": {"mean": 0.02},
        "quality_flags": {
            "all_truncated": False,
            "no_interaction_rate": 0.0,
        },
        "n_rollouts": 64,
        "reject_reason": None,
    }


class SampleDiverseTest(unittest.TestCase):
    def test_profile_targets_split_odd_total_deterministically(self) -> None:
        targets = sample_diverse._profile_targets(
            5,
            sample_diverse.CORNER_PROFILES,
        )

        self.assertEqual(targets, {"bottom_left": 3, "top_right": 2})

    def test_directional_gates_keep_the_two_profiles_separate(self) -> None:
        config = sample_diverse.DiverseConfig()
        bottom_left = {
            "team_sizes": [6, 7],
            "total_units": 13,
            "n_zone": 3,
            "map_scale": 1.12,
        }
        top_right = {
            "team_sizes": [4, 5],
            "total_units": 9,
            "n_zone": 1,
            "map_scale": 0.95,
        }

        self.assertTrue(
            sample_diverse._matches_corner(bottom_left, "bottom_left", config)
        )
        self.assertFalse(
            sample_diverse._matches_corner(bottom_left, "top_right", config)
        )
        self.assertTrue(
            sample_diverse._matches_corner(top_right, "top_right", config)
        )
        self.assertFalse(
            sample_diverse._matches_corner(top_right, "bottom_left", config)
        )

    def test_final_win_rate_band_cannot_be_relaxed_by_cli_config(self) -> None:
        config = replace(sample_diverse.DiverseConfig(), win_rate_min=0.3)

        with self.assertRaisesRegex(ValueError, r"fixed to \[0.4, 0.6\]"):
            sample_diverse._validate_config(config)

    def test_confirmation_call_uses_exact_required_band(self) -> None:
        config = sample_diverse.DiverseConfig()
        job = sample_diverse._CornerJob(
            job_id=3,
            profile="top_right",
            generation_seed=123,
            config=config,
        )
        task = {"metadata": {}}
        screen = _evaluation()
        confirm = _evaluation()

        with patch.object(
            sample_diverse.base,
            "_passes_win_rate_filter",
            side_effect=[
                (True, copy.deepcopy(screen)),
                (True, copy.deepcopy(confirm)),
            ],
        ) as evaluate:
            accepted, reason = sample_diverse._evaluate_corner_task(task, job)

        self.assertTrue(accepted)
        self.assertEqual(reason, "accepted")
        confirmation_kwargs = evaluate.call_args_list[1].kwargs
        self.assertEqual(confirmation_kwargs["win_rate_min"], 0.4)
        self.assertEqual(confirmation_kwargs["win_rate_max"], 0.6)
        self.assertEqual(
            task["metadata"]["evaluation_protocol"]["win_rate_band"],
            [0.4, 0.6],
        )


if __name__ == "__main__":
    unittest.main()
