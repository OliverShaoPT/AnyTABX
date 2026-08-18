from __future__ import annotations

import random
import unittest
from collections import Counter
from dataclasses import replace

import numpy as np

from src.tabx import sampleV1
from src.tabx import sample_task_balanced_diverse as qd


class SampleV1Test(unittest.TestCase):
    def test_defaults_use_requested_limits_and_wider_win_rate_band(self) -> None:
        config = sampleV1.SampleV1Config()

        self.assertEqual(
            (config.max_n_ally, config.max_n_enemy, config.max_n_zone),
            (20, 20, 20),
        )
        self.assertEqual((config.win_rate_min, config.win_rate_max), (0.3, 0.7))
        self.assertEqual(
            (
                config.confirmation_win_rate_min,
                config.confirmation_win_rate_max,
            ),
            (0.3, 0.7),
        )
        self.assertEqual(sampleV1._rollout_batch_sizes(config), (16,))
        self.assertTrue(config.jax_persistent_cache)
        self.assertTrue(config.jax_cache_prewarm)
        self.assertEqual(
            (
                config.confirmation_team_win_rate_min,
                config.confirmation_team_win_rate_max,
            ),
            (0.3, 0.7),
        )

    def test_proposal_mix_matches_global_quality_diversity_policy(self) -> None:
        weights = sampleV1._proposal_base_weights(sampleV1.SampleV1Config())

        self.assertAlmostEqual(weights["global_constructive"], 0.65)
        self.assertAlmostEqual(weights["programmatic"], 0.05)
        self.assertAlmostEqual(weights["radical_mutation"], 0.30)
        self.assertAlmostEqual(weights["distant_crossover"], 0.0)
        self.assertAlmostEqual(weights["local_mutation"], 0.0)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_scale_stratum_targets_put_eighty_percent_outside_old_cap(self) -> None:
        config = sampleV1.SampleV1Config(n_tasks=100)
        targets = sampleV1._ratio_targets(
            config.n_tasks,
            sampleV1._scale_stratum_ratios(config),
        )

        self.assertEqual(sum(targets.values()), 100)
        self.assertEqual(targets["legacy_scale"], 20)
        self.assertEqual(
            sum(
                count
                for name, count in targets.items()
                if name != "legacy_scale"
            ),
            80,
        )

    def test_team_count_sampler_reaches_new_large_buckets(self) -> None:
        config = sampleV1.SampleV1Config()
        rng = random.Random(7)
        values = [
            sampleV1._sample_team_count(
                rng,
                config.max_n_ally,
                config.team_size_bucket_weights,
            )
            for _ in range(500)
        ]

        self.assertGreater(max(values), 15)
        self.assertTrue(all(1 <= value <= 20 for value in values))
        self.assertTrue(any(value > 10 for value in values))

    def test_single_unit_radical_mutation_replaces_one_unit(self) -> None:
        config = replace(
            sampleV1.SampleV1Config(),
            team_size_bucket_weights=(1.0, 0.0, 0.0, 0.0, 0.0),
        )
        values, metadata = sampleV1._resize_and_radically_mutate_roster(
            random.Random(17),
            [3],
            1,
            config,
        )

        self.assertEqual(len(values), 1)
        self.assertEqual(metadata["replace_count"], 1)

    def test_zero_zone_has_own_count_bucket(self) -> None:
        self.assertEqual(sampleV1._v1_count_bucket(0), "0")

    def test_vectorized_parameter_distance_matches_existing_metric(self) -> None:
        rng = np.random.default_rng(5)
        query = rng.normal(size=38)
        references = [rng.normal(size=38) for _ in range(8)]

        actual = sampleV1._parameter_distances(query, references)
        expected = np.asarray(
            [qd._parameter_distance(query, reference) for reference in references]
        )

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)

    def test_parent_sources_are_not_drawn_before_parents_exist(self) -> None:
        config = replace(
            sampleV1.SampleV1Config(),
            source_adaptation_rate=0.0,
        )
        generated: Counter[str] = Counter()
        confirmed: Counter[str] = Counter()
        anchor_sums: Counter[str] = Counter()
        rng = random.Random(11)

        draws = {
            sampleV1._draw_proposal_kind(
                rng,
                config,
                [],
                generated,
                confirmed,
                anchor_sums,
                0.1,
            )
            for _ in range(200)
        }

        self.assertLessEqual(draws, {"global_constructive", "programmatic"})

    def test_invalid_reference_quantile_is_rejected(self) -> None:
        config = replace(sampleV1.SampleV1Config(), anchor_nn_quantile=1.1)

        with self.assertRaisesRegex(ValueError, "anchor_nn_quantile"):
            sampleV1._validate_config(config)


if __name__ == "__main__":
    unittest.main()
