"""Unit tests for generate/ helpers (no full env rollout required)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from generate.agent_centric import (
    build_policy_mask,
    flat_agent_dirname,
    split_one_record,
    split_records_root,
)
from generate.behavior_mix import (
    POLICY_TAG_MASK,
    BehaviorSwitcher,
    DEFAULT_BEHAVIOR_MIX,
    maybe_switch_policy_ids,
    mix_cdf_jax,
    policy_tag_for_spec,
    sample_policy_indices,
)
from generate.dump_schema import (
    OWN_FEATURE_DIM,
    OTHER_FEATURE_DIM,
    ZONE_FEATURE_DIM,
    discover_agent_centric_dirs,
    split_flat_obs,
    write_env_centric_record,
)
from generate.winrate_adapt import (
    choose_adapt_params,
    choose_strength,
    in_win_rate_band,
    is_strong_spec,
    reweight_mix,
    summarize_arrays,
    win_rate_from_counts,
)


class IndependentPolicySampleTest(unittest.TestCase):
    def test_sample_policy_indices_shape_and_range(self) -> None:
        import jax
        import jax.numpy as jnp

        cdf = mix_cdf_jax(DEFAULT_BEHAVIOR_MIX)
        ids = sample_policy_indices(jax.random.key(0), cdf, 5)
        self.assertEqual(tuple(ids.shape), (5,))
        self.assertTrue(bool(jnp.all(ids >= 0)))
        self.assertTrue(bool(jnp.all(ids < len(DEFAULT_BEHAVIOR_MIX))))

    def test_maybe_switch_policy_ids_independent(self) -> None:
        import jax
        import jax.numpy as jnp

        cdf = mix_cdf_jax(DEFAULT_BEHAVIOR_MIX)
        n = 4
        key = jax.random.key(1)
        ids0 = jnp.zeros((n,), dtype=jnp.int32)
        # Past cooldown + switch_prob=1 → all resample (may coincide by chance).
        _key, ids1, steps = maybe_switch_policy_ids(
            key,
            policy_ids=ids0,
            steps_since_switch=jnp.full((n,), 100, dtype=jnp.int32),
            cdf=cdf,
            min_behavior_steps=1,
            behavior_switch_prob=1.0,
        )
        self.assertEqual(tuple(ids1.shape), (n,))
        self.assertTrue(bool(jnp.all(steps == 0)))


class BehaviorSwitcherTest(unittest.TestCase):
    def test_mid_episode_switch_can_be_disabled(self) -> None:
        switcher = BehaviorSwitcher(
            mix=DEFAULT_BEHAVIOR_MIX,
            min_behavior_steps=1,
            behavior_switch_prob=1.0,
            mid_episode_policy_switch=False,
            rng=np.random.default_rng(0),
        )
        first = switcher.current.policy_id
        for _ in range(20):
            self.assertEqual(switcher.maybe_switch().policy_id, first)
        # Reset path still resamples.
        nxt = switcher.force_sample()
        # May equal by chance; just ensure force_sample runs and resets cooldown.
        self.assertEqual(switcher.steps_since_switch, 0)
        self.assertIsNotNone(nxt)

    def test_min_steps_blocks_switch(self) -> None:
        switcher = BehaviorSwitcher(
            mix=DEFAULT_BEHAVIOR_MIX,
            min_behavior_steps=10,
            behavior_switch_prob=1.0,
            rng=np.random.default_rng(0),
        )
        first = switcher.current.policy_id
        for _ in range(9):
            self.assertEqual(switcher.maybe_switch().policy_id, first)
        # With prob=1.0, the 10th eligible call should resample (may equal by chance).
        switcher.maybe_switch()
        self.assertEqual(switcher.steps_since_switch, 0)


class SplitFlatObsTest(unittest.TestCase):
    def test_shapes(self) -> None:
        n_units = 4
        max_n_zone = 2
        n_other = n_units - 1
        dim = OWN_FEATURE_DIM + OTHER_FEATURE_DIM * n_other + ZONE_FEATURE_DIM * max_n_zone
        obs = np.zeros(dim, dtype=np.float32)
        obs[0] = 1.0
        obs[OWN_FEATURE_DIM] = 2.0  # first other slot non-zero
        static, dyn, mask = split_flat_obs(obs, n_units=n_units, max_n_zone=max_n_zone)
        self.assertEqual(static.shape[0], OWN_FEATURE_DIM + ZONE_FEATURE_DIM * max_n_zone)
        self.assertEqual(dyn.shape, (n_other, OTHER_FEATURE_DIM))
        self.assertEqual(mask.shape, (n_other,))
        self.assertEqual(int(mask[0]), 1)


class AgentCentricSplitTest(unittest.TestCase):
    def test_split_writes_reset_aligned_files(self) -> None:
        t, n_ally, n_units, max_n_zone = 5, 2, 4, 1
        obs_dim = (
            OWN_FEATURE_DIM
            + OTHER_FEATURE_DIM * (n_units - 1)
            + ZONE_FEATURE_DIM * max_n_zone
        )
        action_dim = 8
        dist = np.zeros((t, n_ally, action_dim), dtype=np.float32)
        dist[..., 1] = 1.0
        arrays = {
            "actions_behavior": np.zeros((t, n_ally), dtype=np.int32),
            "actions_reference": np.ones((t, n_ally), dtype=np.int32),
            "actions_reference_distribution": dist,
            "reward_team": np.zeros((t, n_ally), dtype=np.float32),
            "reward_individual": np.zeros((t, n_ally), dtype=np.float32),
            "done": np.array([0, 0, 1, 0, 0], dtype=np.uint8),
            "truncation": np.array([0, 0, 1, 0, 0], dtype=np.uint8),
            "is_win": np.array([0, 0, 0, 0, 0], dtype=np.uint8),
            "reset": np.array([1, 0, 0, 1, 0], dtype=np.uint8),
            "episode_id": np.array([0, 0, 0, 1, 1], dtype=np.int32),
            "behavior_policy_id": np.array(
                [[0, 0], [0, 0], [1, 1], [1, 1], [2, 2]], dtype=np.int32
            ),
            "policy_tag": np.array(
                [[0, 0], [0, 0], [3, 3], [3, 3], [7, 7]], dtype=np.int32
            ),
            "obs_flat": np.zeros((t, n_ally, obs_dim), dtype=np.float32),
        }
        meta = {
            "ally_keys": ["ally_0", "ally_1"],
            "n_units": n_units,
            "max_n_zone": max_n_zone,
            "task_index": 0,
            "task_id": "dummy",
            "record_id": 0,
            "shared_ally_policy": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            record_dir = Path(tmp) / "record-000000"
            write_env_centric_record(record_dir, arrays, meta)
            paths = split_one_record(record_dir, mask_prob=1.0, mask_seed=0)
            self.assertEqual(len(paths), 2)
            ally0 = record_dir / "agent_centric" / "ally_0"
            reset = np.load(ally0 / "reset.npy")
            done = np.load(ally0 / "done.npy")
            trunc = np.load(ally0 / "truncation.npy")
            is_win = np.load(ally0 / "is_win.npy")
            self.assertTrue(np.array_equal(reset, arrays["reset"]))
            self.assertTrue(np.array_equal(done, arrays["done"]))
            self.assertTrue(np.array_equal(trunc, arrays["truncation"]))
            self.assertTrue(np.array_equal(is_win, arrays["is_win"]))
            self.assertEqual(np.load(ally0 / "behavior_action.npy").shape, (t,))
            ref_dist = np.load(ally0 / "reference_action_distribution.npy")
            self.assertEqual(ref_dist.shape, (t, action_dim))
            self.assertTrue(np.allclose(ref_dist.sum(axis=-1), 1.0))
            self.assertTrue(
                np.array_equal(
                    np.argmax(ref_dist, axis=-1),
                    np.load(ally0 / "reference_action.npy"),
                )
            )
            # mask_prob=1 → every segment masked; tags overwritten to 8
            self.assertTrue(np.all(np.load(ally0 / "policy_mask.npy") == 1))
            self.assertTrue(
                np.all(np.load(ally0 / "policy_tag.npy") == POLICY_TAG_MASK)
            )

    def test_flat_output_root(self) -> None:
        t, n_ally, n_units, max_n_zone = 3, 2, 4, 1
        obs_dim = (
            OWN_FEATURE_DIM
            + OTHER_FEATURE_DIM * (n_units - 1)
            + ZONE_FEATURE_DIM * max_n_zone
        )
        action_dim = 4
        dist = np.zeros((t, n_ally, action_dim), dtype=np.float32)
        dist[..., 0] = 1.0
        arrays = {
            "actions_behavior": np.zeros((t, n_ally), dtype=np.int32),
            "actions_reference": np.zeros((t, n_ally), dtype=np.int32),
            "actions_reference_distribution": dist,
            "reward_team": np.zeros((t, n_ally), dtype=np.float32),
            "reward_individual": np.zeros((t, n_ally), dtype=np.float32),
            "done": np.zeros((t,), dtype=np.uint8),
            "truncation": np.zeros((t,), dtype=np.uint8),
            "is_win": np.zeros((t,), dtype=np.uint8),
            "reset": np.array([1, 0, 0], dtype=np.uint8),
            "episode_id": np.zeros((t,), dtype=np.int32),
            "behavior_policy_id": np.zeros((t,), dtype=np.int32),
            "obs_flat": np.zeros((t, n_ally, obs_dim), dtype=np.float32),
        }
        meta = {
            "ally_keys": ["unit_00", "unit_01"],
            "n_units": n_units,
            "max_n_zone": max_n_zone,
            "task_index": 36,
            "task_id": "dummy",
            "record_id": 0,
            "shared_ally_policy": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record_dir = root / "task_foo" / "record-000000"
            out_root = root / "agent_flat"
            write_env_centric_record(record_dir, arrays, meta)
            paths = split_one_record(record_dir, output_root=out_root)
            self.assertEqual(len(paths), 2)
            self.assertFalse((record_dir / "agent_centric").exists())
            name0 = flat_agent_dirname(record_dir, "unit_00")
            self.assertTrue((out_root / name0 / "obs_static.npy").exists())
            discovered = discover_agent_centric_dirs(out_root)
            self.assertEqual(len(discovered), 2)

    def test_split_records_root_parallel_flat(self) -> None:
        t, n_ally, n_units, max_n_zone = 2, 1, 2, 0
        obs_dim = OWN_FEATURE_DIM + OTHER_FEATURE_DIM * (n_units - 1)
        arrays = {
            "actions_behavior": np.zeros((t, n_ally), dtype=np.int32),
            "actions_reference": np.zeros((t, n_ally), dtype=np.int32),
            "actions_reference_distribution": np.ones((t, n_ally, 2), dtype=np.float32) / 2,
            "reward_team": np.zeros((t, n_ally), dtype=np.float32),
            "reward_individual": np.zeros((t, n_ally), dtype=np.float32),
            "done": np.zeros((t,), dtype=np.uint8),
            "truncation": np.zeros((t,), dtype=np.uint8),
            "is_win": np.zeros((t,), dtype=np.uint8),
            "reset": np.array([1, 0], dtype=np.uint8),
            "episode_id": np.zeros((t,), dtype=np.int32),
            "behavior_policy_id": np.zeros((t,), dtype=np.int32),
            "obs_flat": np.zeros((t, n_ally, obs_dim), dtype=np.float32),
        }
        meta = {
            "ally_keys": ["unit_00"],
            "n_units": n_units,
            "max_n_zone": max_n_zone,
            "task_index": 1,
            "task_id": "t",
            "record_id": 0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "env"
            out = Path(tmp) / "flat"
            for task, rid in (("task_a", 0), ("task_b", 1)):
                m = dict(meta)
                m["record_id"] = rid
                write_env_centric_record(root / task / f"record-{rid:06d}", arrays, m)
            split_records_root(root, output_root=out, shuffle=True, seed=0, workers=2)
            self.assertEqual(len(discover_agent_centric_dirs(out)), 2)


class PolicyTagMaskTest(unittest.TestCase):
    def test_policy_tag_ladder(self) -> None:
        by_name = {s.name: policy_tag_for_spec(s) for s in DEFAULT_BEHAVIOR_MIX}
        self.assertEqual(by_name["heuristic_random"], 0)
        self.assertEqual(by_name["heuristic_novice"], 1)
        self.assertEqual(by_name["heuristic_medium_eps0.5"], 2)
        self.assertEqual(by_name["heuristic_medium"], 3)
        self.assertEqual(by_name["heuristic_advanced"], 4)
        self.assertEqual(by_name["oracle_eps0.3"], 5)
        self.assertEqual(by_name["oracle_eps0.1"], 6)
        self.assertEqual(by_name["oracle_pure"], 7)
        self.assertEqual(POLICY_TAG_MASK, 8)
        # Quality-ordered mix: policy_id matches policy_tag for defaults.
        for spec in DEFAULT_BEHAVIOR_MIX:
            self.assertEqual(int(spec.policy_id), policy_tag_for_spec(spec), spec.name)

    def test_build_policy_mask_holds_until_switch(self) -> None:
        ids = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)

        class Scripted:
            def __init__(self, values):
                self.values = list(values)
                self.i = 0

            def random(self):
                v = self.values[self.i]
                self.i += 1
                return v

        # p=0.5: 0.0→mask, 0.9→keep, 0.0→mask; held within each policy segment.
        scripted = Scripted([0.0, 0.9, 0.0])
        mask = build_policy_mask(ids, mask_prob=0.5, rng=scripted)  # type: ignore[arg-type]
        self.assertTrue(np.array_equal(mask, np.array([1, 1, 1, 0, 0, 1], dtype=np.uint8)))


class WinrateAdaptTest(unittest.TestCase):
    def test_strong_group_excludes_high_eps_oracle(self) -> None:
        by_name = {s.name: s for s in DEFAULT_BEHAVIOR_MIX}
        self.assertFalse(is_strong_spec(by_name["oracle_eps0.3"]))
        self.assertTrue(is_strong_spec(by_name["oracle_eps0.1"]))
        self.assertTrue(is_strong_spec(by_name["oracle_pure"]))
        self.assertTrue(is_strong_spec(by_name["heuristic_advanced"]))
        self.assertFalse(is_strong_spec(by_name["heuristic_medium"]))

    def test_reweight_monotonic_strong_mass(self) -> None:
        weak = reweight_mix(DEFAULT_BEHAVIOR_MIX, 0.0)
        mid = reweight_mix(DEFAULT_BEHAVIOR_MIX, 0.5)
        capped = reweight_mix(DEFAULT_BEHAVIOR_MIX, 1.0, strength_max=0.7)

        def strong_mass(mix):
            return sum(s.weight for s in mix if is_strong_spec(s))

        def weak_mass(mix):
            return sum(s.weight for s in mix if not is_strong_spec(s))

        self.assertLess(strong_mass(weak), strong_mass(mid))
        self.assertLess(strong_mass(mid), strong_mass(capped))
        self.assertAlmostEqual(sum(s.weight for s in capped), 1.0, places=5)
        # strength is clipped to strength_max → weak mass stays ≥ 1 - max.
        self.assertAlmostEqual(strong_mass(capped), 0.7, places=5)
        self.assertAlmostEqual(weak_mass(capped), 0.3, places=5)
        self.assertGreater(
            {s.name: s.weight for s in capped}["oracle_eps0.3"], 0.0
        )

    def test_oracle_focus_boosts_pure_inside_strong(self) -> None:
        base = reweight_mix(DEFAULT_BEHAVIOR_MIX, 0.7, oracle_focus=0.0)
        focused = reweight_mix(DEFAULT_BEHAVIOR_MIX, 0.7, oracle_focus=1.0)
        base_w = {s.name: s.weight for s in base}
        focused_w = {s.name: s.weight for s in focused}
        self.assertGreater(focused_w["oracle_pure"], base_w["oracle_pure"])
        # At full focus, all strong mass sits on oracle_pure.
        self.assertAlmostEqual(focused_w["oracle_pure"], 0.7, places=5)
        self.assertAlmostEqual(focused_w["heuristic_advanced"], 0.0, places=5)
        self.assertAlmostEqual(focused_w["oracle_eps0.1"], 0.0, places=5)
        # Weak policies retained.
        self.assertGreater(focused_w["oracle_eps0.3"], 0.0)

    def test_focus_policy_advanced_concentrates_strong(self) -> None:
        focused = reweight_mix(
            DEFAULT_BEHAVIOR_MIX,
            0.7,
            oracle_focus=1.0,
            focus_policy="heuristic_advanced",
        )
        focused_w = {s.name: s.weight for s in focused}
        self.assertAlmostEqual(focused_w["heuristic_advanced"], 0.7, places=5)
        self.assertAlmostEqual(focused_w["oracle_pure"], 0.0, places=5)
        self.assertAlmostEqual(focused_w["oracle_eps0.1"], 0.0, places=5)
        self.assertGreater(focused_w["oracle_eps0.3"], 0.0)

    def test_win_rate_max_none_skips_upper_bound(self) -> None:
        self.assertTrue(in_win_rate_band(0.95, win_rate_min=0.3, win_rate_max=None))
        self.assertFalse(in_win_rate_band(0.95, win_rate_min=0.3, win_rate_max=0.7))
        self.assertFalse(in_win_rate_band(0.1, win_rate_min=0.3, win_rate_max=None))

    def test_choose_strength_directions(self) -> None:
        up = choose_strength(0.05, win_rate_min=0.3, win_rate_max=None, current_strength=0.5)
        self.assertGreater(up, 0.5)
        self.assertLessEqual(up, 0.7)
        down = choose_strength(0.9, win_rate_min=0.3, win_rate_max=0.7, current_strength=0.5)
        self.assertLess(down, 0.5)
        stay = choose_strength(0.5, win_rate_min=0.3, win_rate_max=None, current_strength=0.5)
        self.assertEqual(stay, 0.5)

    def test_choose_adapt_params_focus_after_strength_cap(self) -> None:
        nxt = choose_adapt_params(
            0.05,
            win_rate_min=0.3,
            win_rate_max=None,
            strength=0.7,
            oracle_focus=0.0,
            strength_max=0.7,
        )
        self.assertIsNotNone(nxt)
        assert nxt is not None
        s, f = nxt
        self.assertAlmostEqual(s, 0.7, places=5)
        self.assertGreater(f, 0.0)
        stuck = choose_adapt_params(
            0.05,
            win_rate_min=0.3,
            win_rate_max=None,
            strength=0.7,
            oracle_focus=1.0,
            strength_max=0.7,
        )
        self.assertIsNone(stuck)

    def test_summarize_arrays_hp_draw_and_win(self) -> None:
        done = np.array([0, 1, 0, 1], dtype=np.uint8)
        is_win = np.array([0, 0, 0, 0], dtype=np.uint8)
        # N=2 units: team0, team1
        health = np.array(
            [
                [10.0, 10.0],
                [5.0, 5.0],  # draw
                [10.0, 10.0],
                [8.0, 2.0],  # win
            ],
            dtype=np.float32,
        )
        team = np.array(
            [
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
            ],
            dtype=np.int32,
        )
        counts = summarize_arrays(
            done=done, is_win=is_win, unit_health=health, unit_team=team
        )
        self.assertEqual(counts["episodes"], 2)
        self.assertEqual(counts["draw"], 1)
        self.assertEqual(counts["win"], 1)
        self.assertEqual(counts["loss"], 0)
        self.assertAlmostEqual(win_rate_from_counts(counts), 1.0)


if __name__ == "__main__":
    unittest.main()
