"""Unit tests for generate/ helpers (no full env rollout required)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from generate.agent_centric import flat_agent_dirname, split_one_record, split_records_root
from generate.behavior_mix import BehaviorSwitcher, DEFAULT_BEHAVIOR_MIX
from generate.dump_schema import (
    OWN_FEATURE_DIM,
    OTHER_FEATURE_DIM,
    ZONE_FEATURE_DIM,
    discover_agent_centric_dirs,
    split_flat_obs,
    write_env_centric_record,
)


class BehaviorSwitcherTest(unittest.TestCase):
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
            "behavior_policy_id": np.zeros((t,), dtype=np.int32),
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
            paths = split_one_record(record_dir)
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


if __name__ == "__main__":
    unittest.main()
