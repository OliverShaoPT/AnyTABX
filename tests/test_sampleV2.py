from __future__ import annotations

import ast
import copy
import json
import math
import random
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_V2 = ROOT / "src" / "tabx" / "sampleV2.py"
PURE_NAMES = {
    "DescriptorLayout",
    "ReferenceScaler",
    "_flat",
    "_descriptor_variant",
    "_environment_descriptor",
    "_descriptor_matrix",
    "_robust_scale",
    "_pooled_feature_scales",
    "_fit_reference_scaler",
    "_group_distance_matrix",
    "_nearest_environment_distance",
    "_reference_nn_values",
    "_parameter_pairwise_matrix",
    "_behavior_pairwise_matrix",
    "_largest_remainder_quotas",
    "_scale_quotas",
    "_count_bucket",
    "_task_scale_stratum",
    "_stratum_bounds",
    "_build_exact_free_zones",
}
PURE_CONSTANTS = {
    "N_UNIT_TYPES",
    "N_ZONE_TYPES",
    "ATTACK_TYPE_VALUES",
    "SCALE_STRATA",
    "GROUP_WEIGHTS",
}


def load_pure_sample_v2() -> SimpleNamespace:
    """Load descriptor/quota code without importing the heavy JAX runtime."""

    tree = ast.parse(SAMPLE_V2.read_text(encoding="utf-8"))
    nodes: list[ast.stmt] = [
        ast.ImportFrom(
            module="__future__",
            names=[ast.alias(name="annotations")],
            level=0,
        )
    ]
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            if node.name in PURE_NAMES:
                nodes.append(node)
        elif isinstance(node, ast.Assign):
            names = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if names & PURE_CONSTANTS:
                nodes.append(node)
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace = {
        "Any": Any,
        "Iterable": Iterable,
        "Sequence": Sequence,
        "dataclass": dataclass,
        "math": math,
        "np": np,
        "random": random,
        "v1": SimpleNamespace(
            _DISTANCE_GROUPS=(
                (np.r_[0:9, 16:25], 0.35),
                (np.r_[9:13, 25:29, 32], 0.30),
                (np.r_[13:16, 29:32], 0.10),
                (np.r_[33:38], 0.25),
            )
        ),
    }
    exec(compile(module, str(SAMPLE_V2), "exec"), namespace)
    return SimpleNamespace(**namespace)


class SampleV2PureCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v2 = load_pure_sample_v2()
        bank = json.loads(
            (ROOT / "task_files" / "balanced_tasks_v1.json").read_text(
                encoding="utf-8"
            )
        )
        cls.tasks = bank["tasks"][:24]

    def test_fixed_descriptor_is_827_dimensional(self) -> None:
        matrix, layout = self.v2._descriptor_matrix(self.tasks)
        self.assertEqual(matrix.shape, (24, 827))
        self.assertEqual(layout.dimension, 827)

        scaler = self.v2._fit_reference_scaler(matrix, layout)
        self.assertTrue(np.all(np.isfinite(scaler.scale)))
        self.assertTrue(np.all(scaler.scale > 0.0))

    def test_descriptor_is_invariant_to_x_mirror(self) -> None:
        task = copy.deepcopy(self.tasks[0])
        mirrored = copy.deepcopy(task)
        positions = np.asarray(
            mirrored["scenario"]["positions"],
            dtype=float,
        )
        positions[:, 0] *= -1.0
        mirrored["scenario"]["positions"] = positions.tolist()
        rotations = np.asarray(
            mirrored["scenario"]["rotations"],
            dtype=float,
        )
        mirrored["scenario"]["rotations"] = (
            np.pi - rotations
        ).tolist()
        zone_positions = np.asarray(
            mirrored["zone_scenario"]["position"],
            dtype=float,
        ).reshape(-1, 2)
        zone_positions[:, 0] *= -1.0
        mirrored["zone_scenario"]["position"] = zone_positions.tolist()

        first, _ = self.v2._environment_descriptor(task)
        second, _ = self.v2._environment_descriptor(mirrored)
        np.testing.assert_allclose(first, second, rtol=0.0, atol=1e-10)

    def test_default_scale_quotas_put_eighty_percent_outside_legacy(self) -> None:
        config = SimpleNamespace(
            n_tasks=300,
            scale_stratum_ratios=(0.20, 0.20, 0.15, 0.15, 0.10, 0.20),
        )
        quotas = self.v2._scale_quotas(config)

        self.assertEqual(sum(quotas.values()), 300)
        self.assertEqual(quotas["legacy_scale"], 60)
        self.assertEqual(
            sum(
                value
                for name, value in quotas.items()
                if name != "legacy_scale"
            ),
            240,
        )

    def test_scale_strata_are_mutually_exclusive(self) -> None:
        cases = {
            "legacy_scale": (10, 10),
            "large_team": (11, 10),
            "xlarge_team": (16, 10),
            "high_zone": (10, 11),
            "xhigh_zone": (10, 16),
            "joint_ood": (11, 11),
        }
        for expected, (team_count, zone_count) in cases.items():
            task = {
                "scenario": {"teams": [0] * team_count + [1] * team_count},
                "zone_scenario": {"n_zone": zone_count},
            }
            self.assertEqual(
                self.v2._task_scale_stratum(task),
                expected,
            )

    def test_exact_zone_builder_returns_requested_count(self) -> None:
        task = self.tasks[0]
        zones = self.v2._build_exact_free_zones(
            random.Random(42),
            task["scenario"],
            task["grid_info"],
            20,
        )
        self.assertEqual(zones["n_zone"], 20)
        for key in ("zone_type", "position", "axes", "effect_value"):
            self.assertEqual(len(zones[key]), 20)

    def test_pairwise_matrices_are_symmetric_with_zero_diagonal(self) -> None:
        rng = np.random.default_rng(42)
        parameter = self.v2._parameter_pairwise_matrix(
            rng.normal(size=(12, 38))
        )
        behavior = self.v2._behavior_pairwise_matrix(
            rng.normal(size=(12, 6))
        )
        for matrix in (parameter, behavior):
            np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
            np.testing.assert_allclose(np.diag(matrix), 0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
