from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "train_parallel.py"
SPEC = importlib.util.spec_from_file_location("train_parallel", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
train_parallel = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = train_parallel
SPEC.loader.exec_module(train_parallel)


class TrainParallelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.task_path = self.root / "tasks.json"
        self.task_path.write_text(
            json.dumps({"schema_version": "1.0", "tasks": [{"id": i} for i in range(4)]}),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_config(self, **overrides: object) -> Path:
        config = {
            "algorithm": "mappo",
            "task_file_path": "tasks.json",
            "save_path": "runs",
            "gpu_ids": "0, 2",
            "marl_per_gpu": 1,
            "seed_base": 10,
            "NUM_ENVS": 8,
            "NUM_STEPS": 16,
            "TOTAL_TIMESTEPS": 1024,
            "early_stop": {"enabled": True, "debug_mode": True, "patience": 5},
            "wandb": {"enabled": False},
            "algorithm_args": {"LR": 0.001},
        }
        config.update(overrides)
        path = self.root / "parallel.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_load_config_resolves_paths_and_selects_tasks(self) -> None:
        config = train_parallel.load_config(self.write_config(task_indices=[3, 1]))

        self.assertEqual(config["gpu_ids"], ["0", "2"])
        self.assertEqual(config["task_indices"], [3, 1])
        self.assertEqual(config["task_file_path"], str(self.task_path.resolve()))
        self.assertEqual(config["save_path"], str((self.root / "runs").resolve()))
        self.assertEqual(config["cpu_core_reserve"], 8)
        self.assertIsInstance(config["threads_per_coach"], int)
        self.assertGreaterEqual(config["threads_per_coach"], 1)

    def test_threads_per_coach_auto_from_cpu_cores(self) -> None:
        config = train_parallel.load_config(
            self.write_config(
                gpu_ids="cpu",
                marl_per_gpu=32,
                threads_per_coach=None,
                cpu_cores=256,
                cpu_core_reserve=8,
            )
        )

        # floor((256 - 8) / 32) = 7
        self.assertEqual(config["threads_per_coach"], 7)

    def test_threads_per_coach_explicit_override(self) -> None:
        config = train_parallel.load_config(
            self.write_config(threads_per_coach=4, cpu_cores=256, marl_per_gpu=64)
        )
        self.assertEqual(config["threads_per_coach"], 4)

    def test_child_process_env_sets_thread_caps(self) -> None:
        config = train_parallel.load_config(self.write_config(threads_per_coach=3))
        with mock.patch.dict(os.environ, {"XLA_FLAGS": "--xla_dump_to=/tmp"}, clear=False):
            env = train_parallel.child_process_env(config, "7")

        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "7")
        self.assertEqual(env["OMP_NUM_THREADS"], "3")
        self.assertEqual(env["MKL_NUM_THREADS"], "3")
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "3")
        self.assertEqual(env["TF_NUM_INTRAOP_THREADS"], "3")
        self.assertEqual(env["TF_NUM_INTEROP_THREADS"], "1")
        self.assertIn("--xla_cpu_multi_thread_eigen=false", env["XLA_FLAGS"])
        self.assertIn("--xla_force_host_platform_device_count=1", env["XLA_FLAGS"])
        self.assertIn("--xla_dump_to=/tmp", env["XLA_FLAGS"])

    def test_dry_run_writes_job_configs_logs_and_manifest(self) -> None:
        config = train_parallel.load_config(self.write_config(task_indices=[0, 2, 3]))

        result = train_parallel.run(config, dry_run=True, python_executable="/test/python")

        self.assertEqual(result, 0)
        manifest = json.loads((self.root / "runs" / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "dry_run")
        self.assertEqual([job["gpu_id"] for job in manifest["jobs"]], ["0", "2", "0"])
        for job in manifest["jobs"]:
            self.assertEqual(
                job["command"][:3],
                ["/test/python", "-m", "src.baseline.marl_baseline"],
            )
            self.assertIn("--task-index", job["command"])
            self.assertIn("--save-path", job["command"])
            self.assertTrue(Path(job["stdout_path"]).is_file())
            self.assertTrue(Path(job["stderr_path"]).is_file())
            trainer_config = json.loads(Path(job["config_path"]).read_text())
            self.assertEqual(trainer_config["task_index"], job["task_index"])
            self.assertEqual(trainer_config["seed"], 10 + job["task_index"])
            self.assertEqual(trainer_config["save_path"], job["output_dir"])
            self.assertNotIn("gpu_ids", trainer_config)
            self.assertEqual(trainer_config["LR"], 0.001)
            self.assertTrue(trainer_config["early_stop_enabled"])
            self.assertTrue(trainer_config["early_stop_debug_mode"])
            self.assertEqual(trainer_config["early_stop_patience"], 5)
            self.assertEqual(trainer_config["wandb_mode"], "disabled")

    def test_failed_job_does_not_prevent_refill(self) -> None:
        config = train_parallel.load_config(
            self.write_config(gpu_ids=["5"], marl_per_gpu=1, task_indices=[0, 1, 2])
        )
        return_codes = iter([1, 0, 0])
        launches: list[dict[str, object]] = []

        class FakeProcess:
            next_pid = 100

            def __init__(self, command: list[str], **kwargs: object) -> None:
                self.command = command
                self.return_code = next(return_codes)
                self.pid = FakeProcess.next_pid
                FakeProcess.next_pid += 1
                launches.append(kwargs)

            def poll(self) -> int:
                return self.return_code

        with mock.patch.object(train_parallel.subprocess, "Popen", FakeProcess):
            result = train_parallel.run(config, poll_interval=0)

        self.assertEqual(result, 1)
        self.assertEqual(len(launches), 3)
        self.assertTrue(
            all(launch["env"]["CUDA_VISIBLE_DEVICES"] == "5" for launch in launches)
        )
        self.assertTrue(
            all(
                launch["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
                for launch in launches
            )
        )
        self.assertTrue(
            all(launch["env"]["OMP_NUM_THREADS"] == str(config["threads_per_coach"]) for launch in launches)
        )
        manifest = json.loads((self.root / "runs" / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["summary"]["succeeded"], 2)
        self.assertEqual(manifest["summary"]["failed"], 1)
        self.assertEqual(
            [job["status"] for job in manifest["jobs"]],
            ["failed", "succeeded", "succeeded"],
        )

    def test_rejects_invalid_task_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "out of range"):
            train_parallel.load_config(self.write_config(task_indices=[4]))

    def test_rejects_unknown_algorithm(self) -> None:
        with self.assertRaisesRegex(ValueError, "algorithm must be one of"):
            train_parallel.load_config(self.write_config(algorithm="unknown"))


if __name__ == "__main__":
    unittest.main()
