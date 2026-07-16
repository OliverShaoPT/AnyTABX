#!/usr/bin/env python3
"""Run one MARL trainer process per task with fixed per-GPU concurrency."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

TRAINER_MODULE = "src.baseline.marl_baseline"
MANIFEST_NAME = "manifest.json"
SUPPORTED_ALGORITHMS = {"ippo", "mappo", "mappo_rnd", "iql", "vdn", "qmix"}
TRAINER_TOP_LEVEL_KEYS = {
    "algorithm",
    "task_file_path",
    "NUM_ENVS",
    "NUM_STEPS",
    "TOTAL_TIMESTEPS",
    "host_chunk_updates",
    "early_stop_enabled",
    "early_stop_window",
    "early_stop_patience",
    "early_stop_min_delta",
    "early_stop_warmup",
    "wandb_mode",
    "wandb_run_name",
    "wandb_project",
    "jit",
    "PHYSICS",
    "HEURISTIC",
    "WORLD_STATE_TYPE",
    "POSITION_PERMUTATION",
    "FLIP",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    """Atomically write JSON so the live manifest is never partially written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def parse_gpu_ids(value: Any) -> list[str]:
    if isinstance(value, str):
        gpu_ids = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, list):
        gpu_ids = [str(item).strip() for item in value]
    else:
        raise ValueError("gpu_ids must be a comma-separated string or a list")
    if not gpu_ids or any(not gpu_id for gpu_id in gpu_ids):
        raise ValueError("gpu_ids must contain at least one non-empty GPU id")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("gpu_ids must not contain duplicates")
    return gpu_ids


def load_task_count(path: Path) -> int:
    try:
        with path.open(encoding="utf-8") as stream:
            task_bank = json.load(stream)
    except FileNotFoundError as error:
        raise ValueError(f"task_file_path does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid task bank JSON: {error}") from error

    tasks = task_bank.get("tasks") if isinstance(task_bank, dict) else task_bank
    if not isinstance(tasks, list):
        raise ValueError("task bank must be a list or an object containing a 'tasks' list")
    return len(tasks)


def _positive_int(config: dict[str, Any], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_config(config_path: Path) -> dict[str, Any]:
    try:
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
    except FileNotFoundError as error:
        raise ValueError(f"config does not exist: {config_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid config JSON: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("config root must be a JSON object")

    for name in ("algorithm", "task_file_path", "save_path"):
        if not isinstance(config.get(name), str) or not config[name].strip():
            raise ValueError(f"{name} must be a non-empty string")
    if config["algorithm"] not in SUPPORTED_ALGORITHMS:
        supported = ", ".join(sorted(SUPPORTED_ALGORITHMS))
        raise ValueError(f"algorithm must be one of: {supported}")
    config["gpu_ids"] = parse_gpu_ids(config.get("gpu_ids"))
    config["marl_per_gpu"] = _positive_int(config, "marl_per_gpu")

    seed_base = config.get("seed_base", 0)
    if isinstance(seed_base, bool) or not isinstance(seed_base, int):
        raise ValueError("seed_base must be an integer")
    config["seed_base"] = seed_base

    for name in ("NUM_ENVS", "NUM_STEPS", "TOTAL_TIMESTEPS"):
        if name in config:
            _positive_int(config, name)
    for name in ("algorithm_args", "early_stop", "wandb"):
        if name in config and not isinstance(config[name], dict):
            raise ValueError(f"{name} must be a JSON object")

    config_dir = config_path.resolve().parent
    task_path = Path(config["task_file_path"]).expanduser()
    save_path = Path(config["save_path"]).expanduser()
    if not task_path.is_absolute():
        task_path = config_dir / task_path
    if not save_path.is_absolute():
        save_path = config_dir / save_path
    config["task_file_path"] = str(task_path.resolve())
    config["save_path"] = str(save_path.resolve())

    task_count = load_task_count(task_path.resolve())
    task_indices = config.get("task_indices")
    if task_indices is None:
        task_indices = list(range(task_count))
    if not isinstance(task_indices, list):
        raise ValueError("task_indices must be a list of integers")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in task_indices):
        raise ValueError("task_indices must be a list of integers")
    if len(task_indices) != len(set(task_indices)):
        raise ValueError("task_indices must not contain duplicates")
    invalid = [index for index in task_indices if index < 0 or index >= task_count]
    if invalid:
        raise ValueError(f"task_indices out of range for {task_count} tasks: {invalid}")
    config["task_indices"] = task_indices
    return config


@dataclass
class RunningJob:
    process: subprocess.Popen[Any]
    record: dict[str, Any]
    stdout: Any
    stderr: Any
    slot: tuple[str, int]


def make_trainer_config(
    config: dict[str, Any], task_index: int, seed: int, job_dir: Path
) -> dict[str, Any]:
    trainer_config = {
        key: copy.deepcopy(value)
        for key, value in config.items()
        if key in TRAINER_TOP_LEVEL_KEYS
    }
    trainer_config.update(copy.deepcopy(config.get("algorithm_args", {})))

    early_stop = config.get("early_stop", {})
    early_aliases = {
        "enabled": "early_stop_enabled",
        "window": "early_stop_window",
        "patience": "early_stop_patience",
        "min_delta": "early_stop_min_delta",
        "warmup": "early_stop_warmup",
        "host_chunk_updates": "host_chunk_updates",
    }
    for source, destination in early_aliases.items():
        if source in early_stop:
            trainer_config[destination] = early_stop[source]
    wandb = config.get("wandb", {})
    if "enabled" in wandb and "mode" not in wandb:
        trainer_config["wandb_mode"] = "online" if wandb["enabled"] else "disabled"
    wandb_aliases = {
        "mode": "wandb_mode",
        "run_name": "wandb_run_name",
        "project": "wandb_project",
    }
    for source, destination in wandb_aliases.items():
        if source in wandb:
            trainer_config[destination] = wandb[source]

    trainer_config.update(
        {
            "task_index": task_index,
            "seed": seed,
            "save_path": str(job_dir),
        }
    )
    return trainer_config


def trainer_command(python_executable: str, trainer_config: dict[str, Any]) -> list[str]:
    command = [python_executable, "-m", TRAINER_MODULE]
    for key, value in trainer_config.items():
        option = f"--{key.replace('_', '-')}"
        if value is None:
            continue
        if isinstance(value, bool):
            command.append(option if value else f"--no-{option[2:]}")
        elif isinstance(value, (str, int, float)):
            command.extend((option, str(value)))
        else:
            raise ValueError(f"trainer option {key!r} must be a scalar value")
    return command


def build_job(
    config: dict[str, Any],
    task_index: int,
    gpu_id: str,
    slot_index: int,
    python_executable: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed = config["seed_base"] + task_index
    job_dir = Path(config["save_path"]) / f"task_{task_index:05d}_seed_{seed}"
    stdout_path = job_dir / "stdout.log"
    stderr_path = job_dir / "stderr.log"
    job_config_path = job_dir / "trainer_config.json"

    trainer_config = make_trainer_config(config, task_index, seed, job_dir)
    command = trainer_command(python_executable, trainer_config)
    record = {
        "task_index": task_index,
        "seed": seed,
        "gpu_id": gpu_id,
        "slot_index": slot_index,
        "output_dir": str(job_dir),
        "config_path": str(job_config_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "command": command,
        "status": "pending",
        "return_code": None,
        "pid": None,
        "started_at": None,
        "finished_at": None,
    }
    return trainer_config, record


def _save_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = utc_now()
    write_json(manifest_path, manifest)


def run(
    config: dict[str, Any],
    *,
    dry_run: bool = False,
    python_executable: str = sys.executable,
    poll_interval: float = 0.2,
) -> int:
    save_path = Path(config["save_path"])
    save_path.mkdir(parents=True, exist_ok=True)
    manifest_path = save_path / MANIFEST_NAME
    slots = [
        (gpu_id, slot_index)
        for gpu_id in config["gpu_ids"]
        for slot_index in range(config["marl_per_gpu"])
    ]
    available_slots = deque(slots)
    pending_tasks = deque(config["task_indices"])
    running: list[RunningJob] = []
    records: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "version": 1,
        "algorithm": config["algorithm"],
        "task_file_path": config["task_file_path"],
        "save_path": config["save_path"],
        "gpu_ids": config["gpu_ids"],
        "marl_per_gpu": config["marl_per_gpu"],
        "dry_run": dry_run,
        "created_at": utc_now(),
        "updated_at": None,
        "status": "planning" if dry_run else "running",
        "summary": {},
        "jobs": records,
    }

    # Materialize all dry-run jobs in deterministic task/slot order.
    if dry_run:
        for position, task_index in enumerate(config["task_indices"]):
            gpu_id, slot_index = slots[position % len(slots)]
            trainer_config, record = build_job(
                config, task_index, gpu_id, slot_index, python_executable
            )
            job_dir = Path(record["output_dir"])
            job_dir.mkdir(parents=True, exist_ok=True)
            write_json(Path(record["config_path"]), trainer_config)
            Path(record["stdout_path"]).touch()
            Path(record["stderr_path"]).touch()
            record["status"] = "dry_run"
            records.append(record)
        manifest["status"] = "dry_run"
        manifest["summary"] = {"total": len(records), "dry_run": len(records)}
        _save_manifest(manifest_path, manifest)
        return 0

    _save_manifest(manifest_path, manifest)
    interrupted = False
    try:
        while pending_tasks or running:
            while pending_tasks and available_slots:
                task_index = pending_tasks.popleft()
                slot = available_slots.popleft()
                gpu_id, slot_index = slot
                trainer_config, record = build_job(
                    config, task_index, gpu_id, slot_index, python_executable
                )
                records.append(record)
                job_dir = Path(record["output_dir"])
                job_dir.mkdir(parents=True, exist_ok=True)
                write_json(Path(record["config_path"]), trainer_config)
                stdout = Path(record["stdout_path"]).open("w", encoding="utf-8")
                stderr = Path(record["stderr_path"]).open("w", encoding="utf-8")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
                env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
                try:
                    process = subprocess.Popen(
                        record["command"],
                        cwd=str(Path(__file__).resolve().parents[1]),
                        env=env,
                        stdout=stdout,
                        stderr=stderr,
                    )
                except OSError as error:
                    stderr.write(f"Failed to start trainer: {error}\n")
                    stdout.close()
                    stderr.close()
                    record.update(
                        {
                            "status": "failed",
                            "return_code": 127,
                            "started_at": utc_now(),
                            "finished_at": utc_now(),
                        }
                    )
                    available_slots.append(slot)
                else:
                    record.update(
                        {
                            "status": "running",
                            "pid": process.pid,
                            "started_at": utc_now(),
                        }
                    )
                    running.append(RunningJob(process, record, stdout, stderr, slot))
                _save_manifest(manifest_path, manifest)

            completed: list[RunningJob] = []
            for job in running:
                return_code = job.process.poll()
                if return_code is None:
                    continue
                job.stdout.close()
                job.stderr.close()
                job.record.update(
                    {
                        "status": "succeeded" if return_code == 0 else "failed",
                        "return_code": return_code,
                        "finished_at": utc_now(),
                    }
                )
                available_slots.append(job.slot)
                completed.append(job)
            if completed:
                running = [job for job in running if job not in completed]
                _save_manifest(manifest_path, manifest)
            elif running:
                time.sleep(poll_interval)
    except KeyboardInterrupt:
        interrupted = True
        for job in running:
            job.process.terminate()
        for job in running:
            try:
                return_code = job.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                job.process.kill()
                return_code = job.process.wait()
            job.stdout.close()
            job.stderr.close()
            job.record.update(
                {
                    "status": "interrupted",
                    "return_code": return_code,
                    "finished_at": utc_now(),
                }
            )
        for task_index in pending_tasks:
            _, record = build_job(config, task_index, "", -1, python_executable)
            record["status"] = "not_started"
            records.append(record)

    succeeded = sum(record["status"] == "succeeded" for record in records)
    failed = sum(record["status"] == "failed" for record in records)
    manifest["status"] = "interrupted" if interrupted else ("failed" if failed else "succeeded")
    manifest["summary"] = {
        "total": len(config["task_indices"]),
        "succeeded": succeeded,
        "failed": failed,
        "not_started": len(config["task_indices"]) - succeeded - failed,
    }
    _save_manifest(manifest_path, manifest)
    return 130 if interrupted else (1 if failed else 0)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="parallel training JSON configuration")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write job configs and manifest without starting trainers",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        dest="python_executable",
        help="Python executable used for trainer subprocesses",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        config = load_config(args.config.resolve())
        return run(
            config,
            dry_run=args.dry_run,
            python_executable=args.python_executable,
        )
    except ValueError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
