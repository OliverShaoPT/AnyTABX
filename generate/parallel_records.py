"""Orchestrate even, parallel env-centric record generation (GPU production path).

``device=cpu`` remains available for local debug only — not for mass production.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue as queue_mod
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from generate.dump_schema import ensure_dir
from generate.progress import LocalQueue, ProgressTracker
from generate.record_worker import force_jax_cpu_env, worker_main

# NOTE: do not import generate.task_package / src.tabx at module import time.
# Those pull in JAX; for device=cpu we must set CUDA/JAX env first.


def _planned_record_count(jobs: list[dict[str, Any]]) -> int:
    return sum(len(item["record_ids"]) for job in jobs for item in job["items"])


def _attach_progress_queue(
    jobs: list[dict[str, Any]], queue: Any
) -> list[dict[str, Any]]:
    attached: list[dict[str, Any]] = []
    for job in jobs:
        payload = dict(job)
        payload["progress_queue"] = queue
        attached.append(payload)
    return attached


def _drain_progress_queue(
    queue: Any, tracker: ProgressTracker, stop: threading.Event
) -> None:
    """Pull progress events until stop, or until the Manager dies (EOF).

    ``EOFError`` / broken pipe are normal when the parent is SIGTERM'd or the
    Manager shuts down before this daemon thread exits — swallow them quietly.
    """

    def _get(timeout: float | None = None) -> Any:
        if timeout is None:
            return queue.get_nowait()
        return queue.get(timeout=timeout)

    while not stop.is_set():
        try:
            event = _get(0.2)
        except queue_mod.Empty:
            tracker.maybe_heartbeat()
            continue
        except (EOFError, BrokenPipeError, ConnectionError, OSError):
            return
        try:
            tracker.handle(event)
        except Exception as exc:  # noqa: BLE001 — keep progress thread alive
            print(f"[parallel_records] progress handle error: {exc}", flush=True)

    while True:
        try:
            event = _get()
        except queue_mod.Empty:
            break
        except (EOFError, BrokenPipeError, ConnectionError, OSError):
            return
        try:
            tracker.handle(event)
        except Exception as exc:  # noqa: BLE001
            print(f"[parallel_records] progress handle error: {exc}", flush=True)
            break


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "Reading YAML configs requires PyYAML. "
                "Install with `pip install pyyaml`, or pass a .json config."
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


@dataclass(frozen=True)
class WorkerSlot:
    worker_id: int
    device: str
    gpu_id: int | None


def build_worker_slots(
    *,
    device: str,
    cpu_workers: int,
    gpu_ids: list[int],
    workers_per_gpu: int,
) -> list[WorkerSlot]:
    device = device.lower()
    if device == "cpu":
        n = max(1, int(cpu_workers))
        return [WorkerSlot(i, "cpu", None) for i in range(n)]
    if device == "gpu":
        if not gpu_ids:
            raise ValueError("gpu mode requires non-empty gpu_ids")
        if workers_per_gpu <= 0:
            raise ValueError("workers_per_gpu must be positive")
        slots: list[WorkerSlot] = []
        worker_id = 0
        for gpu_id in gpu_ids:
            for _ in range(workers_per_gpu):
                slots.append(WorkerSlot(worker_id, "gpu", int(gpu_id)))
                worker_id += 1
        return slots
    raise ValueError(f"Unsupported device: {device!r} (expected cpu|gpu)")


def even_records_per_task(total_records: int, n_tasks: int) -> list[int]:
    """Return per-task record counts that sum to total_records.

    Everyone gets ``total_records // n_tasks``; the remainder is given one-by-one
    to the first tasks.
    """

    if total_records <= 0:
        raise ValueError("total_records must be positive")
    if n_tasks <= 0:
        raise ValueError("n_tasks must be positive")
    base, rem = divmod(total_records, n_tasks)
    return [base + (1 if i < rem else 0) for i in range(n_tasks)]


def flatten_record_units(
    packages: list[Any],
    counts: list[int],
    start_index: int,
) -> list[tuple[str, int]]:
    """Flatten all (package_name, record_id) work units across tasks."""

    units: list[tuple[str, int]] = []
    for package, count in zip(packages, counts):
        if count <= 0:
            continue
        for record_id in range(start_index, start_index + count):
            units.append((package.name, int(record_id)))
    return units


def _group_units_by_package(
    units: list[tuple[str, int]],
) -> list[dict[str, Any]]:
    """Preserve assignment order while grouping record ids under each package."""

    items: list[dict[str, Any]] = []
    index: dict[str, dict[str, Any]] = {}
    for package_name, record_id in units:
        item = index.get(package_name)
        if item is None:
            item = {"package_name": package_name, "record_ids": []}
            index[package_name] = item
            items.append(item)
        item["record_ids"].append(record_id)
    return items


def plan_jobs_task_affinity(
    *,
    packages: list[Any],
    counts: list[int],
    start_index: int,
    slots: list[WorkerSlot],
) -> list[tuple[WorkerSlot, list[dict[str, Any]]]]:
    """Assign whole tasks to workers (round-robin over tasks).

    Each worker owns all records of its tasks so JAX JIT is amortized in-process.
    Active workers = min(n_slots, n_tasks_with_work).
    """

    task_items: list[dict[str, Any]] = []
    for package, count in zip(packages, counts):
        if count <= 0:
            continue
        task_items.append(
            {
                "package_name": package.name,
                "record_ids": list(
                    range(start_index, start_index + int(count))
                ),
            }
        )
    if not task_items:
        return []

    n_workers = min(len(slots), len(task_items))
    active_slots = slots[:n_workers]
    owned: list[list[dict[str, Any]]] = [[] for _ in range(n_workers)]
    for index, item in enumerate(task_items):
        owned[index % n_workers].append(item)
    return [
        (slot, items)
        for slot, items in zip(active_slots, owned)
        if items
    ]


def plan_jobs_record_stripe(
    *,
    packages: list[Any],
    counts: list[int],
    start_index: int,
    slots: list[WorkerSlot],
) -> list[tuple[WorkerSlot, list[dict[str, Any]]]]:
    """LEGACY: flatten (task, record_id) and round-robin (avoid for GPU production)."""

    units = flatten_record_units(packages, counts, start_index)
    if not units:
        return []

    active_slots = slots[: min(len(slots), len(units))]
    n_workers = len(active_slots)
    owned: list[list[tuple[str, int]]] = [[] for _ in range(n_workers)]
    for index, unit in enumerate(units):
        owned[index % n_workers].append(unit)
    return [
        (slot, _group_units_by_package(units_for_worker))
        for slot, units_for_worker in zip(active_slots, owned)
        if units_for_worker
    ]


def plan_jobs(
    *,
    packages_root: Path,
    output_root: Path,
    packages: list[Any],
    counts: list[int],
    start_index: int,
    slots: list[WorkerSlot],
    total_timesteps: int,
    seed: int | None,
    min_behavior_steps: int,
    behavior_switch_prob: float,
    behavior_mix: list[dict[str, Any]] | None,
    jax_platform: str = "cuda",
    schedule: str = "task",
    stagger_s: float = 1.0,
    scan_rollout: bool = True,
    parallel_envs: int = 1,
    winrate_adapt: bool = False,
    win_rate_min: float = 0.30,
    win_rate_max: float | None = None,
    adapt_pilot_records: int = 4,
    adapt_max_iters: int = 3,
    adapt_strength_max: float = 0.7,
    independent_ally_policies: bool = False,
    mid_episode_policy_switch: bool = True,
    best_teacher_reference: bool = False,
) -> list[dict[str, Any]]:
    """Build one job payload per active worker.

    ``schedule``:
      - ``task`` (default): whole-task affinity; best for JAX compile reuse.
      - ``record``: LEGACY stripe of individual records; avoid for GPU production
        (multi-process cold JIT).
    """

    schedule = (schedule or "task").lower()
    if schedule == "task":
        assignments = plan_jobs_task_affinity(
            packages=packages,
            counts=counts,
            start_index=start_index,
            slots=slots,
        )
    elif schedule == "record":
        assignments = plan_jobs_record_stripe(
            packages=packages,
            counts=counts,
            start_index=start_index,
            slots=slots,
        )
    else:
        raise ValueError(f"Unsupported schedule={schedule!r} (expected task|record)")

    jobs: list[dict[str, Any]] = []
    for slot, items in assignments:
        jobs.append(
            {
                "packages_root": str(packages_root),
                "items": items,
                "output_root": str(output_root),
                "total_timesteps": total_timesteps,
                "seed": seed,
                "min_behavior_steps": min_behavior_steps,
                "behavior_switch_prob": behavior_switch_prob,
                "behavior_mix": behavior_mix,
                "device": slot.device,
                "gpu_id": slot.gpu_id,
                "worker_id": slot.worker_id,
                "jax_platform": jax_platform,
                "schedule": schedule,
                "stagger_s": float(stagger_s),
                "scan_rollout": bool(scan_rollout),
                "parallel_envs": max(1, int(parallel_envs)),
                "winrate_adapt": bool(winrate_adapt),
                "win_rate_min": float(win_rate_min),
                "win_rate_max": win_rate_max,
                "adapt_pilot_records": int(adapt_pilot_records),
                "adapt_max_iters": int(adapt_max_iters),
                "adapt_strength_max": float(adapt_strength_max),
                "independent_ally_policies": bool(independent_ally_policies),
                "mid_episode_policy_switch": bool(mid_episode_policy_switch),
                "best_teacher_reference": bool(best_teacher_reference),
            }
        )
    return jobs


def run_from_config(config: dict[str, Any]) -> None:
    device = str(config.get("device", "gpu")).lower()
    # Parent only discovers packages / orchestrates workers. Keep it off GPUs:
    # importing task_package → sample_task → jax would otherwise init CUDA on
    # physical GPU 0 and inflate nvidia-smi memory there (~tens of GB).
    force_jax_cpu_env()
    if device == "cpu":
        print(
            "[parallel_records] device=cpu (debug only, not for production) → "
            "JAX_PLATFORMS=cpu, JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1",
            flush=True,
        )
    else:
        print(
            "[parallel_records] parent pinned to CPU (workers set CUDA_VISIBLE_DEVICES)",
            flush=True,
        )

    from generate.task_package import discover_task_packages, pack_task_packages

    coach_root = config.get("coach_root") or config.get("task_packages_root")
    if not coach_root:
        raise ValueError("coach_root (or legacy task_packages_root) is required")
    packages_root = Path(coach_root)
    output_root = Path(config["output_root"])
    task_bank = config.get("task_bank")
    ckpt_root = config.get("ckpt_root")

    # Legacy: pack bank + ckpt tree into packages_root when empty.
    if task_bank and ckpt_root:
        packages_root.mkdir(parents=True, exist_ok=True)
        if not any(packages_root.rglob("task.json")):
            print(
                f"[parallel_records] packing packages from {task_bank} → {packages_root}",
                flush=True,
            )
            pack_task_packages(
                task_bank_path=task_bank,
                ckpt_root=ckpt_root,
                output_root=packages_root,
                algorithm=config.get("algorithm"),
                seed=config.get("pack_seed"),
            )

    packages = discover_task_packages(
        packages_root,
        seed=config.get("coach_seed", config.get("pack_seed")),
    )
    task_filter = config.get("task_index")
    if task_filter is not None:
        allowed = set(int(x) for x in task_filter)
        packages = [p for p in packages if p.task_index in allowed]
    if not packages:
        raise RuntimeError("No task packages to generate")

    total_records = int(config["total_records"])
    counts = even_records_per_task(total_records, len(packages))
    start_index = int(config.get("start_index", 0))
    slots = build_worker_slots(
        device=device,
        cpu_workers=int(config.get("cpu_workers", 8)),
        gpu_ids=[int(x) for x in (config.get("gpu_ids") or [])],
        workers_per_gpu=int(config.get("workers_per_gpu", 1)),
    )

    schedule = str(config.get("schedule", "task")).lower()
    stagger_s = float(config.get("stagger_s", 1.0) or 0.0)
    ensure_dir(output_root)
    jobs = plan_jobs(
        packages_root=packages_root,
        output_root=output_root,
        packages=packages,
        counts=counts,
        start_index=start_index,
        slots=slots,
        total_timesteps=int(config["total_timesteps"]),
        seed=config.get("seed"),
        min_behavior_steps=int(config.get("min_behavior_steps", 64)),
        behavior_switch_prob=float(config.get("behavior_switch_prob", 0.2)),
        behavior_mix=config.get("behavior_mix"),
        jax_platform=str(config.get("jax_platform", "cuda")),
        schedule=schedule,
        stagger_s=stagger_s,
        scan_rollout=bool(config.get("scan_rollout", True)),
        parallel_envs=max(1, int(config.get("parallel_envs", 1) or 1)),
        winrate_adapt=bool(config.get("winrate_adapt", False)),
        win_rate_min=float(config.get("win_rate_min", 0.30)),
        win_rate_max=(
            None
            if config.get("win_rate_max", None) is None
            else float(config.get("win_rate_max"))
        ),
        adapt_pilot_records=int(config.get("adapt_pilot_records", 4) or 4),
        adapt_max_iters=int(config.get("adapt_max_iters", 3) or 3),
        adapt_strength_max=float(config.get("adapt_strength_max", 0.7) or 0.7),
        independent_ally_policies=bool(
            config.get("independent_ally_policies", False)
        ),
        mid_episode_policy_switch=bool(
            config.get("mid_episode_policy_switch", True)
        ),
        best_teacher_reference=bool(config.get("best_teacher_reference", False)),
    )

    planned = _planned_record_count(jobs)
    parallel_envs = max(1, int(config.get("parallel_envs", 1) or 1))
    print(
        f"[parallel_records] schedule={schedule} stagger_s={stagger_s} "
        f"scan_rollout={config.get('scan_rollout', True)} "
        f"parallel_envs={parallel_envs} "
        f"winrate_adapt={config.get('winrate_adapt', False)} "
        f"best_teacher_reference={config.get('best_teacher_reference', False)} "
        f"independent_ally_policies={config.get('independent_ally_policies', False)} "
        f"mid_episode_policy_switch={config.get('mid_episode_policy_switch', True)} "
        f"win_rate_min={config.get('win_rate_min', 0.30)} "
        f"win_rate_max={config.get('win_rate_max', None)} "
        f"tasks={len(packages)} total_records={total_records} "
        f"planned={planned} "
        f"per_task={counts[0] if len(set(counts)) == 1 else counts} "
        f"slots={len(slots)} active_workers={len(jobs)} device={slots[0].device}",
        flush=True,
    )
    print(
        f"[parallel_records] progress → {output_root / 'generation_progress.json'} "
        f"(jsonl: generation_progress.jsonl)",
        flush=True,
    )
    for job in jobs:
        sample = [
            f"{item['package_name']}:{rid}"
            for item in job["items"]
            for rid in item["record_ids"]
        ]
        print(
            f"  worker={job['worker_id']:02d} device={job['device']} gpu={job['gpu_id']} "
            f"n_records={len(sample)} sample={sample[:4]}{'...' if len(sample) > 4 else ''}",
            flush=True,
        )

    tracker = ProgressTracker(total=planned, output_root=output_root)
    status = "done"
    try:
        if len(jobs) <= 1:
            queue = LocalQueue(tracker.handle)
            for job in _attach_progress_queue(jobs, queue):
                worker_main(job)
            return

        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        queue = manager.Queue()
        stop = threading.Event()
        drain = threading.Thread(
            target=_drain_progress_queue,
            args=(queue, tracker, stop),
            name="record-progress",
            daemon=True,
        )
        drain.start()
        try:
            # Re-assert CPU env in each child before it imports JAX.
            pool_kwargs: dict[str, Any] = {"processes": len(jobs)}
            if device == "cpu":
                pool_kwargs["initializer"] = force_jax_cpu_env
            with ctx.Pool(**pool_kwargs) as pool:
                for _ in pool.imap_unordered(
                    worker_main, _attach_progress_queue(jobs, queue)
                ):
                    pass
        finally:
            # Give the drain thread a moment to consume late events.
            time.sleep(0.3)
            stop.set()
            drain.join(timeout=5.0)
    except Exception:
        status = "failed"
        raise
    finally:
        tracker.finish(status=status)


def _parse_gpu_ids(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip() != ""]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evenly generate env-centric records in parallel (GPU production)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="generate/configs/record_gen.yaml",
        help="YAML/JSON config (behavior mix + defaults)",
    )
    parser.add_argument(
        "--coach_root",
        type=str,
        default=None,
        help="Root of self-contained coach leaves (task.json + safetensors)",
    )
    parser.add_argument(
        "--task_packages_root",
        type=str,
        default=None,
        help="Alias for --coach_root (legacy name)",
    )
    parser.add_argument(
        "--task_bank",
        type=str,
        default=None,
        help="Legacy: task bank JSON used only when packing into coach_root",
    )
    parser.add_argument("--ckpt_root", type=str, default=None)
    parser.add_argument("--algorithm", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--total_records", type=int, default=None)
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=None)
    parser.add_argument("--device", type=str, choices=("cpu", "gpu"), default=None)
    parser.add_argument("--cpu_workers", type=int, default=None)
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated GPU ids, e.g. 0,1,3",
    )
    parser.add_argument("--workers_per_gpu", type=int, default=None)
    parser.add_argument(
        "--schedule",
        type=str,
        choices=("task", "record"),
        default=None,
        help="task=whole-task affinity (default); record=LEGACY stripe (avoid in production)",
    )
    parser.add_argument(
        "--stagger_s",
        type=float,
        default=None,
        help="Sleep worker_id * stagger_s before each worker starts (JIT stagger)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--coach_seed", type=int, default=None, help="Filter coach leaves by seed")
    parser.add_argument("--min_behavior_steps", type=int, default=None)
    parser.add_argument("--behavior_switch_prob", type=float, default=None)
    parser.add_argument(
        "--scan_rollout",
        type=str,
        choices=("true", "false"),
        default=None,
        help="Device-side lax.scan per record (default true). false=legacy step loop.",
    )
    parser.add_argument(
        "--parallel_envs",
        type=int,
        default=None,
        help="vmap batch size B per scan call (pad last batch). Requires scan_rollout.",
    )
    parser.add_argument(
        "--winrate_adapt",
        type=str,
        choices=("true", "false"),
        default=None,
        help="Per-task ally mix reweight to steer episode win rate (enemy unchanged).",
    )
    parser.add_argument("--win_rate_min", type=float, default=None)
    parser.add_argument(
        "--win_rate_max",
        type=str,
        default=None,
        help="Upper WR bound, or 'null' / 'none' for no upper bound.",
    )
    parser.add_argument("--adapt_pilot_records", type=int, default=None)
    parser.add_argument("--adapt_max_iters", type=int, default=None)
    parser.add_argument(
        "--adapt_strength_max",
        type=float,
        default=None,
        help="Max strong-group mass (default 0.7; keeps weak policies).",
    )
    parser.add_argument(
        "--independent_ally_policies",
        type=str,
        choices=("true", "false"),
        default=None,
        help="If true, each ally samples/resamples its behavior independently.",
    )
    parser.add_argument(
        "--mid_episode_policy_switch",
        type=str,
        choices=("true", "false"),
        default=None,
        help="If false, only resample behavior on episode reset (no mid-trial switch).",
    )
    parser.add_argument(
        "--best_teacher_reference",
        type=str,
        choices=("true", "false"),
        default=None,
        help=(
            "If true, hard reference actions follow task.json coach_eval.best_policy; "
            "soft dist stays RL. Also steers winrate_adapt focus."
        ),
    )
    parser.add_argument(
        "--task_index",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of task indices",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    overrides = {
        "coach_root": args.coach_root,
        "task_packages_root": args.task_packages_root,
        "task_bank": args.task_bank,
        "ckpt_root": args.ckpt_root,
        "algorithm": args.algorithm,
        "output_root": args.output_root,
        "total_records": args.total_records,
        "total_timesteps": args.total_timesteps,
        "start_index": args.start_index,
        "device": args.device,
        "cpu_workers": args.cpu_workers,
        "workers_per_gpu": args.workers_per_gpu,
        "schedule": args.schedule,
        "stagger_s": args.stagger_s,
        "seed": args.seed,
        "coach_seed": args.coach_seed,
        "min_behavior_steps": args.min_behavior_steps,
        "behavior_switch_prob": args.behavior_switch_prob,
        "win_rate_min": args.win_rate_min,
        "adapt_pilot_records": args.adapt_pilot_records,
        "adapt_max_iters": args.adapt_max_iters,
        "adapt_strength_max": args.adapt_strength_max,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    if gpu_ids is not None:
        config["gpu_ids"] = gpu_ids
    if args.task_index is not None:
        config["task_index"] = args.task_index
    if args.scan_rollout is not None:
        config["scan_rollout"] = args.scan_rollout == "true"
    if args.parallel_envs is not None:
        config["parallel_envs"] = max(1, int(args.parallel_envs))
    if args.winrate_adapt is not None:
        config["winrate_adapt"] = args.winrate_adapt == "true"
    if args.independent_ally_policies is not None:
        config["independent_ally_policies"] = (
            args.independent_ally_policies == "true"
        )
    if args.mid_episode_policy_switch is not None:
        config["mid_episode_policy_switch"] = (
            args.mid_episode_policy_switch == "true"
        )
    if args.best_teacher_reference is not None:
        config["best_teacher_reference"] = args.best_teacher_reference == "true"
    if args.win_rate_max is not None:
        token = str(args.win_rate_max).strip().lower()
        if token in {"null", "none", ""}:
            config["win_rate_max"] = None
        else:
            config["win_rate_max"] = float(args.win_rate_max)

    run_from_config(config)
    print(f"[parallel_records] done → {config['output_root']}", flush=True)


if __name__ == "__main__":
    main()
