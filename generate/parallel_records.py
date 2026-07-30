"""Orchestrate even, parallel env-centric record generation (CPU or GPU)."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from generate.dump_schema import ensure_dir
from generate.record_worker import worker_main
from generate.task_package import discover_task_packages, pack_task_packages


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


def even_records_per_task(
    total_records: int,
    n_tasks: int,
    *,
    allow_uneven: bool = False,
) -> list[int]:
    """Return per-task record counts that sum to total_records."""

    if total_records <= 0:
        raise ValueError("total_records must be positive")
    if n_tasks <= 0:
        raise ValueError("n_tasks must be positive")
    base, rem = divmod(total_records, n_tasks)
    if rem and not allow_uneven:
        raise ValueError(
            f"total_records={total_records} is not divisible by n_tasks={n_tasks}. "
            "Adjust total_records, or set allow_uneven: true."
        )
    return [base + (1 if i < rem else 0) for i in range(n_tasks)]


def stripe_record_ids(
    start_index: int,
    count: int,
    n_workers: int,
    worker_id: int,
) -> list[int]:
    """Round-robin ids: worker0 → start, start+n, …; worker1 → start+1, …"""

    if count <= 0:
        return []
    return [
        record_id
        for offset, record_id in enumerate(range(start_index, start_index + count))
        if offset % n_workers == worker_id
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
) -> list[dict[str, Any]]:
    """One job per worker slot (keeps GPU occupancy at workers_per_gpu)."""

    n_workers = len(slots)
    jobs: list[dict[str, Any]] = []
    for slot in slots:
        items: list[dict[str, Any]] = []
        for package, count in zip(packages, counts):
            record_ids = stripe_record_ids(start_index, count, n_workers, slot.worker_id)
            if record_ids:
                items.append(
                    {
                        "package_name": package.name,
                        "record_ids": record_ids,
                    }
                )
        if not items:
            continue
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
            }
        )
    return jobs


def run_from_config(config: dict[str, Any]) -> None:
    packages_root = Path(config["task_packages_root"])
    output_root = Path(config["output_root"])
    task_bank = config.get("task_bank")
    ckpt_root = config.get("ckpt_root")

    if task_bank and ckpt_root:
        packages_root.mkdir(parents=True, exist_ok=True)
        if not any(packages_root.glob("task_*")):
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

    packages = discover_task_packages(packages_root)
    task_filter = config.get("task_index")
    if task_filter is not None:
        allowed = set(int(x) for x in task_filter)
        packages = [p for p in packages if p.task_index in allowed]
    if not packages:
        raise RuntimeError("No task packages to generate")

    total_records = int(config["total_records"])
    counts = even_records_per_task(
        total_records,
        len(packages),
        allow_uneven=bool(config.get("allow_uneven", False)),
    )
    start_index = int(config.get("start_index", 0))
    slots = build_worker_slots(
        device=str(config.get("device", "cpu")),
        cpu_workers=int(config.get("cpu_workers", 8)),
        gpu_ids=[int(x) for x in (config.get("gpu_ids") or [])],
        workers_per_gpu=int(config.get("workers_per_gpu", 1)),
    )

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
    )

    print(
        f"[parallel_records] tasks={len(packages)} total_records={total_records} "
        f"per_task={counts[0] if len(set(counts)) == 1 else counts} "
        f"workers={len(slots)} device={slots[0].device} jobs={len(jobs)}",
        flush=True,
    )
    for slot in slots:
        owned = [
            rid
            for count in counts
            for rid in stripe_record_ids(start_index, count, len(slots), slot.worker_id)
        ]
        print(
            f"  worker={slot.worker_id:02d} device={slot.device} gpu={slot.gpu_id} "
            f"n_records={len(owned)} ids_sample={owned[:6]}{'...' if len(owned) > 6 else ''}",
            flush=True,
        )

    if len(slots) <= 1:
        for job in jobs:
            worker_main(job)
        return

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(slots)) as pool:
        for _ in pool.imap_unordered(worker_main, jobs):
            pass


def _parse_gpu_ids(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip() != ""]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evenly generate env-centric records in parallel (CPU/GPU)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="generate/configs/record_gen.yaml",
        help="YAML/JSON config (behavior mix + defaults)",
    )
    parser.add_argument("--task_packages_root", type=str, default=None)
    parser.add_argument("--task_bank", type=str, default=None, help="Task bank JSON (taskfile)")
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min_behavior_steps", type=int, default=None)
    parser.add_argument("--behavior_switch_prob", type=float, default=None)
    parser.add_argument("--allow_uneven", action="store_true", default=None)
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
        "seed": args.seed,
        "min_behavior_steps": args.min_behavior_steps,
        "behavior_switch_prob": args.behavior_switch_prob,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    if gpu_ids is not None:
        config["gpu_ids"] = gpu_ids
    if args.allow_uneven:
        config["allow_uneven"] = True
    if args.task_index is not None:
        config["task_index"] = args.task_index

    run_from_config(config)
    print(f"[parallel_records] done → {config['output_root']}", flush=True)


if __name__ == "__main__":
    main()
