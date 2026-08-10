#!/usr/bin/env python3
"""Compare ally ``oracle_pure`` vs ``heuristic_advanced`` win rates on coach packages.

Enemy stays the package heuristic (same ``TABXEnemyHeuristicWrapper`` path as
record generation). Multi-GPU: one process per worker slot with
``CUDA_VISIBLE_DEVICES`` set before JAX import.

Usage:
  python tools/compare_oracle_vs_advanced.py \\
    --coach_root /path/to/coach_root \\
    --gpu_ids 4,5,6,7 \\
    --workers_per_gpu 1 \\
    --num_episodes 64 \\
    --parallel_envs 32 \\
    --write_task_json \\
    --json_out /tmp/oracle_vs_advanced.json

``--parallel_envs``: vmap batch size for episode rollouts inside each package
(default 32). Multi-GPU ``--gpu_ids`` still parallelizes across packages.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ``python tools/...`` does not put the repo root on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Bump when changing eval logic; printed at startup so you can verify the remote copy.
SCRIPT_VERSION = "2026-08-10-coach-eval-lib"


def _parse_gpu_ids(value: str | None) -> list[str]:
    if value is None or not str(value).strip():
        return []
    ids = [part.strip() for part in str(value).split(",") if part.strip()]
    if not ids:
        raise ValueError("gpu_ids is empty")
    if len(ids) != len(set(ids)):
        raise ValueError("gpu_ids must not contain duplicates")
    return ids


def worker_main(payload: dict[str, Any]) -> str:
    """Evaluate assigned packages; write JSON to ``result_path`` and return that path."""

    result_path = Path(payload["result_path"])
    worker_id = int(payload.get("worker_id", -1))
    print(
        f"[compare_oracle] worker={worker_id} start version={SCRIPT_VERSION} "
        f"gpu={payload.get('gpu_id')}",
        flush=True,
    )

    device = str(payload.get("device", "gpu")).lower()
    if device == "gpu":
        from generate.record_worker import force_jax_gpu_env

        force_jax_gpu_env(
            payload["gpu_id"],
            jax_platform=str(payload.get("jax_platform", "cuda")),
        )
    else:
        from generate.record_worker import force_jax_cpu_env

        force_jax_cpu_env()

    from generate.coach_eval import (
        DEFAULT_TIE_EPS,
        evaluate_package_oracle_vs_advanced,
        write_coach_eval_to_task_json,
        coach_eval_for_task_metadata,
    )
    from generate.task_package import discover_task_packages

    coach_root = Path(payload["coach_root"])
    assigned = [str(Path(p).resolve()) for p in payload["package_paths"]]
    want = set(assigned)
    by_path = {
        str(p.path.resolve()): p
        for p in discover_task_packages(coach_root, seed=payload.get("coach_seed"))
        if str(p.path.resolve()) in want
    }
    ordered = [by_path[p] for p in assigned if p in by_path]
    missing = [p for p in assigned if p not in by_path]
    if missing:
        print(
            f"[compare_oracle] worker={worker_id} "
            f"missing {len(missing)} package(s), e.g. {missing[0]}",
            flush=True,
        )

    write_task = bool(payload.get("write_task_json", False))
    tie_eps = float(payload.get("tie_eps", DEFAULT_TIE_EPS))
    results: list[dict[str, Any]] = []
    for i, package in enumerate(ordered):
        print(
            f"[compare_oracle] worker={worker_id} gpu={payload.get('gpu_id')} "
            f"({i + 1}/{len(ordered)}) {package.name}",
            flush=True,
        )
        try:
            row = evaluate_package_oracle_vs_advanced(
                package,
                num_episodes=int(payload["num_episodes"]),
                seed=int(payload["seed"]) + int(package.task_index) * 997,
                max_episode_steps=int(payload["max_episode_steps"]),
                tie_eps=tie_eps,
                parallel_envs=payload.get("parallel_envs"),
            )
            if write_task:
                write_coach_eval_to_task_json(
                    package.task_json, coach_eval_for_task_metadata(row)
                )
            results.append(row)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[compare_oracle] worker={worker_id} ERROR {package.name}: {exc}\n"
                f"{traceback.format_exc()}",
                flush=True,
            )
            results.append(
                {
                    "package_name": package.name,
                    "task_id": package.task_id,
                    "task_index": int(package.task_index),
                    "package_path": str(package.path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(results) + "\n", encoding="utf-8")
    print(
        f"[compare_oracle] worker={worker_id} wrote {result_path} "
        f"n={len(results)}",
        flush=True,
    )
    return str(result_path)


@dataclass(frozen=True)
class WorkerSlot:
    worker_id: int
    device: str
    gpu_id: str | None


def _build_slots(
    *,
    device: str,
    gpu_ids: list[str],
    workers_per_gpu: int,
    cpu_workers: int,
) -> list[WorkerSlot]:
    if device == "cpu":
        n = max(1, int(cpu_workers))
        return [WorkerSlot(i, "cpu", None) for i in range(n)]
    if not gpu_ids:
        raise ValueError("device=gpu requires --gpu_ids")
    wpg = max(1, int(workers_per_gpu))
    slots: list[WorkerSlot] = []
    wid = 0
    for gid in gpu_ids:
        for _ in range(wpg):
            slots.append(WorkerSlot(wid, "gpu", gid))
            wid += 1
    return slots


def _print_table(rows: list[dict[str, Any]]) -> None:
    ok = [r for r in rows if "error" not in r]
    err = [r for r in rows if "error" in r]
    print(
        f"{'task':<48} {'oracle':>8} {'advanced':>8} {'delta':>8} "
        f"{'best':>18} {'o_w/d/l':>12} {'a_w/d/l':>12}",
        flush=True,
    )
    print("-" * 130, flush=True)
    for r in sorted(ok, key=lambda x: (int(x.get("task_index", 0)), x["package_name"])):
        o = r["oracle_pure"]
        a = r["heuristic_advanced"]
        print(
            f"{r['package_name']:<48} "
            f"{o['win_rate']:>7.1%} "
            f"{a['win_rate']:>7.1%} "
            f"{r['delta_oracle_minus_advanced']:>+7.1%} "
            f"{str(r.get('best_policy', '')):>18} "
            f"{o['win']}/{o['draw']}/{o['loss']:>4} "
            f"{a['win']}/{a['draw']}/{a['loss']:>4}",
            flush=True,
        )
    if ok:
        mean_o = sum(r["oracle_pure"]["win_rate"] for r in ok) / len(ok)
        mean_a = sum(r["heuristic_advanced"]["win_rate"] for r in ok) / len(ok)
        oracle_better = sum(
            1 for r in ok if r.get("best_policy") == "oracle_pure"
            and r["delta_oracle_minus_advanced"] > 1e-9
        )
        advanced_better = sum(
            1 for r in ok if r.get("best_policy") == "heuristic_advanced"
        )
        print("-" * 130, flush=True)
        print(
            f"[compare_oracle] tasks={len(ok)} "
            f"mean oracle={mean_o:.1%} mean advanced={mean_a:.1%} "
            f"delta={mean_o - mean_a:+.1%} "
            f"(best_oracle={oracle_better} best_advanced={advanced_better})",
            flush=True,
        )
    if err:
        print(f"[compare_oracle] {len(err)} package(s) failed:", flush=True)
        for r in err:
            print(f"  {r['package_name']}: {r['error']}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare oracle_pure vs heuristic_advanced win rates on coach_root."
    )
    parser.add_argument("--coach_root", type=Path, required=True)
    parser.add_argument(
        "--coach_seed",
        type=int,
        default=None,
        help="Optional: only packages whose seed matches (discover filter).",
    )
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated physical GPU ids, e.g. 4,5,6,7",
    )
    parser.add_argument("--workers_per_gpu", type=int, default=1)
    parser.add_argument("--cpu_workers", type=int, default=1)
    parser.add_argument("--jax_platform", type=str, default="cuda")
    parser.add_argument("--num_episodes", type=int, default=64)
    parser.add_argument("--max_episode_steps", type=int, default=512)
    parser.add_argument(
        "--parallel_envs",
        type=int,
        default=32,
        help="vmap batch size for parallel episode rollouts (default 32).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tie_eps", type=float, default=0.02)
    parser.add_argument(
        "--write_task_json",
        action="store_true",
        help="Write coach_eval into each package task.json metadata.",
    )
    parser.add_argument(
        "--task_index",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of task indices.",
    )
    parser.add_argument("--json_out", type=Path, default=None)
    args = parser.parse_args(argv)

    coach_root = Path(args.coach_root)
    if not coach_root.is_dir():
        raise FileNotFoundError(f"coach_root not found: {coach_root}")

    from generate.task_package import discover_task_packages

    packages = discover_task_packages(coach_root, seed=args.coach_seed)
    if args.task_index is not None:
        want = {int(i) for i in args.task_index}
        packages = [p for p in packages if int(p.task_index) in want]
    if not packages:
        raise FileNotFoundError(f"No packages to evaluate under {coach_root}")

    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    slots = _build_slots(
        device=args.device,
        gpu_ids=gpu_ids,
        workers_per_gpu=args.workers_per_gpu,
        cpu_workers=args.cpu_workers,
    )

    assignments: list[list[str]] = [[] for _ in slots]
    for i, package in enumerate(packages):
        assignments[i % len(slots)].append(str(package.path.resolve()))

    import tempfile

    scratch = Path(tempfile.mkdtemp(prefix="compare_oracle_"))
    jobs = []
    for slot, paths in zip(slots, assignments):
        if not paths:
            continue
        jobs.append(
            {
                "coach_root": str(coach_root.resolve()),
                "coach_seed": args.coach_seed,
                "package_paths": paths,
                "device": slot.device,
                "gpu_id": slot.gpu_id,
                "worker_id": slot.worker_id,
                "jax_platform": args.jax_platform,
                "num_episodes": int(args.num_episodes),
                "max_episode_steps": int(args.max_episode_steps),
                "parallel_envs": int(args.parallel_envs),
                "seed": int(args.seed),
                "tie_eps": float(args.tie_eps),
                "write_task_json": bool(args.write_task_json),
                "result_path": str(scratch / f"worker_{slot.worker_id:02d}.json"),
            }
        )

    print(
        f"[compare_oracle] version={SCRIPT_VERSION} file={Path(__file__).resolve()}",
        flush=True,
    )
    print(
        f"[compare_oracle] coach_root={coach_root} packages={len(packages)} "
        f"device={args.device} slots={len(jobs)} "
        f"episodes/policy={args.num_episodes} parallel_envs={args.parallel_envs} "
        f"max_steps={args.max_episode_steps}",
        flush=True,
    )
    for job in jobs:
        print(
            f"  worker={job['worker_id']:02d} device={job['device']} "
            f"gpu={job['gpu_id']} n_packages={len(job['package_paths'])}",
            flush=True,
        )

    ctx = mp.get_context("spawn")
    rows: list[dict[str, Any]] = []
    with ctx.Pool(processes=len(jobs)) as pool:
        result_paths = pool.map(worker_main, jobs)
    for path in result_paths:
        part = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.extend(part)

    _print_table(rows)

    if args.json_out is not None:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "script_version": SCRIPT_VERSION,
            "coach_root": str(coach_root.resolve()),
            "num_episodes": int(args.num_episodes),
            "max_episode_steps": int(args.max_episode_steps),
            "parallel_envs": int(args.parallel_envs),
            "seed": int(args.seed),
            "tie_eps": float(args.tie_eps),
            "device": args.device,
            "gpu_ids": gpu_ids,
            "results": sorted(
                rows,
                key=lambda x: (int(x.get("task_index", 10**9)), x.get("package_name", "")),
            ),
        }
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[compare_oracle] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
