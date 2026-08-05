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
    --json_out /tmp/oracle_vs_advanced.json
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
SCRIPT_VERSION = "2026-08-05-iswin-v2"


ORACLE_SPEC = {
    "policy_id": 7,
    "name": "oracle_pure",
    "kind": "oracle",
    "weight": 1.0,
    "epsilon": 0.0,
}
ADVANCED_SPEC = {
    "policy_id": 4,
    "name": "heuristic_advanced",
    "kind": "heuristic",
    "weight": 1.0,
    "heuristic": "advanced",
}


def _parse_gpu_ids(value: str | None) -> list[str]:
    if value is None or not str(value).strip():
        return []
    ids = [part.strip() for part in str(value).split(",") if part.strip()]
    if not ids:
        raise ValueError("gpu_ids is empty")
    if len(ids) != len(set(ids)):
        raise ValueError("gpu_ids must not contain duplicates")
    return ids


def _wr(counts: dict[str, int]) -> float:
    wins = int(counts.get("win", 0))
    losses = int(counts.get("loss", 0))
    decisive = wins + losses
    if decisive <= 0:
        return 0.0
    return wins / decisive


def _run_policy_episodes(
    *,
    ctx,
    spec_dict: dict[str, Any],
    num_episodes: int,
    seed: int,
    max_episode_steps: int,
) -> dict[str, int]:
    """Roll out episodes; outcome uses env ``info['is_win']`` only (no unit.health)."""

    import jax
    from generate.behavior_mix import BehaviorSpec
    from generate.policies import SharedAllyPolicy
    from src.tabx.heuristic_policy import LastVisibleTarget

    spec = BehaviorSpec(
        policy_id=int(spec_dict["policy_id"]),
        name=str(spec_dict["name"]),
        kind=str(spec_dict["kind"]),
        weight=float(spec_dict.get("weight", 1.0)),
        heuristic=spec_dict.get("heuristic"),
        epsilon=None
        if spec_dict.get("epsilon") is None
        else float(spec_dict["epsilon"]),
    )
    shared = SharedAllyPolicy.from_spec(
        spec,
        oracle=ctx.oracle,
        ally_keys=ctx.ally_keys,
        n_agents=ctx.n_units_total,
        max_n_zone=ctx.env.max_n_zone,
    )
    counts = {"win": 0, "draw": 0, "loss": 0, "episodes": 0, "truncated": 0}
    key = jax.random.key(int(seed) % (2**32))
    env = ctx.env
    env_params = ctx.env_params
    ally_keys = ctx.ally_keys

    for _ep in range(int(num_episodes)):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key, env_params)
        shared.last_visible = {agent: LastVisibleTarget() for agent in ally_keys}
        outcome = "loss"
        truncated = False
        for _t in range(int(max_episode_steps)):
            avail = env.get_avail_actions(state)
            key, bkey, skey = jax.random.split(key, 3)
            behavior = shared.act(
                key=bkey,
                obs_by_agent=obs,
                avail_by_agent=avail,
                ally_keys=ally_keys,
                physics_params=state["physics_params"],
            )
            obs, state, _rewards, dones, info = env.step(skey, state, dict(behavior))
            if bool(dones["__all__"]):
                is_win = int(info["is_win"].reshape(-1)[0])
                truncated = bool(info["truncation"].reshape(-1)[0])
                outcome = "win" if is_win else "loss"
                break
        else:
            truncated = True
            outcome = "loss"

        counts[outcome] += 1
        counts["episodes"] += 1
        if truncated:
            counts["truncated"] += 1
    return counts


def _eval_package(
    package,
    *,
    num_episodes: int,
    seed: int,
    max_episode_steps: int,
) -> dict[str, Any]:
    from generate.env_centric import build_record_gen_context

    ctx = build_record_gen_context(package)
    oracle_counts = _run_policy_episodes(
        ctx=ctx,
        spec_dict=ORACLE_SPEC,
        num_episodes=num_episodes,
        seed=seed,
        max_episode_steps=max_episode_steps,
    )
    advanced_counts = _run_policy_episodes(
        ctx=ctx,
        spec_dict=ADVANCED_SPEC,
        num_episodes=num_episodes,
        seed=seed + 10_000_003,
        max_episode_steps=max_episode_steps,
    )
    o_wr = _wr(oracle_counts)
    a_wr = _wr(advanced_counts)
    return {
        "package_name": package.name,
        "task_id": package.task_id,
        "task_index": int(package.task_index),
        "package_path": str(package.path),
        "enemy_heuristic": (ctx.manifest or {}).get("heuristic"),
        "oracle_pure": {
            **oracle_counts,
            "win_rate": o_wr,
        },
        "heuristic_advanced": {
            **advanced_counts,
            "win_rate": a_wr,
        },
        "delta_oracle_minus_advanced": o_wr - a_wr,
    }


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

    results: list[dict[str, Any]] = []
    for i, package in enumerate(ordered):
        print(
            f"[compare_oracle] worker={worker_id} gpu={payload.get('gpu_id')} "
            f"({i + 1}/{len(ordered)}) {package.name}",
            flush=True,
        )
        try:
            results.append(
                _eval_package(
                    package,
                    num_episodes=int(payload["num_episodes"]),
                    seed=int(payload["seed"]) + int(package.task_index) * 997,
                    max_episode_steps=int(payload["max_episode_steps"]),
                )
            )
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
        f"{'o_w/d/l':>12} {'a_w/d/l':>12}",
        flush=True,
    )
    print("-" * 110, flush=True)
    for r in sorted(ok, key=lambda x: (int(x.get("task_index", 0)), x["package_name"])):
        o = r["oracle_pure"]
        a = r["heuristic_advanced"]
        print(
            f"{r['package_name']:<48} "
            f"{o['win_rate']:>7.1%} "
            f"{a['win_rate']:>7.1%} "
            f"{r['delta_oracle_minus_advanced']:>+7.1%} "
            f"{o['win']}/{o['draw']}/{o['loss']:>4} "
            f"{a['win']}/{a['draw']}/{a['loss']:>4}",
            flush=True,
        )
    if ok:
        mean_o = sum(r["oracle_pure"]["win_rate"] for r in ok) / len(ok)
        mean_a = sum(r["heuristic_advanced"]["win_rate"] for r in ok) / len(ok)
        oracle_better = sum(
            1 for r in ok if r["delta_oracle_minus_advanced"] > 1e-9
        )
        advanced_better = sum(
            1 for r in ok if r["delta_oracle_minus_advanced"] < -1e-9
        )
        print("-" * 110, flush=True)
        print(
            f"[compare_oracle] tasks={len(ok)} "
            f"mean oracle={mean_o:.1%} mean advanced={mean_a:.1%} "
            f"delta={mean_o - mean_a:+.1%} "
            f"(oracle_better={oracle_better} advanced_better={advanced_better})",
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
    parser.add_argument("--seed", type=int, default=0)
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

    # Discover packages in the parent (no JAX needed).
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
                "seed": int(args.seed),
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
        f"episodes/policy={args.num_episodes} max_steps={args.max_episode_steps}",
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
            "seed": int(args.seed),
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
