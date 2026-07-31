"""Spawned worker entry for parallel record generation.

Keep this module free of JAX imports at top-level so CUDA_VISIBLE_DEVICES /
JAX_PLATFORMS can be set before JAX initializes in each process.
"""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Any


def force_jax_cpu_env() -> None:
    """Debug-only: force host/CPU before any JAX import.

    Not used for mass record production (prefer ``force_jax_gpu_env``). With
    ``jax-cuda12-plugin`` installed, import still calls
    ``jax_plugins.xla_cuda12.initialize()`` → ``cuInit``; skip the CUDA
    constraints check so quiet CPU debug runs do not touch the driver.
    """

    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Avoid cuInit / version probe inside jax_plugins.xla_cuda12 (see jax#35105).
    os.environ["JAX_SKIP_CUDA_CONSTRAINTS_CHECK"] = "1"
    # Cap host BLAS threads so many workers do not oversubscribe cores.
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(key, "1")
    xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_cpu_multi_thread_eigen=false" not in xla_flags:
        os.environ["XLA_FLAGS"] = (
            (xla_flags + " --xla_cpu_multi_thread_eigen=false").strip()
        )


def force_jax_gpu_env(gpu_id: int | str, *, jax_platform: str = "cuda") -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["JAX_PLATFORMS"] = str(jax_platform)
    os.environ.pop("JAX_PLATFORM_NAME", None)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _report(payload: dict[str, Any], event: dict[str, Any]) -> None:
    queue = payload.get("progress_queue")
    if queue is None:
        return
    queue.put(event)


def worker_main(payload: dict[str, Any]) -> str:
    """Generate all pre-assigned records for one dedicated worker slot."""

    device = str(payload["device"])
    worker_id = int(payload.get("worker_id", -1))
    stagger_s = float(payload.get("stagger_s", 0.0) or 0.0)
    if stagger_s > 0 and worker_id > 0:
        time.sleep(stagger_s * worker_id)

    if device == "gpu":
        gpu_id = payload["gpu_id"]
        if gpu_id is None:
            raise ValueError("gpu worker requires gpu_id")
        force_jax_gpu_env(
            gpu_id, jax_platform=str(payload.get("jax_platform", "cuda"))
        )
    else:
        force_jax_cpu_env()

    # Import after device env is configured.
    from generate.behavior_mix import behavior_mix_from_config
    from generate.env_centric import (
        build_record_gen_context,
        generate_one_record,
        warmup_record_gen_context,
    )
    from generate.task_package import discover_task_packages

    packages = {p.name: p for p in discover_task_packages(payload["packages_root"])}
    mix = behavior_mix_from_config(payload.get("behavior_mix"))
    scan_rollout = bool(payload.get("scan_rollout", True))
    total_timesteps = int(payload["total_timesteps"])
    min_behavior_steps = int(payload["min_behavior_steps"])
    behavior_switch_prob = float(payload["behavior_switch_prob"])
    # Reuse env+oracle (+ optional scan fn) per package inside this process.
    contexts: dict[str, Any] = {}
    rollout_fns: dict[str, Any] = {}
    paths: list[str] = []
    try:
        for item in payload["items"]:
            package = packages[item["package_name"]]
            ctx = contexts.get(package.name)
            if ctx is None:
                t_setup = time.perf_counter()
                ctx = build_record_gen_context(package)
                setup_s = time.perf_counter() - t_setup
                # Warm *every* behavior in the mix so generate-time switches
                # do not re-enter XLA compile (as much as JAX allows).
                compile_s = warmup_record_gen_context(
                    ctx,
                    behavior_mix=mix,
                    steps=2,
                    seed=int(payload.get("seed") or 0) + worker_id,
                )
                if scan_rollout:
                    from generate.scan_rollout import (
                        build_scan_rollout_fn,
                        warmup_scan_rollout,
                    )

                    rollout_fn = build_scan_rollout_fn(
                        ctx,
                        mix=mix,
                        min_behavior_steps=min_behavior_steps,
                        behavior_switch_prob=behavior_switch_prob,
                        total_timesteps=total_timesteps,
                    )
                    compile_s += warmup_scan_rollout(
                        rollout_fn,
                        seed=int(payload.get("seed") or 0) + worker_id,
                    )
                    rollout_fns[package.name] = rollout_fn
                contexts[package.name] = ctx
                _report(
                    payload,
                    {
                        "event": "compile_done",
                        "worker_id": worker_id,
                        "device": device,
                        "gpu_id": payload.get("gpu_id"),
                        "package_name": item["package_name"],
                        "setup_s": round(setup_s, 3),
                        "compile_s": round(compile_s, 3),
                    },
                )
            for record_id in item["record_ids"]:
                t_gen = time.perf_counter()
                path = generate_one_record(
                    package,
                    record_id=int(record_id),
                    output_root=Path(payload["output_root"]),
                    total_timesteps=total_timesteps,
                    seed=payload.get("seed"),
                    min_behavior_steps=min_behavior_steps,
                    behavior_switch_prob=behavior_switch_prob,
                    behavior_mix=mix,
                    ctx=ctx,
                    scan_rollout=scan_rollout,
                    rollout_fn=rollout_fns.get(package.name),
                )
                generate_s = time.perf_counter() - t_gen
                paths.append(str(path))
                _report(
                    payload,
                    {
                        "event": "record_done",
                        "worker_id": worker_id,
                        "device": device,
                        "gpu_id": payload.get("gpu_id"),
                        "package_name": item["package_name"],
                        "record_id": int(record_id),
                        "path": str(path),
                        "generate_s": round(generate_s, 3),
                    },
                )
                # When a progress queue is attached, the parent draws the bar.
                if payload.get("progress_queue") is None:
                    print(
                        f"[record_worker] worker={worker_id} device={device} "
                        f"gpu={payload.get('gpu_id')} wrote {path} "
                        f"generate_s={generate_s:.1f}",
                        flush=True,
                    )
    except Exception as exc:
        _report(
            payload,
            {
                "event": "worker_error",
                "worker_id": worker_id,
                "device": device,
                "gpu_id": payload.get("gpu_id"),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    return ",".join(paths)
