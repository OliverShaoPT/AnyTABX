"""Compile later tasks' coach-eval programs while the GPU worker runs.

Each process pins the same GPU as its worker and writes the JAX compilation
cache. It does not roll out episodes. The parent stays JAX-free; this module
imports JAX only inside ``prefetch_main``, after CUDA_VISIBLE_DEVICES is set.
"""

from __future__ import annotations

import traceback
from typing import Any


def claim_prefetch_index(
    *,
    worker_index: int,
    next_compile: int,
    n_items: int,
) -> tuple[int | None, int]:
    """Next task index strictly after the worker, or None when none remain.

    ``next_compile`` is the first index not yet claimed. If the worker has
    already reached it, jump to ``worker_index + 1`` so both sides do not
    start the same task.
    """

    target = max(int(next_compile), int(worker_index) + 1)
    if target >= int(n_items):
        return None, target
    return target, target + 1


def prefetch_main(payload: dict[str, Any]) -> str:
    """Compile coach-eval arms for tasks this GPU worker has not started."""

    from generate.record_worker import enable_compilation_cache, force_jax_gpu_env

    gpu_id = payload.get("gpu_id")
    if gpu_id is None:
        return "no-gpu"
    force_jax_gpu_env(gpu_id, jax_platform=str(payload.get("jax_platform", "cuda")))
    enable_compilation_cache()

    from generate.coach_eval import compile_coach_eval_programs
    from generate.task_package import discover_task_packages

    items = list(payload.get("items") or [])
    n_items = len(items)
    slot = int(payload.get("prefetch_slot", 0))
    cursor = payload["worker_index"]
    nxt = payload["next_compile"]
    lock = payload["prefetch_lock"]
    packages = {
        package.name: package
        for package in discover_task_packages(payload["packages_root"])
    }
    num_episodes = max(1, int(payload.get("coach_eval_episodes", 64) or 64))
    max_episode_steps = max(
        1, int(payload.get("coach_eval_max_episode_steps", 512) or 512)
    )
    parallel_envs = max(1, int(payload.get("coach_eval_parallel_envs", 32) or 32))
    seed_base = int(payload.get("seed") or 0)

    while True:
        with lock:
            claimed, updated = claim_prefetch_index(
                worker_index=int(cursor.value),
                next_compile=int(nxt.value),
                n_items=n_items,
            )
            nxt.value = int(updated)
        if claimed is None:
            return "done"
        item = items[claimed]
        package = packages[item["package_name"]]
        try:
            compiled = compile_coach_eval_programs(
                package,
                num_episodes=num_episodes,
                max_episode_steps=max_episode_steps,
                parallel_envs=parallel_envs,
                seed=seed_base + 997 * int(package.task_index),
            )
        except Exception as exc:
            print(
                f"[compile_prefetch] gpu={gpu_id} slot={slot} "
                f"package={package.name} failed {type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
            continue
        print(
            f"[compile_prefetch] gpu={gpu_id} slot={slot} "
            f"package={package.name} compiled={compiled or 'cached'}",
            flush=True,
        )
