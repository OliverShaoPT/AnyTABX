"""Spawned worker entry for parallel record generation.

Keep this module free of JAX imports at top-level so CUDA_VISIBLE_DEVICES /
JAX_PLATFORMS can be set before JAX initializes in each process.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def worker_main(payload: dict[str, Any]) -> str:
    """Generate all pre-assigned records for one dedicated worker slot."""

    device = str(payload["device"])
    worker_id = int(payload.get("worker_id", -1))
    if device == "gpu":
        gpu_id = payload["gpu_id"]
        if gpu_id is None:
            raise ValueError("gpu worker requires gpu_id")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["JAX_PLATFORMS"] = str(payload.get("jax_platform", "cuda"))
    else:
        # Hide GPUs so JAX stays on CPU even if CUDA is installed.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_PLATFORMS"] = "cpu"

    # Import after device env is configured.
    from generate.behavior_mix import behavior_mix_from_config
    from generate.env_centric import generate_one_record
    from generate.task_package import discover_task_packages

    packages = {p.name: p for p in discover_task_packages(payload["packages_root"])}
    mix = behavior_mix_from_config(payload.get("behavior_mix"))
    paths: list[str] = []
    for item in payload["items"]:
        package = packages[item["package_name"]]
        for record_id in item["record_ids"]:
            path = generate_one_record(
                package,
                record_id=int(record_id),
                output_root=Path(payload["output_root"]),
                total_timesteps=int(payload["total_timesteps"]),
                seed=payload.get("seed"),
                min_behavior_steps=int(payload["min_behavior_steps"]),
                behavior_switch_prob=float(payload["behavior_switch_prob"]),
                behavior_mix=mix,
            )
            paths.append(str(path))
            print(
                f"[record_worker] worker={worker_id} device={device} "
                f"gpu={payload.get('gpu_id')} wrote {path}",
                flush=True,
            )
    return ",".join(paths)
