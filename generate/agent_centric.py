"""Step 2: split env-centric records into ally agent-centric sequences."""

from __future__ import annotations

import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from generate.dump_schema import (
    ensure_dir,
    read_env_centric_record,
    save_meta,
    save_npy,
    split_flat_obs,
)


def flat_agent_dirname(record_dir: Path, ally: str) -> str:
    """Unique flat folder name: ``{task_dir}__{record-XXXXXX}__{ally}``."""

    record_dir = Path(record_dir)
    return f"{record_dir.parent.name}__{record_dir.name}__{ally}"


def discover_env_record_dirs(records_root: Path) -> list[Path]:
    """Find env-centric ``record-*`` dirs under a root (or a single record dir)."""

    records_root = Path(records_root)
    record_dirs = sorted(p for p in records_root.glob("**/record-*") if p.is_dir())
    if not record_dirs:
        if (records_root / "meta.json").exists():
            return [records_root]
        raise FileNotFoundError(f"No record-* directories under {records_root}")
    return [p for p in record_dirs if (p / "meta.json").exists()]


def split_one_record(
    record_dir: Path,
    *,
    output_root: Path | None = None,
) -> list[Path]:
    """Split one env-centric record into per-ally agent sequences.

    - ``output_root is None``: write under ``{record_dir}/agent_centric/{ally}/``
      (legacy nested layout).
    - ``output_root`` set: write flat ``{output_root}/{task}__{record}__{ally}/``.
    """

    record_dir = Path(record_dir)
    arrays, meta = read_env_centric_record(record_dir)
    ally_keys = list(meta["ally_keys"])
    n_units = int(meta["n_units"])
    max_n_zone = int(meta["max_n_zone"])
    t = int(arrays["actions_behavior"].shape[0])

    required = [
        "obs_flat",
        "actions_behavior",
        "actions_reference",
        "actions_reference_distribution",
        "reward_team",
        "reward_individual",
        "done",
        "truncation",
        "is_win",
        "reset",
        "episode_id",
        "behavior_policy_id",
    ]
    for name in required:
        if name not in arrays:
            raise KeyError(f"{record_dir} missing {name}.npy")

    nested_root = None if output_root is not None else ensure_dir(record_dir / "agent_centric")
    flat_root = ensure_dir(Path(output_root)) if output_root is not None else None
    written: list[Path] = []

    for ally_index, ally in enumerate(ally_keys):
        static_list = []
        dyn_list = []
        mask_list = []
        for step in range(t):
            static, dyn, mask = split_flat_obs(
                arrays["obs_flat"][step, ally_index],
                n_units=n_units,
                max_n_zone=max_n_zone,
            )
            static_list.append(static)
            dyn_list.append(dyn)
            mask_list.append(mask)

        if flat_root is not None:
            agent_dir = ensure_dir(flat_root / flat_agent_dirname(record_dir, ally))
        else:
            assert nested_root is not None
            agent_dir = ensure_dir(nested_root / ally)

        save_npy(agent_dir / "obs_static.npy", np.stack(static_list, axis=0))
        save_npy(agent_dir / "obs_dynamic.npy", np.stack(dyn_list, axis=0))
        save_npy(agent_dir / "obs_dynamic_mask.npy", np.stack(mask_list, axis=0))
        save_npy(agent_dir / "behavior_action.npy", arrays["actions_behavior"][:, ally_index])
        save_npy(agent_dir / "reference_action.npy", arrays["actions_reference"][:, ally_index])
        # Soft oracle π(a|o); HVAC label_action_distribution equivalent, shape (T, A).
        save_npy(
            agent_dir / "reference_action_distribution.npy",
            arrays["actions_reference_distribution"][:, ally_index],
        )
        save_npy(agent_dir / "reward_team.npy", arrays["reward_team"][:, ally_index])
        save_npy(agent_dir / "reward_individual.npy", arrays["reward_individual"][:, ally_index])
        save_npy(agent_dir / "done.npy", arrays["done"])
        save_npy(agent_dir / "truncation.npy", arrays["truncation"])
        save_npy(agent_dir / "is_win.npy", arrays["is_win"])
        save_npy(agent_dir / "reset.npy", arrays["reset"])
        save_npy(agent_dir / "episode_id.npy", arrays["episode_id"])
        save_npy(agent_dir / "behavior_policy_id.npy", arrays["behavior_policy_id"])

        agent_meta = {
            "schema": "agent_centric_v1",
            "source_record": str(record_dir.resolve()),
            "agent": ally,
            "agent_index": ally_index,
            "task_index": meta.get("task_index"),
            "task_id": meta.get("task_id"),
            "record_id": meta.get("record_id"),
            "total_timesteps": t,
            "n_units": n_units,
            "max_n_zone": max_n_zone,
            "obs_static_dim": int(static_list[0].shape[0]),
            "obs_dynamic_shape": list(dyn_list[0].shape),
            "shared_ally_policy": meta.get("shared_ally_policy", True),
            "layout": "flat" if flat_root is not None else "nested",
            "notes": {
                "reset": "1 marks the first step after env.reset (episode boundary jump)",
                "done": "1 marks the step whose transition ended the episode",
                "truncation": "1 if episode ended by hitting max_episode_steps",
                "is_win": "1 if team 0 (ally) won on this terminal step; else 0",
                "reference_action": "argmax of oracle RL policy (hard label)",
                "reference_action_distribution": (
                    "oracle RL softmax probs (T, action_dim); KL soft target "
                    "(HVAC label_action_distribution)"
                ),
                "alignment": "All arrays share the same time index t",
            },
        }
        save_meta(agent_dir / "meta.json", agent_meta)
        written.append(agent_dir)
    return written


def _split_one_record_job(payload: tuple[str, str | None]) -> tuple[str, int, str | None]:
    """Picklable worker entry: ``(record_dir, output_root|None) -> (record, n_agents, err)``."""

    record_dir_s, output_root_s = payload
    try:
        paths = split_one_record(
            Path(record_dir_s),
            output_root=Path(output_root_s) if output_root_s else None,
        )
        return record_dir_s, len(paths), None
    except Exception as exc:  # noqa: BLE001 - surface per-record failures to parent
        return record_dir_s, 0, f"{type(exc).__name__}: {exc}"


def split_records_root(
    records_root: Path,
    *,
    output_root: Path | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    workers: int = 1,
) -> list[Path]:
    """Convert all env records under ``records_root``.

    Returns the list of env ``record-*`` dirs that were processed (post-shuffle order).
    """

    record_dirs = discover_env_record_dirs(records_root)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(record_dirs)

    out_s = str(Path(output_root).resolve()) if output_root is not None else None
    n_workers = max(1, int(workers))
    jobs = [(str(p), out_s) for p in record_dirs]
    errors: list[str] = []

    if n_workers == 1:
        for job in jobs:
            path_s, n_agents, err = _split_one_record_job(job)
            if err:
                errors.append(f"{path_s}: {err}")
                print(f"[agent_centric] ERROR {path_s}: {err}", flush=True)
            else:
                print(f"[agent_centric] {path_s} -> {n_agents} agents", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_split_one_record_job, job): job[0] for job in jobs}
            for fut in as_completed(futures):
                path_s, n_agents, err = fut.result()
                if err:
                    errors.append(f"{path_s}: {err}")
                    print(f"[agent_centric] ERROR {path_s}: {err}", flush=True)
                else:
                    print(f"[agent_centric] {path_s} -> {n_agents} agents", flush=True)

    if errors:
        raise RuntimeError(f"agent_centric failed on {len(errors)} record(s); first: {errors[0]}")
    return record_dirs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Split env-centric records to agent-centric")
    parser.add_argument(
        "--record_dir",
        type=str,
        default=None,
        help="Single record-XXXXXX directory",
    )
    parser.add_argument(
        "--records_root",
        type=str,
        default=None,
        help="Root containing task_*/record-* trees (or a single record dir)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help=(
            "Write all agent sequences as flat dirs under this root "
            "(recommended). Default: nested {record}/agent_centric/{ally}/"
        ),
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle env-record processing order (output order is unordered flat dirs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for --shuffle",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 1))),
        help="Process-pool workers for parallel conversion (default: min(8, cpu_count))",
    )
    args = parser.parse_args(argv)
    output_root = Path(args.output_root) if args.output_root else None

    if args.record_dir:
        paths = split_one_record(Path(args.record_dir), output_root=output_root)
        print(f"[agent_centric] {args.record_dir} -> {len(paths)} agents", flush=True)
    elif args.records_root:
        split_records_root(
            Path(args.records_root),
            output_root=output_root,
            shuffle=bool(args.shuffle),
            seed=args.seed,
            workers=int(args.workers),
        )
    else:
        raise SystemExit("Provide --record_dir or --records_root")


if __name__ == "__main__":
    main()
