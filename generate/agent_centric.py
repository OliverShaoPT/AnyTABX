"""Step 2: split env-centric records into ally agent-centric sequences."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from generate.dump_schema import (
    ensure_dir,
    read_env_centric_record,
    save_meta,
    save_npy,
    split_flat_obs,
)


def split_one_record(record_dir: Path) -> list[Path]:
    arrays, meta = read_env_centric_record(record_dir)
    ally_keys = list(meta["ally_keys"])
    n_units = int(meta["n_units"])
    max_n_zone = int(meta["max_n_zone"])
    t = int(arrays["actions_behavior"].shape[0])

    required = [
        "obs_flat",
        "actions_behavior",
        "actions_reference",
        "reward_team",
        "reward_individual",
        "done",
        "reset",
        "episode_id",
        "behavior_policy_id",
    ]
    for name in required:
        if name not in arrays:
            raise KeyError(f"{record_dir} missing {name}.npy")

    out_root = ensure_dir(record_dir / "agent_centric")
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

        agent_dir = ensure_dir(out_root / ally)
        save_npy(agent_dir / "obs_static.npy", np.stack(static_list, axis=0))
        save_npy(agent_dir / "obs_dynamic.npy", np.stack(dyn_list, axis=0))
        save_npy(agent_dir / "obs_dynamic_mask.npy", np.stack(mask_list, axis=0))
        save_npy(agent_dir / "behavior_action.npy", arrays["actions_behavior"][:, ally_index])
        save_npy(agent_dir / "reference_action.npy", arrays["actions_reference"][:, ally_index])
        save_npy(agent_dir / "reward_team.npy", arrays["reward_team"][:, ally_index])
        save_npy(agent_dir / "reward_individual.npy", arrays["reward_individual"][:, ally_index])
        save_npy(agent_dir / "done.npy", arrays["done"])
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
            "notes": {
                "reset": "1 marks the first step after env.reset (episode boundary jump)",
                "done": "1 marks the step whose transition ended the episode",
                "alignment": "All arrays share the same time index t",
            },
        }
        save_meta(agent_dir / "meta.json", agent_meta)
        written.append(agent_dir)
    return written


def split_records_root(records_root: Path) -> None:
    records_root = Path(records_root)
    record_dirs = sorted(records_root.glob("**/record-*"))
    if not record_dirs:
        # Allow passing a single record dir.
        if (records_root / "meta.json").exists():
            record_dirs = [records_root]
        else:
            raise FileNotFoundError(f"No record-* directories under {records_root}")
    for record_dir in record_dirs:
        if not (record_dir / "meta.json").exists():
            continue
        paths = split_one_record(record_dir)
        print(f"[agent_centric] {record_dir} -> {len(paths)} agents", flush=True)


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
        help="Root containing task_*/record-* trees",
    )
    args = parser.parse_args(argv)
    if args.record_dir:
        split_one_record(Path(args.record_dir))
    elif args.records_root:
        split_records_root(Path(args.records_root))
    else:
        raise SystemExit("Provide --record_dir or --records_root")


if __name__ == "__main__":
    main()
