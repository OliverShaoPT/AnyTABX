"""Step 2: split env-centric records into ally agent-centric sequences."""

from __future__ import annotations

import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from generate.behavior_mix import (
    POLICY_TAG_MASK,
    behavior_mix_from_config,
    policy_id_mapping_json,
    policy_id_to_tag,
    policy_tag_legend_json,
)
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


def _per_agent_series(arr: np.ndarray, ally_index: int, t: int) -> np.ndarray:
    """Accept ``(T,)`` or ``(T, n_ally)`` → ``(T,)`` for one ally."""

    arr = np.asarray(arr)
    if arr.ndim == 1:
        if arr.shape[0] != t:
            raise ValueError(f"expected length {t}, got {arr.shape}")
        return arr
    if arr.ndim == 2:
        if arr.shape[0] != t:
            raise ValueError(f"expected T={t}, got {arr.shape}")
        return arr[:, ally_index]
    raise ValueError(f"unsupported policy array shape {arr.shape}")


def resolve_policy_tag(
    arrays: dict[str, np.ndarray],
    meta: dict,
    *,
    ally_index: int,
    t: int,
) -> np.ndarray:
    """Return ``(T,)`` quality tags for one ally (prefer dumped ``policy_tag``)."""

    if "policy_tag" in arrays:
        return _per_agent_series(arrays["policy_tag"], ally_index, t).astype(np.int32)

    policy_ids = _per_agent_series(arrays["behavior_policy_id"], ally_index, t).astype(np.int32)
    mix = behavior_mix_from_config(meta.get("behavior_mix"))
    id2tag = policy_id_to_tag(mix)
    return np.asarray([id2tag.get(int(pid), int(pid)) for pid in policy_ids], dtype=np.int32)


def build_policy_mask(
    policy_ids: np.ndarray,
    *,
    mask_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Segment mask: decide at each policy switch; hold until next switch.

    ``1`` = masked. Decision at t=0 and whenever ``policy_ids[t] != policy_ids[t-1]``.
    """

    policy_ids = np.asarray(policy_ids).reshape(-1)
    t = int(policy_ids.shape[0])
    mask = np.zeros((t,), dtype=np.uint8)
    if t == 0:
        return mask
    p = float(np.clip(mask_prob, 0.0, 1.0))
    masked = bool(rng.random() < p)
    mask[0] = np.uint8(masked)
    for i in range(1, t):
        if int(policy_ids[i]) != int(policy_ids[i - 1]):
            masked = bool(rng.random() < p)
        mask[i] = np.uint8(masked)
    return mask


def split_one_record(
    record_dir: Path,
    *,
    output_root: Path | None = None,
    mask_prob: float = 0.3,
    mask_seed: int | None = None,
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
    base_seed = 0 if mask_seed is None else int(mask_seed)
    # Stable per-record salt so workers shuffle order does not change masks.
    record_salt = int(meta.get("record_id", 0)) + 1009 * int(meta.get("task_index", 0) or 0)

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

        policy_id_1d = _per_agent_series(
            arrays["behavior_policy_id"], ally_index, t
        ).astype(np.int32)
        policy_tag_1d = resolve_policy_tag(
            arrays, meta, ally_index=ally_index, t=t
        )
        rng = np.random.default_rng(base_seed + record_salt + 17 * ally_index)
        # Switch detection uses per-agent policy_id (tag changes with it today).
        policy_mask = build_policy_mask(policy_id_1d, mask_prob=mask_prob, rng=rng)
        # Tag 8 = mask: overwrite quality tag on masked steps.
        policy_tag_1d = policy_tag_1d.copy()
        policy_tag_1d[policy_mask.astype(bool)] = np.int32(POLICY_TAG_MASK)

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
        save_npy(agent_dir / "behavior_policy_id.npy", policy_id_1d)
        save_npy(agent_dir / "policy_tag.npy", policy_tag_1d)
        save_npy(agent_dir / "policy_mask.npy", policy_mask)

        mix_for_meta = behavior_mix_from_config(meta.get("behavior_mix"))
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
            "mask_prob": float(mask_prob),
            "policy_tag_legend": policy_tag_legend_json(),
            "policy_id_mapping": policy_id_mapping_json(mix_for_meta),
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
                "policy_tag": (
                    "see policy_id_mapping / policy_tag_legend; "
                    f"masked steps overwritten to tag={POLICY_TAG_MASK}"
                ),
                "policy_mask": (
                    "1=masked. Drawn at each behavior_policy_id switch with "
                    "mask_prob; held until the next switch; coincides with policy_tag==8"
                ),
                "alignment": "All arrays share the same time index t",
            },
        }
        save_meta(agent_dir / "meta.json", agent_meta)
        written.append(agent_dir)
    return written


def _split_one_record_job(
    payload: tuple[str, str | None, float, int | None],
) -> tuple[str, int, str | None]:
    """Picklable worker: ``(record_dir, output_root, mask_prob, mask_seed)``."""

    record_dir_s, output_root_s, mask_prob, mask_seed = payload
    try:
        paths = split_one_record(
            Path(record_dir_s),
            output_root=Path(output_root_s) if output_root_s else None,
            mask_prob=float(mask_prob),
            mask_seed=mask_seed,
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
    mask_prob: float = 0.3,
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
    jobs = [(str(p), out_s, float(mask_prob), seed) for p in record_dirs]
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
        help="RNG seed for --shuffle and policy_mask sampling",
    )
    parser.add_argument(
        "--mask_prob",
        type=float,
        default=0.3,
        help=(
            "At each behavior_policy_id switch, probability to mask until the "
            "next switch (writes policy_mask.npy; 1=masked). Default: 0.3"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 1))),
        help="Process-pool workers for parallel conversion (default: min(8, cpu_count))",
    )
    args = parser.parse_args(argv)
    output_root = Path(args.output_root) if args.output_root else None
    mask_prob = float(args.mask_prob)

    if args.record_dir:
        paths = split_one_record(
            Path(args.record_dir),
            output_root=output_root,
            mask_prob=mask_prob,
            mask_seed=args.seed,
        )
        print(f"[agent_centric] {args.record_dir} -> {len(paths)} agents", flush=True)
    elif args.records_root:
        split_records_root(
            Path(args.records_root),
            output_root=output_root,
            shuffle=bool(args.shuffle),
            seed=args.seed,
            workers=int(args.workers),
            mask_prob=mask_prob,
        )
    else:
        raise SystemExit("Provide --record_dir or --records_root")


if __name__ == "__main__":
    main()
