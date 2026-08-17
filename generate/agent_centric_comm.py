"""Agent-centric split that writes teacher intent communication fields.

Calls the existing ``split_one_record`` (unchanged leaf schema), then adds:

  reference_message.npy   [T]
  visible_ally.npz        keys msg/id/valid, each [T, n_real_ally-1]
"""

from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from generate.agent_centric import (
    _ally_is_padding,
    discover_env_record_dirs,
    split_one_record,
)
from generate.dump_schema import (
    OWN_IS_ALIVE_IDX,
    read_env_centric_record,
    save_meta,
    save_npy,
)
from generate.intent_label import (
    DEFAULT_MAX_N_UNITS,
    intent_from_reference,
    visible_ally_messages,
)


def _real_ally_indices(arrays: dict[str, np.ndarray], n_ally: int) -> list[int]:
    return [i for i in range(n_ally) if not _ally_is_padding(arrays, i)]


def annotate_agent_leaves(
    record_dir: Path,
    agent_dirs: list[Path],
    *,
    max_n_units: int = DEFAULT_MAX_N_UNITS,
) -> None:
    arrays, meta = read_env_centric_record(record_dir)
    if "attack_target" not in arrays:
        raise KeyError(
            f"{record_dir} missing attack_target.npy; "
            "generate with python -m generate.env_centric_comm"
        )
    if "visible_matrix" not in arrays:
        raise KeyError(f"{record_dir} missing visible_matrix.npy")
    if "actions_reference" not in arrays:
        raise KeyError(f"{record_dir} missing actions_reference.npy")

    ally_keys = list(meta["ally_keys"])
    n_ally = len(ally_keys)
    unit_keys = list(meta.get("unit_keys") or ally_keys)
    t = int(arrays["actions_reference"].shape[0])
    real_idx = np.asarray(_real_ally_indices(arrays, n_ally), dtype=np.int32)
    if real_idx.size == 0:
        return

    # unit id for schema ally i is typically i (unit_keys = ally_keys + enemy_keys)
    ally_unit_ids = np.asarray(
        [unit_keys.index(ally_keys[int(i)]) if ally_keys[int(i)] in unit_keys else int(i) for i in real_idx],
        dtype=np.int32,
    )

    ref = np.asarray(arrays["actions_reference"], dtype=np.int32)
    if ref.ndim == 1:
        ref = np.repeat(ref[:, None], n_ally, axis=1)
    atk_all = np.asarray(arrays["attack_target"], dtype=np.int32)
    # attack_target is [T, n_units]; take the ally unit rows
    if atk_all.ndim == 1:
        atk_all = np.repeat(atk_all[:, None], len(unit_keys), axis=1)
    atk_ally = atk_all[:, ally_unit_ids]

    if "unit_is_alive" in arrays:
        alive_u = np.asarray(arrays["unit_is_alive"]).reshape(t, -1)
        alive_ally = alive_u[:, ally_unit_ids]
    else:
        obs = np.asarray(arrays["obs_flat"])
        alive_ally = obs[:, real_idx, OWN_IS_ALIVE_IDX] > 0.5

    ref_real = ref[:, real_idx]
    _, _, msg_all = intent_from_reference(
        ref_real, atk_ally, alive_ally, max_n_units=max_n_units
    )
    vis = np.asarray(arrays["visible_matrix"])

    by_index = {}
    for d in agent_dirs:
        ameta_path = d / "meta.json"
        if not ameta_path.exists():
            continue
        ameta = json.loads(ameta_path.read_text(encoding="utf-8"))
        by_index[int(ameta["agent_index"])] = (d, ameta)

    for recv in real_idx.tolist():
        if recv not in by_index:
            continue
        agent_dir, ameta = by_index[recv]
        recv_col = int(np.where(real_idx == recv)[0][0])
        msg, uids, valid = visible_ally_messages(
            receiver_index=recv,
            ally_indices=real_idx,
            ally_unit_ids=ally_unit_ids,
            msg_all=msg_all,
            visible_matrix=vis,
            alive_all=alive_ally,
            receiver_alive=alive_ally[:, recv_col],
        )
        save_npy(agent_dir / "reference_message.npy", msg_all[:, recv_col])
        np.savez_compressed(
            agent_dir / "visible_ally.npz",
            msg=np.asarray(msg, dtype=np.int32),
            id=np.asarray(uids, dtype=np.int32),
            valid=np.asarray(valid, dtype=np.uint8),
        )
        notes = dict(ameta.get("notes") or {})
        notes["reference_message"] = (
            "teacher intent packed id (mode+focus); MessageHead target only"
        )
        notes["visible_ally"] = (
            "npz keys msg/id/valid; other real allies in agent_index order; "
            "teacher m*; invalid slots PAD / -1 / 0"
        )
        ameta["notes"] = notes
        ameta["comm_schema"] = "agent_centric_v1_comm"
        ameta["max_n_units_comm"] = int(max_n_units)
        ameta["n_other_ally"] = int(msg.shape[1])
        save_meta(agent_dir / "meta.json", ameta)


def split_one_record_comm(
    record_dir: Path,
    *,
    output_root: Path | None = None,
    mask_prob: float = 0.3,
    mask_seed: int | None = None,
    max_n_units: int = DEFAULT_MAX_N_UNITS,
) -> list[Path]:
    paths = split_one_record(
        record_dir,
        output_root=output_root,
        mask_prob=mask_prob,
        mask_seed=mask_seed,
    )
    annotate_agent_leaves(record_dir, paths, max_n_units=max_n_units)
    return paths


def _split_one_record_job(
    payload: tuple[str, str | None, float, int | None, int],
) -> tuple[str, int, str | None]:
    record_dir_s, output_root_s, mask_prob, mask_seed, max_n_units = payload
    try:
        paths = split_one_record_comm(
            Path(record_dir_s),
            output_root=Path(output_root_s) if output_root_s else None,
            mask_prob=float(mask_prob),
            mask_seed=mask_seed,
            max_n_units=int(max_n_units),
        )
        return record_dir_s, len(paths), None
    except Exception as exc:  # noqa: BLE001
        return record_dir_s, 0, f"{type(exc).__name__}: {exc}"


def split_records_root_comm(
    records_root: Path,
    *,
    output_root: Path | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    workers: int = 1,
    mask_prob: float = 0.3,
    max_n_units: int = DEFAULT_MAX_N_UNITS,
) -> list[Path]:
    record_dirs = discover_env_record_dirs(records_root)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(record_dirs)
    out_s = str(Path(output_root).resolve()) if output_root is not None else None
    n_workers = max(1, int(workers))
    jobs = [
        (str(p), out_s, float(mask_prob), seed, int(max_n_units)) for p in record_dirs
    ]
    errors: list[str] = []
    if n_workers == 1:
        for job in jobs:
            path_s, n_agents, err = _split_one_record_job(job)
            if err:
                errors.append(f"{path_s}: {err}")
                print(f"[agent_centric_comm] ERROR {path_s}: {err}", flush=True)
            else:
                print(f"[agent_centric_comm] {path_s} -> {n_agents} agents", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_split_one_record_job, job): job[0] for job in jobs}
            for fut in as_completed(futures):
                path_s, n_agents, err = fut.result()
                if err:
                    errors.append(f"{path_s}: {err}")
                    print(f"[agent_centric_comm] ERROR {path_s}: {err}", flush=True)
                else:
                    print(f"[agent_centric_comm] {path_s} -> {n_agents} agents", flush=True)
    if errors:
        raise RuntimeError(
            f"agent_centric_comm failed on {len(errors)} record(s); first: {errors[0]}"
        )
    return record_dirs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Split env-centric records and write teacher comm fields"
    )
    parser.add_argument("--record_dir", type=str, default=None)
    parser.add_argument("--records_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mask_prob", type=float, default=0.3)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 1))),
    )
    parser.add_argument("--max_n_units", type=int, default=DEFAULT_MAX_N_UNITS)
    args = parser.parse_args(argv)
    output_root = Path(args.output_root) if args.output_root else None
    if args.record_dir:
        paths = split_one_record_comm(
            Path(args.record_dir),
            output_root=output_root,
            mask_prob=float(args.mask_prob),
            mask_seed=args.seed,
            max_n_units=int(args.max_n_units),
        )
        print(f"[agent_centric_comm] {args.record_dir} -> {len(paths)} agents", flush=True)
    elif args.records_root:
        split_records_root_comm(
            Path(args.records_root),
            output_root=output_root,
            shuffle=bool(args.shuffle),
            seed=args.seed,
            workers=int(args.workers),
            mask_prob=float(args.mask_prob),
            max_n_units=int(args.max_n_units),
        )
    else:
        raise SystemExit("Provide --record_dir or --records_root")


if __name__ == "__main__":
    main()
