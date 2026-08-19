"""Agent-centric split that writes teacher/behavior intent communication fields.

Calls the existing ``split_one_record`` (env obs layout unchanged), then adds:

  reference_message.npy   [T]   teacher packed id (MessageHead target)
  behavior_message.npy    [T]   behavior packed id (on disk; not L_msg GT)
  visible_ally.npz        msg / msg_behavior / move / move_behavior / id / valid
  obs_dynamic last channel  global unit_keys index (reconstructed from roll;
                            TABX.get_obs / obs_flat stay 16-d)
"""

from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from generate.agent_centric import (
    _ally_is_padding,
    discover_env_record_dirs,
    split_one_record,
)
from generate.dump_schema import (
    OWN_IS_ALIVE_IDX,
    append_other_unit_ids,
    discover_agent_centric_dirs,
    load_meta,
    read_env_centric_record,
    save_meta,
    save_npy,
)
from generate.intent_label import (
    DEFAULT_MAX_N_UNITS,
    intent_from_reference,
    move_from_action,
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
    if "actions_behavior" not in arrays:
        raise KeyError(f"{record_dir} missing actions_behavior.npy")

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
    _, _, msg_teacher = intent_from_reference(
        ref_real, atk_ally, alive_ally, max_n_units=max_n_units
    )
    move_teacher = move_from_action(ref_real)

    beh = np.asarray(arrays["actions_behavior"], dtype=np.int32)
    if beh.ndim == 1:
        beh = np.repeat(beh[:, None], n_ally, axis=1)
    beh_real = beh[:, real_idx]
    _, _, msg_behavior = intent_from_reference(
        beh_real, atk_ally, alive_ally, max_n_units=max_n_units
    )
    move_behavior = move_from_action(beh_real)
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
        vis_kw = dict(
            receiver_index=recv,
            ally_indices=real_idx,
            ally_unit_ids=ally_unit_ids,
            visible_matrix=vis,
            alive_all=alive_ally,
            receiver_alive=alive_ally[:, recv_col],
        )
        msg, uids, valid = visible_ally_messages(msg_all=msg_teacher, **vis_kw)
        msg_b, _, _ = visible_ally_messages(msg_all=msg_behavior, **vis_kw)
        move, _, _ = visible_ally_messages(msg_all=move_teacher, **vis_kw)
        move_b, _, _ = visible_ally_messages(msg_all=move_behavior, **vis_kw)
        save_npy(agent_dir / "reference_message.npy", msg_teacher[:, recv_col])
        save_npy(agent_dir / "behavior_message.npy", msg_behavior[:, recv_col])
        np.savez_compressed(
            agent_dir / "visible_ally.npz",
            msg=np.asarray(msg, dtype=np.int32),
            msg_behavior=np.asarray(msg_b, dtype=np.int32),
            move=np.asarray(move, dtype=np.int32),
            move_behavior=np.asarray(move_b, dtype=np.int32),
            id=np.asarray(uids, dtype=np.int32),
            valid=np.asarray(valid, dtype=np.uint8),
        )
        n_units = int(ameta.get("n_units") or len(unit_keys))
        ego_uid = int(ally_unit_ids[recv_col])
        dyn_path = agent_dir / "obs_dynamic.npy"
        if dyn_path.is_file():
            dyn = np.load(dyn_path)
            dyn = append_other_unit_ids(dyn, ego_uid, n_units)
            save_npy(dyn_path, dyn)
            ameta["obs_dynamic_shape"] = list(dyn.shape[1:])
        notes = dict(ameta.get("notes") or {})
        notes["reference_message"] = (
            "teacher intent packed id (mode+focus); MessageHead target only"
        )
        notes["behavior_message"] = (
            "behavior-action packed id (mode+focus); on disk only, not L_msg GT"
        )
        notes["visible_ally"] = (
            "npz keys msg/msg_behavior/move/move_behavior/id/valid; "
            "other real allies in agent_index order; invalid slots PAD / 0 / -1 / 0"
        )
        notes["obs_dynamic_unit_id"] = (
            "last channel = global unit_keys index of that roll slot; "
            "not present in env get_obs / obs_flat"
        )
        ameta["notes"] = notes
        ameta["comm_schema"] = "agent_centric_v2_comm"
        ameta["max_n_units_comm"] = int(max_n_units)
        ameta["n_other_ally"] = int(msg.shape[1])
        ameta["unit_id"] = ego_uid
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


def _annotate_one_record_job(
    payload: tuple[str, list[str], int],
) -> tuple[str, int, str | None]:
    record_dir_s, leaf_dirs_s, max_n_units = payload
    try:
        annotate_agent_leaves(
            Path(record_dir_s),
            [Path(p) for p in leaf_dirs_s],
            max_n_units=int(max_n_units),
        )
        return record_dir_s, len(leaf_dirs_s), None
    except Exception as exc:  # noqa: BLE001
        return record_dir_s, 0, f"{type(exc).__name__}: {exc}"


def annotate_existing_leaves(
    leaves_root: Path,
    *,
    max_n_units: int = DEFAULT_MAX_N_UNITS,
    workers: int = 1,
) -> int:
    """Re-write comm fields + dyn unit_id on existing leaves. Does not re-split."""

    leaves_root = Path(leaves_root)
    print(f"[agent_centric_comm] scanning {leaves_root} ...", flush=True)
    leaves = discover_agent_centric_dirs(leaves_root)
    by_record: dict[Path, list[Path]] = {}
    skipped = 0
    for d in leaves:
        meta_path = d / "meta.json"
        if not meta_path.is_file():
            skipped += 1
            continue
        ameta = load_meta(meta_path)
        src = ameta.get("source_record")
        if not src:
            skipped += 1
            continue
        by_record.setdefault(Path(src), []).append(d)
    jobs = [
        (str(rec), [str(p) for p in dirs], int(max_n_units))
        for rec, dirs in by_record.items()
    ]
    n_workers = max(1, min(int(workers), len(jobs) or 1))
    print(
        f"[agent_centric_comm] annotate_only: {len(leaves)} leaves, "
        f"{len(jobs)} source records, skip={skipped}, workers={n_workers}",
        flush=True,
    )
    if not jobs:
        return 0

    errors: list[str] = []
    n = 0
    pbar = tqdm(total=len(jobs), desc="annotate_only", unit="record")

    def _consume(path_s: str, n_leaves: int, err: str | None) -> None:
        nonlocal n
        if err:
            errors.append(f"{path_s}: {err}")
            tqdm.write(f"[agent_centric_comm] ERROR {path_s}: {err}")
        else:
            n += n_leaves
        pbar.set_postfix(leaves=n, errors=len(errors))
        pbar.update(1)

    try:
        if n_workers == 1:
            for job in jobs:
                _consume(*_annotate_one_record_job(job))
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = [
                    pool.submit(_annotate_one_record_job, job) for job in jobs
                ]
                for fut in as_completed(futures):
                    _consume(*fut.result())
    finally:
        pbar.close()
    if errors:
        raise RuntimeError(
            f"annotate_only failed on {len(errors)} record(s); first: {errors[0]}"
        )
    return n


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Split env-centric records and write teacher/behavior comm fields"
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
    parser.add_argument(
        "--annotate_only",
        action="store_true",
        help="Patch existing agent-centric leaves (unit_id + dual msg/move); do not re-split",
    )
    args = parser.parse_args(argv)
    output_root = Path(args.output_root) if args.output_root else None
    if args.annotate_only:
        root = Path(args.output_root or args.records_root or args.record_dir or "")
        if not root:
            raise SystemExit("--annotate_only needs --output_root or --records_root")
        n = annotate_existing_leaves(
            root,
            max_n_units=int(args.max_n_units),
            workers=int(args.workers),
        )
        print(f"[agent_centric_comm] annotate_only updated {n} leaves", flush=True)
        return
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
