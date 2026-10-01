#!/usr/bin/env python3
"""Split coach task directories into N shard JSON files.

Only directory names are listed. task.json and checkpoints are not opened,
so this stays a single directory listing on a flat coach root. Pass one
file to generate so that machine only builds the tasks in it.

  python tools/split_coach_tasks.py /path/to/coach_root --shards 4 --out-dir ./task_shards

  python -m generate.env_centric_comm \\
    --config generate/configs/0930_4800tasks_train_comm.yaml \\
    --task_list ./task_shards/shard_00_of_04.json \\
    --output_root /path/to/part_00
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from generate.task_package import (  # noqa: E402
    list_task_leaf_refs,
    shard_payload,
    split_even,
)

DEFAULT_COACH_ROOT = (
    "/mnt/cfs/shaopt/data/MARL/yxq/TABX/coach_runs/merged_tasks"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "coach_root",
        nargs="?",
        default=DEFAULT_COACH_ROOT,
        help=f"Coach root (default: {DEFAULT_COACH_ROOT})",
    )
    parser.add_argument("--shards", type=int, required=True, help="Number of parts")
    parser.add_argument(
        "--records-per-task",
        type=int,
        default=10,
        help="Stored on each shard so generate keeps this many records per task",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="task_shards",
        help="Directory for shard_XX_of_NN.json",
    )
    args = parser.parse_args()
    if args.records_per_task < 1:
        raise SystemExit("--records-per-task must be >= 1")

    root = Path(args.coach_root)
    print(f"listing {root}", flush=True)
    packages = list_task_leaf_refs(root)
    shards = split_even(packages, int(args.shards))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    width = max(2, len(str(len(shards) - 1)))
    print(f"coach_root={root}")
    print(f"leaves={len(packages)} shards={len(shards)}")
    for index, group in enumerate(shards):
        payload = shard_payload(
            group,
            coach_root=root,
            shard=index,
            n_shards=len(shards),
            records_per_task=int(args.records_per_task),
        )
        path = out_dir / f"shard_{index:0{width}d}_of_{len(shards):0{width}d}.json"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        first = group[0].name if group else ""
        last = group[-1].name if group else ""
        print(
            f"  {path} count={payload['count']} "
            f"total_records={payload['total_records']} "
            f"first={first} last={last}"
        )


if __name__ == "__main__":
    main()
