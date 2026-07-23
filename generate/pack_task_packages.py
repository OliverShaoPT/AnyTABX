"""Pack a task bank + training ckpt tree into per-task packages."""

from __future__ import annotations

import argparse
from pathlib import Path

from generate.task_package import pack_task_packages


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pack task.json + oracle ckpt per task")
    parser.add_argument("--task_bank", type=str, required=True)
    parser.add_argument("--ckpt_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--algorithm", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--task_index", type=int, nargs="*", default=None)
    args = parser.parse_args(argv)
    written = pack_task_packages(
        task_bank_path=args.task_bank,
        ckpt_root=args.ckpt_root,
        output_root=args.output_root,
        algorithm=args.algorithm,
        seed=args.seed,
        task_indices=args.task_index,
    )
    print(f"Packed {len(written)} task packages into {args.output_root}")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
