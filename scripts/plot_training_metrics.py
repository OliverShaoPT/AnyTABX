#!/usr/bin/env python3
"""Plot live MARL reward and early-stop diagnostics from training_metrics.csv."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Sequence


def read_metrics(path: Path) -> dict[str, list[float]]:
    columns: dict[str, list[float]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            for key, value in row.items():
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                columns.setdefault(key, []).append(number)
    return columns


def draw(path: Path, output: Path, *, show: bool) -> None:
    import matplotlib.pyplot as plt

    metrics = read_metrics(path)
    updates = metrics.get("update_steps", [])
    if not updates:
        raise ValueError(f"No completed updates found in {path}")

    figure = plt.figure("TABX training convergence", figsize=(12, 8), clear=True)
    reward_axis, rollout_axis = figure.subplots(2, 1, sharex=True)

    reward_axis.plot(
        updates,
        metrics.get("episode_returns", []),
        alpha=0.45,
        label="episode return",
    )
    reward_axis.plot(
        updates,
        metrics.get("early_stop/window_return", []),
        linewidth=2,
        label="early-stop rolling mean",
    )
    best = metrics.get("early_stop/best_return", [])
    if best:
        reward_axis.plot(updates, best, linestyle="--", label="best rolling mean")
    reward_axis.set_ylabel("Return")
    reward_axis.grid(alpha=0.25)
    reward_axis.legend()

    rollout_axis.plot(
        updates,
        metrics.get("rollout_reward_mean", []),
        color="tab:green",
        label="rollout reward mean",
    )
    rollout_axis.set_xlabel("Training update")
    rollout_axis.set_ylabel("Rollout reward")
    rollout_axis.grid(alpha=0.25)
    rollout_axis.legend()

    triggered = metrics.get("early_stop/triggered", [])
    trigger_updates = [
        update for update, flag in zip(updates, triggered) if flag >= 0.5
    ]
    for index, update in enumerate(trigger_updates):
        label = "early-stop trigger" if index == 0 else None
        reward_axis.axvline(update, color="tab:red", alpha=0.7, label=label)
        rollout_axis.axvline(update, color="tab:red", alpha=0.7)
    if trigger_updates:
        reward_axis.legend()

    figure.suptitle(f"{path.parent.name}: reward and early-stop diagnostics")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    if show:
        plt.show(block=False)
        plt.pause(0.01)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="path to training_metrics.csv")
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG output path (default: training_curve.png beside the CSV)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="redraw while training appends new updates",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="watch refresh interval in seconds (default: 5)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="only write the PNG; do not open an interactive window",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    metrics = args.metrics.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else metrics.with_name("training_curve.png")
    )
    if args.interval <= 0:
        raise SystemExit("--interval must be positive")
    try:
        while True:
            try:
                draw(metrics, output, show=not args.no_show)
            except (FileNotFoundError, ValueError) as error:
                if not args.watch:
                    raise SystemExit(str(error)) from error
                print(f"Waiting for metrics: {error}")
            if not args.watch:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 130
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
