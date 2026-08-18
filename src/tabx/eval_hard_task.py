"""Re-evaluate and filter an asymmetric hard-task bank.

Unlike :mod:`src.tabx.eval_task`, this command emits no separate evaluation
artifact.  Its single output is another standard TABX task-bank JSON, so the
result can be handed directly to the training pipeline.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


def _early_compute_backend(argv: Sequence[str]) -> str:
    """Read the backend before importing modules that initialize JAX."""

    for index, argument in enumerate(argv):
        if argument.startswith("--compute-backend="):
            return argument.split("=", 1)[1]
        if argument == "--compute-backend" and index + 1 < len(argv):
            return argv[index + 1]
    return "cpu"


if __name__ == "__main__":
    _CLI_COMPUTE_BACKEND = _early_compute_backend(sys.argv[1:])
    if _CLI_COMPUTE_BACKEND == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_SKIP_CUDA_CONSTRAINTS_CHECK"] = "1"
    elif _CLI_COMPUTE_BACKEND == "gpu":
        os.environ["JAX_PLATFORMS"] = "cuda"

import jax
import numpy as np
import tyro

from src.tabx import sample_task as base
from src.tabx.eval_task import evaluate_tasks, flip_task


@dataclass(frozen=True)
class HardEvalConfig:
    """Configuration for fresh original/flipped hard-task certification."""

    task_file: str
    output: str
    heuristic: str | None = None
    physics: str | None = None
    num_seeds: int = 256
    seed: int = 200_000
    epsilon_override: float | None = 0.05
    max_episode_steps: int = 512
    win_rate_min: float = 0.20
    win_rate_max: float = 0.40
    min_band_probability: float = 0.85
    max_abs_side_bias: float = 0.10
    max_truncation_rate: float = 0.15
    max_no_interaction_rate: float = 0.0
    max_abs_hp_margin: float = 0.65
    max_abs_hp_side_bias: float = 0.20
    min_weak_damage_share: float = 0.15
    compute_backend: str = "cpu"
    cpu_cores: int = 1


def _validate_config(config: HardEvalConfig) -> None:
    if config.compute_backend not in {"cpu", "gpu"}:
        raise ValueError("compute_backend must be 'cpu' or 'gpu'.")
    if config.cpu_cores <= 0:
        raise ValueError("cpu_cores must be positive.")
    if config.compute_backend == "gpu" and config.cpu_cores != 1:
        raise ValueError("GPU evaluation requires cpu_cores=1.")
    if config.num_seeds <= 0:
        raise ValueError("num_seeds must be positive.")
    if not 0.0 < config.win_rate_min < config.win_rate_max < 0.5:
        raise ValueError("Hard win-rate band must satisfy 0 < min < max < 0.5.")
    unit_interval = {
        "min_band_probability": config.min_band_probability,
        "max_abs_side_bias": config.max_abs_side_bias,
        "max_truncation_rate": config.max_truncation_rate,
        "max_no_interaction_rate": config.max_no_interaction_rate,
        "max_abs_hp_margin": config.max_abs_hp_margin,
        "max_abs_hp_side_bias": config.max_abs_hp_side_bias,
        "min_weak_damage_share": config.min_weak_damage_share,
    }
    invalid = [
        name for name, value in unit_interval.items() if not 0.0 <= value <= 1.0
    ]
    if invalid:
        raise ValueError(f"Hard evaluation thresholds must lie in [0, 1]: {invalid}.")


def _hard_bands(config: HardEvalConfig) -> tuple[tuple[float, float], tuple[float, float]]:
    original = (config.win_rate_min, config.win_rate_max)
    flipped = (1.0 - config.win_rate_max, 1.0 - config.win_rate_min)
    return original, flipped


def _damage_share(result: dict[str, Any], team: int) -> float:
    values = np.asarray(result.get("damage_share", (0.0, 0.0)), dtype=float)
    return float(values[team]) if values.size > team else 0.0


def assess_hard_result(
    original: dict[str, Any],
    flipped: dict[str, Any],
    config: HardEvalConfig,
) -> dict[str, Any]:
    """Apply the same asymmetric, side-invariant gate used by hard sampling."""

    original_band, flipped_band = _hard_bands(config)
    original_win_rate = float(original["win_rate"])
    flipped_win_rate = float(flipped["win_rate"])
    original_wins = int(round(original_win_rate * config.num_seeds))
    flipped_wins = int(round(flipped_win_rate * config.num_seeds))
    original_probability = base._balance_posterior_probability(
        original_wins, config.num_seeds, *original_band
    )
    flipped_probability = base._balance_posterior_probability(
        flipped_wins, config.num_seeds, *flipped_band
    )
    team_invariant_win_rate = (
        original_win_rate + 1.0 - flipped_win_rate
    ) * 0.5
    side_bias = original_win_rate + flipped_win_rate - 1.0
    original_hp = float(original["hp_margin"]["mean"])
    flipped_hp = float(flipped["hp_margin"]["mean"])
    hp_side_bias = original_hp + flipped_hp
    weak_damage_share = min(
        _damage_share(original, 0),
        _damage_share(flipped, 1),
    )

    reasons: list[str] = []
    if not original_band[0] <= original_win_rate <= original_band[1]:
        reasons.append("original_win_rate")
    if not flipped_band[0] <= flipped_win_rate <= flipped_band[1]:
        reasons.append("flipped_win_rate")
    if original_probability < config.min_band_probability:
        reasons.append("original_band_probability")
    if flipped_probability < config.min_band_probability:
        reasons.append("flipped_band_probability")
    if not original_band[0] <= team_invariant_win_rate <= original_band[1]:
        reasons.append("team_invariant_win_rate")
    if abs(side_bias) > config.max_abs_side_bias:
        reasons.append("side_bias")
    if abs(original_hp) > config.max_abs_hp_margin:
        reasons.append("original_hp_margin")
    if abs(flipped_hp) > config.max_abs_hp_margin:
        reasons.append("flipped_hp_margin")
    if abs(hp_side_bias) > config.max_abs_hp_side_bias:
        reasons.append("hp_side_bias")
    if weak_damage_share < config.min_weak_damage_share:
        reasons.append("weak_damage_share")

    for label, result in (("original", original), ("flipped", flipped)):
        truncation_rate = float(result["truncation_rate"])
        truncations = int(round(truncation_rate * config.num_seeds))
        if (
            truncation_rate > 0.0
            and base._wilson_upper(truncations, config.num_seeds)
            > config.max_truncation_rate
        ):
            reasons.append(f"{label}_truncation")
        no_interaction_rate = float(result["quality_flags"]["no_interaction_rate"])
        if no_interaction_rate > config.max_no_interaction_rate:
            reasons.append(f"{label}_no_interaction")

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "original_win_rate": original_win_rate,
        "flipped_win_rate": flipped_win_rate,
        "original_band_probability": original_probability,
        "flipped_band_probability": flipped_probability,
        "team_invariant_win_rate": team_invariant_win_rate,
        "side_bias": side_bias,
        "original_hp_margin": original_hp,
        "flipped_hp_margin": flipped_hp,
        "hp_side_bias": hp_side_bias,
        "weak_damage_share": weak_damage_share,
    }


def _protocol(config: HardEvalConfig, physics: str, heuristic: str) -> dict[str, Any]:
    original_band, flipped_band = _hard_bands(config)
    return {
        "evaluator": "src.tabx.eval_hard_task",
        "physics": physics,
        "heuristic": heuristic,
        "num_seeds_per_orientation": config.num_seeds,
        "seed": config.seed,
        "epsilon": config.epsilon_override,
        "max_episode_steps": config.max_episode_steps,
        "original_win_rate": list(original_band),
        "flipped_win_rate": list(flipped_band),
        "team_invariant_win_rate": list(original_band),
        "minimum_band_probability": config.min_band_probability,
        "max_abs_side_bias": config.max_abs_side_bias,
        "max_truncation_rate": config.max_truncation_rate,
        "max_no_interaction_rate": config.max_no_interaction_rate,
        "max_abs_hp_margin": config.max_abs_hp_margin,
        "max_abs_hp_side_bias": config.max_abs_hp_side_bias,
        "min_weak_damage_share": config.min_weak_damage_share,
        "paired_original_flipped_seeds": True,
    }


def build_filtered_hard_bank(
    bank: dict[str, Any],
    results: Sequence[dict[str, Any]],
    config: HardEvalConfig,
    *,
    physics: str,
    heuristic: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return one standard task bank containing only freshly certified tasks."""

    tasks = bank["tasks"]
    task_ids = [str(task["task_id"]) for task in tasks]
    duplicate_task_ids = sorted(
        task_id for task_id, count in Counter(task_ids).items() if count > 1
    )
    if duplicate_task_ids:
        raise ValueError(f"Input task bank contains duplicate task IDs: {duplicate_task_ids}.")
    if len(results) != 2 * len(tasks):
        raise ValueError(
            "Expected one original and one flipped result per input task; "
            f"received {len(results)} results for {len(tasks)} tasks."
        )
    original_results = results[: len(tasks)]
    flipped_results = results[len(tasks) :]
    kept_tasks: list[dict[str, Any]] = []
    removed_task_ids: list[str] = []
    reason_counts: Counter[str] = Counter()

    for task, original, flipped in zip(
        tasks, original_results, flipped_results, strict=True
    ):
        task_id = str(task["task_id"])
        if str(original["task_id"]) != task_id:
            raise ValueError(
                f"Original result mismatch: expected {task_id!r}, "
                f"received {original['task_id']!r}."
            )
        expected_flipped_id = f"{task_id}__flipped"
        if str(flipped["task_id"]) != expected_flipped_id:
            raise ValueError(
                f"Flipped result mismatch: expected {expected_flipped_id!r}, "
                f"received {flipped['task_id']!r}."
            )
        assessment = assess_hard_result(original, flipped, config)
        if not assessment["accepted"]:
            removed_task_ids.append(task_id)
            reason_counts.update(assessment["reasons"])
            continue
        retained = copy.deepcopy(task)
        metadata = retained.setdefault("metadata", {})
        metadata["task_profile"] = "hard"
        metadata["hard_task_metrics"] = {
            key: value
            for key, value in assessment.items()
            if key not in {"accepted", "reasons"}
        }
        metadata["hard_task_evaluation"] = {
            "accepted": True,
            "assessment": assessment,
            "original": copy.deepcopy(original),
            "flipped": copy.deepcopy(flipped),
        }
        kept_tasks.append(retained)

    if not kept_tasks:
        raise RuntimeError(
            "Hard evaluation rejected every task; no empty training bank was written."
        )

    summary = {
        "enabled": True,
        "input_task_file": str(Path(config.task_file)),
        "n_input_tasks": len(tasks),
        "n_kept_tasks": len(kept_tasks),
        "n_removed_tasks": len(removed_task_ids),
        "removed_task_ids": removed_task_ids,
        "reason_counts": dict(reason_counts),
        "protocol": _protocol(config, physics, heuristic),
    }
    output_bank = copy.deepcopy(bank)
    output_bank["tasks"] = kept_tasks
    output_bank["manifest"]["n_tasks"] = len(kept_tasks)
    output_bank["manifest"]["hard_task_evaluation"] = copy.deepcopy(summary)
    return output_bank, summary


def _write_json_atomic(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write safely, including when output intentionally equals the input path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
            temporary_path = Path(file.name)
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output


def main() -> None:
    config = tyro.cli(HardEvalConfig)
    _validate_config(config)
    print(
        f"Hard-task evaluation backend: {jax.default_backend()} "
        f"({len(jax.devices())} visible device(s)).",
        flush=True,
    )
    bank = base.load_task_bank(config.task_file)
    manifest = bank["manifest"]
    physics = config.physics or str(manifest["physics"])
    heuristic = config.heuristic or str(manifest["heuristic"])
    tasks = bank["tasks"]
    evaluation_tasks = list(tasks) + [flip_task(task) for task in tasks]
    schema = manifest.get("schema", {})
    team_max = max(
        int(schema.get("max_n_ally", 0)),
        int(schema.get("max_n_enemy", 0)),
    ) or None
    max_n_zone = int(schema.get("max_n_zone", 0)) or None
    results = evaluate_tasks(
        evaluation_tasks,
        physics=physics,
        heuristic=heuristic,
        num_seeds=config.num_seeds,
        seed=config.seed,
        epsilon_override=config.epsilon_override,
        max_episode_steps=config.max_episode_steps,
        max_n_ally=team_max,
        max_n_enemy=team_max,
        max_n_zone=max_n_zone,
        show_progress=config.cpu_cores == 1,
        cpu_cores=config.cpu_cores,
    )
    output_bank, summary = build_filtered_hard_bank(
        bank,
        results,
        config,
        physics=physics,
        heuristic=heuristic,
    )
    output = _write_json_atomic(config.output, output_bank)
    print(
        f"Hard-task filter kept {summary['n_kept_tasks']}/"
        f"{summary['n_input_tasks']} tasks. Saved training bank to {output}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
