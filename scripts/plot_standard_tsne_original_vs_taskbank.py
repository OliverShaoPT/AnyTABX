"""Draw a standard exact t-SNE comparison of Original-33 and one task bank.

Only the 33 built-in scenes and the explicitly supplied JSON task bank are
used.  Both groups participate in descriptor configuration, robust scaling,
and one ordinary exact t-SNE fit.  The result is a pure scatter plot without
hulls, contours, or manually adjusted coordinates.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from plot_environment_randomness_tsne import (
    GROUP_WEIGHTS,
    N_ITER,
    PERPLEXITY,
    RANDOM_SEED,
    descriptor_matrix,
    fit_robust_scaler,
    group_active_indexes,
    infer_descriptor_config,
    neighborhood_trustworthiness,
    render_distribution_plot,
    weighted_linear_features,
)
from plot_standard_environment_tsne import (
    squared_euclidean_distances,
    standard_exact_tsne,
)
from plot_tsne_diversity import load_original_scenes, read_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def run(
    *,
    task_bank_path: Path,
    task_key: str,
    task_label: str,
    output_tag: str,
) -> tuple[Path, Path]:
    if not re.fullmatch(r"[A-Za-z0-9_]+", task_key):
        raise ValueError(
            "task key may contain only letters, digits, and underscores."
        )
    if not re.fullmatch(r"[A-Za-z0-9_]+", output_tag):
        raise ValueError(
            "output tag may contain only letters, digits, and underscores."
        )

    original_tasks, _, original_counts = load_original_scenes()
    task_bank = read_json(task_bank_path)
    bank_tasks: list[dict[str, Any]] = task_bank["tasks"]
    if original_counts["total_original_scenes"] != 33:
        raise ValueError(
            "Expected exactly 33 built-in scenes, found "
            f"{original_counts['total_original_scenes']}."
        )
    if not bank_tasks:
        raise ValueError("The supplied task bank contains no tasks.")

    descriptor_config = infer_descriptor_config(
        (original_tasks, bank_tasks)
    )
    original_raw, layout = descriptor_matrix(
        original_tasks,
        descriptor_config,
    )
    bank_raw, bank_layout = descriptor_matrix(
        bank_tasks,
        descriptor_config,
    )
    if bank_layout != layout:
        raise AssertionError("Environment descriptor layouts do not match.")

    combined_raw = np.vstack([original_raw, bank_raw])
    scaler = fit_robust_scaler(combined_raw)
    active_groups = group_active_indexes(layout, scaler)
    combined_scaled = scaler.transform(combined_raw)
    features = weighted_linear_features(
        combined_scaled,
        active_groups,
    )
    embedding, kl_divergence = standard_exact_tsne(features)
    high_distances = np.sqrt(
        squared_euclidean_distances(features)
    )
    trustworthiness = neighborhood_trustworthiness(
        high_distances,
        embedding,
        n_neighbors=10,
    )

    original_indexes = np.arange(
        0,
        len(original_tasks),
        dtype=int,
    )
    bank_indexes = np.arange(
        len(original_tasks),
        len(original_tasks) + len(bank_tasks),
        dtype=int,
    )
    active_dimension_count = sum(
        len(indexes)
        for indexes in active_groups.values()
    )
    footer = (
        f"{layout.dimension}-D descriptor -> "
        f"{active_dimension_count} active features | "
        f"standard exact t-SNE n={len(embedding)} | "
        f"perplexity={PERPLEXITY:g} | seed={RANDOM_SEED} | "
        f"KL={kl_divergence:.3f} | T(10)={trustworthiness:.3f}"
    )

    output_path = (
        OUTPUTS / f"standard_tsne_original_vs_{output_tag}.png"
    )
    render_distribution_plot(
        output_path,
        embedding,
        [
            {
                "indexes": bank_indexes.tolist(),
                "label": task_label,
                "color": (211, 78, 100, 210),
                "marker": "diamond",
                "radius": 7,
            },
            {
                "indexes": original_indexes.tolist(),
                "label": "Original 33 scenes",
                "color": (30, 96, 162, 235),
                "marker": "triangle",
                "radius": 16,
            },
        ],
        [],
        title=f"Standard t-SNE: Original vs {task_label}",
        subtitle=(
            "One shared exact t-SNE fit; pure two-group scatter with "
            "no enclosing boundary"
        ),
        footer=footer,
        summary_lines=[
            (
                f"Original: n={len(original_tasks)} | "
                f"{task_label}: n={len(bank_tasks)}"
            ),
            "Descriptor and robust scaler fitted on these two groups only",
            "Original scenes use enlarged triangle markers",
        ],
        extent_embedding=embedding,
    )

    metrics = {
        "method": {
            "algorithm": (
                "standard exact t-SNE with Euclidean input distances"
            ),
            "implementation": "self-contained NumPy",
            "descriptor_and_scaler_fit_on": [
                "original",
                task_key,
            ],
            "global_embedding_fit_on": [
                "original",
                task_key,
            ],
            "descriptor_dimension": layout.dimension,
            "active_weighted_feature_dimensions": (
                active_dimension_count
            ),
            "descriptor_config": {
                "max_units_per_team": (
                    descriptor_config.max_units_per_team
                ),
                "max_zones": descriptor_config.max_zones,
                "attack_type_values": (
                    descriptor_config.attack_type_values
                ),
            },
            "preprocessing": (
                "combined robust scaling followed by "
                "group-balanced linear weighting"
            ),
            "group_weights": GROUP_WEIGHTS,
            "plot_style": "scatter only; no hull or enclosing boundary",
            "original_marker_radius": 16,
        },
        "sources": {
            "original": "src/tabx/scenarios/{units,zones,challenges}",
            task_key: str(task_bank_path),
        },
        "inventory": {
            "original": original_counts,
            task_key: len(bank_tasks),
            "tsne_fit_points": len(embedding),
        },
        "embedding": {
            "n_fit_points": len(embedding),
            "perplexity": PERPLEXITY,
            "iterations": N_ITER,
            "random_seed": RANDOM_SEED,
            "kl_divergence": kl_divergence,
            "trustworthiness_at_10": trustworthiness,
        },
        "plot_file": output_path.name,
    }
    metrics_path = (
        OUTPUTS / f"standard_tsne_original_vs_{output_tag}_metrics.json"
    )
    with metrics_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")

    print(
        json.dumps(
            {
                "plot": str(output_path),
                "metrics": str(metrics_path),
                "embedding": metrics["embedding"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return output_path, metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Draw one standard exact t-SNE plot using only Original-33 "
            "and a supplied JSON task bank."
        )
    )
    parser.add_argument("--task-bank-path", type=Path, required=True)
    parser.add_argument("--task-key", required=True)
    parser.add_argument("--task-label", required=True)
    parser.add_argument("--output-tag", required=True)
    args = parser.parse_args()
    run(
        task_bank_path=args.task_bank_path,
        task_key=args.task_key,
        task_label=args.task_label,
        output_tag=args.output_tag,
    )


if __name__ == "__main__":
    main()
