"""Draw standard t-SNE comparisons involving ``hard_diverse_tasks``.

The two panels share one Euclidean t-SNE embedding fitted to the original
33 scenes, ``tasks1_balanced``, and ``hard_diverse_tasks``.  Preprocessing and
the exact t-SNE implementation are shared with
``plot_standard_environment_tsne.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from plot_environment_randomness_tsne import (
    COLORS,
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
    within_metrics,
)
from plot_standard_environment_tsne import (
    squared_euclidean_distances,
    standard_exact_tsne,
)
from plot_tsne_diversity import load_original_scenes, read_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def main() -> None:
    original_tasks, _, original_counts = load_original_scenes()
    tasks1_tasks = read_json(OUTPUTS / "tasks1_balanced.json")["tasks"]
    hard_tasks = read_json(OUTPUTS / "hard_diverse_tasks.json")["tasks"]

    descriptor_config = infer_descriptor_config(
        (original_tasks, tasks1_tasks, hard_tasks)
    )
    original_raw, layout = descriptor_matrix(original_tasks, descriptor_config)
    tasks1_raw, tasks1_layout = descriptor_matrix(tasks1_tasks, descriptor_config)
    hard_raw, hard_layout = descriptor_matrix(hard_tasks, descriptor_config)
    if layout != tasks1_layout or layout != hard_layout:
        raise AssertionError("Environment descriptor layouts do not match.")

    scaler = fit_robust_scaler(
        np.vstack([original_raw, tasks1_raw, hard_raw])
    )
    original_scaled = scaler.transform(original_raw)
    tasks1_scaled = scaler.transform(tasks1_raw)
    hard_scaled = scaler.transform(hard_raw)
    active_groups = group_active_indexes(layout, scaler)

    global_scaled = np.vstack([original_scaled, tasks1_scaled, hard_scaled])
    global_features = weighted_linear_features(global_scaled, active_groups)
    embedding, kl_divergence = standard_exact_tsne(global_features)
    high_distances = np.sqrt(squared_euclidean_distances(global_features))
    trustworthiness = neighborhood_trustworthiness(
        high_distances,
        embedding,
        n_neighbors=10,
    )

    within = {
        "original": within_metrics(original_tasks, original_scaled, active_groups),
        "tasks1": within_metrics(tasks1_tasks, tasks1_scaled, active_groups),
        "hard": within_metrics(hard_tasks, hard_scaled, active_groups),
    }

    original_indexes = np.arange(len(original_tasks), dtype=int)
    tasks1_indexes = np.arange(
        len(original_tasks),
        len(original_tasks) + len(tasks1_tasks),
        dtype=int,
    )
    hard_indexes = np.arange(
        len(original_tasks) + len(tasks1_tasks),
        len(global_scaled),
        dtype=int,
    )

    comparison_specs: list[dict[str, Any]] = [
        {
            "output": OUTPUTS
            / "standard_tsne_original_vs_hard_diverse_tasks.png",
            "groups": [
                {
                    "indexes": original_indexes.tolist(),
                    "label": "Original scenes",
                    "color": COLORS["original"],
                    "marker": "triangle",
                    "radius": 9,
                },
                {
                    "indexes": hard_indexes.tolist(),
                    "label": "hard_diverse_tasks",
                    "color": COLORS["modified"],
                    "marker": "diamond",
                    "radius": 8,
                },
            ],
            "title": "Standard t-SNE: original vs hard_diverse_tasks",
            "subtitle": "Shared Euclidean embedding of original, tasks1, and hard scenes",
            "summary": [
                f"Original: n={len(original_tasks)}, eff.dim={within['original']['effective_dimension_participation_ratio']:.2f}",
                f"Hard:     n={len(hard_tasks)}, eff.dim={within['hard']['effective_dimension_participation_ratio']:.2f}",
                "Pure scatter plot; no hull or enclosing boundary",
            ],
        },
        {
            "output": OUTPUTS
            / "standard_tsne_tasks1_balanced_vs_hard_diverse_tasks.png",
            "groups": [
                {
                    "indexes": tasks1_indexes.tolist(),
                    "label": "tasks1_balanced",
                    "color": COLORS["author"],
                    "marker": "circle",
                    "radius": 7,
                },
                {
                    "indexes": hard_indexes.tolist(),
                    "label": "hard_diverse_tasks",
                    "color": COLORS["modified"],
                    "marker": "diamond",
                    "radius": 8,
                },
            ],
            "title": "Standard t-SNE: tasks1_balanced vs hard_diverse_tasks",
            "subtitle": "Shared Euclidean embedding of original, tasks1, and hard scenes",
            "summary": [
                f"tasks1: n={len(tasks1_tasks)}, eff.dim={within['tasks1']['effective_dimension_participation_ratio']:.2f}",
                f"Hard:   n={len(hard_tasks)}, eff.dim={within['hard']['effective_dimension_participation_ratio']:.2f}",
                "Pure scatter plot; no hull or enclosing boundary",
            ],
        },
    ]

    active_dimension_count = sum(len(indexes) for indexes in active_groups.values())
    footer = (
        f"{layout.dimension}-D descriptor -> {active_dimension_count} active weighted features | "
        f"standard exact t-SNE (Euclidean) | perplexity={PERPLEXITY:g} | "
        f"seed={RANDOM_SEED} | KL={kl_divergence:.3f} | T(10)={trustworthiness:.3f}"
    )
    for spec in comparison_specs:
        render_distribution_plot(
            spec["output"],
            embedding,
            spec["groups"],
            [],
            title=spec["title"],
            subtitle=spec["subtitle"],
            footer=footer,
            summary_lines=spec["summary"],
        )

    metrics = {
        "method": {
            "algorithm": "standard exact t-SNE with Euclidean input distances",
            "implementation": "self-contained NumPy",
            "global_embedding_fit_on": [
                "original scenes",
                "tasks1_balanced",
                "hard_diverse_tasks",
            ],
            "descriptor_dimension": layout.dimension,
            "active_weighted_feature_dimensions": active_dimension_count,
            "descriptor_config": {
                "max_units_per_team": descriptor_config.max_units_per_team,
                "max_zones": descriptor_config.max_zones,
                "attack_type_values": descriptor_config.attack_type_values,
            },
            "preprocessing": "robust scaling followed by group-balanced linear weighting",
            "group_weights": GROUP_WEIGHTS,
            "plot_style": "scatter only; no convex hull or central envelope",
        },
        "inventory": {
            "original": original_counts,
            "tasks1_balanced": len(tasks1_tasks),
            "hard_diverse_tasks": len(hard_tasks),
        },
        "within_set_randomness": within,
        "embedding": {
            "n_points": len(embedding),
            "perplexity": PERPLEXITY,
            "iterations": N_ITER,
            "random_seed": RANDOM_SEED,
            "kl_divergence": kl_divergence,
            "trustworthiness_at_10": trustworthiness,
        },
        "panel_files": [spec["output"].name for spec in comparison_specs],
    }
    metrics_path = OUTPUTS / "standard_tsne_hard_metrics.json"
    with metrics_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")

    print(
        json.dumps(
            {
                "plots": [str(spec["output"]) for spec in comparison_specs],
                "metrics": str(metrics_path),
                "embedding": metrics["embedding"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
