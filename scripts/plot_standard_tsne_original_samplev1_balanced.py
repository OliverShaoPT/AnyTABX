"""Draw one shared standard exact t-SNE for three environment groups."""

from __future__ import annotations

import json
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
SAMPLE_PATH = OUTPUTS / "sampleV1_tasks.json"
BALANCED_PATH = ROOT / "task_files" / "balanced_tasks_v1.json"


def load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = read_json(path)["tasks"]
    if not tasks:
        raise ValueError(f"{path} contains no tasks.")
    return tasks


def run() -> tuple[Path, Path]:
    original_tasks, _, original_counts = load_original_scenes()
    sample_tasks = load_tasks(SAMPLE_PATH)
    balanced_tasks = load_tasks(BALANCED_PATH)
    if original_counts["total_original_scenes"] != 33:
        raise ValueError(
            "Expected exactly 33 built-in scenes, found "
            f"{original_counts['total_original_scenes']}."
        )

    task_groups = (original_tasks, sample_tasks, balanced_tasks)
    descriptor_config = infer_descriptor_config(task_groups)
    raw_matrices = []
    layout = None
    for tasks in task_groups:
        raw, current_layout = descriptor_matrix(tasks, descriptor_config)
        if layout is not None and current_layout != layout:
            raise AssertionError("Environment descriptor layouts do not match.")
        layout = current_layout
        raw_matrices.append(raw)
    if layout is None:
        raise AssertionError("Descriptor layout was not created.")

    combined_raw = np.vstack(raw_matrices)
    scaler = fit_robust_scaler(combined_raw)
    active_groups = group_active_indexes(layout, scaler)
    features = weighted_linear_features(
        scaler.transform(combined_raw),
        active_groups,
    )
    embedding, kl_divergence = standard_exact_tsne(features)
    high_distances = np.sqrt(squared_euclidean_distances(features))
    trustworthiness = neighborhood_trustworthiness(
        high_distances,
        embedding,
        n_neighbors=10,
    )

    original_end = len(original_tasks)
    sample_end = original_end + len(sample_tasks)
    balanced_end = sample_end + len(balanced_tasks)
    original_indexes = np.arange(0, original_end, dtype=int)
    sample_indexes = np.arange(original_end, sample_end, dtype=int)
    balanced_indexes = np.arange(sample_end, balanced_end, dtype=int)
    active_dimension_count = sum(
        len(indexes) for indexes in active_groups.values()
    )

    footer = (
        f"{layout.dimension}-D descriptor -> "
        f"{active_dimension_count} active features | "
        f"standard exact t-SNE n={len(embedding)} | "
        f"perplexity={PERPLEXITY:g} | seed={RANDOM_SEED} | "
        f"KL={kl_divergence:.3f} | T(10)={trustworthiness:.3f}"
    )
    output_path = (
        OUTPUTS
        / "standard_tsne_original_sampleV1_balanced_tasks_v1.png"
    )
    render_distribution_plot(
        output_path,
        embedding,
        [
            {
                "indexes": balanced_indexes.tolist(),
                "label": "Balanced-tasks-v1",
                "color": (54, 155, 118, 190),
                "marker": "circle",
                "radius": 6,
            },
            {
                "indexes": sample_indexes.tolist(),
                "label": "SampleV1-tasks",
                "color": (211, 78, 100, 225),
                "marker": "diamond",
                "radius": 8,
            },
            {
                "indexes": original_indexes.tolist(),
                "label": "Original 33 scenes",
                "color": (30, 96, 162, 240),
                "marker": "triangle",
                "radius": 16,
            },
        ],
        [],
        title="Standard t-SNE: Original, SampleV1, and Balanced-v1",
        subtitle=(
            "One shared exact t-SNE fit in one feature space; "
            "pure three-group scatter"
        ),
        footer=footer,
        summary_lines=[
            (
                f"Original: n={len(original_tasks)} | "
                f"SampleV1: n={len(sample_tasks)} | "
                f"Balanced-v1: n={len(balanced_tasks)}"
            ),
            "Descriptor and robust scaler fitted jointly on these groups only",
            "Original scenes use enlarged triangle markers",
        ],
        extent_embedding=embedding,
    )

    metrics = {
        "method": {
            "algorithm": "standard exact t-SNE with Euclidean input distances",
            "implementation": "self-contained NumPy",
            "descriptor_and_scaler_fit_on": [
                "original",
                "sampleV1_tasks",
                "balanced_tasks_v1",
            ],
            "global_embedding_fit_on": [
                "original",
                "sampleV1_tasks",
                "balanced_tasks_v1",
            ],
            "descriptor_dimension": layout.dimension,
            "active_weighted_feature_dimensions": active_dimension_count,
            "descriptor_config": {
                "max_units_per_team": descriptor_config.max_units_per_team,
                "max_zones": descriptor_config.max_zones,
                "attack_type_values": descriptor_config.attack_type_values,
            },
            "preprocessing": (
                "joint robust scaling followed by "
                "group-balanced linear weighting"
            ),
            "group_weights": GROUP_WEIGHTS,
            "plot_style": "scatter only; no hull or enclosing boundary",
            "original_marker_radius": 16,
        },
        "sources": {
            "original": "src/tabx/scenarios/{units,zones,challenges}",
            "sampleV1_tasks": str(SAMPLE_PATH),
            "balanced_tasks_v1": str(BALANCED_PATH),
        },
        "inventory": {
            "original": original_counts,
            "sampleV1_tasks": len(sample_tasks),
            "balanced_tasks_v1": len(balanced_tasks),
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
        OUTPUTS
        / "standard_tsne_original_sampleV1_balanced_tasks_v1_metrics.json"
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


if __name__ == "__main__":
    run()
