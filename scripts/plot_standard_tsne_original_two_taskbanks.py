"""Draw one shared standard exact t-SNE for Original-33 and two task banks."""

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


def load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = read_json(path)["tasks"]
    if not tasks:
        raise ValueError(f"{path} contains no tasks.")
    return tasks


def validate_slug(value: str, name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_]+", value):
        raise ValueError(
            f"{name} may contain only letters, digits, and underscores."
        )


def run(
    *,
    candidate_path: Path,
    candidate_key: str,
    candidate_label: str,
    baseline_path: Path,
    baseline_key: str,
    baseline_label: str,
    output_tag: str,
) -> tuple[Path, Path]:
    validate_slug(candidate_key, "candidate key")
    validate_slug(baseline_key, "baseline key")
    validate_slug(output_tag, "output tag")

    original_tasks, _, original_counts = load_original_scenes()
    candidate_tasks = load_tasks(candidate_path)
    baseline_tasks = load_tasks(baseline_path)
    if original_counts["total_original_scenes"] != 33:
        raise ValueError(
            "Expected exactly 33 built-in scenes, found "
            f"{original_counts['total_original_scenes']}."
        )

    task_groups = (original_tasks, candidate_tasks, baseline_tasks)
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
    candidate_end = original_end + len(candidate_tasks)
    baseline_end = candidate_end + len(baseline_tasks)
    original_indexes = np.arange(0, original_end, dtype=int)
    candidate_indexes = np.arange(
        original_end,
        candidate_end,
        dtype=int,
    )
    baseline_indexes = np.arange(
        candidate_end,
        baseline_end,
        dtype=int,
    )
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

    output_path = OUTPUTS / f"standard_tsne_{output_tag}.png"
    render_distribution_plot(
        output_path,
        embedding,
        [
            {
                "indexes": baseline_indexes.tolist(),
                "label": baseline_label,
                "color": (54, 155, 118, 190),
                "marker": "circle",
                "radius": 6,
            },
            {
                "indexes": candidate_indexes.tolist(),
                "label": candidate_label,
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
        title=(
            f"Standard t-SNE: Original, {candidate_label}, "
            f"and {baseline_label}"
        ),
        subtitle=(
            "One shared exact t-SNE fit in one feature space; "
            "pure three-group scatter"
        ),
        footer=footer,
        summary_lines=[
            (
                f"Original: n={len(original_tasks)} | "
                f"{candidate_label}: n={len(candidate_tasks)} | "
                f"{baseline_label}: n={len(baseline_tasks)}"
            ),
            "Descriptor and robust scaler fitted jointly on these groups only",
            "Original scenes use enlarged triangle markers",
        ],
        extent_embedding=embedding,
        legend_position="bottom",
    )

    metrics = {
        "method": {
            "algorithm": "standard exact t-SNE with Euclidean input distances",
            "implementation": "self-contained NumPy",
            "descriptor_and_scaler_fit_on": [
                "original",
                candidate_key,
                baseline_key,
            ],
            "global_embedding_fit_on": [
                "original",
                candidate_key,
                baseline_key,
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
            candidate_key: str(candidate_path.resolve()),
            baseline_key: str(baseline_path.resolve()),
        },
        "inventory": {
            "original": original_counts,
            candidate_key: len(candidate_tasks),
            baseline_key: len(baseline_tasks),
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
    metrics_path = OUTPUTS / f"standard_tsne_{output_tag}_metrics.json"
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
            "Draw one standard exact t-SNE using Original-33 "
            "and two supplied JSON task banks."
        )
    )
    parser.add_argument("--candidate-path", type=Path, required=True)
    parser.add_argument("--candidate-key", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--baseline-path", type=Path, required=True)
    parser.add_argument("--baseline-key", required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--output-tag", required=True)
    args = parser.parse_args()
    run(
        candidate_path=args.candidate_path,
        candidate_key=args.candidate_key,
        candidate_label=args.candidate_label,
        baseline_path=args.baseline_path,
        baseline_key=args.baseline_key,
        baseline_label=args.baseline_label,
        output_tag=args.output_tag,
    )


if __name__ == "__main__":
    main()
