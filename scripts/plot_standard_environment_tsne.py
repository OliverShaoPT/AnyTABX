"""Draw three standard Euclidean t-SNE comparisons of TABX scene banks.

The environment descriptor and its robust, group-balanced preprocessing are
shared with ``plot_environment_randomness_tsne.py``.  The important difference
is the embedding: this script applies ordinary exact t-SNE to Euclidean
distances in the preprocessed feature matrix.  It does not use the custom
sum-of-group-distances affinity from the earlier analysis.

All panels reuse one global embedding of the 33 original scenes plus the
complete ``balance_tasks`` bank.  Because ``tasks1_balanced`` is a subset of
``balance_tasks``, this makes point coordinates directly comparable across the
three exported figures.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from plot_environment_randomness_tsne import (
    COLORS,
    GROUP_WEIGHTS,
    N_ITER,
    PERPLEXITY,
    RANDOM_SEED,
    cluster_coverage,
    descriptor_matrix,
    fit_robust_scaler,
    group_active_indexes,
    infer_descriptor_config,
    joint_probabilities,
    kmeans,
    neighborhood_trustworthiness,
    render_distribution_plot,
    student_probabilities,
    weighted_linear_features,
    within_metrics,
)
from plot_tsne_diversity import load_original_scenes, read_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def squared_euclidean_distances(features: np.ndarray) -> np.ndarray:
    """Return a numerically stable all-pairs squared Euclidean matrix."""

    squared_norms = np.sum(np.square(features), axis=1)
    distances_squared = (
        squared_norms[:, None]
        + squared_norms[None, :]
        - 2.0 * features @ features.T
    )
    distances_squared = np.maximum(distances_squared, 0.0)
    np.fill_diagonal(distances_squared, 0.0)
    return distances_squared


def standard_exact_tsne(
    features: np.ndarray,
    *,
    perplexity: float = PERPLEXITY,
    n_iter: int = N_ITER,
    random_seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, float]:
    """Apply standard exact t-SNE to a Euclidean feature matrix.

    This is a small NumPy implementation of the conventional exact algorithm:
    Gaussian high-dimensional affinities, a two-dimensional Student-t kernel,
    early exaggeration, momentum, adaptive gains, and PCA initialization.
    """

    if features.ndim != 2 or len(features) < 3:
        raise ValueError("t-SNE expects a 2-D feature matrix with at least 3 rows.")
    if not np.all(np.isfinite(features)):
        raise ValueError("t-SNE input contains NaN or infinite values.")

    distances_squared = squared_euclidean_distances(features)
    probabilities = joint_probabilities(distances_squared, perplexity)

    centered = features - features.mean(axis=0, keepdims=True)
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    embedding = centered @ right_vectors[:2].T
    embedding /= max(float(np.std(embedding)), 1e-12)
    embedding *= 1e-4
    rng = np.random.default_rng(random_seed)
    embedding += rng.normal(0.0, 1e-6, size=embedding.shape)

    velocity = np.zeros_like(embedding)
    gains = np.ones_like(embedding)
    early_exaggeration = 12.0
    learning_rate = max(len(features) / (early_exaggeration * 4.0), 50.0)

    for iteration in range(n_iter):
        low_probabilities, numerator = student_probabilities(embedding)
        target_probabilities = (
            probabilities * early_exaggeration
            if iteration < 250
            else probabilities
        )
        affinities = (target_probabilities - low_probabilities) * numerator
        gradient = 4.0 * (
            affinities.sum(axis=1)[:, None] * embedding
            - affinities @ embedding
        )

        momentum = 0.5 if iteration < 250 else 0.8
        gains = np.where(
            np.sign(gradient) != np.sign(velocity),
            gains + 0.2,
            gains * 0.8,
        )
        gains = np.maximum(gains, 0.01)
        velocity = momentum * velocity - learning_rate * gains * gradient
        embedding += velocity
        embedding -= embedding.mean(axis=0, keepdims=True)

    low_probabilities, _ = student_probabilities(embedding)
    mask = probabilities > 0.0
    kl_divergence = float(
        np.sum(
            probabilities[mask]
            * np.log(probabilities[mask] / low_probabilities[mask])
        )
    )
    return embedding, kl_divergence


def main() -> None:
    original_tasks, _, original_counts = load_original_scenes()
    tasks1_tasks = read_json(OUTPUTS / "tasks1_balanced.json")["tasks"]
    balance_tasks = read_json(OUTPUTS / "balance_tasks.json")["tasks"]

    tasks1_hashes = {task["metadata"]["canonical_hash"] for task in tasks1_tasks}
    balance_hashes = [task["metadata"]["canonical_hash"] for task in balance_tasks]
    tasks1_subset_mask = np.asarray(
        [task_hash in tasks1_hashes for task_hash in balance_hashes],
        dtype=bool,
    )
    if int(tasks1_subset_mask.sum()) != len(tasks1_tasks):
        raise ValueError(
            "tasks1_balanced is not preserved as an exact subset of balance_tasks."
        )
    additions_mask = ~tasks1_subset_mask

    descriptor_config = infer_descriptor_config(
        (original_tasks, tasks1_tasks, balance_tasks)
    )
    original_raw, layout = descriptor_matrix(original_tasks, descriptor_config)
    tasks1_raw, tasks1_layout = descriptor_matrix(tasks1_tasks, descriptor_config)
    balance_raw, balance_layout = descriptor_matrix(balance_tasks, descriptor_config)
    if layout != tasks1_layout or layout != balance_layout:
        raise AssertionError("Environment descriptor layouts do not match.")

    scaler = fit_robust_scaler(np.vstack([original_raw, balance_raw]))
    original_scaled = scaler.transform(original_raw)
    tasks1_scaled = scaler.transform(tasks1_raw)
    balance_scaled = scaler.transform(balance_raw)
    active_groups = group_active_indexes(layout, scaler)

    global_scaled = np.vstack([original_scaled, balance_scaled])
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
        "balance": within_metrics(balance_tasks, balance_scaled, active_groups),
    }
    balance_features = weighted_linear_features(balance_scaled, active_groups)
    mode_centers = kmeans(balance_features, 50, seed=RANDOM_SEED)
    modes = {
        "original": cluster_coverage(
            weighted_linear_features(original_scaled, active_groups),
            mode_centers,
        ),
        "tasks1": cluster_coverage(
            weighted_linear_features(tasks1_scaled, active_groups),
            mode_centers,
        ),
        "balance": cluster_coverage(balance_features, mode_centers),
    }

    original_indexes = np.arange(len(original_tasks), dtype=int)
    balance_indexes = np.arange(
        len(original_tasks),
        len(original_tasks) + len(balance_tasks),
        dtype=int,
    )
    tasks1_indexes = balance_indexes[tasks1_subset_mask]
    addition_indexes = balance_indexes[additions_mask]

    comparison_specs: list[dict[str, Any]] = [
        {
            "output": OUTPUTS / "standard_tsne_original_vs_tasks1_balanced.png",
            "groups": [
                {
                    "indexes": original_indexes.tolist(),
                    "label": "Original scenes",
                    "color": COLORS["original"],
                    "marker": "triangle",
                    "radius": 9,
                },
                {
                    "indexes": tasks1_indexes.tolist(),
                    "label": "tasks1_balanced",
                    "color": COLORS["author"],
                    "marker": "circle",
                    "radius": 7,
                },
            ],
            "title": "Standard t-SNE: original vs tasks1_balanced",
            "subtitle": "One shared Euclidean t-SNE embedding of the full environment descriptor",
            "summary": [
                f"Original: n={len(original_tasks)}, eff.dim={within['original']['effective_dimension_participation_ratio']:.2f}",
                f"tasks1:   n={len(tasks1_tasks)}, eff.dim={within['tasks1']['effective_dimension_participation_ratio']:.2f}",
                "Pure scatter plot; no hull or enclosing boundary",
            ],
        },
        {
            "output": OUTPUTS / "standard_tsne_original_vs_balance_tasks.png",
            "groups": [
                {
                    "indexes": original_indexes.tolist(),
                    "label": "Original scenes",
                    "color": COLORS["original"],
                    "marker": "triangle",
                    "radius": 9,
                },
                {
                    "indexes": balance_indexes.tolist(),
                    "label": "balance_tasks",
                    "color": COLORS["modified"],
                    "marker": "circle",
                    "radius": 7,
                },
            ],
            "title": "Standard t-SNE: original vs balance_tasks",
            "subtitle": "One shared Euclidean t-SNE embedding of the full environment descriptor",
            "summary": [
                f"Original: n={len(original_tasks)}, eff.dim={within['original']['effective_dimension_participation_ratio']:.2f}",
                f"Balance:  n={len(balance_tasks)}, eff.dim={within['balance']['effective_dimension_participation_ratio']:.2f}",
                "Pure scatter plot; no hull or enclosing boundary",
            ],
        },
        {
            "output": OUTPUTS / "standard_tsne_tasks1_balanced_vs_balance_tasks.png",
            "groups": [
                {
                    "indexes": tasks1_indexes.tolist(),
                    "label": "tasks1_balanced subset",
                    "color": COLORS["author"],
                    "marker": "circle",
                    "radius": 7,
                },
                {
                    "indexes": addition_indexes.tolist(),
                    "label": "Additional balance_tasks",
                    "color": COLORS["modified"],
                    "marker": "diamond",
                    "radius": 8,
                },
            ],
            "title": "Standard t-SNE: tasks1_balanced vs balance_tasks",
            "subtitle": "Blue diamonds are the added tasks; both groups form balance_tasks",
            "summary": [
                f"tasks1:  n={len(tasks1_tasks)}, occupied modes={modes['tasks1']['occupied_clusters']}/50",
                f"Balance: n={len(balance_tasks)}, occupied modes={modes['balance']['occupied_clusters']}/50",
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
            "global_embedding_reused_across_panels": True,
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
            "balance_tasks": len(balance_tasks),
            "balance_additions_beyond_tasks1": int(additions_mask.sum()),
        },
        "embedding": {
            "n_points": len(embedding),
            "perplexity": PERPLEXITY,
            "iterations": N_ITER,
            "random_seed": RANDOM_SEED,
            "kl_divergence": kl_divergence,
            "trustworthiness_at_10": trustworthiness,
        },
        "mode_coverage_k50": modes,
        "panel_files": [spec["output"].name for spec in comparison_specs],
    }
    metrics_path = OUTPUTS / "standard_tsne_metrics.json"
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
