"""Create pairwise t-SNE plots and diversity metrics for TABX task banks.

The script intentionally depends only on NumPy and Pillow.  It uses the
project's 38-dimensional parameter signature and group-weighted parameter
distance, with mirror/rotation/team-label canonicalization, so visual
separation is driven by task structure rather than arbitrary orientation.
"""

from __future__ import annotations

import copy
import json
import math
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
SCENARIOS = ROOT / "src" / "tabx" / "scenarios"

RANDOM_SEED = 42
PERPLEXITY = 30.0
N_ITER = 1_200
PARAMETER_DUPLICATE_THRESHOLD = 0.04

GROUPS: tuple[tuple[np.ndarray, float], ...] = (
    (np.r_[0:9, 16:25], 0.35),
    (np.r_[9:13, 25:29, 32], 0.30),
    (np.r_[13:16, 29:32], 0.10),
    (np.r_[33:38], 0.25),
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_original_scenes() -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    """Expand all unit/zone compositions and append all challenge scenes."""

    unit_paths = sorted((SCENARIOS / "units").glob("*.json"))
    zone_paths = sorted((SCENARIOS / "zones").glob("*.json"))
    challenge_paths = sorted((SCENARIOS / "challenges").glob("*.json"))

    units = [(path.stem, read_json(path)) for path in unit_paths]
    zones = [(path.stem, read_json(path)) for path in zone_paths]
    tasks: list[dict[str, Any]] = []
    labels: list[str] = []

    for (unit_name, unit), (zone_name, zone) in product(units, zones):
        tasks.append(
            {
                "grid_info": copy.deepcopy(unit.get("grid_info", {})),
                "scenario": copy.deepcopy(unit["scenario"]),
                "zone_scenario": copy.deepcopy(zone["zone_scenario"]),
            }
        )
        labels.append(f"composed:{unit_name}+{zone_name}")

    for path in challenge_paths:
        challenge = read_json(path)
        tasks.append(
            {
                "grid_info": copy.deepcopy(challenge.get("grid_info", {})),
                "scenario": copy.deepcopy(challenge["scenario"]),
                "zone_scenario": copy.deepcopy(challenge["zone_scenario"]),
            }
        )
        labels.append(f"challenge:{path.stem}")

    counts = {
        "unit_assets": len(units),
        "zone_assets": len(zones),
        "composed_scenes": len(units) * len(zones),
        "challenge_scenes": len(challenge_paths),
        "total_original_scenes": len(tasks),
    }
    return tasks, labels, counts


def parameter_signature(
    task: dict[str, Any], *, mirror_x: bool, mirror_y: bool, flip_teams: bool
) -> np.ndarray:
    """Match ``sample_task._parameter_signature`` for a transformed task."""

    scenario = task["scenario"]
    teams = np.asarray(scenario["teams"], dtype=int).reshape(-1)
    if flip_teams:
        teams = 1 - teams
    unit_ids = np.asarray(scenario["unit_ids"], dtype=int).reshape(-1)
    if np.any(unit_ids < 0) or np.any(unit_ids > 8):
        raise ValueError("The 38-D parameter signature expects unit IDs in [0, 8].")
    positions = np.asarray(scenario["positions"], dtype=float).reshape(-1, 2).copy()
    positions[:, 0] *= -1.0 if mirror_x else 1.0
    positions[:, 1] *= -1.0 if mirror_y else 1.0
    health = np.asarray(scenario["healths"], dtype=float).reshape(-1)
    damage = np.asarray(scenario["attack_damages"], dtype=float).reshape(-1)
    speed = np.asarray(scenario["speeds"], dtype=float).reshape(-1)

    grid = task.get("grid_info", {})
    width = float(grid.get("max_field_width", 121.0))
    height = float(grid.get("max_field_height", 78.0))
    diagonal = max(math.hypot(width, height), 1.0)

    features: list[float] = []
    team_centers: list[np.ndarray] = []
    for team in (0, 1):
        mask = teams == team
        if not np.any(mask):
            raise ValueError("Every scene must contain both team 0 and team 1.")
        counts = np.bincount(unit_ids[mask], minlength=9).astype(float)[:9]
        features.extend((counts / max(counts.sum(), 1.0)).tolist())
        team_positions = positions[mask]
        center = team_positions.mean(axis=0)
        team_centers.append(center)
        spread = float(np.linalg.norm(team_positions - center, axis=1).mean())
        features.extend(
            [
                len(team_positions) / 10.0,
                center[0] / max(width, 1.0),
                center[1] / max(height, 1.0),
                spread / diagonal,
                float(np.mean(health[mask])) / 500.0,
                float(np.mean(np.abs(damage[mask]))) / 100.0,
                float(np.mean(speed[mask])) / 2.0,
            ]
        )

    features.append(float(np.linalg.norm(team_centers[0] - team_centers[1])) / diagonal)
    zones = task["zone_scenario"]
    zone_types = np.asarray(zones["zone_type"], dtype=int).reshape(-1)
    features.append(float(zones["n_zone"]) / 4.0)
    for zone_type in (1, 2, 3):
        features.append(float(np.count_nonzero(zone_types == zone_type)) / 4.0)
    if int(zones["n_zone"]) > 0:
        axes = np.asarray(zones["axes"], dtype=float).reshape(-1, 2)
        coverage = float(
            np.sum(math.pi * axes[:, 0] * axes[:, 1]) / max(width * height, 1.0)
        )
    else:
        coverage = 0.0
    features.append(min(coverage, 2.0) / 2.0)
    result = np.asarray(features, dtype=float)
    if result.shape != (38,):
        raise AssertionError(f"Expected a 38-D signature, got {result.shape}.")
    return result


def canonical_parameter_signature(task: dict[str, Any]) -> np.ndarray:
    """Make the signature invariant to map symmetries and team labels."""

    variants = [
        parameter_signature(
            task,
            mirror_x=mirror_x,
            mirror_y=mirror_y,
            flip_teams=flip_teams,
        )
        for mirror_x, mirror_y, flip_teams in product((False, True), repeat=3)
    ]
    return min(variants, key=lambda value: tuple(np.round(value, 10).tolist()))


def signature_matrix(tasks: Iterable[dict[str, Any]]) -> np.ndarray:
    return np.stack([canonical_parameter_signature(task) for task in tasks])


def parameter_distance_matrix(first: np.ndarray, second: np.ndarray | None = None) -> np.ndarray:
    """Vectorized form of the project's group-weighted parameter distance."""

    second = first if second is None else second
    distances = np.zeros((len(first), len(second)), dtype=float)
    for indexes, weight in GROUPS:
        delta = first[:, None, indexes] - second[None, :, indexes]
        distances += weight * np.sqrt(np.mean(np.square(delta), axis=2))
    return distances


def weighted_linear_features(signatures: np.ndarray) -> np.ndarray:
    """Linear weighting used only for effective-dimension diagnostics/PCA init."""

    weighted = np.zeros_like(signatures)
    for indexes, weight in GROUPS:
        weighted[:, indexes] = signatures[:, indexes] * math.sqrt(weight / len(indexes))
    return weighted


def conditional_probabilities(distances_squared: np.ndarray, perplexity: float) -> np.ndarray:
    """Binary-search Gaussian bandwidths to obtain the requested perplexity."""

    n_samples = len(distances_squared)
    target_entropy = math.log(min(perplexity, n_samples - 1.0))
    probabilities = np.zeros_like(distances_squared)
    for row in range(n_samples):
        mask = np.arange(n_samples) != row
        values = distances_squared[row, mask]
        beta = 1.0
        beta_min = -math.inf
        beta_max = math.inf
        row_probabilities = np.empty_like(values)
        for _ in range(60):
            row_probabilities = np.exp(-values * beta)
            total = max(float(row_probabilities.sum()), 1e-300)
            entropy = math.log(total) + beta * float(np.dot(values, row_probabilities)) / total
            difference = entropy - target_entropy
            if abs(difference) < 1e-5:
                break
            if difference > 0.0:
                beta_min = beta
                beta = beta * 2.0 if math.isinf(beta_max) else (beta + beta_max) / 2.0
            else:
                beta_max = beta
                beta = beta / 2.0 if math.isinf(beta_min) else (beta + beta_min) / 2.0
        probabilities[row, mask] = row_probabilities / max(row_probabilities.sum(), 1e-300)

    probabilities = (probabilities + probabilities.T) / (2.0 * n_samples)
    probabilities = np.maximum(probabilities, 1e-12)
    probabilities /= probabilities.sum()
    return probabilities


def tsne_embedding(
    signatures: np.ndarray,
    *,
    random_seed: int = RANDOM_SEED,
    perplexity: float = PERPLEXITY,
    n_iter: int = N_ITER,
) -> tuple[np.ndarray, float]:
    """Exact O(n^2) t-SNE suitable for the few hundred scenes used here."""

    n_samples = len(signatures)
    distances = parameter_distance_matrix(signatures)
    probabilities = conditional_probabilities(np.square(distances), perplexity)

    linear = weighted_linear_features(signatures)
    centered = linear - linear.mean(axis=0, keepdims=True)
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    embedding = centered @ right_vectors[:2].T
    scale = max(float(np.std(embedding)), 1e-12)
    embedding = embedding / scale * 1e-4
    rng = np.random.default_rng(random_seed)
    embedding += rng.normal(0.0, 1e-6, size=embedding.shape)

    velocity = np.zeros_like(embedding)
    gains = np.ones_like(embedding)
    learning_rate = 100.0
    for iteration in range(n_iter):
        squared_norm = np.sum(np.square(embedding), axis=1)
        low_distances = (
            squared_norm[:, None] + squared_norm[None, :] - 2.0 * embedding @ embedding.T
        )
        low_distances = np.maximum(low_distances, 0.0)
        numerator = 1.0 / (1.0 + low_distances)
        np.fill_diagonal(numerator, 0.0)
        q_values = np.maximum(numerator / max(float(numerator.sum()), 1e-300), 1e-12)
        p_values = probabilities * 12.0 if iteration < 250 else probabilities
        affinities = (p_values - q_values) * numerator
        gradient = 4.0 * (
            affinities.sum(axis=1)[:, None] * embedding - affinities @ embedding
        )

        momentum = 0.5 if iteration < 250 else 0.8
        gains = np.where(np.sign(gradient) != np.sign(velocity), gains + 0.2, gains * 0.8)
        gains = np.maximum(gains, 0.01)
        velocity = momentum * velocity - learning_rate * gains * gradient
        embedding += velocity
        embedding -= embedding.mean(axis=0, keepdims=True)

    squared_norm = np.sum(np.square(embedding), axis=1)
    low_distances = np.maximum(
        squared_norm[:, None] + squared_norm[None, :] - 2.0 * embedding @ embedding.T,
        0.0,
    )
    numerator = 1.0 / (1.0 + low_distances)
    np.fill_diagonal(numerator, 0.0)
    q_values = np.maximum(numerator / max(float(numerator.sum()), 1e-300), 1e-12)
    kl_divergence = float(np.sum(probabilities * np.log(probabilities / q_values)))
    return embedding, kl_divergence


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def profile_counts(tasks: list[dict[str, Any]]) -> dict[str, int]:
    roster_profiles: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    team_size_profiles: set[tuple[int, int]] = set()
    zone_profiles: set[tuple[int, int, int, int]] = set()
    for task in tasks:
        scenario = task["scenario"]
        teams = np.asarray(scenario["teams"], dtype=int).reshape(-1)
        unit_ids = np.asarray(scenario["unit_ids"], dtype=int).reshape(-1)
        team_counts = [tuple(np.bincount(unit_ids[teams == team], minlength=9)[:9]) for team in (0, 1)]
        roster_profiles.add(tuple(sorted(team_counts)))
        team_size_profiles.add(tuple(sorted((int(np.sum(teams == 0)), int(np.sum(teams == 1))))))
        zone_types = np.asarray(task["zone_scenario"]["zone_type"], dtype=int).reshape(-1)
        zone_profiles.add(
            (
                int(task["zone_scenario"]["n_zone"]),
                int(np.count_nonzero(zone_types == 1)),
                int(np.count_nonzero(zone_types == 2)),
                int(np.count_nonzero(zone_types == 3)),
            )
        )
    return {
        "unique_roster_profiles": len(roster_profiles),
        "unique_team_size_profiles": len(team_size_profiles),
        "unique_zone_type_profiles": len(zone_profiles),
    }


def within_set_metrics(tasks: list[dict[str, Any]], signatures: np.ndarray) -> dict[str, Any]:
    distances = parameter_distance_matrix(signatures)
    upper = distances[np.triu_indices(len(signatures), k=1)]
    nearest = np.min(np.where(np.eye(len(signatures), dtype=bool), np.inf, distances), axis=1)
    linear = weighted_linear_features(signatures)
    centered = linear - linear.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    variances = np.square(singular_values) / max(len(signatures) - 1, 1)
    effective_dimension = float(np.square(variances.sum()) / max(np.square(variances).sum(), 1e-300))
    rounded_signatures = {tuple(np.round(row, 8)) for row in signatures}
    return {
        "n_scenes": len(tasks),
        "unique_parameter_signatures_rounded_8": len(rounded_signatures),
        "pairwise_parameter_distance": percentile_summary(upper),
        "nearest_neighbor_parameter_distance": percentile_summary(nearest),
        "pairs_below_0_04": int(np.count_nonzero(upper < PARAMETER_DUPLICATE_THRESHOLD)),
        "effective_dimension_participation_ratio": effective_dimension,
        **profile_counts(tasks),
    }


def cross_set_metrics(
    first_name: str,
    first: np.ndarray,
    second_name: str,
    second: np.ndarray,
) -> dict[str, Any]:
    distances = parameter_distance_matrix(first, second)
    first_to_second = np.min(distances, axis=1)
    second_to_first = np.min(distances, axis=0)
    return {
        f"{first_name}_to_nearest_{second_name}": percentile_summary(first_to_second),
        f"{second_name}_to_nearest_{first_name}": percentile_summary(second_to_first),
        "cross_pairs_below_0_04": int(np.count_nonzero(distances < PARAMETER_DUPLICATE_THRESHOLD)),
        "cross_exact_signature_pairs": int(np.count_nonzero(distances < 1e-12)),
    }


COLORS = {
    "original": (52, 73, 94, 205),
    "tasks1": (238, 123, 48, 190),
    "additional": (40, 142, 166, 205),
}


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size=size)


def draw_marker(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    *,
    color: tuple[int, int, int, int],
    marker: str,
    radius: int,
) -> None:
    if marker == "triangle":
        draw.polygon(
            [(x, y - radius), (x - radius, y + radius), (x + radius, y + radius)],
            fill=color,
            outline=(255, 255, 255, 210),
            width=1,
        )
    elif marker == "diamond":
        draw.polygon(
            [(x, y - radius), (x - radius, y), (x, y + radius), (x + radius, y)],
            fill=color,
            outline=(255, 255, 255, 210),
            width=1,
        )
    else:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline=(255, 255, 255, 210),
            width=1,
        )


def render_plot(
    output_path: Path,
    embedding: np.ndarray,
    groups: list[dict[str, Any]],
    *,
    title: str,
    subtitle: str,
    kl_divergence: float,
) -> None:
    width, height = 1800, 1300
    image = Image.new("RGB", (width, height), (247, 249, 252))
    draw = ImageDraw.Draw(image, "RGBA")

    plot_left, plot_top, plot_right, plot_bottom = 150, 190, 1660, 1120
    draw.rounded_rectangle(
        (plot_left, plot_top, plot_right, plot_bottom),
        radius=18,
        fill=(255, 255, 255, 255),
        outline=(205, 214, 224, 255),
        width=2,
    )

    draw.text((90, 55), title, fill=(25, 35, 48, 255), font=font(46, bold=True))
    draw.text((92, 117), subtitle, fill=(89, 103, 119, 255), font=font(24))

    low = np.percentile(embedding, 1, axis=0)
    high = np.percentile(embedding, 99, axis=0)
    span = np.maximum(high - low, 1e-9)
    low -= span * 0.10
    high += span * 0.10

    for step in range(1, 5):
        x = plot_left + (plot_right - plot_left) * step / 5
        y = plot_top + (plot_bottom - plot_top) * step / 5
        draw.line((x, plot_top, x, plot_bottom), fill=(222, 228, 235, 180), width=1)
        draw.line((plot_left, y, plot_right, y), fill=(222, 228, 235, 180), width=1)

    def map_point(point: np.ndarray) -> tuple[float, float]:
        normalized = np.clip((point - low) / np.maximum(high - low, 1e-9), 0.0, 1.0)
        return (
            plot_left + normalized[0] * (plot_right - plot_left),
            plot_bottom - normalized[1] * (plot_bottom - plot_top),
        )

    for group in groups:
        for index in group["indexes"]:
            x, y = map_point(embedding[index])
            draw_marker(
                draw,
                x,
                y,
                color=group["color"],
                marker=group["marker"],
                radius=group.get("radius", 7),
            )

    legend_x, legend_y = 1130, 215
    legend_width = 500
    legend_height = 62 + 55 * len(groups)
    draw.rounded_rectangle(
        (legend_x, legend_y, legend_x + legend_width, legend_y + legend_height),
        radius=14,
        fill=(255, 255, 255, 235),
        outline=(207, 215, 224, 240),
        width=2,
    )
    draw.text((legend_x + 24, legend_y + 17), "Scene sets", fill=(38, 50, 64, 255), font=font(23, bold=True))
    for row, group in enumerate(groups):
        cy = legend_y + 78 + row * 55
        draw_marker(
            draw,
            legend_x + 34,
            cy,
            color=group["color"],
            marker=group["marker"],
            radius=8,
        )
        draw.text(
            (legend_x + 58, cy - 14),
            f"{group['label']}  (n={len(group['indexes'])})",
            fill=(45, 57, 70, 255),
            font=font(22),
        )

    draw.text(
        ((plot_left + plot_right) / 2 - 48, 1163),
        "t-SNE 1",
        fill=(79, 92, 108, 255),
        font=font(23),
    )
    vertical_label = Image.new("RGBA", (180, 55), (0, 0, 0, 0))
    vertical_draw = ImageDraw.Draw(vertical_label)
    vertical_draw.text((0, 12), "t-SNE 2", fill=(79, 92, 108, 255), font=font(23))
    vertical_label = vertical_label.rotate(90, expand=True)
    image.paste(vertical_label, (43, 590), vertical_label)

    note = (
        f"Canonical 38-D TABX parameter signature | project-weighted distance | "
        f"perplexity={PERPLEXITY:g} | seed={RANDOM_SEED} | KL={kl_divergence:.3f}"
    )
    draw.text((92, 1246), note, fill=(108, 120, 134, 255), font=font(18))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, optimize=True)


def main() -> None:
    original_tasks, original_labels, original_counts = load_original_scenes()
    tasks1_bank = read_json(OUTPUTS / "tasks1_balanced.json")
    balance_bank = read_json(OUTPUTS / "balance_tasks.json")
    tasks1_tasks = tasks1_bank["tasks"]
    balance_tasks = balance_bank["tasks"]

    original_signatures = signature_matrix(original_tasks)
    tasks1_signatures = signature_matrix(tasks1_tasks)
    balance_signatures = signature_matrix(balance_tasks)

    tasks1_hashes = {task["metadata"]["canonical_hash"] for task in tasks1_tasks}
    balance_hashes = [task["metadata"]["canonical_hash"] for task in balance_tasks]
    subset_mask = np.asarray([value in tasks1_hashes for value in balance_hashes], dtype=bool)
    additional_mask = ~subset_mask
    if int(subset_mask.sum()) != len(tasks1_tasks):
        raise AssertionError("balance_tasks must contain every tasks1_balanced task exactly once.")

    plot_specs = [
        {
            "output": OUTPUTS / "tsne_original_vs_tasks1_balanced.png",
            "signatures": np.vstack([original_signatures, tasks1_signatures]),
            "groups": [
                {
                    "indexes": list(range(len(original_tasks))),
                    "label": "Original composed + challenge",
                    "color": COLORS["original"],
                    "marker": "triangle",
                    "radius": 9,
                },
                {
                    "indexes": list(range(len(original_tasks), len(original_tasks) + len(tasks1_tasks))),
                    "label": "tasks1_balanced",
                    "color": COLORS["tasks1"],
                    "marker": "circle",
                    "radius": 7,
                },
            ],
            "title": "Original TABX scenes vs tasks1_balanced",
            "subtitle": "20 composed scenes + 13 challenge scenes compared with 276 balanced generated tasks",
        },
        {
            "output": OUTPUTS / "tsne_original_vs_balance_tasks.png",
            "signatures": np.vstack([original_signatures, balance_signatures]),
            "groups": [
                {
                    "indexes": list(range(len(original_tasks))),
                    "label": "Original composed + challenge",
                    "color": COLORS["original"],
                    "marker": "triangle",
                    "radius": 9,
                },
                {
                    "indexes": list(range(len(original_tasks), len(original_tasks) + len(balance_tasks))),
                    "label": "balance_tasks",
                    "color": COLORS["additional"],
                    "marker": "circle",
                    "radius": 7,
                },
            ],
            "title": "Original TABX scenes vs balance_tasks",
            "subtitle": "20 composed scenes + 13 challenge scenes compared with the 376-task merged bank",
        },
        {
            "output": OUTPUTS / "tsne_tasks1_balanced_vs_balance_tasks.png",
            "signatures": balance_signatures,
            "groups": [
                {
                    "indexes": np.flatnonzero(subset_mask).tolist(),
                    "label": "tasks1_balanced subset",
                    "color": COLORS["tasks1"],
                    "marker": "circle",
                    "radius": 7,
                },
                {
                    "indexes": np.flatnonzero(additional_mask).tolist(),
                    "label": "Added balanced_diverse",
                    "color": COLORS["additional"],
                    "marker": "diamond",
                    "radius": 8,
                },
            ],
            "title": "tasks1_balanced vs balance_tasks",
            "subtitle": "The merged bank contains all 276 tasks1 scenes plus 100 additional balanced-diverse scenes",
        },
    ]

    embedding_reports: dict[str, Any] = {}
    for spec in plot_specs:
        embedding, kl_divergence = tsne_embedding(spec["signatures"])
        render_plot(
            spec["output"],
            embedding,
            spec["groups"],
            title=spec["title"],
            subtitle=spec["subtitle"],
            kl_divergence=kl_divergence,
        )
        embedding_reports[spec["output"].name] = {
            "n_points": len(embedding),
            "kl_divergence": kl_divergence,
            "perplexity": PERPLEXITY,
            "random_seed": RANDOM_SEED,
            "iterations": N_ITER,
        }

    additional_tasks = [task for task, keep in zip(balance_tasks, additional_mask) if keep]
    additional_signatures = balance_signatures[additional_mask]
    metrics = {
        "method": {
            "feature": "canonical 38-D TABX parameter signature",
            "invariances": ["mirror_x", "mirror_y", "rotate_180", "team_label_swap"],
            "distance": "project group-weighted parameter distance",
            "parameter_duplicate_threshold": PARAMETER_DUPLICATE_THRESHOLD,
            "tsne_note": "Each comparison is fit separately; axes/coordinates are not comparable across plots.",
        },
        "original_scene_inventory": original_counts,
        "original_scene_labels": original_labels,
        "within_set_diversity": {
            "original": within_set_metrics(original_tasks, original_signatures),
            "tasks1_balanced": within_set_metrics(tasks1_tasks, tasks1_signatures),
            "balance_tasks": within_set_metrics(balance_tasks, balance_signatures),
            "balance_tasks_added_100": within_set_metrics(additional_tasks, additional_signatures),
        },
        "cross_set_comparison": {
            "original_vs_tasks1_balanced": cross_set_metrics(
                "original", original_signatures, "tasks1_balanced", tasks1_signatures
            ),
            "original_vs_balance_tasks": cross_set_metrics(
                "original", original_signatures, "balance_tasks", balance_signatures
            ),
            "tasks1_balanced_vs_added_100": cross_set_metrics(
                "tasks1_balanced", tasks1_signatures, "added_100", additional_signatures
            ),
        },
        "subset_relationship": {
            "tasks1_balanced_count": len(tasks1_tasks),
            "tasks1_hashes_found_in_balance_tasks": int(subset_mask.sum()),
            "new_tasks_in_balance_tasks": int(additional_mask.sum()),
        },
        "embeddings": embedding_reports,
    }
    metrics_path = OUTPUTS / "tsne_diversity_metrics.json"
    with metrics_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")

    print(json.dumps({
        "plots": [str(spec["output"]) for spec in plot_specs],
        "metrics": str(metrics_path),
        "original_counts": original_counts,
        "tasks1_count": len(tasks1_tasks),
        "balance_count": len(balance_tasks),
        "added_count": int(additional_mask.sum()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
