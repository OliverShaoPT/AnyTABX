"""Create a paper-style diversity comparison with data-independent scaling.

The 827-D descriptor layout and every normalization constant are fixed before
loading the evaluated task banks. Original-33 is used only as the semantic
anchor set. Balanced-v1 and SampleV2 are both treated as evaluated datasets.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

from plot_environment_randomness_tsne import (
    GROUP_WEIGHTS,
    DescriptorConfig,
    DescriptorLayout,
    descriptor_matrix,
    joint_probabilities,
    neighborhood_trustworthiness,
    student_probabilities,
    weighted_linear_features,
)
from plot_tsne_diversity import (
    draw_marker,
    font,
    load_original_scenes,
    read_json,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
BALANCED_PATH = ROOT / "task_files" / "balanced_tasks_v1.json"
SAMPLE_PATH = OUTPUTS / "sampleV2_tasks.json"
PLOT_PATH = OUTPUTS / "paper_semantic_diversity_comparison.png"
METRICS_PATH = OUTPUTS / "paper_semantic_diversity_comparison_metrics.json"
RAREFACTION_PATH = OUTPUTS / "paper_semantic_diversity_rarefaction.csv"

DESCRIPTOR_CONFIG = DescriptorConfig(
    max_units_per_team=20,
    max_zones=20,
    attack_type_values=(0, 1),
)
TSNE_PERPLEXITY = 20.0
TSNE_ITERATIONS = 1_200
RANDOM_SEED = 42
RAREFACTION_REPEATS = 500
RAREFACTION_SIZES = (20, 40, 60, 80, 100, 120, 150)

ORIGINAL_COLOR = (37, 104, 174, 238)
BALANCED_COLOR = (50, 155, 119, 195)
SAMPLE_COLOR = (214, 78, 100, 225)
GRID_COLOR = (218, 225, 233, 230)
TEXT_COLOR = (38, 50, 64, 255)
MUTED_COLOR = (91, 104, 119, 255)

# These constants come from the task definition, not from evaluated samples.
SEMANTIC_BOUNDS = {
    "max_units_per_team": 20.0,
    "max_zones": 20.0,
    "base_map_width": 121.0,
    "base_map_height": 78.0,
    "max_map_scale": 1.50,
    "max_stat_scale": 1.55,
    "template_health_max": 685.0,
    "template_abs_damage_max": 80.0,
    "template_cooldown_max": 10.0,
    "template_attack_range_max": 40.0,
    "template_speed_max": 1.4,
    "template_sight_angle_max": math.pi / 2.0,
    "template_body_radius_max": 4.25,
    "template_body_weight_max": 50.0,
    "lava_effect_max": 25.0,
    "swamp_effect_max": 1.0,
}


def semantic_normalize(
    values: np.ndarray,
    layout: DescriptorLayout,
) -> np.ndarray:
    """Apply a fixed, zero-preserving semantic normalization."""

    normalized = np.asarray(values, dtype=float).copy()

    roster = normalized[:, layout.group_slices["roster"]]
    roster /= SEMANTIC_BOUNDS["max_units_per_team"]

    # Geometry is already data-independent: positions and axes are divided by
    # map width/height, headings are sin/cos, and presence is binary.
    geometry = normalized[:, layout.group_slices["initial_geometry"]]
    geometry.reshape(len(values), 2, 20, 5)

    attributes = normalized[
        :,
        layout.group_slices["unit_attributes"],
    ].reshape(len(values), 2, 20, 10)
    stat_scale = SEMANTIC_BOUNDS["max_stat_scale"]
    attribute_scales = np.asarray(
        [
            SEMANTIC_BOUNDS["template_health_max"] * stat_scale,
            SEMANTIC_BOUNDS["template_abs_damage_max"] * stat_scale,
            SEMANTIC_BOUNDS["template_cooldown_max"],
            SEMANTIC_BOUNDS["template_attack_range_max"],
            1.0,
            1.0,
            SEMANTIC_BOUNDS["template_speed_max"] * stat_scale,
            SEMANTIC_BOUNDS["template_sight_angle_max"],
            SEMANTIC_BOUNDS["template_body_radius_max"],
            SEMANTIC_BOUNDS["template_body_weight_max"],
        ],
        dtype=float,
    )
    attributes /= attribute_scales

    terrain = normalized[
        :,
        layout.group_slices["terrain_and_map"],
    ]
    max_width = (
        SEMANTIC_BOUNDS["base_map_width"]
        * SEMANTIC_BOUNDS["max_map_scale"]
    )
    max_height = (
        SEMANTIC_BOUNDS["base_map_height"]
        * SEMANTIC_BOUNDS["max_map_scale"]
    )
    terrain[:, :7] /= np.asarray(
        [
            28.0,
            18.0,
            max_width,
            max_height,
            max_width / 2.0,
            max_height / 2.0,
            SEMANTIC_BOUNDS["max_zones"],
        ],
        dtype=float,
    )
    zone_slots = terrain[:, 7:].reshape(len(values), 20, 10)
    zone_types = np.argmax(zone_slots[..., 1:5], axis=2)
    zone_present = zone_slots[..., 0] > 0.5
    effect_scale = np.ones(zone_types.shape, dtype=float)
    effect_scale[(zone_types == 1) & zone_present] = (
        SEMANTIC_BOUNDS["lava_effect_max"]
    )
    effect_scale[(zone_types == 3) & zone_present] = (
        SEMANTIC_BOUNDS["swamp_effect_max"]
    )
    zone_slots[..., 9] /= effect_scale

    if normalized.shape[1] != 827:
        raise AssertionError(
            f"Expected fixed 827-D descriptor, got {normalized.shape[1]}."
        )
    if not np.all(np.isfinite(normalized)):
        raise ValueError("Semantic normalization produced non-finite values.")
    return normalized


def all_group_indexes(
    layout: DescriptorLayout,
) -> dict[str, np.ndarray]:
    return {
        name: np.arange(section.start, section.stop, dtype=int)
        for name, section in layout.group_slices.items()
    }


def group_distance_matrix(
    first: np.ndarray,
    second: np.ndarray,
    layout: DescriptorLayout,
) -> np.ndarray:
    """Compute the pre-registered weighted sum of group RMS distances."""

    total = np.zeros((len(first), len(second)), dtype=float)
    for name, weight in GROUP_WEIGHTS.items():
        section = layout.group_slices[name]
        left = first[:, section]
        right = second[:, section]
        squared = (
            np.sum(np.square(left), axis=1)[:, None]
            + np.sum(np.square(right), axis=1)[None, :]
            - 2.0 * left @ right.T
        )
        group_rms = np.sqrt(
            np.maximum(squared, 0.0) / max(left.shape[1], 1)
        )
        total += weight * group_rms
    return total


def exact_tsne_from_distances(
    distances: np.ndarray,
    initialization_features: np.ndarray,
    *,
    perplexity: float,
    random_seed: int,
    n_iter: int,
) -> tuple[np.ndarray, float]:
    """Run conventional exact t-SNE using a fixed precomputed metric."""

    probabilities = joint_probabilities(
        np.square(distances),
        perplexity,
    )
    centered = (
        initialization_features
        - initialization_features.mean(axis=0, keepdims=True)
    )
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    embedding = centered @ right_vectors[:2].T
    embedding /= max(float(np.std(embedding)), 1e-12)
    embedding *= 1e-4
    embedding += np.random.default_rng(random_seed).normal(
        0.0,
        1e-6,
        embedding.shape,
    )

    velocity = np.zeros_like(embedding)
    gains = np.ones_like(embedding)
    early_exaggeration = 12.0
    learning_rate = max(
        len(embedding) / (early_exaggeration * 4.0),
        50.0,
    )
    for iteration in range(n_iter):
        low_probabilities, numerator = student_probabilities(embedding)
        targets = (
            probabilities * early_exaggeration
            if iteration < 250
            else probabilities
        )
        affinities = (targets - low_probabilities) * numerator
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
            * np.log(
                probabilities[mask] / low_probabilities[mask]
            )
        )
    )
    return embedding, kl_divergence


def pca_projection(
    features: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float]]:
    centered = features - features.mean(axis=0, keepdims=True)
    _, singular_values, right_vectors = np.linalg.svd(
        centered,
        full_matrices=False,
    )
    projection = centered @ right_vectors[:2].T
    variance = np.square(singular_values)
    ratio = variance / max(float(variance.sum()), 1e-300)
    return projection, (float(ratio[0]), float(ratio[1]))


def percentile_record(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
    }


def jackknife_interval(estimates: np.ndarray) -> dict[str, float]:
    values = np.asarray(estimates, dtype=float)
    mean = float(np.mean(values))
    standard_error = math.sqrt(
        (len(values) - 1)
        / len(values)
        * float(np.sum(np.square(values - mean)))
    )
    return {
        "leave_one_anchor_mean": mean,
        "standard_error": standard_error,
        "ci95_low": mean - 1.96 * standard_error,
        "ci95_high": mean + 1.96 * standard_error,
    }


def dataset_metrics(
    candidate_to_original: np.ndarray,
    candidate_internal: np.ndarray,
    *,
    original_thresholds: dict[str, float],
) -> dict[str, Any]:
    novelty = np.min(candidate_to_original, axis=1)
    internal = candidate_internal.copy()
    np.fill_diagonal(internal, math.inf)
    internal_nn = np.min(internal, axis=1)
    upper = candidate_internal[np.triu_indices(len(candidate_internal), k=1)]
    original_to_candidate = np.min(candidate_to_original, axis=0)

    leave_one_out = []
    for anchor in range(candidate_to_original.shape[1]):
        kept = np.delete(candidate_to_original, anchor, axis=1)
        leave_one_out.append(float(np.median(np.min(kept, axis=1))))

    return {
        "novelty_candidate_to_original": percentile_record(novelty),
        "internal_nearest_neighbor": percentile_record(internal_nn),
        "internal_pairwise_dispersion": percentile_record(upper),
        "original_to_candidate_coverage_distance": percentile_record(
            original_to_candidate
        ),
        "fraction_beyond_original_loo": {
            key: float(np.mean(novelty > threshold))
            for key, threshold in original_thresholds.items()
        },
        "novelty_anchor_jackknife": jackknife_interval(
            np.asarray(leave_one_out)
        ),
    }


def rarefaction(
    name: str,
    candidate_to_original: np.ndarray,
    candidate_internal: np.ndarray,
    *,
    coverage_threshold: float,
    seed: int,
) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(seed)
    result: list[dict[str, float | int | str]] = []
    for size in RAREFACTION_SIZES:
        if size > len(candidate_internal):
            continue
        records = {
            "novelty": [],
            "internal_nn": [],
            "pairwise": [],
            "coverage_distance": [],
            "anchor_coverage_rate": [],
        }
        for _ in range(RAREFACTION_REPEATS):
            subset = rng.choice(
                len(candidate_internal),
                size=size,
                replace=False,
            )
            to_original = candidate_to_original[subset]
            within = candidate_internal[np.ix_(subset, subset)].copy()
            np.fill_diagonal(within, math.inf)
            records["novelty"].append(
                float(np.median(np.min(to_original, axis=1)))
            )
            records["internal_nn"].append(
                float(np.median(np.min(within, axis=1)))
            )
            upper = within[np.triu_indices(size, k=1)]
            records["pairwise"].append(float(np.median(upper)))
            original_to_subset = np.min(to_original, axis=0)
            records["coverage_distance"].append(
                float(np.median(original_to_subset))
            )
            records["anchor_coverage_rate"].append(
                float(np.mean(original_to_subset <= coverage_threshold))
            )
        for metric, values in records.items():
            array = np.asarray(values, dtype=float)
            result.append(
                {
                    "dataset": name,
                    "sample_size": size,
                    "metric": metric,
                    "mean": float(np.mean(array)),
                    "ci95_low": float(np.percentile(array, 2.5)),
                    "ci95_high": float(np.percentile(array, 97.5)),
                }
            )
    return result


def task_structure_metrics(
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    combinations = Counter()
    unit_types = Counter()
    zone_types = Counter()
    strata = Counter()
    for task in tasks:
        teams = np.asarray(
            task["scenario"]["teams"],
            dtype=int,
        ).reshape(-1)
        ally = int(np.count_nonzero(teams == 0))
        enemy = int(np.count_nonzero(teams == 1))
        zones = int(task["zone_scenario"]["n_zone"])
        combinations[(ally, enemy, zones)] += 1
        unit_types.update(
            np.asarray(
                task["scenario"]["unit_ids"],
                dtype=int,
            ).reshape(-1).tolist()
        )
        zone_types.update(
            np.asarray(
                task["zone_scenario"]["zone_type"],
                dtype=int,
            ).reshape(-1).tolist()
        )
        if max(ally, enemy) > 10 and zones > 10:
            strata["joint_ood"] += 1
        elif max(ally, enemy) >= 16:
            strata["xlarge_team"] += 1
        elif max(ally, enemy) > 10:
            strata["large_team"] += 1
        elif zones >= 16:
            strata["xhigh_zone"] += 1
        elif zones > 10:
            strata["high_zone"] += 1
        else:
            strata["legacy_scale"] += 1

    def normalized_entropy(counter: Counter[int]) -> float:
        counts = np.asarray(list(counter.values()), dtype=float)
        if len(counts) <= 1:
            return 0.0
        probabilities = counts / counts.sum()
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        return entropy / math.log(len(counts))

    return {
        "unique_ally_enemy_zone_combinations": len(combinations),
        "unit_type_normalized_entropy": normalized_entropy(unit_types),
        "zone_type_normalized_entropy": normalized_entropy(zone_types),
        "scale_strata": dict(sorted(strata.items())),
    }


def dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: tuple[int, int, int, int],
    width: int = 2,
    dash: float = 9.0,
    gap: float = 6.0,
) -> None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return
    position = 0.0
    while position < length:
        finish = min(position + dash, length)
        draw.line(
            (
                start[0] + dx * position / length,
                start[1] + dy * position / length,
                start[0] + dx * finish / length,
                start[1] + dy * finish / length,
            ),
            fill=fill,
            width=width,
        )
        position += dash + gap


def panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    title: str,
    subtitle: str,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    draw.rounded_rectangle(
        box,
        radius=18,
        fill=(255, 255, 255, 255),
        outline=(205, 214, 224, 255),
        width=2,
    )
    draw.text(
        (left + 24, top + 19),
        title,
        fill=TEXT_COLOR,
        font=font(27, bold=True),
    )
    draw.text(
        (left + 25, top + 58),
        subtitle,
        fill=MUTED_COLOR,
        font=font(18),
    )
    return left + 100, top + 105, right - 28, bottom - 66


def chart_axes(
    draw: ImageDraw.ImageDraw,
    chart: tuple[int, int, int, int],
    *,
    x_label: str,
    y_label: str,
) -> None:
    left, top, right, bottom = chart
    for step in range(5):
        x = left + (right - left) * step / 4
        y = top + (bottom - top) * step / 4
        draw.line((x, top, x, bottom), fill=GRID_COLOR, width=1)
        draw.line((left, y, right, y), fill=GRID_COLOR, width=1)
    draw.line((left, bottom, right, bottom), fill=(136, 149, 164, 255), width=2)
    draw.line((left, top, left, bottom), fill=(136, 149, 164, 255), width=2)
    draw.text(
        ((left + right) / 2 - 60, bottom + 27),
        x_label,
        fill=MUTED_COLOR,
        font=font(18),
    )
    label = Image.new("RGBA", (260, 42), (0, 0, 0, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((0, 7), y_label, fill=MUTED_COLOR, font=font(18))
    label = label.rotate(90, expand=True)
    draw._image.paste(
        label,
        (left - 95, int((top + bottom) / 2 - 75)),
        label,
    )


def bounds_with_margin(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = np.min(values, axis=0)
    high = np.max(values, axis=0)
    span = np.maximum(high - low, 1e-9)
    return low - 0.08 * span, high + 0.08 * span


def map_xy(
    point: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    chart: tuple[int, int, int, int],
) -> tuple[float, float]:
    left, top, right, bottom = chart
    unit = (point - low) / np.maximum(high - low, 1e-12)
    return (
        left + unit[0] * (right - left),
        bottom - unit[1] * (bottom - top),
    )


def scatter_panel(
    draw: ImageDraw.ImageDraw,
    chart: tuple[int, int, int, int],
    values: np.ndarray,
    groups: list[dict[str, Any]],
    *,
    x_label: str,
    y_label: str,
) -> None:
    chart_axes(draw, chart, x_label=x_label, y_label=y_label)
    low, high = bounds_with_margin(values)
    for group in groups:
        for index in group["indexes"]:
            x, y = map_xy(values[index], low, high, chart)
            draw_marker(
                draw,
                x,
                y,
                color=group["color"],
                marker=group["marker"],
                radius=group["radius"],
            )


def line_chart(
    draw: ImageDraw.ImageDraw,
    chart: tuple[int, int, int, int],
    series: list[dict[str, Any]],
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    x_label: str,
    y_label: str,
    threshold: float | None = None,
) -> None:
    chart_axes(draw, chart, x_label=x_label, y_label=y_label)
    left, top, right, bottom = chart

    def map_value(x: float, y: float) -> tuple[float, float]:
        px = left + (x - x_range[0]) / max(
            x_range[1] - x_range[0],
            1e-12,
        ) * (right - left)
        py = bottom - (y - y_range[0]) / max(
            y_range[1] - y_range[0],
            1e-12,
        ) * (bottom - top)
        return px, py

    if threshold is not None:
        x, _ = map_value(threshold, y_range[0])
        dashed_line(
            draw,
            (x, top),
            (x, bottom),
            fill=(108, 120, 134, 190),
            width=2,
        )
        draw.text(
            (x + 6, top + 4),
            "Original LOO P75",
            fill=MUTED_COLOR,
            font=font(15),
        )

    for item in series:
        x_values = np.asarray(item["x"], dtype=float)
        y_values = np.asarray(item["y"], dtype=float)
        if "low" in item and "high" in item:
            low_values = np.asarray(item["low"], dtype=float)
            high_values = np.asarray(item["high"], dtype=float)
            upper = [
                map_value(float(x), float(y))
                for x, y in zip(x_values, high_values)
            ]
            lower = [
                map_value(float(x), float(y))
                for x, y in zip(
                    x_values[::-1],
                    low_values[::-1],
                )
            ]
            draw.polygon(
                upper + lower,
                fill=(*item["color"][:3], 32),
            )
        points = [
            map_value(float(x), float(y))
            for x, y in zip(x_values, y_values)
        ]
        if item.get("dashed"):
            for start, end in zip(points, points[1:]):
                dashed_line(
                    draw,
                    start,
                    end,
                    fill=item["color"],
                    width=item.get("width", 4),
                )
        else:
            draw.line(
                points,
                fill=item["color"],
                width=item.get("width", 4),
                joint="curve",
            )
        for x, y in points:
            draw.ellipse(
                (x - 4, y - 4, x + 4, y + 4),
                fill=item["color"],
            )

    def format_tick(value: float, span: float) -> str:
        if span >= 10.0:
            return f"{value:.0f}"
        if span >= 1.0:
            return f"{value:.1f}"
        return f"{value:.2f}".rstrip("0").rstrip(".")

    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]
    for step in range(5):
        x_value = x_range[0] + (x_range[1] - x_range[0]) * step / 4
        y_value = y_range[1] - (y_range[1] - y_range[0]) * step / 4
        x, _ = map_value(x_value, y_range[0])
        _, y = map_value(x_range[0], y_value)
        draw.text(
            (x - 20, bottom + 6),
            format_tick(x_value, x_span),
            fill=MUTED_COLOR,
            font=font(14),
        )
        draw.text(
            (left - 52, y - 9),
            format_tick(y_value, y_span),
            fill=MUTED_COLOR,
            font=font(14),
        )


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(values, dtype=float))
    cumulative = np.arange(1, len(ordered) + 1, dtype=float) / len(ordered)
    return ordered, cumulative


def rarefaction_series(
    records: list[dict[str, float | int | str]],
    dataset: str,
    metric: str,
) -> dict[str, np.ndarray]:
    selected = [
        row
        for row in records
        if row["dataset"] == dataset and row["metric"] == metric
    ]
    selected.sort(key=lambda row: int(row["sample_size"]))
    return {
        "x": np.asarray([row["sample_size"] for row in selected]),
        "mean": np.asarray([row["mean"] for row in selected]),
        "low": np.asarray([row["ci95_low"] for row in selected]),
        "high": np.asarray([row["ci95_high"] for row in selected]),
    }


def draw_global_legend(draw: ImageDraw.ImageDraw) -> None:
    entries = [
        ("Original 33", ORIGINAL_COLOR, "triangle", 9),
        ("Balanced-v1 (n=550)", BALANCED_COLOR, "circle", 6),
        ("SampleV2 (n=155)", SAMPLE_COLOR, "diamond", 7),
    ]
    start_x = 620
    for index, (label, color, marker, radius) in enumerate(entries):
        x = start_x + index * 430
        y = 134
        draw_marker(
            draw,
            x,
            y,
            color=color,
            marker=marker,
            radius=radius,
        )
        draw.text(
            (x + 22, y - 13),
            label,
            fill=TEXT_COLOR,
            font=font(20),
        )


def render_figure(
    pca_values: np.ndarray,
    pca_ratio: tuple[float, float],
    tsne_values: np.ndarray,
    tsne_groups: list[dict[str, Any]],
    novelty_values: dict[str, np.ndarray],
    original_p75: float,
    rarefaction_records: list[dict[str, float | int | str]],
    *,
    kl_divergence: float,
    trustworthiness: float,
) -> None:
    width, height = 2200, 1900
    image = Image.new("RGB", (width, height), (247, 249, 252))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text(
        (70, 35),
        "Data-independent semantic diversity comparison",
        fill=(24, 34, 47, 255),
        font=font(46, bold=True),
    )
    draw.text(
        (72, 91),
        (
            "Fixed 827-D semantic scale | Original-33 anchors | "
            "matched-size t-SNE and rarefaction"
        ),
        fill=MUTED_COLOR,
        font=font(23),
    )
    draw_global_legend(draw)

    first = (65, 175, 1070, 930)
    second = (1130, 175, 2135, 930)
    third = (65, 985, 1070, 1745)
    fourth = (1130, 985, 2135, 1745)

    pca_chart = panel(
        draw,
        first,
        title="A  Global coverage (PCA)",
        subtitle=(
            f"PC1 {pca_ratio[0] * 100:.1f}% | "
            f"PC2 {pca_ratio[1] * 100:.1f}% explained variance"
        ),
    )
    original_end = 33
    balanced_end = original_end + 550
    scatter_panel(
        draw,
        pca_chart,
        pca_values,
        [
            {
                "indexes": range(original_end, balanced_end),
                "color": BALANCED_COLOR,
                "marker": "circle",
                "radius": 4,
            },
            {
                "indexes": range(balanced_end, len(pca_values)),
                "color": SAMPLE_COLOR,
                "marker": "diamond",
                "radius": 5,
            },
            {
                "indexes": range(0, original_end),
                "color": ORIGINAL_COLOR,
                "marker": "triangle",
                "radius": 10,
            },
        ],
        x_label="PC1",
        y_label="PC2",
    )

    tsne_chart = panel(
        draw,
        second,
        title="B  Local structure (exact t-SNE)",
        subtitle=(
            "Balanced-v1 subsampled to n=155 | "
            f"p=20, seed=42, KL={kl_divergence:.3f}, T(10)={trustworthiness:.3f}"
        ),
    )
    scatter_panel(
        draw,
        tsne_chart,
        tsne_values,
        tsne_groups,
        x_label="t-SNE 1",
        y_label="t-SNE 2",
    )

    ecdf_chart = panel(
        draw,
        third,
        title="C  Expansion from Original-33",
        subtitle=(
            "ECDF of distance to nearest Original anchor; "
            "right-shift indicates greater novelty"
        ),
    )
    ecdf_series = []
    for name, color, dashed in (
        ("original_loo", (108, 120, 134, 235), True),
        ("balanced", BALANCED_COLOR, False),
        ("sample", SAMPLE_COLOR, False),
    ):
        x_values, y_values = ecdf(novelty_values[name])
        ecdf_series.append(
            {
                "x": x_values,
                "y": y_values,
                "color": color,
                "dashed": dashed,
                "width": 4,
            }
        )
    ecdf_max = max(
        float(np.max(values))
        for values in novelty_values.values()
    )
    line_chart(
        draw,
        ecdf_chart,
        ecdf_series,
        x_range=(0.0, ecdf_max * 1.04),
        y_range=(0.0, 1.0),
        x_label="Nearest Original distance",
        y_label="Cumulative fraction",
        threshold=original_p75,
    )
    draw.text(
        (third[0] + 690, third[1] + 65),
        "gray dashed: Original leave-one-out",
        fill=MUTED_COLOR,
        font=font(16),
    )

    left, top, right, bottom = fourth
    draw.rounded_rectangle(
        fourth,
        radius=18,
        fill=(255, 255, 255, 255),
        outline=(205, 214, 224, 255),
        width=2,
    )
    draw.text(
        (left + 24, top + 19),
        "D  Matched-size rarefaction (500 repeats)",
        fill=TEXT_COLOR,
        font=font(27, bold=True),
    )
    draw.text(
        (left + 25, top + 58),
        "Mean with 95% resampling interval",
        fill=MUTED_COLOR,
        font=font(18),
    )

    pair_balanced = rarefaction_series(
        rarefaction_records,
        "balanced",
        "pairwise",
    )
    pair_sample = rarefaction_series(
        rarefaction_records,
        "sample",
        "pairwise",
    )
    coverage_balanced = rarefaction_series(
        rarefaction_records,
        "balanced",
        "anchor_coverage_rate",
    )
    coverage_sample = rarefaction_series(
        rarefaction_records,
        "sample",
        "anchor_coverage_rate",
    )
    pair_max = max(
        float(np.max(pair_balanced["high"])),
        float(np.max(pair_sample["high"])),
    )
    upper_chart = (left + 110, top + 105, right - 28, top + 365)
    lower_chart = (left + 110, top + 445, right - 28, bottom - 64)
    line_chart(
        draw,
        upper_chart,
        [
            {
                "x": pair_balanced["x"],
                "y": pair_balanced["mean"],
                "low": pair_balanced["low"],
                "high": pair_balanced["high"],
                "color": BALANCED_COLOR,
            },
            {
                "x": pair_sample["x"],
                "y": pair_sample["mean"],
                "low": pair_sample["low"],
                "high": pair_sample["high"],
                "color": SAMPLE_COLOR,
            },
        ],
        x_range=(20.0, 150.0),
        y_range=(0.0, pair_max * 1.05),
        x_label="Matched task count",
        y_label="Pairwise dispersion",
    )
    line_chart(
        draw,
        lower_chart,
        [
            {
                "x": coverage_balanced["x"],
                "y": coverage_balanced["mean"],
                "low": coverage_balanced["low"],
                "high": coverage_balanced["high"],
                "color": BALANCED_COLOR,
            },
            {
                "x": coverage_sample["x"],
                "y": coverage_sample["mean"],
                "low": coverage_sample["low"],
                "high": coverage_sample["high"],
                "color": SAMPLE_COLOR,
            },
        ],
        x_range=(20.0, 150.0),
        y_range=(0.0, 1.0),
        x_label="Matched task count",
        y_label="Original-anchor coverage@P75",
    )

    draw.text(
        (72, 1810),
        (
            "Semantic distance = sum of weighted group RMS "
            "(roster .25, geometry .30, attributes .20, terrain/map .25). "
            "No evaluated dataset fits the feature scale."
        ),
        fill=MUTED_COLOR,
        font=font(18),
    )
    image.save(PLOT_PATH, optimize=True)


def main() -> None:
    original_tasks, _, original_counts = load_original_scenes()
    balanced_tasks = read_json(BALANCED_PATH)["tasks"]
    sample_tasks = read_json(SAMPLE_PATH)["tasks"]
    if original_counts["total_original_scenes"] != 33:
        raise ValueError("Expected exactly 33 built-in Original scenes.")
    if len(balanced_tasks) != 550 or len(sample_tasks) != 155:
        raise ValueError(
            "Expected Balanced-v1 n=550 and SampleV2 n=155 for this figure."
        )

    raw_matrices = []
    layout = None
    for tasks in (original_tasks, balanced_tasks, sample_tasks):
        raw, current_layout = descriptor_matrix(tasks, DESCRIPTOR_CONFIG)
        if layout is not None and current_layout != layout:
            raise AssertionError("Descriptor layouts do not match.")
        layout = current_layout
        raw_matrices.append(raw)
    if layout is None or layout.dimension != 827:
        raise AssertionError("Fixed descriptor must have 827 dimensions.")

    original, balanced, sample = [
        semantic_normalize(raw, layout)
        for raw in raw_matrices
    ]
    group_indexes = all_group_indexes(layout)
    all_scaled = np.vstack([original, balanced, sample])
    all_features = weighted_linear_features(all_scaled, group_indexes)
    pca_values, pca_ratio = pca_projection(all_features)

    original_distances = group_distance_matrix(
        original,
        original,
        layout,
    )
    original_loo_matrix = original_distances.copy()
    np.fill_diagonal(original_loo_matrix, math.inf)
    original_loo = np.min(original_loo_matrix, axis=1)
    original_thresholds = {
        "p50": float(np.percentile(original_loo, 50)),
        "p75": float(np.percentile(original_loo, 75)),
        "p90": float(np.percentile(original_loo, 90)),
    }

    balanced_to_original = group_distance_matrix(
        balanced,
        original,
        layout,
    )
    sample_to_original = group_distance_matrix(
        sample,
        original,
        layout,
    )
    balanced_internal = group_distance_matrix(
        balanced,
        balanced,
        layout,
    )
    sample_internal = group_distance_matrix(
        sample,
        sample,
        layout,
    )

    balanced_metrics = dataset_metrics(
        balanced_to_original,
        balanced_internal,
        original_thresholds=original_thresholds,
    )
    sample_metrics = dataset_metrics(
        sample_to_original,
        sample_internal,
        original_thresholds=original_thresholds,
    )

    rarefaction_records = (
        rarefaction(
            "balanced",
            balanced_to_original,
            balanced_internal,
            coverage_threshold=original_thresholds["p75"],
            seed=RANDOM_SEED,
        )
        + rarefaction(
            "sample",
            sample_to_original,
            sample_internal,
            coverage_threshold=original_thresholds["p75"],
            seed=RANDOM_SEED + 1,
        )
    )

    rng = np.random.default_rng(RANDOM_SEED)
    balanced_subset = np.sort(
        rng.choice(len(balanced), size=len(sample), replace=False)
    )
    tsne_scaled = np.vstack(
        [original, balanced[balanced_subset], sample]
    )
    tsne_features = weighted_linear_features(tsne_scaled, group_indexes)
    tsne_distances = group_distance_matrix(
        tsne_scaled,
        tsne_scaled,
        layout,
    )
    embedding, kl_divergence = exact_tsne_from_distances(
        tsne_distances,
        tsne_features,
        perplexity=TSNE_PERPLEXITY,
        random_seed=RANDOM_SEED,
        n_iter=TSNE_ITERATIONS,
    )
    trustworthiness = neighborhood_trustworthiness(
        tsne_distances,
        embedding,
        n_neighbors=10,
    )
    original_end = len(original)
    balanced_end = original_end + len(balanced_subset)
    tsne_groups = [
        {
            "indexes": range(original_end, balanced_end),
            "color": BALANCED_COLOR,
            "marker": "circle",
            "radius": 5,
        },
        {
            "indexes": range(balanced_end, len(embedding)),
            "color": SAMPLE_COLOR,
            "marker": "diamond",
            "radius": 6,
        },
        {
            "indexes": range(0, original_end),
            "color": ORIGINAL_COLOR,
            "marker": "triangle",
            "radius": 11,
        },
    ]

    render_figure(
        pca_values,
        pca_ratio,
        embedding,
        tsne_groups,
        {
            "original_loo": original_loo,
            "balanced": np.min(balanced_to_original, axis=1),
            "sample": np.min(sample_to_original, axis=1),
        },
        original_thresholds["p75"],
        rarefaction_records,
        kl_divergence=kl_divergence,
        trustworthiness=trustworthiness,
    )

    with RAREFACTION_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset",
                "sample_size",
                "metric",
                "mean",
                "ci95_low",
                "ci95_high",
            ],
        )
        writer.writeheader()
        writer.writerows(rarefaction_records)

    metrics = {
        "method": {
            "descriptor_dimension": layout.dimension,
            "descriptor_config": {
                "max_units_per_team": 20,
                "max_zones": 20,
                "attack_type_values": [0, 1],
            },
            "scaler_fit_on": [],
            "normalization": (
                "data-independent semantic bounds; zero-preserving for "
                "padded unit and zone slots"
            ),
            "semantic_bounds": SEMANTIC_BOUNDS,
            "group_weights": GROUP_WEIGHTS,
            "distance": "weighted sum of group RMS distances",
            "original_role": "reference anchors only",
        },
        "inventory": {
            "original": len(original_tasks),
            "balanced_v1": len(balanced_tasks),
            "sample_v2": len(sample_tasks),
        },
        "original_leave_one_out_nearest_distance": {
            **percentile_record(original_loo),
            "thresholds": original_thresholds,
        },
        "balanced_v1": {
            **balanced_metrics,
            "structure": task_structure_metrics(balanced_tasks),
        },
        "sample_v2": {
            **sample_metrics,
            "structure": task_structure_metrics(sample_tasks),
        },
        "pca": {
            "fit_on": ["original", "balanced_v1", "sample_v2"],
            "explained_variance_ratio": list(pca_ratio),
        },
        "tsne": {
            "fit_on": [
                "original",
                "balanced_v1_matched_subsample",
                "sample_v2",
            ],
            "n_original": len(original),
            "n_balanced_subsample": len(balanced_subset),
            "n_sample_v2": len(sample),
            "balanced_subsample_indexes": balanced_subset.tolist(),
            "perplexity": TSNE_PERPLEXITY,
            "iterations": TSNE_ITERATIONS,
            "random_seed": RANDOM_SEED,
            "kl_divergence": kl_divergence,
            "trustworthiness_at_10": trustworthiness,
        },
        "rarefaction": {
            "sizes": list(RAREFACTION_SIZES),
            "repeats": RAREFACTION_REPEATS,
            "coverage_threshold": (
                "Original leave-one-out nearest-distance P75"
            ),
            "csv_file": RAREFACTION_PATH.name,
        },
        "plot_file": PLOT_PATH.name,
    }
    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")

    print(
        json.dumps(
            {
                "plot": str(PLOT_PATH),
                "metrics": str(METRICS_PATH),
                "rarefaction": str(RAREFACTION_PATH),
                "pca_explained_variance": pca_ratio,
                "tsne": metrics["tsne"],
                "balanced_medians": {
                    key: value["median"]
                    for key, value in balanced_metrics.items()
                    if isinstance(value, dict) and "median" in value
                },
                "sample_v2_medians": {
                    key: value["median"]
                    for key, value in sample_metrics.items()
                    if isinstance(value, dict) and "median" in value
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
