"""Visualize TABX environment-randomness coverage with a full scene descriptor.

Unlike the compact 38-D task-deduplication signature, this descriptor retains
the sampled environment variables themselves: roster counts, per-unit initial
pose and combat/physics values, map dimensions, and every terrain zone's type,
position, axes, and strength.  Feature groups are robust-scaled and weighted
separately so the many per-unit fields cannot drown out terrain or roster
variation.
"""

from __future__ import annotations

import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

from plot_tsne_diversity import (
    draw_marker,
    font,
    load_original_scenes,
    percentile_summary,
    profile_counts,
    read_json,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

RANDOM_SEED = 42
PERPLEXITY = 30.0
N_ITER = 1_200
N_UNIT_TYPES = 9
N_ZONE_TYPES = 4

GROUP_WEIGHTS = {
    "roster": 0.25,
    "initial_geometry": 0.30,
    "unit_attributes": 0.20,
    "terrain_and_map": 0.25,
}


@dataclass(frozen=True)
class DescriptorLayout:
    group_slices: dict[str, slice]
    dimension: int


@dataclass(frozen=True)
class DescriptorConfig:
    max_units_per_team: int
    max_zones: int
    attack_type_values: tuple[int, ...]


def infer_descriptor_config(task_sets: Iterable[Iterable[dict[str, Any]]]) -> DescriptorConfig:
    max_units_per_team = 0
    max_zones = 0
    attack_type_values: set[int] = set()
    for tasks in task_sets:
        for task in tasks:
            scenario = task["scenario"]
            teams = flat_field(scenario, "teams", dtype=int)
            max_units_per_team = max(
                max_units_per_team,
                int(np.count_nonzero(teams == 0)),
                int(np.count_nonzero(teams == 1)),
            )
            max_zones = max(max_zones, int(task["zone_scenario"]["n_zone"]))
            attack_type_values.update(
                int(value) for value in flat_field(scenario, "attack_types", dtype=int)
            )
    if max_units_per_team <= 0 or not attack_type_values:
        raise ValueError("Cannot infer a descriptor layout from empty or invalid task sets.")
    return DescriptorConfig(
        max_units_per_team=max_units_per_team,
        max_zones=max_zones,
        attack_type_values=tuple(sorted(attack_type_values)),
    )


def flat_field(scenario: dict[str, Any], name: str, *, dtype: Any = float) -> np.ndarray:
    return np.asarray(scenario[name], dtype=dtype).reshape(-1)


def descriptor_variant(
    task: dict[str, Any],
    config: DescriptorConfig,
    *,
    mirror_x: bool,
    mirror_y: bool,
    flip_teams: bool,
) -> tuple[np.ndarray, DescriptorLayout]:
    scenario = task["scenario"]
    teams = flat_field(scenario, "teams", dtype=int)
    unit_ids = flat_field(scenario, "unit_ids", dtype=int)
    if np.any(unit_ids < 0) or np.any(unit_ids >= N_UNIT_TYPES):
        raise ValueError("Environment descriptor expects unit IDs in [0, 8].")

    positions = np.asarray(scenario["positions"], dtype=float).reshape(-1, 2).copy()
    rotations = flat_field(scenario, "rotations")
    sx = -1.0 if mirror_x else 1.0
    sy = -1.0 if mirror_y else 1.0
    positions[:, 0] *= sx
    positions[:, 1] *= sy
    heading_x = np.cos(rotations) * sx
    heading_y = np.sin(rotations) * sy

    grid = task.get("grid_info", {})
    width = max(float(grid.get("max_field_width", 121.0)), 1e-6)
    height = max(float(grid.get("max_field_height", 78.0)), 1e-6)
    normalized_positions = positions / np.asarray([width, height])

    health = flat_field(scenario, "healths")
    damage = flat_field(scenario, "attack_damages")
    cooldown = flat_field(scenario, "attack_cooldowns")
    attack_range = flat_field(scenario, "attack_ranges")
    attack_type = flat_field(scenario, "attack_types", dtype=int)
    speed = flat_field(scenario, "speeds")
    sight_angle = flat_field(scenario, "sight_angles")
    radius_key = "body_radii" if "body_radii" in scenario else "body_radiuss"
    body_radius = flat_field(scenario, radius_key)
    body_weight = flat_field(scenario, "body_weights")
    attack_type_index = {
        value: index for index, value in enumerate(config.attack_type_values)
    }

    roster_features: list[float] = []
    geometry_features: list[float] = []
    attribute_features: list[float] = []
    physical_team_order = (1, 0) if flip_teams else (0, 1)
    for physical_team in physical_team_order:
        indexes = np.flatnonzero(teams == physical_team)
        if len(indexes) > config.max_units_per_team:
            raise ValueError(
                f"Team has {len(indexes)} units; inferred maximum is "
                f"{config.max_units_per_team}."
            )
        order = np.lexsort(
            (
                health[indexes],
                normalized_positions[indexes, 1],
                normalized_positions[indexes, 0],
                unit_ids[indexes],
            )
        )
        indexes = indexes[order]
        counts = np.bincount(unit_ids[indexes], minlength=N_UNIT_TYPES)[:N_UNIT_TYPES]
        roster_features.extend(counts.astype(float).tolist())
        roster_features.append(float(len(indexes)))

        for slot in range(config.max_units_per_team):
            if slot < len(indexes):
                index = indexes[slot]
                geometry_features.extend(
                    [
                        1.0,
                        float(normalized_positions[index, 0]),
                        float(normalized_positions[index, 1]),
                        float(heading_x[index]),
                        float(heading_y[index]),
                    ]
                )
                attack_one_hot = [0.0] * len(config.attack_type_values)
                attack_one_hot[attack_type_index[int(attack_type[index])]] = 1.0
                attribute_features.extend(
                    [
                        float(health[index]),
                        float(damage[index]),
                        float(cooldown[index]),
                        float(attack_range[index]),
                        *attack_one_hot,
                        float(speed[index]),
                        float(sight_angle[index]),
                        float(body_radius[index]),
                        float(body_weight[index]),
                    ]
                )
            else:
                geometry_features.extend([0.0] * 5)
                attribute_features.extend([0.0] * (8 + len(config.attack_type_values)))

    zones = task["zone_scenario"]
    n_zone = int(zones["n_zone"])
    if n_zone > config.max_zones:
        raise ValueError(
            f"Scene has {n_zone} zones; inferred maximum is {config.max_zones}."
        )
    zone_types = np.asarray(zones["zone_type"], dtype=int).reshape(-1)[:n_zone]
    zone_positions = np.asarray(zones["position"], dtype=float).reshape(-1, 2)[:n_zone].copy()
    zone_axes = np.asarray(zones["axes"], dtype=float).reshape(-1, 2)[:n_zone]
    zone_effects = np.asarray(zones["effect_value"], dtype=float).reshape(-1)[:n_zone]
    zone_positions[:, 0] *= sx
    zone_positions[:, 1] *= sy
    zone_positions /= np.asarray([width, height])
    normalized_axes = zone_axes / np.asarray([width, height])
    if n_zone:
        zone_order = np.lexsort(
            (
                zone_effects,
                normalized_axes[:, 1],
                normalized_axes[:, 0],
                zone_positions[:, 1],
                zone_positions[:, 0],
                zone_types,
            )
        )
    else:
        zone_order = np.asarray([], dtype=int)

    terrain_features: list[float] = [
        float(grid.get("grid_width", 28.0)),
        float(grid.get("grid_height", 18.0)),
        width,
        height,
        float(grid.get("margin_width", 0.0)),
        float(grid.get("margin_height", 0.0)),
        float(n_zone),
    ]
    for slot in range(config.max_zones):
        if slot < n_zone:
            index = int(zone_order[slot])
            zone_type = int(zone_types[index])
            one_hot = [float(zone_type == value) for value in range(N_ZONE_TYPES)]
            terrain_features.extend(
                [
                    1.0,
                    *one_hot,
                    float(zone_positions[index, 0]),
                    float(zone_positions[index, 1]),
                    float(normalized_axes[index, 0]),
                    float(normalized_axes[index, 1]),
                    float(zone_effects[index]),
                ]
            )
        else:
            terrain_features.extend([0.0] * 10)

    groups = {
        "roster": np.asarray(roster_features, dtype=float),
        "initial_geometry": np.asarray(geometry_features, dtype=float),
        "unit_attributes": np.asarray(attribute_features, dtype=float),
        "terrain_and_map": np.asarray(terrain_features, dtype=float),
    }
    values: list[np.ndarray] = []
    group_slices: dict[str, slice] = {}
    start = 0
    for name in GROUP_WEIGHTS:
        group = groups[name]
        values.append(group)
        group_slices[name] = slice(start, start + len(group))
        start += len(group)
    return np.concatenate(values), DescriptorLayout(group_slices=group_slices, dimension=start)


def full_environment_descriptor(
    task: dict[str, Any], config: DescriptorConfig
) -> tuple[np.ndarray, DescriptorLayout]:
    variants = [
        descriptor_variant(
            task,
            config,
            mirror_x=mirror_x,
            mirror_y=mirror_y,
            flip_teams=flip_teams,
        )
        for mirror_x in (False, True)
        for mirror_y in (False, True)
        for flip_teams in (False, True)
    ]
    quantized = [
        (np.round(values, 12), layout) for values, layout in variants
    ]
    return min(quantized, key=lambda item: tuple(item[0].tolist()))


def descriptor_matrix(
    tasks: Iterable[dict[str, Any]], config: DescriptorConfig
) -> tuple[np.ndarray, DescriptorLayout]:
    rows: list[np.ndarray] = []
    layout: DescriptorLayout | None = None
    for task in tasks:
        row, current_layout = full_environment_descriptor(task, config)
        if layout is None:
            layout = current_layout
        elif current_layout != layout:
            raise AssertionError("Descriptor layout changed between tasks.")
        rows.append(row)
    if layout is None:
        raise ValueError("Cannot describe an empty task collection.")
    return np.stack(rows), layout


@dataclass(frozen=True)
class RobustScaler:
    center: np.ndarray
    scale: np.ndarray
    active: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        scaled = (values - self.center) / self.scale
        scaled[:, ~self.active] = 0.0
        return scaled


def fit_robust_scaler(values: np.ndarray) -> RobustScaler:
    center = np.median(values, axis=0)
    q25, q75 = np.percentile(values, [25, 75], axis=0)
    interquartile = q75 - q25
    standard_deviation = np.std(values, axis=0)
    scale = np.where(interquartile > 1e-9, interquartile, standard_deviation)
    active = scale > 1e-9
    scale = np.where(active, scale, 1.0)
    return RobustScaler(center=center, scale=scale, active=active)


def group_active_indexes(layout: DescriptorLayout, scaler: RobustScaler) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, section in layout.group_slices.items():
        indexes = np.arange(section.start, section.stop)
        result[name] = indexes[scaler.active[indexes]]
    return result


def environment_distance_matrix(
    first: np.ndarray,
    second: np.ndarray | None,
    active_groups: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    second = first if second is None else second
    total = np.zeros((len(first), len(second)), dtype=float)
    group_distances: dict[str, np.ndarray] = {}
    for name, weight in GROUP_WEIGHTS.items():
        indexes = active_groups[name]
        if not len(indexes):
            group_distance = np.zeros_like(total)
        else:
            delta = first[:, None, indexes] - second[None, :, indexes]
            group_distance = np.sqrt(np.mean(np.square(delta), axis=2))
        group_distances[name] = group_distance
        total += weight * group_distance
    return total, group_distances


def weighted_linear_features(
    scaled: np.ndarray, active_groups: dict[str, np.ndarray]
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for name, weight in GROUP_WEIGHTS.items():
        indexes = active_groups[name]
        if len(indexes):
            columns.append(scaled[:, indexes] * math.sqrt(weight / len(indexes)))
    if not columns:
        raise ValueError(
            "All environment descriptor dimensions are constant; "
            "a distribution embedding cannot be computed."
        )
    return np.concatenate(columns, axis=1)


def joint_probabilities(
    distances_squared: np.ndarray, perplexity: float
) -> np.ndarray:
    """Build the symmetric high-dimensional probabilities used by exact t-SNE."""

    n_samples = len(distances_squared)
    if n_samples < 3:
        raise ValueError("t-SNE requires at least three samples.")
    target_entropy = math.log(min(perplexity, n_samples - 1.0))
    conditional = np.zeros_like(distances_squared)
    for row in range(n_samples):
        mask = np.arange(n_samples) != row
        values = distances_squared[row, mask]
        values = values - np.min(values)
        beta = 1.0
        beta_min = -math.inf
        beta_max = math.inf
        row_probabilities = np.empty_like(values)
        for _ in range(60):
            row_probabilities = np.exp(-values * beta)
            total = max(float(row_probabilities.sum()), 1e-300)
            entropy = (
                math.log(total)
                + beta * float(np.dot(values, row_probabilities)) / total
            )
            difference = entropy - target_entropy
            if abs(difference) < 1e-5:
                break
            if difference > 0.0:
                beta_min = beta
                beta = (
                    beta * 2.0
                    if math.isinf(beta_max)
                    else (beta + beta_max) / 2.0
                )
            else:
                beta_max = beta
                beta = (
                    beta / 2.0
                    if math.isinf(beta_min)
                    else (beta + beta_min) / 2.0
                )
        conditional[row, mask] = row_probabilities / max(
            float(row_probabilities.sum()), 1e-300
        )

    probabilities = (conditional + conditional.T) / (2.0 * n_samples)
    probabilities = np.maximum(probabilities, 1e-12)
    np.fill_diagonal(probabilities, 0.0)
    probabilities /= probabilities.sum()
    return probabilities


def student_probabilities(embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norm = np.sum(np.square(embedding), axis=1)
    distances = np.maximum(
        norm[:, None] + norm[None, :] - 2.0 * embedding @ embedding.T,
        0.0,
    )
    numerator = 1.0 / (1.0 + distances)
    np.fill_diagonal(numerator, 0.0)
    probabilities = np.maximum(
        numerator / max(float(numerator.sum()), 1e-300),
        1e-12,
    )
    np.fill_diagonal(probabilities, 0.0)
    probabilities /= probabilities.sum()
    return probabilities, numerator


def exact_tsne_from_environment(
    scaled: np.ndarray,
    active_groups: dict[str, np.ndarray],
) -> tuple[np.ndarray, float]:
    """Run deterministic exact t-SNE using the precomputed environment distance."""

    distances, _ = environment_distance_matrix(scaled, None, active_groups)
    probabilities = joint_probabilities(np.square(distances), PERPLEXITY)
    linear = weighted_linear_features(scaled, active_groups)
    centered = linear - linear.mean(axis=0, keepdims=True)
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    embedding = centered @ right_vectors[:2].T
    embedding /= max(float(np.std(embedding)), 1e-12)
    embedding *= 1e-4
    embedding += np.random.default_rng(RANDOM_SEED).normal(0.0, 1e-6, embedding.shape)

    velocity = np.zeros_like(embedding)
    gains = np.ones_like(embedding)
    early_exaggeration = 12.0
    learning_rate = max(len(scaled) / (early_exaggeration * 4.0), 50.0)
    for iteration in range(N_ITER):
        q_values, numerator = student_probabilities(embedding)
        p_values = (
            probabilities * early_exaggeration
            if iteration < 250
            else probabilities
        )
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

    q_values, _ = student_probabilities(embedding)
    mask = probabilities > 0.0
    kl_divergence = float(
        np.sum(probabilities[mask] * np.log(probabilities[mask] / q_values[mask]))
    )
    return embedding, kl_divergence


def convex_hull(points: np.ndarray) -> list[tuple[float, float]]:
    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) <= 2:
        return unique

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def central_hull(points: np.ndarray, fraction: float = 0.90) -> list[tuple[float, float]]:
    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    keep = distances <= np.quantile(distances, fraction)
    return convex_hull(points[keep])


def draw_dashed_polygon(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    *,
    fill: tuple[int, int, int, int],
    width: int = 3,
    dash: float = 13.0,
    gap: float = 8.0,
) -> None:
    if len(points) < 2:
        return
    closed = points + [points[0]]
    for start, end in zip(closed, closed[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue
        position = 0.0
        while position < length:
            segment_end = min(position + dash, length)
            first = (start[0] + dx * position / length, start[1] + dy * position / length)
            second = (
                start[0] + dx * segment_end / length,
                start[1] + dy * segment_end / length,
            )
            draw.line((first, second), fill=fill, width=width)
            position += dash + gap


COLORS = {
    "original": (66, 82, 99, 215),
    "author": (235, 126, 57, 195),
    "modified": (34, 145, 169, 210),
}


def render_distribution_plot(
    output_path: Path,
    embedding: np.ndarray,
    groups: list[dict[str, Any]],
    hulls: list[dict[str, Any]],
    *,
    title: str,
    subtitle: str,
    footer: str,
    summary_lines: list[str],
    extent_embedding: np.ndarray | None = None,
    legend_position: str = "inside",
) -> None:
    if legend_position not in {"inside", "bottom"}:
        raise ValueError("legend_position must be 'inside' or 'bottom'.")
    bottom_legend = legend_position == "bottom"
    width, height = 1800, 1540 if bottom_legend else 1300
    image = Image.new("RGB", (width, height), (247, 249, 252))
    draw = ImageDraw.Draw(image, "RGBA")
    plot_left, plot_top, plot_right = 150, 190, 1660
    plot_bottom = 1080 if bottom_legend else 1120
    draw.rounded_rectangle(
        (plot_left, plot_top, plot_right, plot_bottom),
        radius=18,
        fill=(255, 255, 255, 255),
        outline=(205, 214, 224, 255),
        width=2,
    )
    draw.text((90, 55), title, fill=(25, 35, 48, 255), font=font(46, bold=True))
    draw.text((92, 117), subtitle, fill=(89, 103, 119, 255), font=font(24))

    extent_source = embedding if extent_embedding is None else extent_embedding
    low = np.percentile(extent_source, 1, axis=0)
    high = np.percentile(extent_source, 99, axis=0)
    span = np.maximum(high - low, 1e-9)
    low -= span * 0.10
    high += span * 0.10

    def map_point(point: np.ndarray) -> tuple[float, float]:
        normalized = np.clip((point - low) / np.maximum(high - low, 1e-9), 0.0, 1.0)
        return (
            plot_left + normalized[0] * (plot_right - plot_left),
            plot_bottom - normalized[1] * (plot_bottom - plot_top),
        )

    for step in range(1, 5):
        x = plot_left + (plot_right - plot_left) * step / 5
        y = plot_top + (plot_bottom - plot_top) * step / 5
        draw.line((x, plot_top, x, plot_bottom), fill=(222, 228, 235, 180), width=1)
        draw.line((plot_left, y, plot_right, y), fill=(222, 228, 235, 180), width=1)

    for hull_spec in hulls:
        group_points = embedding[hull_spec["indexes"]]
        central = [map_point(np.asarray(point)) for point in central_hull(group_points)]
        outer = [map_point(np.asarray(point)) for point in convex_hull(group_points)]
        color = hull_spec["color"]
        if len(central) >= 3:
            draw.polygon(central, fill=(*color[:3], 24))
            draw_dashed_polygon(draw, central, fill=(*color[:3], 135), width=3)
        if len(outer) >= 3:
            draw.line(outer + [outer[0]], fill=(*color[:3], 190), width=4)

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

    if bottom_legend:
        legend_x, legend_y = plot_left, 1160
        legend_width = plot_right - plot_left
        wrapped_summary = [
            wrapped
            for line in summary_lines
            for wrapped in (textwrap.wrap(line, width=138) or [""])
        ]
        legend_height = 138 + 31 * len(wrapped_summary)
        draw.rounded_rectangle(
            (
                legend_x,
                legend_y,
                legend_x + legend_width,
                legend_y + legend_height,
            ),
            radius=14,
            fill=(255, 255, 255, 255),
            outline=(207, 215, 224, 240),
            width=2,
        )
        draw.text(
            (legend_x + 24, legend_y + 17),
            "Environment distributions",
            fill=(38, 50, 64, 255),
            font=font(23, bold=True),
        )
        entry_width = legend_width / max(len(groups), 1)
        for column, group in enumerate(groups):
            entry_x = legend_x + 28 + column * entry_width
            cy = legend_y + 78
            draw_marker(
                draw,
                entry_x + 8,
                cy,
                color=group["color"],
                marker=group["marker"],
                radius=8,
            )
            draw.text(
                (entry_x + 34, cy - 14),
                f"{group['label']}  (n={len(group['indexes'])})",
                fill=(45, 57, 70, 255),
                font=font(22),
            )
        summary_y = legend_y + 111
        draw.line(
            (
                legend_x + 22,
                summary_y,
                legend_x + legend_width - 22,
                summary_y,
            ),
            fill=(211, 219, 227, 220),
            width=1,
        )
        for row, line in enumerate(wrapped_summary):
            draw.text(
                (legend_x + 24, summary_y + 15 + row * 31),
                line,
                fill=(81, 94, 109, 255),
                font=font(19),
            )
    else:
        legend_x, legend_y = 1060, 215
        legend_width = 570
        wrapped_summary = [
            wrapped
            for line in summary_lines
            for wrapped in (textwrap.wrap(line, width=55) or [""])
        ]
        legend_height = 64 + 58 * len(groups) + 36 + 31 * len(
            wrapped_summary
        )
        if legend_y + legend_height > plot_bottom - 15:
            raise ValueError(
                "Legend content exceeds the available plot height; "
                "shorten the summary or reduce its font size."
            )
        draw.rounded_rectangle(
            (
                legend_x,
                legend_y,
                legend_x + legend_width,
                legend_y + legend_height,
            ),
            radius=14,
            fill=(255, 255, 255, 238),
            outline=(207, 215, 224, 240),
            width=2,
        )
        draw.text(
            (legend_x + 24, legend_y + 17),
            "Environment distributions",
            fill=(38, 50, 64, 255),
            font=font(23, bold=True),
        )
        for row, group in enumerate(groups):
            cy = legend_y + 80 + row * 58
            draw_marker(
                draw,
                legend_x + 34,
                cy,
                color=group["color"],
                marker=group["marker"],
                radius=8,
            )
            draw.text(
                (legend_x + 60, cy - 14),
                f"{group['label']}  (n={len(group['indexes'])})",
                fill=(45, 57, 70, 255),
                font=font(22),
            )
        summary_y = legend_y + 78 + 58 * len(groups)
        draw.line(
            (
                legend_x + 22,
                summary_y,
                legend_x + legend_width - 22,
                summary_y,
            ),
            fill=(211, 219, 227, 220),
            width=1,
        )
        for row, line in enumerate(wrapped_summary):
            draw.text(
                (legend_x + 24, summary_y + 16 + row * 31),
                line,
                fill=(81, 94, 109, 255),
                font=font(19),
            )

    x_axis_y = plot_bottom + 43
    draw.text(
        ((plot_left + plot_right) / 2 - 48, x_axis_y),
        "t-SNE 1",
        fill=(79, 92, 108, 255),
        font=font(23),
    )
    vertical_label = Image.new("RGBA", (180, 55), (0, 0, 0, 0))
    vertical_draw = ImageDraw.Draw(vertical_label)
    vertical_draw.text((0, 12), "t-SNE 2", fill=(79, 92, 108, 255), font=font(23))
    vertical_label = vertical_label.rotate(90, expand=True)
    image.paste(vertical_label, (43, 590), vertical_label)
    footer_y = 1483 if bottom_legend else 1246
    draw.text(
        (92, footer_y),
        footer,
        fill=(108, 120, 134, 255),
        font=font(18),
    )
    image.save(output_path, optimize=True)


def within_metrics(
    tasks: list[dict[str, Any]],
    scaled: np.ndarray,
    active_groups: dict[str, np.ndarray],
) -> dict[str, Any]:
    distances, group_distances = environment_distance_matrix(scaled, None, active_groups)
    upper_indexes = np.triu_indices(len(scaled), k=1)
    pairwise = distances[upper_indexes]
    nearest = np.min(np.where(np.eye(len(scaled), dtype=bool), np.inf, distances), axis=1)
    linear = weighted_linear_features(scaled, active_groups)
    centered = linear - linear.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    variances = np.square(singular_values) / max(len(scaled) - 1, 1)
    effective_dimension = float(np.square(variances.sum()) / max(np.square(variances).sum(), 1e-300))
    rms_radius = float(np.sqrt(np.mean(np.sum(np.square(centered), axis=1))))
    unique_descriptors = len({tuple(np.round(row, 8)) for row in scaled})
    result: dict[str, Any] = {
        "n_scenes": len(tasks),
        "unique_full_environment_descriptors": unique_descriptors,
        "pairwise_environment_distance": percentile_summary(pairwise),
        "nearest_neighbor_environment_distance": percentile_summary(nearest),
        "coverage_rms_radius": rms_radius,
        "effective_dimension_participation_ratio": effective_dimension,
        "mean_pairwise_distance_by_parameter_group": {
            name: float(np.mean(values[upper_indexes])) for name, values in group_distances.items()
        },
        **profile_counts(tasks),
    }
    return result


def cross_metrics(
    first_name: str,
    first: np.ndarray,
    second_name: str,
    second: np.ndarray,
    active_groups: dict[str, np.ndarray],
) -> dict[str, Any]:
    distances, group_distances = environment_distance_matrix(first, second, active_groups)
    result: dict[str, Any] = {
        f"{first_name}_to_nearest_{second_name}": percentile_summary(np.min(distances, axis=1)),
        f"{second_name}_to_nearest_{first_name}": percentile_summary(np.min(distances, axis=0)),
        "cross_exact_descriptor_pairs": int(np.count_nonzero(distances < 1e-12)),
        "mean_cross_distance_by_parameter_group": {
            name: float(np.mean(values)) for name, values in group_distances.items()
        },
    }
    return result


def kmeans(values: np.ndarray, n_clusters: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = [values[int(rng.integers(len(values)))]]
    closest_squared = np.sum(np.square(values - centers[0]), axis=1)
    for _ in range(1, n_clusters):
        probabilities = closest_squared / max(float(closest_squared.sum()), 1e-300)
        centers.append(values[int(rng.choice(len(values), p=probabilities))])
        candidate_squared = np.sum(np.square(values - centers[-1]), axis=1)
        closest_squared = np.minimum(closest_squared, candidate_squared)
    centers_array = np.stack(centers)
    for _ in range(100):
        squared = np.sum(np.square(values[:, None, :] - centers_array[None, :, :]), axis=2)
        labels = np.argmin(squared, axis=1)
        updated = centers_array.copy()
        for cluster in range(n_clusters):
            if np.any(labels == cluster):
                updated[cluster] = values[labels == cluster].mean(axis=0)
        if np.allclose(updated, centers_array, atol=1e-7, rtol=0.0):
            centers_array = updated
            break
        centers_array = updated
    return centers_array


def cluster_coverage(values: np.ndarray, centers: np.ndarray) -> dict[str, float | int]:
    squared = np.sum(np.square(values[:, None, :] - centers[None, :, :]), axis=2)
    labels = np.argmin(squared, axis=1)
    counts = np.bincount(labels, minlength=len(centers))
    probabilities = counts[counts > 0] / counts.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return {
        "occupied_clusters": int(np.count_nonzero(counts)),
        "cluster_coverage_fraction": float(np.count_nonzero(counts) / len(centers)),
        "effective_clusters_exp_entropy": float(math.exp(entropy)),
    }


def percentage_change(before: float, after: float) -> float:
    return (after / before - 1.0) * 100.0


def neighborhood_trustworthiness(
    high_distances: np.ndarray, embedding: np.ndarray, *, n_neighbors: int = 10
) -> float:
    """Measure how well low-dimensional neighbors are ranked in high dimensions."""

    n_samples = len(embedding)
    n_neighbors = min(n_neighbors, max(1, (n_samples - 1) // 2))
    high_ranks = np.argsort(np.argsort(high_distances, axis=1), axis=1)
    low_distances = np.linalg.norm(
        embedding[:, None, :] - embedding[None, :, :], axis=2
    )
    low_neighbors = np.argsort(low_distances, axis=1)[:, 1 : n_neighbors + 1]
    penalty = 0.0
    for row in range(n_samples):
        ranks = high_ranks[row, low_neighbors[row]]
        penalty += float(np.sum(np.maximum(ranks - n_neighbors, 0)))
    normalizer = 2.0 / (
        n_samples
        * n_neighbors
        * max(2 * n_samples - 3 * n_neighbors - 1, 1)
    )
    return 1.0 - normalizer * penalty


def main() -> None:
    original_tasks, _, original_counts = load_original_scenes()
    tasks1_tasks = read_json(OUTPUTS / "tasks1_balanced.json")["tasks"]
    balance_tasks = read_json(OUTPUTS / "balance_tasks.json")["tasks"]
    tasks1_hashes = {task["metadata"]["canonical_hash"] for task in tasks1_tasks}
    balance_hashes = [task["metadata"]["canonical_hash"] for task in balance_tasks]
    subset_mask = np.asarray([value in tasks1_hashes for value in balance_hashes], dtype=bool)
    additional_mask = ~subset_mask
    additional_tasks = [task for task, keep in zip(balance_tasks, additional_mask) if keep]

    descriptor_config = infer_descriptor_config(
        (original_tasks, tasks1_tasks, balance_tasks)
    )
    original_raw, layout = descriptor_matrix(original_tasks, descriptor_config)
    tasks1_raw, layout_tasks1 = descriptor_matrix(tasks1_tasks, descriptor_config)
    balance_raw, layout_balance = descriptor_matrix(balance_tasks, descriptor_config)
    if layout != layout_tasks1 or layout != layout_balance:
        raise AssertionError("Descriptor layouts do not match.")

    scaler = fit_robust_scaler(np.vstack([original_raw, balance_raw]))
    original = scaler.transform(original_raw)
    tasks1 = scaler.transform(tasks1_raw)
    balance = scaler.transform(balance_raw)
    additional = balance[additional_mask]
    active_groups = group_active_indexes(layout, scaler)

    within = {
        "original": within_metrics(original_tasks, original, active_groups),
        "tasks1_balanced_author_sampler": within_metrics(tasks1_tasks, tasks1, active_groups),
        "balance_tasks_merged": within_metrics(balance_tasks, balance, active_groups),
        "modified_sampler_additions": within_metrics(additional_tasks, additional, active_groups),
    }

    linear_balance = weighted_linear_features(balance, active_groups)
    centers = kmeans(linear_balance, 50, seed=RANDOM_SEED)
    cluster_metrics = {
        "original": cluster_coverage(weighted_linear_features(original, active_groups), centers),
        "tasks1_balanced_author_sampler": cluster_coverage(
            weighted_linear_features(tasks1, active_groups), centers
        ),
        "balance_tasks_merged": cluster_coverage(linear_balance, centers),
        "modified_sampler_additions": cluster_coverage(
            weighted_linear_features(additional, active_groups), centers
        ),
    }

    original_global = np.arange(len(original), dtype=int)
    balance_global = np.arange(
        len(original), len(original) + len(balance), dtype=int
    )
    author_global = balance_global[subset_mask]
    additions_global = balance_global[additional_mask]
    global_values = np.vstack([original, balance])
    global_embedding, kl_divergence = exact_tsne_from_environment(
        global_values, active_groups
    )
    global_distances, _ = environment_distance_matrix(
        global_values, None, active_groups
    )
    trustworthiness = neighborhood_trustworthiness(
        global_distances, global_embedding, n_neighbors=10
    )

    comparison_specs = [
        {
            "name": "original_vs_tasks1_balanced",
            "output": OUTPUTS / "tsne_env_original_vs_tasks1_balanced.png",
            "groups": [
                {
                    "indexes": original_global.tolist(),
                    "label": "Original scenes",
                    "color": COLORS["original"],
                    "marker": "triangle",
                    "radius": 9,
                },
                {
                    "indexes": author_global.tolist(),
                    "label": "Author sampler: tasks1",
                    "color": COLORS["author"],
                    "marker": "circle",
                    "radius": 7,
                },
            ],
            "hulls": [
                {"indexes": original_global.tolist(), "color": COLORS["original"]},
                {"indexes": author_global.tolist(), "color": COLORS["author"]},
            ],
            "title": "Environment randomness: original vs author sampler",
            "subtitle": "Shared global t-SNE coordinates | roster, pose, attributes, map, and terrain",
            "summary": [
                f"Original: eff.dim={within['original']['effective_dimension_participation_ratio']:.2f}, RMS={within['original']['coverage_rms_radius']:.3f}",
                f"Author:   eff.dim={within['tasks1_balanced_author_sampler']['effective_dimension_participation_ratio']:.2f}, RMS={within['tasks1_balanced_author_sampler']['coverage_rms_radius']:.3f}",
                "Solid=100% boundary; shaded dashed=central 90%",
            ],
        },
        {
            "name": "original_vs_balance_tasks",
            "output": OUTPUTS / "tsne_env_original_vs_balance_tasks.png",
            "groups": [
                {
                    "indexes": original_global.tolist(),
                    "label": "Original scenes",
                    "color": COLORS["original"],
                    "marker": "triangle",
                    "radius": 9,
                },
                {
                    "indexes": balance_global.tolist(),
                    "label": "Merged balance_tasks",
                    "color": COLORS["modified"],
                    "marker": "circle",
                    "radius": 7,
                },
            ],
            "hulls": [
                {"indexes": original_global.tolist(), "color": COLORS["original"]},
                {"indexes": balance_global.tolist(), "color": COLORS["modified"]},
            ],
            "title": "Environment randomness: original vs merged sampler bank",
            "subtitle": "Shared global t-SNE coordinates | roster, pose, attributes, map, and terrain",
            "summary": [
                f"Original: eff.dim={within['original']['effective_dimension_participation_ratio']:.2f}, RMS={within['original']['coverage_rms_radius']:.3f}",
                f"Merged:   eff.dim={within['balance_tasks_merged']['effective_dimension_participation_ratio']:.2f}, RMS={within['balance_tasks_merged']['coverage_rms_radius']:.3f}",
                "Solid=100% boundary; shaded dashed=central 90%",
            ],
        },
        {
            "name": "tasks1_balanced_vs_balance_tasks",
            "output": OUTPUTS / "tsne_env_tasks1_balanced_vs_balance_tasks.png",
            "groups": [
                {
                    "indexes": author_global.tolist(),
                    "label": "Author sampler subset",
                    "color": COLORS["author"],
                    "marker": "circle",
                    "radius": 7,
                },
                {
                    "indexes": additions_global.tolist(),
                    "label": "Modified sampler added",
                    "color": COLORS["modified"],
                    "marker": "diamond",
                    "radius": 8,
                },
            ],
            "hulls": [
                {"indexes": author_global.tolist(), "color": COLORS["author"]},
                {"indexes": balance_global.tolist(), "color": COLORS["modified"]},
            ],
            "title": "Environment randomness: author vs modified sampler",
            "subtitle": "Shared global coordinates | blue additions and envelope show the merged bank",
            "summary": [
                f"Author: eff.dim={within['tasks1_balanced_author_sampler']['effective_dimension_participation_ratio']:.2f}, RMS={within['tasks1_balanced_author_sampler']['coverage_rms_radius']:.3f}",
                f"Merged: eff.dim={within['balance_tasks_merged']['effective_dimension_participation_ratio']:.2f}, RMS={within['balance_tasks_merged']['coverage_rms_radius']:.3f}",
                f"Occupied modes: {cluster_metrics['tasks1_balanced_author_sampler']['occupied_clusters']} -> {cluster_metrics['balance_tasks_merged']['occupied_clusters']} / 50",
                "Solid=100% boundary; shaded dashed=central 90%",
            ],
        },
    ]

    footer = (
        f"{layout.dimension}-D descriptor | global exact t-SNE | "
        f"perplexity={PERPLEXITY:g} | seed={RANDOM_SEED} | "
        f"KL={kl_divergence:.3f} | T(10)={trustworthiness:.3f}"
    )
    for spec in comparison_specs:
        render_distribution_plot(
            spec["output"],
            global_embedding,
            spec["groups"],
            spec["hulls"],
            title=spec["title"],
            subtitle=spec["subtitle"],
            footer=footer,
            summary_lines=spec["summary"],
        )

    author = within["tasks1_balanced_author_sampler"]
    merged = within["balance_tasks_merged"]
    expansion = {
        "scene_count_percent": percentage_change(author["n_scenes"], merged["n_scenes"]),
        "coverage_rms_radius_percent": percentage_change(
            author["coverage_rms_radius"], merged["coverage_rms_radius"]
        ),
        "effective_dimension_percent": percentage_change(
            author["effective_dimension_participation_ratio"],
            merged["effective_dimension_participation_ratio"],
        ),
        "unique_roster_profiles_percent": percentage_change(
            author["unique_roster_profiles"], merged["unique_roster_profiles"]
        ),
        "occupied_50_modes_percent": percentage_change(
            cluster_metrics["tasks1_balanced_author_sampler"]["occupied_clusters"],
            cluster_metrics["balance_tasks_merged"]["occupied_clusters"],
        ),
        "mean_pairwise_distance_percent": percentage_change(
            author["pairwise_environment_distance"]["mean"],
            merged["pairwise_environment_distance"]["mean"],
        ),
    }

    metrics = {
        "method": {
            "descriptor_dimension": layout.dimension,
            "descriptor_config": {
                "max_units_per_team": descriptor_config.max_units_per_team,
                "max_zones": descriptor_config.max_zones,
                "attack_type_values": descriptor_config.attack_type_values,
            },
            "descriptor_contents": {
                "roster": "unit counts and team size",
                "initial_geometry": "per-unit presence, normalized x/y, heading cos/sin",
                "unit_attributes": "health, damage, cooldown, range, one-hot attack type, speed, sight aperture, radius/weight",
                "terrain_and_map": "grid/field dimensions and dynamically sized zones: type, position, axes, effect",
            },
            "group_weights": GROUP_WEIGHTS,
            "robust_scaling_fit_on": "original scenes plus unique balance_tasks scenes",
            "invariances": ["mirror_x", "mirror_y", "rotate_180", "team_label_swap", "unit_order"],
            "embedding": "one deterministic global exact t-SNE fitted once; all panels share coordinates",
        },
        "original_inventory": original_counts,
        "active_dimensions": {
            name: int(len(indexes)) for name, indexes in active_groups.items()
        },
        "within_set_randomness": within,
        "cluster_mode_coverage_k50": cluster_metrics,
        "cross_set_randomness": {
            "original_vs_author": cross_metrics(
                "original", original, "author", tasks1, active_groups
            ),
            "original_vs_merged": cross_metrics(
                "original", original, "merged", balance, active_groups
            ),
            "author_vs_modified_additions": cross_metrics(
                "author", tasks1, "modified_additions", additional, active_groups
            ),
        },
        "author_to_merged_expansion_percent": expansion,
        "subset_relationship": {
            "author_tasks": len(tasks1_tasks),
            "modified_additions": len(additional_tasks),
            "merged_tasks": len(balance_tasks),
            "author_hashes_preserved_in_merged": int(subset_mask.sum()),
        },
        "embedding": {
            "n_points": len(global_embedding),
            "kl_divergence": kl_divergence,
            "trustworthiness_at_10": trustworthiness,
            "perplexity": PERPLEXITY,
            "iterations": N_ITER,
            "random_seed": RANDOM_SEED,
            "panel_files": [spec["output"].name for spec in comparison_specs],
        },
    }
    metrics_path = OUTPUTS / "environment_randomness_metrics.json"
    with metrics_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")

    print(
        json.dumps(
            {
                "plots": [str(spec["output"]) for spec in comparison_specs],
                "metrics": str(metrics_path),
                "descriptor_dimension": layout.dimension,
                "descriptor_config": metrics["method"]["descriptor_config"],
                "active_dimensions": metrics["active_dimensions"],
                "embedding": metrics["embedding"],
                "expansion_percent": expansion,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
