"""Combine the five standard t-SNE comparisons into one shared-coordinate figure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

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
    weighted_linear_features,
)
from plot_standard_environment_tsne import (
    squared_euclidean_distances,
    standard_exact_tsne,
)
from plot_tsne_diversity import draw_marker, font, load_original_scenes, read_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

BACKGROUND = (246, 248, 251, 255)
PANEL_BACKGROUND = (255, 255, 255, 255)
PANEL_BORDER = (202, 213, 225, 255)
GRID_COLOR = (224, 231, 239, 255)
TEXT_COLOR = (31, 42, 55, 255)
MUTED_TEXT = (87, 103, 122, 255)
HARD_COLOR = (139, 92, 185, 210)


def embedding_mapper(
    embedding: np.ndarray,
    plot_box: tuple[int, int, int, int],
) -> Any:
    """Create a mapper using one global extent shared by every panel."""

    left, top, right, bottom = plot_box
    minimum = embedding.min(axis=0)
    maximum = embedding.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-9)
    minimum -= span * 0.06
    maximum += span * 0.06
    span = maximum - minimum

    def map_point(point: np.ndarray) -> tuple[float, float]:
        x = left + (float(point[0]) - minimum[0]) / span[0] * (right - left)
        y = bottom - (float(point[1]) - minimum[1]) / span[1] * (bottom - top)
        return x, y

    return map_point


def draw_panel(
    draw: ImageDraw.ImageDraw,
    embedding: np.ndarray,
    panel_box: tuple[int, int, int, int],
    *,
    panel_label: str,
    title: str,
    groups: list[dict[str, Any]],
    extent_embedding: np.ndarray | None = None,
) -> None:
    left, top, right, bottom = panel_box
    draw.rounded_rectangle(
        panel_box,
        radius=18,
        fill=PANEL_BACKGROUND,
        outline=PANEL_BORDER,
        width=2,
    )
    draw.text(
        (left + 22, top + 17),
        panel_label,
        fill=COLORS["modified"],
        font=font(23, bold=True),
    )
    draw.text(
        (left + 62, top + 16),
        title,
        fill=TEXT_COLOR,
        font=font(23, bold=True),
    )

    plot_box = (left + 50, top + 66, right - 24, bottom - 58)
    plot_left, plot_top, plot_right, plot_bottom = plot_box
    for step in range(5):
        fraction = step / 4
        x = plot_left + fraction * (plot_right - plot_left)
        y = plot_top + fraction * (plot_bottom - plot_top)
        draw.line(
            [(x, plot_top), (x, plot_bottom)],
            fill=GRID_COLOR,
            width=1,
        )
        draw.line(
            [(plot_left, y), (plot_right, y)],
            fill=GRID_COLOR,
            width=1,
        )
    draw.rectangle(plot_box, outline=PANEL_BORDER, width=1)

    extent_source = embedding if extent_embedding is None else extent_embedding
    map_point = embedding_mapper(extent_source, plot_box)
    for group in groups:
        for index in group["indexes"]:
            x, y = map_point(embedding[index])
            draw_marker(
                draw,
                x,
                y,
                color=group["color"],
                marker=group["marker"],
                radius=group["radius"],
            )

    legend_y = bottom - 39
    legend_x = left + 56
    for group in groups:
        draw_marker(
            draw,
            legend_x,
            legend_y,
            color=group["color"],
            marker=group["marker"],
            radius=6,
        )
        label = f"{group['label']} (n={len(group['indexes'])})"
        draw.text(
            (legend_x + 13, legend_y - 10),
            label,
            fill=MUTED_TEXT,
            font=font(17),
        )
        legend_x += int(draw.textlength(label, font=font(17))) + 54


def draw_method_panel(
    draw: ImageDraw.ImageDraw,
    panel_box: tuple[int, int, int, int],
    *,
    descriptor_dimension: int,
    active_dimensions: int,
    kl_divergence: float,
    trustworthiness: float,
) -> None:
    left, top, right, bottom = panel_box
    draw.rounded_rectangle(
        panel_box,
        radius=18,
        fill=(249, 251, 253, 255),
        outline=PANEL_BORDER,
        width=2,
    )
    draw.text(
        (left + 30, top + 26),
        "Shared method and legend",
        fill=TEXT_COLOR,
        font=font(27, bold=True),
    )

    legend_rows = [
        ("Original scenes", COLORS["original"], "triangle"),
        ("tasks1_balanced", COLORS["author"], "circle"),
        ("balance_tasks / additions", COLORS["modified"], "diamond"),
        ("hard_diverse_tasks", HARD_COLOR, "diamond"),
    ]
    y = top + 86
    for label, color, marker in legend_rows:
        draw_marker(draw, left + 43, y + 9, color=color, marker=marker, radius=8)
        draw.text((left + 65, y), label, fill=TEXT_COLOR, font=font(20))
        y += 43

    draw.line(
        [(left + 30, y + 5), (right - 30, y + 5)],
        fill=PANEL_BORDER,
        width=1,
    )
    method_lines = [
        "One global embedding for all five panels",
        f"{descriptor_dimension}-D descriptor -> {active_dimensions} active features",
        f"Exact Euclidean t-SNE, perplexity={PERPLEXITY:g}, seed={RANDOM_SEED}",
        f"KL divergence={kl_divergence:.3f}",
        f"Trustworthiness T(10)={trustworthiness:.3f}",
        "Pure scatter plots; no hulls or enclosing circles",
    ]
    y += 27
    for line in method_lines:
        draw.text((left + 30, y), line, fill=MUTED_TEXT, font=font(18))
        y += 35


def main() -> None:
    original_tasks, _, _ = load_original_scenes()
    tasks1_tasks = read_json(OUTPUTS / "tasks1_balanced.json")["tasks"]
    balance_tasks = read_json(OUTPUTS / "balance_tasks.json")["tasks"]
    hard_tasks = read_json(OUTPUTS / "hard_diverse_tasks.json")["tasks"]

    tasks1_hashes = {task["metadata"]["canonical_hash"] for task in tasks1_tasks}
    tasks1_subset_mask = np.asarray(
        [
            task["metadata"]["canonical_hash"] in tasks1_hashes
            for task in balance_tasks
        ],
        dtype=bool,
    )
    if int(tasks1_subset_mask.sum()) != len(tasks1_tasks):
        raise ValueError(
            "tasks1_balanced is not preserved as an exact subset of balance_tasks."
        )
    addition_mask = ~tasks1_subset_mask

    descriptor_config = infer_descriptor_config(
        (original_tasks, balance_tasks, hard_tasks)
    )
    original_raw, layout = descriptor_matrix(original_tasks, descriptor_config)
    balance_raw, balance_layout = descriptor_matrix(balance_tasks, descriptor_config)
    hard_raw, hard_layout = descriptor_matrix(hard_tasks, descriptor_config)
    if layout != balance_layout or layout != hard_layout:
        raise AssertionError("Environment descriptor layouts do not match.")

    scaler = fit_robust_scaler(
        np.vstack([original_raw, balance_raw, hard_raw])
    )
    original_scaled = scaler.transform(original_raw)
    balance_scaled = scaler.transform(balance_raw)
    hard_scaled = scaler.transform(hard_raw)
    active_groups = group_active_indexes(layout, scaler)
    global_scaled = np.vstack([original_scaled, balance_scaled, hard_scaled])
    global_features = weighted_linear_features(global_scaled, active_groups)
    embedding, kl_divergence = standard_exact_tsne(global_features)
    trustworthiness = neighborhood_trustworthiness(
        np.sqrt(squared_euclidean_distances(global_features)),
        embedding,
        n_neighbors=10,
    )

    original_indexes = np.arange(len(original_tasks), dtype=int)
    balance_indexes = np.arange(
        len(original_tasks),
        len(original_tasks) + len(balance_tasks),
        dtype=int,
    )
    tasks1_indexes = balance_indexes[tasks1_subset_mask]
    addition_indexes = balance_indexes[addition_mask]
    hard_indexes = np.arange(
        len(original_tasks) + len(balance_tasks),
        len(global_scaled),
        dtype=int,
    )

    original_group = {
        "indexes": original_indexes.tolist(),
        "label": "Original",
        "color": COLORS["original"],
        "marker": "triangle",
        "radius": 6,
    }
    tasks1_group = {
        "indexes": tasks1_indexes.tolist(),
        "label": "tasks1",
        "color": COLORS["author"],
        "marker": "circle",
        "radius": 5,
    }
    balance_group = {
        "indexes": balance_indexes.tolist(),
        "label": "balance",
        "color": COLORS["modified"],
        "marker": "circle",
        "radius": 5,
    }
    additions_group = {
        "indexes": addition_indexes.tolist(),
        "label": "Added",
        "color": COLORS["modified"],
        "marker": "diamond",
        "radius": 6,
    }
    hard_group = {
        "indexes": hard_indexes.tolist(),
        "label": "hard",
        "color": HARD_COLOR,
        "marker": "diamond",
        "radius": 6,
    }

    width, height = 2500, 1660
    image = Image.new("RGBA", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text(
        (70, 42),
        "Standard t-SNE comparison of TABX scene banks",
        fill=TEXT_COLOR,
        font=font(44, bold=True),
    )
    draw.text(
        (72, 102),
        "Five panels share one global Euclidean t-SNE coordinate system",
        fill=MUTED_TEXT,
        font=font(25),
    )

    margin_x = 62
    gap_x = 26
    gap_y = 28
    top = 160
    bottom_margin = 76
    panel_width = (width - 2 * margin_x - 2 * gap_x) // 3
    panel_height = (height - top - bottom_margin - gap_y) // 2
    panel_boxes: list[tuple[int, int, int, int]] = []
    for row in range(2):
        for column in range(3):
            left = margin_x + column * (panel_width + gap_x)
            panel_top = top + row * (panel_height + gap_y)
            panel_boxes.append(
                (left, panel_top, left + panel_width, panel_top + panel_height)
            )

    panels = [
        ("A", "Original vs tasks1", [tasks1_group, original_group]),
        ("B", "Original vs balance_tasks", [balance_group, original_group]),
        ("C", "tasks1 vs balance additions", [tasks1_group, additions_group]),
        ("D", "Original vs hard tasks", [hard_group, original_group]),
        ("E", "tasks1 vs hard tasks", [tasks1_group, hard_group]),
    ]
    for panel_box, (panel_label, title, groups) in zip(panel_boxes, panels):
        draw_panel(
            draw,
            embedding,
            panel_box,
            panel_label=panel_label,
            title=title,
            groups=groups,
        )

    active_dimension_count = sum(len(indexes) for indexes in active_groups.values())
    draw_method_panel(
        draw,
        panel_boxes[5],
        descriptor_dimension=layout.dimension,
        active_dimensions=active_dimension_count,
        kl_divergence=kl_divergence,
        trustworthiness=trustworthiness,
    )

    footer = (
        f"n={len(global_scaled)} unique scenes | original=33 | tasks1={len(tasks1_tasks)} "
        f"(subset of balance) | balance={len(balance_tasks)} | hard={len(hard_tasks)}"
    )
    draw.text((70, height - 50), footer, fill=MUTED_TEXT, font=font(19))

    output_path = OUTPUTS / "standard_tsne_five_comparisons.png"
    image.convert("RGB").save(output_path, quality=96, optimize=True)

    metrics = {
        "method": {
            "algorithm": "standard exact t-SNE with Euclidean input distances",
            "global_embedding_fit_on": [
                "original scenes",
                "balance_tasks",
                "hard_diverse_tasks",
            ],
            "tasks1_relationship": "exact subset of balance_tasks",
            "descriptor_dimension": layout.dimension,
            "active_weighted_feature_dimensions": active_dimension_count,
            "group_weights": GROUP_WEIGHTS,
            "preprocessing": "robust scaling followed by group-balanced linear weighting",
        },
        "inventory": {
            "original": len(original_tasks),
            "tasks1_balanced": len(tasks1_tasks),
            "balance_tasks": len(balance_tasks),
            "balance_additions": int(addition_mask.sum()),
            "hard_diverse_tasks": len(hard_tasks),
            "global_embedding_points": len(global_scaled),
        },
        "embedding": {
            "perplexity": PERPLEXITY,
            "iterations": N_ITER,
            "random_seed": RANDOM_SEED,
            "kl_divergence": kl_divergence,
            "trustworthiness_at_10": trustworthiness,
        },
        "output": output_path.name,
    }
    metrics_path = OUTPUTS / "standard_tsne_five_metrics.json"
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


if __name__ == "__main__":
    main()
