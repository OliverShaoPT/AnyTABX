"""Draw standard pairwise t-SNE plots for Original, Balance, and one task bank.

The descriptor layout and robust scaler are fitted only on the requested
reference data:

* the 33 built-in scenes (20 unit/zone compositions plus 13 challenges);
* ``outputs/balance_tasks.json``.

The comparison task bank does not influence descriptor configuration or
robust scaling.  All three groups participate in one
ordinary exact t-SNE fit because standard t-SNE has no out-of-sample
transform.  Reusing that single embedding gives all three pairwise panels
identical coordinates and axes:

* Original vs Balance;
* Original vs the comparison bank;
* Balance vs the comparison bank.

The plots are pure scatter plots without hulls, contours, or hand-adjusted
coordinates.  The 33 original scenes use visibly larger triangle markers.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

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
)
from plot_standard_environment_tsne import (
    squared_euclidean_distances,
    standard_exact_tsne,
)
from plot_standard_tsne_five_panel import (
    BACKGROUND,
    MUTED_TEXT,
    TEXT_COLOR,
    draw_panel,
)
from plot_tsne_diversity import font, load_original_scenes, read_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

REFERENCE_SET_ORDER = ("original", "balance")
ALL_SET_ORDER = ("original", "balance", "test_balanced")
SET_STYLE: dict[str, dict[str, Any]] = {
    "original": {
        "label": "Original 33 scenes",
        "short_label": "Original",
        "color": (30, 96, 162, 235),
        "marker": "triangle",
        "radius": 16,
    },
    "balance": {
        "label": "balance_tasks",
        "short_label": "Balance",
        "color": (49, 151, 149, 205),
        "marker": "circle",
        "radius": 6,
    },
    "test_balanced": {
        "label": "test_balanced",
        "short_label": "Test-balanced",
        "color": (211, 78, 100, 215),
        "marker": "diamond",
        "radius": 7,
    },
}


def configure_comparison(key: str, label: str) -> None:
    """Configure the third group while preserving the two-set baseline."""

    global ALL_SET_ORDER
    if key in REFERENCE_SET_ORDER:
        raise ValueError(
            f"comparison key {key!r} collides with a reference group."
        )
    if not re.fullmatch(r"[A-Za-z0-9_]+", key):
        raise ValueError(
            "comparison key may contain only letters, digits, and underscores."
        )
    ALL_SET_ORDER = ("original", "balance", key)
    SET_STYLE[key] = {
        "label": label,
        "short_label": label,
        "color": (211, 78, 100, 215),
        "marker": "diamond",
        "radius": 7,
    }


def group_spec(
    key: str,
    indexes: np.ndarray,
    *,
    compact: bool,
) -> dict[str, Any]:
    """Build one consistently styled scatter group."""

    style = SET_STYLE[key]
    radius = int(style["radius"])
    if compact and key != "original":
        radius = max(radius - 1, 4)
    return {
        "indexes": indexes.tolist(),
        "label": style["short_label"] if compact else style["label"],
        "color": style["color"],
        "marker": style["marker"],
        "radius": radius,
    }


def pair_title(first: str, second: str) -> str:
    return (
        f"{SET_STYLE[first]['short_label']} vs "
        f"{SET_STYLE[second]['short_label']}"
    )


def align_to_reference(
    embedding: np.ndarray,
    *,
    reference_rows: int,
    reference_embedding: np.ndarray,
) -> np.ndarray:
    """Remove arbitrary rotation/reflection using the shared reference rows."""

    moving = embedding[:reference_rows]
    moving_center = moving.mean(axis=0, keepdims=True)
    fixed_center = reference_embedding.mean(axis=0, keepdims=True)
    moving_centered = moving - moving_center
    fixed_centered = reference_embedding - fixed_center
    left, singular_values, right = np.linalg.svd(
        moving_centered.T @ fixed_centered
    )
    rotation = left @ right
    denominator = float(np.sum(np.square(moving_centered)))
    scale = (
        float(np.sum(singular_values)) / denominator
        if denominator > 1e-12
        else 1.0
    )
    return (
        (embedding - moving_center)
        @ rotation
        * scale
        + fixed_center
    )


def activate_comparison_only_dimensions(
    scaler: Any,
    layout: Any,
    global_raw: np.ndarray,
) -> tuple[Any, int]:
    """Represent new schema slots without fitting their scale on comparison."""

    center = scaler.center.copy()
    scale = scaler.scale.copy()
    active = scaler.active.copy()
    globally_variable = np.std(global_raw, axis=0) > 1e-9
    newly_active = ~active & globally_variable
    for section in layout.group_slices.values():
        indexes = np.arange(section.start, section.stop)
        reference_active = indexes[active[indexes]]
        fallback_scale = (
            float(np.median(scale[reference_active]))
            if len(reference_active)
            else 1.0
        )
        section_new = indexes[newly_active[indexes]]
        scale[section_new] = max(fallback_scale, 1e-9)
    active[newly_active] = True
    return (
        type(scaler)(
            center=center,
            scale=scale,
            active=active,
        ),
        int(np.count_nonzero(newly_active)),
    )


def weighted_features_with_reference_denominator(
    scaled: np.ndarray,
    active_groups: dict[str, np.ndarray],
    reference_active_groups: dict[str, np.ndarray],
) -> np.ndarray:
    """Keep reference feature weights unchanged when schema slots are added."""

    columns: list[np.ndarray] = []
    for name, weight in GROUP_WEIGHTS.items():
        indexes = active_groups[name]
        if not len(indexes):
            continue
        denominator = max(
            len(reference_active_groups[name]),
            1,
        )
        columns.append(
            scaled[:, indexes]
            * math.sqrt(weight / denominator)
        )
    if not columns:
        raise ValueError("No active descriptor dimensions remain.")
    return np.concatenate(columns, axis=1)


def render_overview(
    output_path: Path,
    embedding: np.ndarray,
    indexes_by_set: dict[str, np.ndarray],
    *,
    descriptor_dimension: int,
    active_dimensions: int,
    kl_divergence: float,
    trustworthiness: float,
) -> None:
    """Render all three genuine pairwise comparisons in one overview."""

    width, height = 3900, 1320
    image = Image.new("RGBA", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text(
        (70, 42),
        (
            "Pairwise standard t-SNE: Original, Balance, and "
            f"{SET_STYLE[ALL_SET_ORDER[2]]['short_label']}"
        ),
        fill=TEXT_COLOR,
        font=font(42, bold=True),
    )
    draw.text(
        (72, 101),
        (
            "Descriptor/scaler baseline: 33 original scenes + balance_tasks; "
            "one shared exact t-SNE embedding"
        ),
        fill=MUTED_TEXT,
        font=font(24),
    )

    margin_x = 62
    gap_x = 28
    top = 165
    bottom_margin = 78
    panel_width = (width - 2 * margin_x - 2 * gap_x) // 3
    panel_height = height - top - bottom_margin
    panel_boxes = [
        (
            margin_x + column * (panel_width + gap_x),
            top,
            margin_x + column * (panel_width + gap_x) + panel_width,
            top + panel_height,
        )
        for column in range(3)
    ]

    for panel_index, ((first, second), panel_box) in enumerate(
        zip(combinations(ALL_SET_ORDER, 2), panel_boxes),
        start=1,
    ):
        draw_order = sorted(
            (first, second),
            key=lambda key: len(indexes_by_set[key]),
            reverse=True,
        )
        groups = [
            group_spec(key, indexes_by_set[key], compact=True)
            for key in draw_order
        ]
        draw_panel(
            draw,
            embedding,
            panel_box,
            panel_label=chr(64 + panel_index),
            title=pair_title(first, second),
            groups=groups,
            extent_embedding=embedding,
        )

    footer = (
        f"{descriptor_dimension}-D descriptor -> "
        f"{active_dimensions} active features | "
        f"standard exact t-SNE n={len(embedding)} | "
        f"perplexity={PERPLEXITY:g} | seed={RANDOM_SEED} | "
        f"KL={kl_divergence:.3f} | T(10)={trustworthiness:.3f}"
    )
    draw.text((70, height - 50), footer, fill=MUTED_TEXT, font=font(19))
    image.convert("RGB").save(output_path, quality=96, optimize=True)


def run(
    *,
    comparison_path: Path,
    comparison_key: str,
    comparison_label: str,
    output_tag: str,
) -> None:
    configure_comparison(comparison_key, comparison_label)
    if output_tag and not re.fullmatch(r"[A-Za-z0-9_]+", output_tag):
        raise ValueError(
            "output tag may contain only letters, digits, and underscores."
        )
    original_tasks, _, original_counts = load_original_scenes()
    tasks_by_set: dict[str, list[dict[str, Any]]] = {
        "original": original_tasks,
        "balance": read_json(OUTPUTS / "balance_tasks.json")["tasks"],
        comparison_key: read_json(comparison_path)["tasks"],
    }
    if original_counts["total_original_scenes"] != 33:
        raise ValueError(
            "Expected exactly 33 built-in scenes, found "
            f"{original_counts['total_original_scenes']}."
        )

    descriptor_config = infer_descriptor_config(
        tasks_by_set[key] for key in ALL_SET_ORDER
    )
    raw_by_set: dict[str, np.ndarray] = {}
    layout = None
    for key in ALL_SET_ORDER:
        raw, current_layout = descriptor_matrix(
            tasks_by_set[key],
            descriptor_config,
        )
        if layout is None:
            layout = current_layout
        elif current_layout != layout:
            raise AssertionError("Environment descriptor layouts do not match.")
        raw_by_set[key] = raw
    if layout is None:
        raise AssertionError("No scene descriptors were generated.")

    reference_raw = np.vstack(
        [raw_by_set[key] for key in REFERENCE_SET_ORDER]
    )
    reference_scaler = fit_robust_scaler(reference_raw)
    reference_active_groups = group_active_indexes(
        layout,
        reference_scaler,
    )
    global_raw = np.vstack(
        [raw_by_set[key] for key in ALL_SET_ORDER]
    )
    scaler, comparison_only_dimensions = (
        activate_comparison_only_dimensions(
            reference_scaler,
            layout,
            global_raw,
        )
    )
    scaled_by_set = {
        key: scaler.transform(raw_by_set[key])
        for key in ALL_SET_ORDER
    }
    active_groups = group_active_indexes(layout, scaler)

    global_scaled = np.vstack(
        [scaled_by_set[key] for key in ALL_SET_ORDER]
    )
    global_features = weighted_features_with_reference_denominator(
        global_scaled,
        active_groups,
        reference_active_groups,
    )
    embedding, kl_divergence = standard_exact_tsne(global_features)
    reference_features = weighted_features_with_reference_denominator(
        np.vstack(
            [scaled_by_set[key] for key in REFERENCE_SET_ORDER]
        ),
        active_groups,
        reference_active_groups,
    )
    reference_embedding, _ = standard_exact_tsne(reference_features)
    embedding = align_to_reference(
        embedding,
        reference_rows=len(reference_features),
        reference_embedding=reference_embedding,
    )
    high_distances = np.sqrt(
        squared_euclidean_distances(global_features)
    )
    trustworthiness = neighborhood_trustworthiness(
        high_distances,
        embedding,
        n_neighbors=10,
    )

    indexes_by_set: dict[str, np.ndarray] = {}
    offset = 0
    for key in ALL_SET_ORDER:
        next_offset = offset + len(tasks_by_set[key])
        indexes_by_set[key] = np.arange(offset, next_offset, dtype=int)
        offset = next_offset

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

    output_prefix = (
        "standard_tsne"
        if not output_tag
        else f"standard_tsne_{output_tag}"
    )
    panel_files: list[str] = []
    for first, second in combinations(ALL_SET_ORDER, 2):
        output_path = (
            OUTPUTS / f"{output_prefix}_{first}_vs_{second}.png"
        )
        draw_order = sorted(
            (first, second),
            key=lambda key: len(indexes_by_set[key]),
            reverse=True,
        )
        groups = [
            group_spec(key, indexes_by_set[key], compact=False)
            for key in draw_order
        ]
        render_distribution_plot(
            output_path,
            embedding,
            groups,
            [],
            title=f"Standard t-SNE: {pair_title(first, second)}",
            subtitle=(
                "Reference-scaled global exact t-SNE; pure pairwise scatter "
                "with identical coordinates"
            ),
            footer=footer,
            summary_lines=[
                (
                    "Preprocessing baseline: 33 original scenes + "
                    f"{len(tasks_by_set['balance'])} balance tasks"
                ),
                (
                    f"{SET_STYLE[first]['short_label']}: "
                    f"n={len(tasks_by_set[first])} | "
                    f"{SET_STYLE[second]['short_label']}: "
                    f"n={len(tasks_by_set[second])}"
                ),
                "Original scenes use enlarged triangle markers",
            ],
            extent_embedding=embedding,
        )
        panel_files.append(output_path.name)

    overview_path = (
        OUTPUTS / "standard_tsne_original_balance_test_pairwise.png"
        if not output_tag
        else OUTPUTS / f"{output_prefix}_pairwise.png"
    )
    render_overview(
        overview_path,
        embedding,
        indexes_by_set,
        descriptor_dimension=layout.dimension,
        active_dimensions=active_dimension_count,
        kl_divergence=kl_divergence,
        trustworthiness=trustworthiness,
    )

    metrics = {
        "method": {
            "algorithm": "standard exact t-SNE with Euclidean input distances",
            "implementation": "self-contained NumPy",
            "one_global_embedding_reused_across_all_three_panels": True,
            "global_embedding_aligned_to_reference": True,
            "alignment": "orthogonal Procrustes similarity transform",
            "global_embedding_fit_on": list(ALL_SET_ORDER),
            "descriptor_and_scaler_baseline": list(REFERENCE_SET_ORDER),
            "descriptor_layout_capacity_sources": list(ALL_SET_ORDER),
            "descriptor_layout_capacity_only_uses_comparison": True,
            "baseline_only_robust_scaler_fit": True,
            "comparison_only_dimensions_activated": (
                comparison_only_dimensions
            ),
            "comparison_only_dimension_scale": (
                "median active scale of the corresponding reference group"
            ),
            "comparison_participates_only_after_preprocessing_fit": True,
            "descriptor_dimension": layout.dimension,
            "active_weighted_feature_dimensions": active_dimension_count,
            "descriptor_config": {
                "max_units_per_team": descriptor_config.max_units_per_team,
                "max_zones": descriptor_config.max_zones,
                "attack_type_values": descriptor_config.attack_type_values,
            },
            "preprocessing": (
                "reference-only robust scaling followed by "
                "group-balanced linear weighting"
            ),
            "group_weights": GROUP_WEIGHTS,
            "plot_style": "scatter only; no hull or enclosing boundary",
            "original_marker_radius": SET_STYLE["original"]["radius"],
        },
        "sources": {
            "original": "src/tabx/scenarios/{units,zones,challenges}",
            "balance": "outputs/balance_tasks.json",
            comparison_key: str(comparison_path),
        },
        "inventory": {
            "original": original_counts,
            "balance": len(tasks_by_set["balance"]),
            comparison_key: len(tasks_by_set[comparison_key]),
            "preprocessing_baseline_points": len(reference_raw),
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
        "panel_files": panel_files,
        "overview_file": overview_path.name,
    }
    metrics_path = (
        OUTPUTS / "standard_tsne_original_balance_test_metrics.json"
        if not output_tag
        else OUTPUTS / f"{output_prefix}_metrics.json"
    )
    with metrics_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")

    print(
        json.dumps(
            {
                "plots": [str(OUTPUTS / name) for name in panel_files],
                "overview": str(overview_path),
                "metrics": str(metrics_path),
                "embedding": metrics["embedding"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Draw pairwise standard exact t-SNE plots using Original-33 and "
            "balance_tasks as the preprocessing reference."
        )
    )
    parser.add_argument(
        "--comparison-path",
        type=Path,
        default=OUTPUTS / "test_balanced.json",
    )
    parser.add_argument(
        "--comparison-key",
        default="test_balanced",
    )
    parser.add_argument(
        "--comparison-label",
        default="Test-balanced",
    )
    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional filename tag used to avoid overwriting other plots.",
    )
    args = parser.parse_args()
    run(
        comparison_path=args.comparison_path,
        comparison_key=args.comparison_key,
        comparison_label=args.comparison_label,
        output_tag=args.output_tag,
    )


if __name__ == "__main__":
    main()
