"""Plot ten pairwise comparisons using a fixed four-bank t-SNE reference.

The five banks are:

* the 33 built-in scenes (20 unit/zone compositions plus 13 challenges);
* ``low_winrate_tasks.json``;
* ``outputs/balance_tasks.json``;
* ``outputs/balanced_diverse_tasks.json``;
* one projected task bank selected with ``--bridge-file``.

Only the first four banks fit the descriptor preprocessing and t-SNE.  Bridge
tasks are projected into that fixed coordinate system by inverse-distance
weighted nearest-neighbor interpolation.  Consequently, bridge tasks cannot
move any reference point, and all ten panels can be compared directly.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
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
    render_distribution_plot,
    weighted_linear_features,
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

REFERENCE_SET_ORDER = (
    "original",
    "low_winrate",
    "balance",
    "balanced_diverse",
)
SET_ORDER = REFERENCE_SET_ORDER + ("bridge",)
BRIDGE_PROJECTION_NEIGHBORS = 10

SET_STYLE: dict[str, dict[str, Any]] = {
    "original": {
        "label": "Original scenes",
        "short_label": "Original",
        "color": COLORS["original"],
        "marker": "triangle",
        "radius": 8,
    },
    "low_winrate": {
        "label": "low_winrate_tasks",
        "short_label": "Low-win-rate",
        "color": COLORS["author"],
        "marker": "circle",
        "radius": 7,
    },
    "balance": {
        "label": "balance_tasks",
        "short_label": "Balance",
        "color": COLORS["modified"],
        "marker": "circle",
        "radius": 6,
    },
    "balanced_diverse": {
        "label": "balanced_diverse_tasks",
        "short_label": "Balanced-diverse",
        "color": (139, 92, 185, 210),
        "marker": "diamond",
        "radius": 7,
    },
    "bridge": {
        "label": "bridge_tasks_balanced",
        "short_label": "Bridge",
        "color": (211, 78, 100, 210),
        "marker": "circle",
        "radius": 7,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit t-SNE on the fixed four reference banks and project one "
            "additional task bank without refitting."
        )
    )
    parser.add_argument(
        "--bridge-file",
        type=Path,
        default=OUTPUTS / "bridge_tasks_balanced.json",
        help="JSON task bank to project into the fixed four-bank embedding.",
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help=(
            "Filename-safe tag for projected-bank outputs. "
            "Defaults to the bridge file stem."
        ),
    )
    parser.add_argument(
        "--bridge-label",
        default=None,
        help="Full projected-bank label used in standalone plot legends.",
    )
    parser.add_argument(
        "--bridge-short-label",
        default=None,
        help="Short projected-bank label used in the overview.",
    )
    return parser.parse_args()


def group_spec(
    key: str,
    indexes: np.ndarray,
    *,
    compact: bool,
) -> dict[str, Any]:
    """Build a plotting group with stable styling across every panel."""

    style = SET_STYLE[key]
    return {
        "indexes": indexes.tolist(),
        "label": style["short_label"] if compact else style["label"],
        "color": style["color"],
        "marker": style["marker"],
        "radius": max(int(style["radius"]) - 2, 4) if compact else style["radius"],
    }


def project_into_reference_embedding(
    reference_features: np.ndarray,
    reference_embedding: np.ndarray,
    query_features: np.ndarray,
    *,
    n_neighbors: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project queries without refitting or moving the reference embedding.

    Each query uses inverse-squared-distance weights over its nearest reference
    descriptors.  Exact descriptor matches are placed at the mean coordinates
    of all exact matches among the selected neighbors.
    """

    if len(reference_features) != len(reference_embedding):
        raise ValueError("Reference features and embedding must have equal lengths.")
    if not 1 <= n_neighbors <= len(reference_features):
        raise ValueError("n_neighbors must be between 1 and the reference size.")

    reference_norms = np.sum(np.square(reference_features), axis=1)
    query_norms = np.sum(np.square(query_features), axis=1)
    distances_squared = (
        query_norms[:, None]
        + reference_norms[None, :]
        - 2.0 * query_features @ reference_features.T
    )
    distances_squared = np.maximum(distances_squared, 0.0)
    neighbor_indexes = np.argpartition(
        distances_squared,
        kth=n_neighbors - 1,
        axis=1,
    )[:, :n_neighbors]
    neighbor_distances_squared = np.take_along_axis(
        distances_squared,
        neighbor_indexes,
        axis=1,
    )
    neighbor_order = np.argsort(neighbor_distances_squared, axis=1)
    neighbor_indexes = np.take_along_axis(
        neighbor_indexes,
        neighbor_order,
        axis=1,
    )
    neighbor_distances_squared = np.take_along_axis(
        neighbor_distances_squared,
        neighbor_order,
        axis=1,
    )

    weights = np.zeros_like(neighbor_distances_squared)
    for row, row_distances_squared in enumerate(neighbor_distances_squared):
        exact_matches = row_distances_squared <= 1e-18
        if np.any(exact_matches):
            weights[row, exact_matches] = 1.0 / np.count_nonzero(exact_matches)
        else:
            inverse_squared = 1.0 / np.maximum(row_distances_squared, 1e-18)
            weights[row] = inverse_squared / inverse_squared.sum()

    neighbor_embedding = reference_embedding[neighbor_indexes]
    projected = np.sum(weights[:, :, None] * neighbor_embedding, axis=1)
    nearest_distances = np.sqrt(neighbor_distances_squared[:, 0])
    diagnostics = {
        "method": "inverse-squared-distance weighted k-nearest-neighbor interpolation",
        "n_neighbors": n_neighbors,
        "reference_points": len(reference_features),
        "projected_points": len(query_features),
        "nearest_reference_distance": {
            "minimum": float(np.min(nearest_distances)),
            "median": float(np.median(nearest_distances)),
            "mean": float(np.mean(nearest_distances)),
            "p90": float(np.percentile(nearest_distances, 90)),
            "maximum": float(np.max(nearest_distances)),
        },
    }
    return projected, diagnostics


def render_overview(
    output_path: Path,
    embedding: np.ndarray,
    indexes_by_set: dict[str, np.ndarray],
    *,
    descriptor_dimension: int,
    active_dimensions: int,
    kl_divergence: float,
    trustworthiness: float,
    reference_embedding: np.ndarray,
    projected_points: int,
) -> None:
    """Render a compact 5-by-2 overview using the shared global coordinates."""

    width, height = 2500, 3600
    image = Image.new("RGBA", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text(
        (70, 42),
        "Pairwise standard t-SNE comparison of five TABX scene banks",
        fill=TEXT_COLOR,
        font=font(42, bold=True),
    )
    draw.text(
        (72, 101),
        "Fixed four-bank t-SNE coordinates; Bridge is projected without refitting",
        fill=MUTED_TEXT,
        font=font(24),
    )

    margin_x = 62
    gap_x = 30
    gap_y = 28
    top = 160
    bottom_margin = 76
    n_columns = 2
    n_rows = 5
    panel_width = (width - 2 * margin_x - gap_x) // n_columns
    panel_height = (
        height - top - bottom_margin - (n_rows - 1) * gap_y
    ) // n_rows
    panel_boxes: list[tuple[int, int, int, int]] = []
    for row in range(n_rows):
        for column in range(n_columns):
            left = margin_x + column * (panel_width + gap_x)
            panel_top = top + row * (panel_height + gap_y)
            panel_boxes.append(
                (left, panel_top, left + panel_width, panel_top + panel_height)
            )

    for panel_index, ((first, second), panel_box) in enumerate(
        zip(combinations(SET_ORDER, 2), panel_boxes),
        start=1,
    ):
        keys_by_draw_order = sorted(
            (first, second),
            key=lambda key: len(indexes_by_set[key]),
            reverse=True,
        )
        groups = [
            group_spec(key, indexes_by_set[key], compact=True)
            for key in keys_by_draw_order
        ]
        title = (
            f"{SET_STYLE[first]['short_label']} vs "
            f"{SET_STYLE[second]['short_label']}"
        )
        draw_panel(
            draw,
            embedding,
            panel_box,
            panel_label=chr(64 + panel_index),
            title=title,
            groups=groups,
            extent_embedding=reference_embedding,
        )

    footer = (
        f"{descriptor_dimension}-D descriptor -> {active_dimensions} active features | "
        f"exact t-SNE fit n={len(reference_embedding)} + projected Bridge "
        f"n={projected_points} | perplexity={PERPLEXITY:g} | seed={RANDOM_SEED} | "
        f"KL={kl_divergence:.3f} | T(10)={trustworthiness:.3f}"
    )
    draw.text((70, height - 50), footer, fill=MUTED_TEXT, font=font(19))
    image.convert("RGB").save(output_path, quality=96, optimize=True)


def main() -> None:
    args = parse_args()
    bridge_path = args.bridge_file.resolve()
    if not bridge_path.is_file():
        raise FileNotFoundError(f"Bridge task bank not found: {bridge_path}")
    output_tag = args.output_tag or bridge_path.stem
    if not output_tag or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in output_tag
    ):
        raise ValueError(
            "output_tag may contain only letters, digits, underscores, and hyphens."
        )
    SET_STYLE["bridge"]["label"] = args.bridge_label or bridge_path.stem
    SET_STYLE["bridge"]["short_label"] = (
        args.bridge_short_label
        or bridge_path.stem.replace("_tasks", "").replace("_", " ").title()
    )

    original_tasks, _, original_counts = load_original_scenes()
    tasks_by_set: dict[str, list[dict[str, Any]]] = {
        "original": original_tasks,
        "low_winrate": read_json(ROOT / "low_winrate_tasks.json")["tasks"],
        "balance": read_json(OUTPUTS / "balance_tasks.json")["tasks"],
        "balanced_diverse": read_json(
            OUTPUTS / "balanced_diverse_tasks.json"
        )["tasks"],
        "bridge": read_json(bridge_path)["tasks"],
    }

    descriptor_config = infer_descriptor_config(
        tasks_by_set[key]
        for key in REFERENCE_SET_ORDER
    )
    raw_by_set: dict[str, np.ndarray] = {}
    layout = None
    for key in SET_ORDER:
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
    scaler = fit_robust_scaler(reference_raw)
    scaled_by_set = {
        key: scaler.transform(raw_by_set[key])
        for key in SET_ORDER
    }
    active_groups = group_active_indexes(layout, scaler)
    reference_scaled = np.vstack(
        [scaled_by_set[key] for key in REFERENCE_SET_ORDER]
    )
    reference_features = weighted_linear_features(
        reference_scaled,
        active_groups,
    )
    reference_embedding, kl_divergence = standard_exact_tsne(
        reference_features
    )
    high_distances = np.sqrt(
        squared_euclidean_distances(reference_features)
    )
    trustworthiness = neighborhood_trustworthiness(
        high_distances,
        reference_embedding,
        n_neighbors=10,
    )
    bridge_features = weighted_linear_features(
        scaled_by_set["bridge"],
        active_groups,
    )
    bridge_embedding, bridge_projection = project_into_reference_embedding(
        reference_features,
        reference_embedding,
        bridge_features,
        n_neighbors=BRIDGE_PROJECTION_NEIGHBORS,
    )
    embedding = np.vstack([reference_embedding, bridge_embedding])

    indexes_by_set: dict[str, np.ndarray] = {}
    offset = 0
    for key in SET_ORDER:
        next_offset = offset + len(tasks_by_set[key])
        indexes_by_set[key] = np.arange(offset, next_offset, dtype=int)
        offset = next_offset

    active_dimension_count = sum(
        len(indexes)
        for indexes in active_groups.values()
    )
    reference_footer = (
        f"{layout.dimension}-D descriptor -> {active_dimension_count} active features | "
        f"standard exact t-SNE (Euclidean) | perplexity={PERPLEXITY:g} | "
        f"seed={RANDOM_SEED} | KL={kl_divergence:.3f} | T(10)={trustworthiness:.3f}"
    )
    bridge_footer = (
        f"{layout.dimension}-D descriptor -> {active_dimension_count} active features | "
        f"fixed t-SNE n={len(reference_embedding)} + projected Bridge "
        f"n={len(bridge_embedding)} | k={BRIDGE_PROJECTION_NEIGHBORS} | "
        f"seed={RANDOM_SEED}"
    )

    panel_files: list[str] = []
    for first, second in combinations(SET_ORDER, 2):
        first_filename_key = output_tag if first == "bridge" else first
        second_filename_key = output_tag if second == "bridge" else second
        output_path = (
            OUTPUTS
            / (
                f"standard_tsne_pairwise_{first_filename_key}"
                f"_vs_{second_filename_key}.png"
            )
        )
        keys_by_draw_order = sorted(
            (first, second),
            key=lambda key: len(indexes_by_set[key]),
            reverse=True,
        )
        groups = [
            group_spec(key, indexes_by_set[key], compact=False)
            for key in keys_by_draw_order
        ]
        includes_bridge = "bridge" in (first, second)
        if includes_bridge:
            subtitle = "Fixed four-bank t-SNE; Bridge projected without refitting"
            summary_lines = [
                f"Reference t-SNE fit: n={len(reference_embedding)} scenes",
                (
                    f"Bridge: n={len(bridge_embedding)}, "
                    f"k={BRIDGE_PROJECTION_NEIGHBORS} neighbor projection"
                ),
                "Bridge is excluded from scaling and t-SNE fitting",
            ]
            panel_footer = bridge_footer
        else:
            subtitle = (
                "Shared Euclidean embedding of all four full-environment "
                "scene descriptors"
            )
            summary_lines = [
                f"Global embedding: n={len(reference_embedding)} scenes",
                "Robust-scaled, group-balanced environment features",
                "All six figures use identical t-SNE coordinates",
            ]
            panel_footer = reference_footer
        render_distribution_plot(
            output_path,
            embedding,
            groups,
            [],
            title=(
                f"Standard t-SNE: {SET_STYLE[first]['label']} vs "
                f"{SET_STYLE[second]['label']}"
            ),
            subtitle=subtitle,
            footer=panel_footer,
            summary_lines=summary_lines,
            extent_embedding=reference_embedding,
        )
        panel_files.append(output_path.name)

    overview_path = (
        OUTPUTS
        / f"standard_tsne_{output_tag}_pairwise_comparisons.png"
    )
    render_overview(
        overview_path,
        embedding,
        indexes_by_set,
        descriptor_dimension=layout.dimension,
        active_dimensions=active_dimension_count,
        kl_divergence=kl_divergence,
        trustworthiness=trustworthiness,
        reference_embedding=reference_embedding,
        projected_points=len(bridge_embedding),
    )

    metrics = {
        "method": {
            "algorithm": "standard exact t-SNE with Euclidean input distances",
            "implementation": "self-contained NumPy",
            "fixed_reference_embedding_reused_across_all_ten_panels": True,
            "reference_embedding_fit_on": list(REFERENCE_SET_ORDER),
            "bridge_participates_in_descriptor_config_fit": False,
            "bridge_participates_in_robust_scaler_fit": False,
            "bridge_participates_in_tsne_fit": False,
            "descriptor_dimension": layout.dimension,
            "active_weighted_feature_dimensions": active_dimension_count,
            "descriptor_config": {
                "max_units_per_team": descriptor_config.max_units_per_team,
                "max_zones": descriptor_config.max_zones,
                "attack_type_values": descriptor_config.attack_type_values,
            },
            "preprocessing": (
                "four-reference-bank robust scaling followed by "
                "group-balanced linear weighting"
            ),
            "group_weights": GROUP_WEIGHTS,
            "bridge_projection": bridge_projection,
            "plot_style": "scatter only; no hull or enclosing boundary",
        },
        "sources": {
            "original": "src/tabx/scenarios/{units,zones,challenges}",
            "low_winrate": "low_winrate_tasks.json",
            "balance": "outputs/balance_tasks.json",
            "balanced_diverse": "outputs/balanced_diverse_tasks.json",
            "bridge": str(bridge_path),
        },
        "inventory": {
            "original": original_counts,
            "low_winrate": len(tasks_by_set["low_winrate"]),
            "balance": len(tasks_by_set["balance"]),
            "balanced_diverse": len(tasks_by_set["balanced_diverse"]),
            "bridge": len(tasks_by_set["bridge"]),
            "reference_fit_points": len(reference_embedding),
            "projected_bridge_points": len(bridge_embedding),
            "total_display_points": len(embedding),
        },
        "embedding": {
            "n_fit_points": len(reference_embedding),
            "n_display_points": len(embedding),
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
        OUTPUTS
        / f"standard_tsne_{output_tag}_pairwise_metrics.json"
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


if __name__ == "__main__":
    main()
