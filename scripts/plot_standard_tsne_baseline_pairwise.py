"""Compare three JSON task banks with one baseline-defined standard t-SNE.

The descriptor layout and robust scaler are fitted only on:

* the 33 built-in scenes (20 unit/zone compositions plus 13 challenges);
* ``outputs/balance_tasks.json``.

``bridge_tasks_balanced`` and ``hard_diverse_tasks`` are excluded from
descriptor-layout inference and robust-scaler fitting.  All four banks then
participate in one ordinary exact t-SNE fit, so every displayed point is a
genuine t-SNE result and the three exported JSON-bank comparisons use
identical axes.  Every panel also shows the 33 built-in scenes as a third,
shared reference group:

* balance vs bridge;
* balance vs hard;
* bridge vs hard.
"""

from __future__ import annotations

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

REFERENCE_SET_ORDER = ("original", "balance")
JSON_SET_ORDER = ("balance", "bridge", "hard")
ALL_SET_ORDER = ("original",) + JSON_SET_ORDER
SET_STYLE: dict[str, dict[str, Any]] = {
    "original": {
        "label": "Original scenes",
        "short_label": "Original",
        "color": COLORS["original"],
        "marker": "triangle",
        "radius": 14,
    },
    "balance": {
        "label": "balance_tasks",
        "short_label": "Balance",
        "color": COLORS["modified"],
        "marker": "circle",
        "radius": 6,
    },
    "bridge": {
        "label": "bridge_tasks_balanced",
        "short_label": "Bridge",
        "color": (211, 78, 100, 210),
        "marker": "diamond",
        "radius": 7,
    },
    "hard": {
        "label": "hard_diverse_tasks",
        "short_label": "Hard-diverse",
        "color": (139, 92, 185, 210),
        "marker": "circle",
        "radius": 7,
    },
}


def group_spec(
    key: str,
    indexes: np.ndarray,
    *,
    compact: bool,
) -> dict[str, Any]:
    """Build one consistently styled point group."""

    style = SET_STYLE[key]
    return {
        "indexes": indexes.tolist(),
        "label": style["short_label"] if compact else style["label"],
        "color": style["color"],
        "marker": style["marker"],
        "radius": max(int(style["radius"]) - 2, 4) if compact else style["radius"],
    }


def render_overview(
    output_path: Path,
    embedding: np.ndarray,
    indexes_by_set: dict[str, np.ndarray],
    *,
    descriptor_dimension: int,
    active_dimensions: int,
    kl_divergence: float,
    trustworthiness: float,
    fit_points: int,
) -> None:
    """Render the three JSON-bank comparisons in one horizontal overview."""

    width, height = 3900, 1320
    image = Image.new("RGBA", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text(
        (70, 42),
        "Pairwise standard t-SNE comparison of three TABX task banks",
        fill=TEXT_COLOR,
        font=font(42, bold=True),
    )
    draw.text(
        (72, 101),
        (
            "Descriptor/scaler baseline: 33 original scenes + balance_tasks; "
            "all panels show Original and share one exact t-SNE fit"
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
        zip(combinations(JSON_SET_ORDER, 2), panel_boxes),
        start=1,
    ):
        keys_by_draw_order = sorted(
            ("original", first, second),
            key=lambda key: len(indexes_by_set[key]),
            reverse=True,
        )
        groups = [
            group_spec(key, indexes_by_set[key], compact=True)
            for key in keys_by_draw_order
        ]
        title = (
            f"{SET_STYLE[first]['short_label']} vs "
            f"{SET_STYLE[second]['short_label']} (+ Original)"
        )
        draw_panel(
            draw,
            embedding,
            panel_box,
            panel_label=chr(64 + panel_index),
            title=title,
            groups=groups,
            extent_embedding=embedding,
        )

    footer = (
        f"{descriptor_dimension}-D descriptor -> {active_dimensions} active features | "
        f"standard exact t-SNE n={fit_points} | "
        f"perplexity={PERPLEXITY:g} | seed={RANDOM_SEED} | "
        f"KL={kl_divergence:.3f} | T(10)={trustworthiness:.3f}"
    )
    draw.text((70, height - 50), footer, fill=MUTED_TEXT, font=font(19))
    image.convert("RGB").save(output_path, quality=96, optimize=True)


def main() -> None:
    original_tasks, _, original_counts = load_original_scenes()
    tasks_by_set: dict[str, list[dict[str, Any]]] = {
        "original": original_tasks,
        "balance": read_json(OUTPUTS / "balance_tasks.json")["tasks"],
        "bridge": read_json(OUTPUTS / "bridge_tasks_balanced.json")["tasks"],
        "hard": read_json(OUTPUTS / "hard_diverse_tasks.json")["tasks"],
    }
    if original_counts["total_original_scenes"] != 33:
        raise ValueError(
            "Expected exactly 33 built-in scenes, found "
            f"{original_counts['total_original_scenes']}."
        )

    descriptor_config = infer_descriptor_config(
        tasks_by_set[key] for key in REFERENCE_SET_ORDER
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
    scaler = fit_robust_scaler(reference_raw)
    scaled_by_set = {
        key: scaler.transform(raw_by_set[key])
        for key in ALL_SET_ORDER
    }
    active_groups = group_active_indexes(layout, scaler)

    global_scaled = np.vstack(
        [scaled_by_set[key] for key in ALL_SET_ORDER]
    )
    global_features = weighted_linear_features(
        global_scaled,
        active_groups,
    )
    embedding, kl_divergence = standard_exact_tsne(
        global_features
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
        f"{layout.dimension}-D descriptor -> {active_dimension_count} active features | "
        f"standard exact t-SNE n={len(embedding)} | "
        f"perplexity={PERPLEXITY:g} | seed={RANDOM_SEED} | "
        f"KL={kl_divergence:.3f} | T(10)={trustworthiness:.3f}"
    )

    panel_files: list[str] = []
    for first, second in combinations(JSON_SET_ORDER, 2):
        output_path = (
            OUTPUTS
            / f"standard_tsne_baseline_pairwise_{first}_vs_{second}.png"
        )
        keys_by_draw_order = sorted(
            ("original", first, second),
            key=lambda key: len(indexes_by_set[key]),
            reverse=True,
        )
        groups = [
            group_spec(key, indexes_by_set[key], compact=False)
            for key in keys_by_draw_order
        ]
        render_distribution_plot(
            output_path,
            embedding,
            groups,
            [],
            title=(
                f"Standard t-SNE: {SET_STYLE[first]['label']} vs "
                f"{SET_STYLE[second]['label']}"
            ),
            subtitle=(
                "Baseline-scaled global exact t-SNE; all four task banks "
                "participate, with 33 original scenes shown"
            ),
            footer=footer,
            summary_lines=[
                (
                    f"Preprocessing baseline: 33 original + "
                    f"{len(tasks_by_set['balance'])} balance tasks"
                ),
                (
                    f"{SET_STYLE[first]['short_label']}: "
                    f"n={len(tasks_by_set[first])} | "
                    f"{SET_STYLE[second]['short_label']}: "
                    f"n={len(tasks_by_set[second])}"
                ),
                "Pure scatter plot; identical coordinates in all three panels",
            ],
            extent_embedding=embedding,
        )
        panel_files.append(output_path.name)

    overview_path = (
        OUTPUTS / "standard_tsne_baseline_pairwise_json_comparisons.png"
    )
    render_overview(
        overview_path,
        embedding,
        indexes_by_set,
        descriptor_dimension=layout.dimension,
        active_dimensions=active_dimension_count,
        kl_divergence=kl_divergence,
        trustworthiness=trustworthiness,
        fit_points=len(embedding),
    )

    metrics = {
        "method": {
            "algorithm": "standard exact t-SNE with Euclidean input distances",
            "implementation": "self-contained NumPy",
            "one_global_embedding_reused_across_all_three_panels": True,
            "original_scenes_shown_in_every_panel": True,
            "global_embedding_fit_on": list(ALL_SET_ORDER),
            "descriptor_and_scaler_baseline": list(REFERENCE_SET_ORDER),
            "baseline_only_descriptor_config_fit": True,
            "baseline_only_robust_scaler_fit": True,
            "bridge_participates_in_tsne_fit": True,
            "hard_participates_in_tsne_fit": True,
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
        },
        "sources": {
            "original": "src/tabx/scenarios/{units,zones,challenges}",
            "balance": "outputs/balance_tasks.json",
            "bridge": "outputs/bridge_tasks_balanced.json",
            "hard": "outputs/hard_diverse_tasks.json",
        },
        "inventory": {
            "original": original_counts,
            "balance": len(tasks_by_set["balance"]),
            "bridge": len(tasks_by_set["bridge"]),
            "hard": len(tasks_by_set["hard"]),
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
        OUTPUTS / "standard_tsne_baseline_pairwise_metrics.json"
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
