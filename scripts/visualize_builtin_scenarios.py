"""Render the 33 built-in TABX scenarios as static initial-state maps.

The built-in suite consists of 13 challenge scenarios and the Cartesian
product of four unit scenarios with five zone layouts (including ``void``).
This renderer intentionally operates on the JSON files directly so that it
does not require JAX or a GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from visualize_low_winrate_tasks import (
    BG,
    DIM,
    MUTED,
    PANEL,
    TEAM_COLORS,
    TEXT,
    UNIT_CODES,
    UNIT_NAMES,
    ZONE_COLORS,
    _draw_field,
    _font,
)


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = ROOT / "src" / "tabx" / "scenarios"
ASSET_DIR = ROOT / "src" / "tabx" / "visualize" / "assets" / "units"

ZONE_NAMES = ("void", "1S", "2L", "2L2B2S", "3B")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scenario_task(
    name: str,
    category: str,
    unit_name: str | None = None,
    zone_name: str | None = None,
) -> dict[str, Any]:
    if category == "challenge":
        task = _read_json(SCENARIO_DIR / "challenges" / f"{name}.json")
        zone_bucket = "challenge"
    else:
        if unit_name is None or zone_name is None:
            raise ValueError("unit-zone scenarios require both names")
        units = _read_json(SCENARIO_DIR / "units" / f"{unit_name}.json")
        zones = _read_json(SCENARIO_DIR / "zones" / f"{zone_name}.json")
        task = {
            "grid_info": units["grid_info"],
            "scenario": units["scenario"],
            "zone_scenario": zones["zone_scenario"],
        }
        zone_bucket = zone_name

    # ``_draw_field`` only uses this value to separate labels for generated
    # overlapping-zone tasks. Built-in layouts do not need that adjustment.
    task["metadata"] = {"scenario_bucket": {"zone": zone_bucket}}
    task["_name"] = name
    task["_category"] = category
    task["_unit_name"] = unit_name
    task["_zone_name"] = zone_name
    return task


def load_builtin_scenarios() -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    challenge_names = sorted(path.stem for path in (SCENARIO_DIR / "challenges").glob("*.json"))
    unit_names = sorted(path.stem for path in (SCENARIO_DIR / "units").glob("*.json"))

    challenges = [_scenario_task(name, "challenge") for name in challenge_names]
    matrix = [
        [
            _scenario_task(
                unit_name if zone_name == "void" else f"{unit_name}_{zone_name}",
                "unit-zone",
                unit_name,
                zone_name,
            )
            for zone_name in ZONE_NAMES
        ]
        for unit_name in unit_names
    ]

    total = len(challenges) + sum(len(row) for row in matrix)
    if len(challenges) != 13 or len(unit_names) != 4 or total != 33:
        raise RuntimeError(
            f"expected 13 challenges + 4x5 unit-zone scenarios = 33; got "
            f"{len(challenges)} + {len(unit_names)}x{len(ZONE_NAMES)} = {total}"
        )
    return challenges, matrix


def _draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int, compact: bool = False) -> None:
    font = _font(15 if compact else 18, bold=True)
    entries = (
        ("Ally", TEAM_COLORS[0]),
        ("Enemy", TEAM_COLORS[1]),
        ("Lava", ZONE_COLORS[1]),
        ("Bush", ZONE_COLORS[2]),
        ("Swamp", ZONE_COLORS[3]),
    )
    for label, color in entries:
        draw.rounded_rectangle((x, y + 3, x + 18, y + 21), radius=4, fill=color)
        draw.text((x + 27, y), label, font=font, fill=TEXT)
        x += 112 if compact else 135


def _draw_scene_panel(
    image: Image.Image,
    task: dict[str, Any],
    box: tuple[int, int, int, int],
    subtitle: str | None = None,
) -> None:
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=16, fill=PANEL, outline="#294158", width=2)
    draw.text((left + 17, top + 13), task["_name"], font=_font(20, bold=True), fill=TEXT)
    if subtitle:
        draw.text((left + 17, top + 42), subtitle, font=_font(13), fill=MUTED)
        field_top = top + 67
    else:
        field_top = top + 50
    _draw_field(
        image,
        task,
        (left + 13, field_top, right - 13, bottom - 13),
        ASSET_DIR,
        compact=True,
    )


def render_individual(task: dict[str, Any], output_path: Path) -> None:
    image = Image.new("RGBA", (1800, 1100), BG)
    draw = ImageDraw.Draw(image)
    category = "Challenge scenario" if task["_category"] == "challenge" else "Unit composition × zone layout"
    draw.text((60, 38), task["_name"], font=_font(40, bold=True), fill=TEXT)
    draw.text((60, 94), category, font=_font(19), fill=MUTED)
    _draw_field(image, task, (55, 150, 1745, 1000), ASSET_DIR, compact=False)
    _draw_legend(draw, 60, 1032)
    draw.text((1350, 1035), "Initial state", font=_font(16), fill=DIM)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, quality=95)


def render_challenge_overview(challenges: list[dict[str, Any]], output_path: Path) -> None:
    cols = 5
    panel_w, panel_h = 600, 440
    gap = 24
    width = 2 * 40 + cols * panel_w + (cols - 1) * gap
    rows = (len(challenges) + cols - 1) // cols
    height = 155 + rows * panel_h + (rows - 1) * gap + 45
    image = Image.new("RGBA", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw.text((40, 28), "TABX built-in challenge scenarios", font=_font(38, bold=True), fill=TEXT)
    draw.text((40, 78), "13 handcrafted tactical layouts · exact initial positions and zones", font=_font(18), fill=MUTED)
    _draw_legend(draw, width - 720, 74, compact=True)

    for index, task in enumerate(challenges):
        row, col = divmod(index, cols)
        left = 40 + col * (panel_w + gap)
        top = 135 + row * (panel_h + gap)
        _draw_scene_panel(image, task, (left, top, left + panel_w, top + panel_h))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, quality=95)


def render_unit_zone_matrix(matrix: list[list[dict[str, Any]]], output_path: Path) -> None:
    rows, cols = len(matrix), len(ZONE_NAMES)
    row_label_w = 325
    panel_w, panel_h = 570, 450
    gap = 20
    top = 190
    width = 50 + row_label_w + cols * panel_w + (cols - 1) * gap + 50
    height = top + rows * panel_h + (rows - 1) * gap + 45
    image = Image.new("RGBA", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw.text((50, 28), "TABX unit composition × zone matrix", font=_font(38, bold=True), fill=TEXT)
    draw.text((50, 79), "4 unit scenarios × 5 zone layouts = 20 built-in combinations", font=_font(18), fill=MUTED)
    _draw_legend(draw, width - 720, 76, compact=True)

    for col, zone_name in enumerate(ZONE_NAMES):
        x = 50 + row_label_w + col * (panel_w + gap)
        label = "No zone (void)" if zone_name == "void" else zone_name
        draw.text((x + 15, 141), label, font=_font(21, bold=True), fill=TEXT)

    for row, tasks in enumerate(matrix):
        y = top + row * (panel_h + gap)
        unit_name = tasks[0]["_unit_name"]
        draw.rounded_rectangle((50, y, 50 + row_label_w - 20, y + panel_h), radius=16, fill=PANEL, outline="#294158", width=2)
        draw.text((72, y + 25), f"Unit set {row + 1}", font=_font(17, bold=True), fill=MUTED)
        draw.multiline_text((72, y + 67), unit_name, font=_font(22, bold=True), fill=TEXT, spacing=7)
        draw.text((72, y + panel_h - 50), "same units →", font=_font(16), fill=DIM)
        for col, task in enumerate(tasks):
            x = 50 + row_label_w + col * (panel_w + gap)
            _draw_scene_panel(image, task, (x, y, x + panel_w, y + panel_h))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, quality=95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "builtin_scenarios_visualization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    challenges, matrix = load_builtin_scenarios()
    all_scenarios = challenges + [task for row in matrix for task in row]
    scenes_dir = args.output_dir / "scenes"
    for index, task in enumerate(all_scenarios, start=1):
        render_individual(task, scenes_dir / f"{index:02d}_{task['_name']}.png")
    render_challenge_overview(challenges, args.output_dir / "01_challenge_scenarios.png")
    render_unit_zone_matrix(matrix, args.output_dir / "02_unit_zone_matrix.png")
    manifest = {
        "count": len(all_scenarios),
        "challenge_count": len(challenges),
        "unit_zone_count": sum(len(row) for row in matrix),
        "scenarios": [task["_name"] for task in all_scenarios],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
