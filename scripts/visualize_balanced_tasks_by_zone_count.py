"""Render every zoned task in a task bank as four grouped contact sheets.

The default grouping is 1--5, 6--10, 11--15, and 16--20 zones.  Each
thumbnail contains the complete map, terrain ellipses, units, and headings.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "task_files" / "balanced_tasks_v2.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "balanced_tasks_v2_zone_visualization"

GROUPS = ((1, 5), (6, 10), (11, 15), (16, 20))
TILE_W, TILE_H = 360, 270
GAP = 8
OUTER_MARGIN = 28
HEADER_H = 168
TARGET_ASPECT = 16 / 9

PAGE_BG = "#E8EDF3"
CARD_BG = "#FFFFFF"
CARD_BORDER = "#B9C5D1"
TEXT = "#17212B"
MUTED = "#5D6B78"
GRID = "#D8E0E8"
ALLY = "#2478D4"
ENEMY = "#E14B45"

ZONE_STYLE = {
    1: ((255, 111, 78, 92), "#D9472B", "L"),   # lava
    2: ((75, 181, 105, 88), "#238A45", "B"),   # bush
    3: ((95, 126, 153, 92), "#506B82", "S"),   # swamp
}
ZONE_NAMES = {1: "Lava", 2: "Bush", 3: "Swamp"}
UNIT_CODES = {1: "F", 2: "S", 3: "K", 4: "M", 5: "A", 6: "C", 7: "D", 8: "H", 9: "P"}


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    font_dir = Path("C:/Windows/Fonts")
    candidates = (
        ("msyhbd.ttc", "seguisb.ttf", "arialbd.ttf")
        if bold
        else ("msyh.ttc", "segoeui.ttf", "arial.ttf")
    )
    for name in candidates:
        path = font_dir / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def flat(values: list[Any]) -> list[Any]:
    return [value[0] if isinstance(value, list) else value for value in values]


def task_zone_count(task: dict[str, Any]) -> int:
    return int(task["zone_scenario"]["n_zone"])


def choose_grid(n_tasks: int) -> tuple[int, int]:
    # Choose a landscape grid while accounting for the thumbnail aspect ratio.
    columns = max(1, math.ceil(math.sqrt(n_tasks * TARGET_ASPECT * TILE_H / TILE_W)))
    return columns, math.ceil(n_tasks / columns)


def draw_legend_chip(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    label: str,
    color: str,
    *,
    circle: bool = True,
) -> int:
    if circle:
        draw.ellipse((x, y + 2, x + 17, y + 19), fill=color, outline="#FFFFFF", width=1)
    else:
        draw.rounded_rectangle((x, y + 2, x + 22, y + 19), radius=4, fill=color)
    draw.text((x + 28, y), label, font=font(16), fill=MUTED)
    return x + 28 + int(draw.textlength(label, font=font(16))) + 28


def draw_header(
    image: Image.Image,
    *,
    source_name: str,
    lo: int,
    hi: int,
    tasks: list[tuple[int, dict[str, Any]]],
    columns: int,
    rows: int,
) -> None:
    draw = ImageDraw.Draw(image)
    draw.text(
        (OUTER_MARGIN, 22),
        f"场景可视化：{lo}–{hi} 个 Zones",
        font=font(34, bold=True),
        fill=TEXT,
    )
    counts = Counter(task_zone_count(task) for _, task in tasks)
    distribution = "   ".join(f"{n} zones: {counts[n]}" for n in range(lo, hi + 1))
    draw.text(
        (OUTER_MARGIN, 72),
        f"{source_name}  ·  共 {len(tasks)} 个场景  ·  {columns} × {rows} 布局",
        font=font(18),
        fill=MUTED,
    )
    draw.text((OUTER_MARGIN, 104), distribution, font=font(16), fill=MUTED)

    x = OUTER_MARGIN
    y = 134
    x = draw_legend_chip(draw, x, y, "Ally", ALLY)
    x = draw_legend_chip(draw, x, y, "Enemy", ENEMY)
    for zone_type in (1, 2, 3):
        _, outline, _ = ZONE_STYLE[zone_type]
        x = draw_legend_chip(draw, x, y, ZONE_NAMES[zone_type], outline, circle=False)
    draw.text(
        (x + 5, y),
        "字母表示单位类型；短线表示朝向",
        font=font(16),
        fill=MUTED,
    )


def draw_scene_tile(task_number: int, task: dict[str, Any]) -> Image.Image:
    tile = Image.new("RGBA", (TILE_W, TILE_H), CARD_BG)
    draw = ImageDraw.Draw(tile, "RGBA")
    draw.rounded_rectangle(
        (0, 0, TILE_W - 1, TILE_H - 1),
        radius=9,
        fill=CARD_BG,
        outline=CARD_BORDER,
        width=1,
    )

    n_zone = task_zone_count(task)
    scenario = task["scenario"]
    n_units = len(scenario["positions"])
    task_id = str(task.get("task_id", f"task_{task_number:06d}"))
    draw.text((10, 7), f"#{task_number:04d}  {task_id}", font=font(13, bold=True), fill=TEXT)
    draw.text(
        (TILE_W - 10, 7),
        f"{n_zone}Z · {n_units}U",
        anchor="ra",
        font=font(13, bold=True),
        fill=MUTED,
    )

    field_left, field_top = 8, 31
    field_right, field_bottom = TILE_W - 8, TILE_H - 8
    field_w = field_right - field_left
    field_h = field_bottom - field_top
    grid_info = task.get("grid_info", {})
    world_w = max(float(grid_info.get("max_field_width", 121.0)), 1e-6)
    world_h = max(float(grid_info.get("max_field_height", 78.0)), 1e-6)
    scale = min(field_w / world_w, field_h / world_h)
    map_w, map_h = world_w * scale, world_h * scale
    map_left = field_left + (field_w - map_w) / 2
    map_top = field_top + (field_h - map_h) / 2
    map_right, map_bottom = map_left + map_w, map_top + map_h
    center_x, center_y = (map_left + map_right) / 2, (map_top + map_bottom) / 2

    draw.rectangle((map_left, map_top, map_right, map_bottom), fill="#F8FAFC", outline="#8FA0AF", width=1)
    draw.line((map_left, center_y, map_right, center_y), fill=GRID, width=1)
    draw.line((center_x, map_top, center_x, map_bottom), fill=GRID, width=1)

    zones = task["zone_scenario"]
    zone_types = [int(value) for value in flat(zones["zone_type"])]
    for index in range(n_zone):
        zone_type = zone_types[index]
        if zone_type not in ZONE_STYLE:
            continue
        zx, zy = (float(value) for value in zones["position"][index])
        axis_x, axis_y = (float(value) for value in zones["axes"][index])
        px, py = center_x + zx * scale, center_y - zy * scale
        rx, ry = axis_x * scale, axis_y * scale
        fill_color, outline_color, label = ZONE_STYLE[zone_type]
        draw.ellipse((px - rx, py - ry, px + rx, py + ry), fill=fill_color, outline=outline_color, width=1)
        if rx >= 8 and ry >= 7:
            draw.text((px, py), label, anchor="mm", font=font(10, bold=True), fill=outline_color)

    teams = [int(value) for value in flat(scenario["teams"])]
    unit_ids = [int(value) for value in flat(scenario["unit_ids"])]
    rotations = [float(value) for value in flat(scenario["rotations"])]
    radius_key = "body_radii" if "body_radii" in scenario else "body_radiuss"
    body_radii = [float(value) for value in flat(scenario.get(radius_key, [[1.0]] * n_units))]
    for index, (wx, wy) in enumerate(scenario["positions"]):
        px = center_x + float(wx) * scale
        py = center_y - float(wy) * scale
        team_color = ALLY if teams[index] == 0 else ENEMY
        radius = max(3.3, min(8.5, body_radii[index] * scale))
        heading_length = radius + 5.0
        angle = rotations[index]
        hx = px + math.cos(angle) * heading_length
        hy = py - math.sin(angle) * heading_length
        draw.line((px, py, hx, hy), fill=team_color, width=2)
        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius),
            fill=team_color,
            outline="#FFFFFF",
            width=1,
        )
        code = UNIT_CODES.get(unit_ids[index], "?")
        draw.text((px, py - 0.5), code, anchor="mm", font=font(9, bold=True), fill="#FFFFFF")

    return tile


def render_group(
    source_path: Path,
    output_dir: Path,
    all_tasks: list[dict[str, Any]],
    lo: int,
    hi: int,
) -> dict[str, Any]:
    tasks = [
        (index, task)
        for index, task in enumerate(all_tasks, start=1)
        if lo <= task_zone_count(task) <= hi
    ]
    tasks.sort(key=lambda item: (task_zone_count(item[1]), item[0]))
    columns, rows = choose_grid(len(tasks))
    width = OUTER_MARGIN * 2 + columns * TILE_W + (columns - 1) * GAP
    height = HEADER_H + OUTER_MARGIN + rows * TILE_H + (rows - 1) * GAP
    sheet = Image.new("RGB", (width, height), PAGE_BG)
    draw_header(
        sheet,
        source_name=source_path.name,
        lo=lo,
        hi=hi,
        tasks=tasks,
        columns=columns,
        rows=rows,
    )
    for slot, (task_number, task) in enumerate(tasks):
        row, column = divmod(slot, columns)
        x = OUTER_MARGIN + column * (TILE_W + GAP)
        y = HEADER_H + row * (TILE_H + GAP)
        tile = draw_scene_tile(task_number, task)
        sheet.paste(tile.convert("RGB"), (x, y))
        tile.close()

    output_path = output_dir / f"zones_{lo:02d}_{hi:02d}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG", optimize=True)
    sheet.close()
    return {
        "file": str(output_path),
        "zone_range": [lo, hi],
        "n_scenes": len(tasks),
        "grid": [columns, rows],
        "image_size": [width, height],
        "counts": dict(sorted(Counter(task_zone_count(task) for _, task in tasks).items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    tasks = payload["tasks"]
    results = [render_group(source_path, output_dir, tasks, lo, hi) for lo, hi in GROUPS]
    manifest = {
        "source": str(source_path),
        "source_task_count": len(tasks),
        "rendered_scene_count": sum(result["n_scenes"] for result in results),
        "excluded_zero_zone_scenes": sum(task_zone_count(task) == 0 for task in tasks),
        "groups": results,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
