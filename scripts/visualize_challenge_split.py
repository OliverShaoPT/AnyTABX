"""Visualize the rationale for the recommended 9/4 challenge split."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from visualize_low_winrate_tasks import (
    BG,
    DIM,
    FIELD_BORDER,
    MUTED,
    PANEL,
    TEAM_COLORS,
    TEXT,
    UNIT_CODES,
    UNIT_NAMES,
    ZONE_COLORS,
    _draw_field,
)


ROOT = Path(__file__).resolve().parents[1]
CHALLENGE_DIR = ROOT / "src" / "tabx" / "scenarios" / "challenges"
ASSET_DIR = ROOT / "src" / "tabx" / "visualize" / "assets" / "units"
TRAIN_SCENARIOS = (
    "ambush",
    "clover",
    "cross",
    "elbow",
    "grid",
    "pair",
    "pingpong",
    "ribbon",
    "superking",
)
UNSEEN_SCENARIOS = ("bypass", "crossfire", "encirclement", "vsrangers")

ACTIVE_UNIT_IDS = tuple(range(8))  # paladin does not occur in any challenge scenario
ZONE_LABELS = {1: "Lava", 2: "Bush", 3: "Swamp"}
TRAIN_COLOR = "#4EA5FF"
UNSEEN_COLOR = "#FFB454"
GRID = "#22384A"
CARD_BORDER = "#294158"


def _font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    font_dir = Path("C:/Windows/Fonts")
    if mono:
        candidates = ("consolab.ttf", "consola.ttf", "msyh.ttc")
    elif bold:
        candidates = ("msyhbd.ttc", "msyh.ttc", "simhei.ttf", "seguisb.ttf")
    else:
        candidates = ("msyh.ttc", "simhei.ttf", "segoeui.ttf", "arial.ttf")
    for candidate in candidates:
        path = font_dir / candidate
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def _flat(values: list[list[Any]]) -> list[Any]:
    return [value[0] if isinstance(value, list) else value for value in values]


def _load_record(name: str) -> dict[str, Any]:
    task = json.loads((CHALLENGE_DIR / f"{name}.json").read_text(encoding="utf-8"))
    scenario = task["scenario"]
    zones = task["zone_scenario"]
    unit_ids = [int(value) for value in _flat(scenario["unit_ids"])]
    teams = [int(value) for value in _flat(scenario["teams"])]
    positions = scenario["positions"]
    cross_team_distances = [
        math.dist(positions[i], positions[j])
        for i, team_i in enumerate(teams)
        for j, team_j in enumerate(teams)
        if team_i == 0 and team_j == 1
    ]
    zone_types = [int(value) for value in _flat(zones["zone_type"])]
    unit_counts = Counter(unit_ids)
    zone_counts = Counter(zone_types)
    split = "train" if name in TRAIN_SCENARIOS else "unseen"
    task["metadata"] = {"scenario_bucket": {"zone": "built-in"}}
    return {
        "name": name,
        "split": split,
        "task": task,
        "unit_counts": unit_counts,
        "zone_counts": zone_counts,
        "n_units": len(unit_ids),
        "n_ally": teams.count(0),
        "n_enemy": teams.count(1),
        "n_unit_types": len(unit_counts),
        "n_zones": len(zone_types),
        "mean_distance": sum(cross_team_distances) / len(cross_team_distances),
        "min_distance": min(cross_team_distances),
        "max_distance": max(cross_team_distances),
    }


def load_records() -> list[dict[str, Any]]:
    records = [_load_record(name) for name in TRAIN_SCENARIOS + UNSEEN_SCENARIOS]
    disk_names = {path.stem for path in CHALLENGE_DIR.glob("*.json")}
    split_names = set(TRAIN_SCENARIOS + UNSEEN_SCENARIOS)
    if len(records) != 13 or disk_names != split_names:
        raise RuntimeError("the split must contain every one of the 13 challenge scenarios exactly once")
    return records


def _rounded_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=20, fill=PANEL, outline=CARD_BORDER, width=2)


def _draw_heatmap(image: Image.Image, records: list[dict[str, Any]], box: tuple[int, int, int, int]) -> None:
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = box
    _rounded_panel(draw, box)
    draw.text((left + 28, top + 22), "特征覆盖热力图", font=_font(28, bold=True), fill=TEXT)
    draw.text(
        (left + 28, top + 62),
        "数字表示场景内单位/地形数量；训练集覆盖了 Unseen 中出现的全部基础元素",
        font=_font(16),
        fill=MUTED,
    )

    x_name = left + 28
    x_group = x_name + 245
    x_cells = x_group + 95
    cell_w = 92
    header_y = top + 112
    row_y = top + 166
    row_h = 74
    name_font = _font(17, bold=True)
    cell_font = _font(15, bold=True)
    small_font = _font(13)

    draw.text((x_name, header_y), "场景", font=_font(15, bold=True), fill=MUTED)
    draw.text((x_group, header_y), "集合", font=_font(15, bold=True), fill=MUTED)
    headers = [UNIT_CODES[i] for i in ACTIVE_UNIT_IDS] + ["L", "B", "S", "N", "距离"]
    for col, header in enumerate(headers):
        x = x_cells + col * cell_w
        draw.text((x + cell_w / 2, header_y), header, anchor="ma", font=_font(15, bold=True), fill=MUTED)
    draw.text(
        (x_cells, header_y + 24),
        "F farmer · S assassin · K king · M mammoth · A archer · C cannon · D deadeye · H healer",
        font=_font(11),
        fill=DIM,
    )

    for row, record in enumerate(records):
        y = row_y + row * row_h
        is_unseen = record["split"] == "unseen"
        row_color = UNSEEN_COLOR if is_unseen else TRAIN_COLOR
        if is_unseen:
            draw.rounded_rectangle(
                (left + 15, y - 6, right - 15, y + row_h - 8),
                radius=10,
                fill="#1B2632",
                outline=UNSEEN_COLOR,
                width=2,
            )
        elif row == len(TRAIN_SCENARIOS):
            draw.line((left + 20, y - 13, right - 20, y - 13), fill=UNSEEN_COLOR, width=3)

        draw.ellipse((x_name, y + 16, x_name + 13, y + 29), fill=row_color)
        draw.text((x_name + 23, y + 10), record["name"], font=name_font, fill=TEXT)
        draw.text(
            (x_group, y + 11),
            "测试" if is_unseen else "训练",
            font=small_font,
            fill=row_color,
        )

        values = [record["unit_counts"].get(i, 0) for i in ACTIVE_UNIT_IDS]
        values += [record["zone_counts"].get(i, 0) for i in (1, 2, 3)]
        for col, value in enumerate(values):
            x = x_cells + col * cell_w
            if value:
                if col < len(ACTIVE_UNIT_IDS):
                    fill = "#244D6B" if not is_unseen else "#624A2A"
                else:
                    fill = ZONE_COLORS[col - len(ACTIVE_UNIT_IDS) + 1]
                draw.rounded_rectangle((x + 14, y + 7, x + cell_w - 14, y + 43), radius=8, fill=fill)
                draw.text((x + cell_w / 2, y + 25), str(value), anchor="mm", font=cell_font, fill=TEXT)
            else:
                draw.text((x + cell_w / 2, y + 25), "·", anchor="mm", font=cell_font, fill=DIM)

        x_n = x_cells + 11 * cell_w
        draw.text((x_n + cell_w / 2, y + 25), str(record["n_units"]), anchor="mm", font=cell_font, fill=TEXT)
        x_dist = x_cells + 12 * cell_w
        draw.text(
            (x_dist + cell_w / 2, y + 25),
            f"{record['mean_distance']:.1f}",
            anchor="mm",
            font=cell_font,
            fill=UNSEEN_COLOR if is_unseen else TEXT,
        )


def _draw_scatter(image: Image.Image, records: list[dict[str, Any]], box: tuple[int, int, int, int]) -> None:
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = box
    _rounded_panel(draw, box)
    draw.text((left + 28, top + 22), "场景空间：规模 × 接敌距离", font=_font(27, bold=True), fill=TEXT)
    draw.text((left + 28, top + 62), "圆点大小代表地形数量", font=_font(15), fill=MUTED)

    plot = (left + 92, top + 125, right - 45, bottom - 72)
    px0, py0, px1, py1 = plot
    x_min, x_max = 20.0, 86.0
    y_min, y_max = 2.0, 13.0
    to_x = lambda value: px0 + (value - x_min) / (x_max - x_min) * (px1 - px0)
    to_y = lambda value: py1 - (value - y_min) / (y_max - y_min) * (py1 - py0)

    for tick in (20, 30, 40, 50, 60, 70, 80):
        x = to_x(tick)
        draw.line((x, py0, x, py1), fill=GRID, width=1)
        draw.text((x, py1 + 17), str(tick), anchor="ma", font=_font(13), fill=MUTED)
    for tick in (3, 5, 7, 9, 11, 13):
        y = to_y(tick)
        draw.line((px0, y, px1, y), fill=GRID, width=1)
        draw.text((px0 - 16, y), str(tick), anchor="rm", font=_font(13), fill=MUTED)
    draw.line((px0, py1, px1, py1), fill=FIELD_BORDER, width=2)
    draw.line((px0, py0, px0, py1), fill=FIELD_BORDER, width=2)
    draw.text((px0 + (px1 - px0) / 2, bottom - 34), "两队平均初始距离", anchor="mm", font=_font(15), fill=MUTED)
    draw.text((left + 30, py0 + (py1 - py0) / 2), "单位数", anchor="mm", font=_font(15), fill=MUTED)

    label_offsets = {
        "ambush": (9, -25),
        "clover": (9, 10),
        "pair": (9, 8),
        "ribbon": (-80, 10),
        "pingpong": (-98, -27),
        "crossfire": (8, -28),
        "vsrangers": (-92, 10),
        "bypass": (-86, -29),
    }
    for record in records:
        x = to_x(record["mean_distance"])
        y = to_y(record["n_units"])
        is_unseen = record["split"] == "unseen"
        color = UNSEEN_COLOR if is_unseen else TRAIN_COLOR
        radius = 9 + 3 * record["n_zones"]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=TEXT, width=2)
        dx, dy = label_offsets.get(record["name"], (9, -22 if is_unseen else -19))
        draw.text((x + dx, y + dy), record["name"], font=_font(13, bold=is_unseen), fill=color)

    draw.rounded_rectangle((right - 265, top + 22, right - 236, top + 43), radius=5, fill=TRAIN_COLOR)
    draw.text((right - 225, top + 19), "训练", font=_font(14), fill=TEXT)
    draw.rounded_rectangle((right - 150, top + 22, right - 121, top + 43), radius=5, fill=UNSEEN_COLOR)
    draw.text((right - 110, top + 19), "Unseen", font=_font(14), fill=TEXT)


def _draw_coverage(image: Image.Image, records: list[dict[str, Any]], box: tuple[int, int, int, int]) -> None:
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = box
    _rounded_panel(draw, box)
    train = [record for record in records if record["split"] == "train"]
    unseen = [record for record in records if record["split"] == "unseen"]
    train_units = set().union(*(set(record["unit_counts"]) for record in train))
    unseen_units = set().union(*(set(record["unit_counts"]) for record in unseen))
    train_zones = set().union(*(set(record["zone_counts"]) for record in train))
    unseen_zones = set().union(*(set(record["zone_counts"]) for record in unseen))
    train_distances = [record["mean_distance"] for record in train]
    in_range = sum(min(train_distances) <= record["mean_distance"] <= max(train_distances) for record in unseen)

    draw.text((left + 28, top + 20), "为什么这组划分适合测“场景泛化”", font=_font(25, bold=True), fill=TEXT)
    metrics = (
        (f"{len(unseen_units & train_units)}/{len(unseen_units)}", "Unseen 单位类型已在训练出现"),
        (f"{len(unseen_zones & train_zones)}/{len(unseen_zones)}", "Unseen 地形类型已在训练出现"),
        (f"{in_range}/{len(unseen)}", "接敌距离位于训练范围内"),
    )
    start_x = left + 32
    for index, (value, label) in enumerate(metrics):
        x = start_x + index * 390
        draw.text((x, top + 77), value, font=_font(39, bold=True), fill=UNSEEN_COLOR if index == 2 else TRAIN_COLOR)
        draw.text((x, top + 127), label, font=_font(14), fill=MUTED)
    draw.line((left + 28, top + 170, right - 28, top + 170), fill=GRID, width=1)
    draw.text(
        (left + 30, top + 190),
        f"训练距离范围：{min(train_distances):.1f}–{max(train_distances):.1f}；bypass = 81.4，是刻意保留的单一长距离外推点。",
        font=_font(15),
        fill=TEXT,
    )
    draw.text(
        (left + 30, top + 226),
        "crossfire 的 4 个 lava 区也高于训练集最大地形数 3，用于检验地形密度外推；其余测试点保持插值。",
        font=_font(15),
        fill=TEXT,
    )


def _draw_unseen_cards(image: Image.Image, records: list[dict[str, Any]], top: int) -> None:
    draw = ImageDraw.Draw(image)
    unseen = [record for record in records if record["split"] == "unseen"]
    notes = {
        "bypass": "长距离接敌 · Bush+Swamp · 异构编队",
        "crossfire": "多方向交火 · 4 个 Lava 区 · 地形密度外推",
        "encirclement": "包围结构 · 中心目标 · 多方向协同",
        "vsrangers": "无地形干扰 · 近战/远程纯战术泛化",
    }
    margin, gap = 50, 24
    card_w = (3600 - 2 * margin - 3 * gap) // 4
    card_h = 775
    draw.text((50, top - 54), "4 个 Unseen 场景分别承担什么测试职责", font=_font(29, bold=True), fill=TEXT)
    for index, record in enumerate(unseen):
        left = margin + index * (card_w + gap)
        box = (left, top, left + card_w, top + card_h)
        _rounded_panel(draw, box)
        draw.rounded_rectangle((left + 20, top + 18, left + 112, top + 51), radius=8, fill=UNSEEN_COLOR)
        draw.text((left + 66, top + 35), "UNSEEN", anchor="mm", font=_font(13, bold=True), fill=BG)
        draw.text((left + 130, top + 18), record["name"], font=_font(23, bold=True), fill=TEXT)
        draw.text((left + 21, top + 64), notes[record["name"]], font=_font(14), fill=MUTED)
        _draw_field(
            image,
            record["task"],
            (left + 18, top + 108, left + card_w - 18, top + 590),
            ASSET_DIR,
            compact=True,
        )
        metrics = (
            f"单位 {record['n_units']}（{record['n_ally']} vs {record['n_enemy']}）",
            f"平均距离 {record['mean_distance']:.1f}",
            f"地形 {record['n_zones']}",
        )
        x = left + 24
        for metric in metrics:
            draw.rounded_rectangle((x, top + 620, x + 240, top + 663), radius=9, fill="#13263A")
            draw.text((x + 120, top + 642), metric, anchor="mm", font=_font(14, bold=True), fill=TEXT)
            x += 255
        unit_names = ", ".join(UNIT_NAMES[unit_id] for unit_id in sorted(record["unit_counts"]))
        draw.text((left + 24, top + 698), f"单位类型：{unit_names}", font=_font(13), fill=DIM)


def render_dashboard(records: list[dict[str, Any]], output_path: Path) -> None:
    image = Image.new("RGBA", (3600, 2540), BG)
    draw = ImageDraw.Draw(image)
    draw.text((50, 32), "13 个 Challenge 场景：9 训练 / 4 Unseen 划分依据", font=_font(42, bold=True), fill=TEXT)
    draw.text(
        (50, 91),
        "目标：基础单位与地形充分覆盖，完整空间布局保持未见，从而测量组合泛化而非元素缺失",
        font=_font(19),
        fill=MUTED,
    )
    draw.rounded_rectangle((2860, 43, 3085, 94), radius=12, fill=TRAIN_COLOR)
    draw.text((2972, 68), "TRAIN  9", anchor="mm", font=_font(17, bold=True), fill=BG)
    draw.rounded_rectangle((3110, 43, 3545, 94), radius=12, fill=UNSEEN_COLOR)
    draw.text((3327, 68), "UNSEEN  4", anchor="mm", font=_font(17, bold=True), fill=BG)

    _draw_heatmap(image, records, (50, 145, 2240, 1510))
    _draw_scatter(image, records, (2270, 145, 3550, 980))
    _draw_coverage(image, records, (2270, 1005, 3550, 1510))
    _draw_unseen_cards(image, records, 1650)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, quality=95)


def write_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    fields = [
        "scenario",
        "split",
        "n_units",
        "n_ally",
        "n_enemy",
        "n_unit_types",
        "n_zones",
        "mean_cross_team_distance",
        "min_cross_team_distance",
        "max_cross_team_distance",
    ] + [f"unit_{UNIT_NAMES[i]}" for i in ACTIVE_UNIT_IDS] + [f"zone_{ZONE_LABELS[i].lower()}" for i in (1, 2, 3)]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {
                "scenario": record["name"],
                "split": record["split"],
                "n_units": record["n_units"],
                "n_ally": record["n_ally"],
                "n_enemy": record["n_enemy"],
                "n_unit_types": record["n_unit_types"],
                "n_zones": record["n_zones"],
                "mean_cross_team_distance": f"{record['mean_distance']:.6f}",
                "min_cross_team_distance": f"{record['min_distance']:.6f}",
                "max_cross_team_distance": f"{record['max_distance']:.6f}",
            }
            row.update({f"unit_{UNIT_NAMES[i]}": record["unit_counts"].get(i, 0) for i in ACTIVE_UNIT_IDS})
            row.update({f"zone_{ZONE_LABELS[i].lower()}": record["zone_counts"].get(i, 0) for i in (1, 2, 3)})
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "challenge_split_visualization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records()
    dashboard = args.output_dir / "challenge_split_rationale.png"
    render_dashboard(records, dashboard)
    write_csv(records, args.output_dir / "challenge_features.csv")
    split = {"train": list(TRAIN_SCENARIOS), "unseen": list(UNSEEN_SCENARIOS)}
    (args.output_dir / "challenge_split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"dashboard": str(dashboard), **split}, ensure_ascii=False))


if __name__ == "__main__":
    main()
