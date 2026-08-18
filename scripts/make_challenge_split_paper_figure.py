"""Create a publication-ready figure for the 9/4 TABX challenge split.

Outputs a vector PDF, a vector SVG, a 600-dpi PNG (when ``pdftoppm`` is
available), and a ready-to-edit figure caption.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import reportlab
from reportlab import rl_config
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Ellipse, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from visualize_challenge_split import TRAIN_SCENARIOS, UNSEEN_SCENARIOS, load_records


ROOT = Path(__file__).resolve().parents[1]

PAGE_W = 522.0  # 7.25 in, suitable for a two-column paper figure
PAGE_H = 466.0

BLACK = colors.HexColor("#1A1A1A")
DARK_GRAY = colors.HexColor("#4D4D4D")
MID_GRAY = colors.HexColor("#777777")
LIGHT_GRAY = colors.HexColor("#E6E6E6")
GRID = colors.HexColor("#D9D9D9")
PANEL_BG = colors.HexColor("#FAFAFA")
WHITE = colors.white

# Okabe-Ito color-blind-safe palette.
TRAIN = colors.HexColor("#0072B2")
UNSEEN = colors.HexColor("#D55E00")
ALLY = colors.HexColor("#0072B2")
ENEMY = colors.HexColor("#CC79A7")
LAVA = colors.HexColor("#E69F00")
BUSH = colors.HexColor("#009E73")
SWAMP = colors.HexColor("#56B4E9")
ZONE_COLORS = {1: LAVA, 2: BUSH, 3: SWAMP}
ZONE_LIGHT = {
    1: colors.HexColor("#FBEBC2"),
    2: colors.HexColor("#D7F0E7"),
    3: colors.HexColor("#DCEEF7"),
}
TRAIN_LIGHT = colors.HexColor("#DCECF5")
UNSEEN_LIGHT = colors.HexColor("#F7E4DA")

UNIT_CODES = ("F", "S", "K", "M", "A", "C", "D", "H", "P")
ACTIVE_UNIT_IDS = tuple(range(8))


def _register_embedded_fonts() -> None:
    windows_fonts = Path("C:/Windows/Fonts")
    reportlab_fonts = Path(reportlab.__file__).resolve().parent / "fonts"
    candidates = {
        "PaperSans": (windows_fonts / "arial.ttf", reportlab_fonts / "Vera.ttf"),
        "PaperSans-Bold": (windows_fonts / "arialbd.ttf", reportlab_fonts / "VeraBd.ttf"),
        "PaperSans-Italic": (windows_fonts / "ariali.ttf", reportlab_fonts / "VeraIt.ttf"),
    }
    for name, paths in candidates.items():
        path = next((candidate for candidate in paths if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(f"no embeddable font found for {name}")
        pdfmetrics.registerFont(TTFont(name, str(path)))
        if name == "PaperSans":
            # ReportLab's vector renderer emits a few empty text-state commands
            # using its Times-Roman default. Rebind that default to the same
            # embedded face so strict conference font checkers see no unembedded
            # resource, even though those commands draw no glyphs.
            pdfmetrics.registerFont(TTFont("Times-Roman", str(path)))


_register_embedded_fonts()
rl_config.canvas_basefontname = "PaperSans"


class FigureCanvas:
    """A small top-left-coordinate wrapper around ReportLab vector shapes."""

    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.drawing = Drawing(width, height)
        self.rect(0, 0, width, height, fill=WHITE, stroke=None)

    def _y(self, y: float) -> float:
        return self.height - y

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        fill: colors.Color | None = None,
        stroke: colors.Color | None = BLACK,
        stroke_width: float = 0.5,
        radius: float = 0,
    ) -> None:
        self.drawing.add(
            Rect(
                x,
                self.height - y - height,
                width,
                height,
                rx=radius,
                ry=radius,
                fillColor=fill,
                strokeColor=stroke,
                strokeWidth=stroke_width,
            )
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: colors.Color = BLACK,
        width: float = 0.5,
        dash: list[float] | None = None,
    ) -> None:
        shape = Line(x1, self._y(y1), x2, self._y(y2), strokeColor=color, strokeWidth=width)
        if dash:
            shape.strokeDashArray = dash
        self.drawing.add(shape)

    def text(
        self,
        x: float,
        y: float,
        value: str,
        size: float = 7,
        color: colors.Color = BLACK,
        bold: bool = False,
        italic: bool = False,
        anchor: str = "start",
    ) -> None:
        if bold:
            font = "PaperSans-Bold"
        elif italic:
            font = "PaperSans-Italic"
        else:
            font = "PaperSans"
        self.drawing.add(
            String(
                x,
                self.height - y - size * 0.80,
                value,
                fontName=font,
                fontSize=size,
                fillColor=color,
                textAnchor=anchor,
            )
        )

    def circle(
        self,
        cx: float,
        cy: float,
        radius: float,
        fill: colors.Color | None,
        stroke: colors.Color | None = BLACK,
        stroke_width: float = 0.5,
    ) -> None:
        self.drawing.add(
            Circle(
                cx,
                self._y(cy),
                radius,
                fillColor=fill,
                strokeColor=stroke,
                strokeWidth=stroke_width,
            )
        )

    def ellipse(
        self,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        fill: colors.Color | None,
        stroke: colors.Color | None,
        stroke_width: float = 0.6,
    ) -> None:
        self.drawing.add(
            Ellipse(
                cx,
                self._y(cy),
                rx,
                ry,
                fillColor=fill,
                strokeColor=stroke,
                strokeWidth=stroke_width,
            )
        )

    def diamond(
        self,
        cx: float,
        cy: float,
        radius: float,
        fill: colors.Color,
        stroke: colors.Color = WHITE,
        stroke_width: float = 0.6,
    ) -> None:
        points = [
            cx,
            self._y(cy - radius),
            cx + radius,
            self._y(cy),
            cx,
            self._y(cy + radius),
            cx - radius,
            self._y(cy),
        ]
        self.drawing.add(
            Polygon(points, fillColor=fill, strokeColor=stroke, strokeWidth=stroke_width)
        )


def _panel(fig: FigureCanvas, x: float, y: float, w: float, h: float) -> None:
    fig.rect(x, y, w, h, fill=PANEL_BG, stroke=LIGHT_GRAY, stroke_width=0.7, radius=3)


def _draw_feature_matrix(
    fig: FigureCanvas,
    records: list[dict[str, Any]],
    x: float,
    y: float,
    w: float,
    h: float,
) -> None:
    _panel(fig, x, y, w, h)
    fig.text(x + 6, y + 5, "(a) Primitive coverage across scenarios", size=8.2, bold=True)
    fig.text(
        x + 6,
        y + 15,
        "Counts of unit and terrain primitives; held-out rows are outlined.",
        size=5.4,
        color=DARK_GRAY,
    )

    name_w = 68.0
    cell_w = 17.7
    table_x = x + name_w + 7
    header_y = y + 29
    row_y = y + 47
    row_h = 14.0
    columns = [UNIT_CODES[i] for i in ACTIVE_UNIT_IDS] + ["La", "Bu", "Sw", "n", "d"]

    fig.text(table_x + 4 * cell_w, header_y - 8, "Units", size=5.4, bold=True, anchor="middle")
    fig.text(table_x + 9.5 * cell_w, header_y - 8, "Terrain", size=5.4, bold=True, anchor="middle")
    fig.text(table_x + 12 * cell_w, header_y - 8, "Stats", size=5.4, bold=True, anchor="middle")
    for col, label in enumerate(columns):
        fig.text(table_x + (col + 0.5) * cell_w, header_y, label, size=5.7, bold=True, anchor="middle")
    fig.text(x + 7, header_y, "Scenario", size=5.7, bold=True)

    for row, record in enumerate(records):
        yy = row_y + row * row_h
        is_unseen = record["split"] == "unseen"
        row_color = UNSEEN if is_unseen else TRAIN
        if is_unseen:
            fig.rect(x + 3, yy - 2, w - 6, row_h, fill=UNSEEN_LIGHT, stroke=UNSEEN, stroke_width=0.7, radius=1.5)
        elif row == len(TRAIN_SCENARIOS) - 1:
            fig.line(x + 4, yy + row_h + 2, x + w - 4, yy + row_h + 2, UNSEEN, 0.9)
        fig.rect(x + 6, yy + 2.4, 2.2, 8.0, fill=row_color, stroke=None)
        fig.text(x + 11, yy + 2.6, record["name"], size=5.6, bold=is_unseen)

        values = [record["unit_counts"].get(i, 0) for i in ACTIVE_UNIT_IDS]
        values += [record["zone_counts"].get(i, 0) for i in (1, 2, 3)]
        for col, value in enumerate(values):
            cx = table_x + (col + 0.5) * cell_w
            if value:
                if col < len(ACTIVE_UNIT_IDS):
                    fill = UNSEEN_LIGHT if is_unseen else TRAIN_LIGHT
                    stroke = row_color
                else:
                    zone_id = col - len(ACTIVE_UNIT_IDS) + 1
                    fill = ZONE_LIGHT[zone_id]
                    stroke = ZONE_COLORS[zone_id]
                fig.rect(cx - 5.7, yy + 1.0, 11.4, 10.2, fill=fill, stroke=stroke, stroke_width=0.45, radius=1.2)
                fig.text(cx, yy + 3.1, str(value), size=5.2, bold=True, anchor="middle")
            else:
                fig.text(cx, yy + 3.2, "-", size=4.8, color=MID_GRAY, anchor="middle")

        stat_values = (str(record["n_units"]), f"{record['mean_distance']:.1f}")
        for offset, value in enumerate(stat_values, start=11):
            cx = table_x + (offset + 0.5) * cell_w
            fig.text(
                cx,
                yy + 3.0,
                value,
                size=5.3,
                color=UNSEEN if is_unseen else BLACK,
                bold=is_unseen,
                anchor="middle",
            )

    foot_y = row_y + len(records) * row_h + 4
    fig.text(
        x + 7,
        foot_y,
        "F farmer; S assassin; K king; M mammoth; A archer; C cannon; D deadeye; H healer.",
        size=4.5,
        color=DARK_GRAY,
    )
    fig.text(
        x + 7,
        foot_y + 7,
        "La lava; Bu bush; Sw swamp; n number of units; d mean cross-team distance.",
        size=4.5,
        color=DARK_GRAY,
    )


def _draw_distribution(
    fig: FigureCanvas,
    records: list[dict[str, Any]],
    x: float,
    y: float,
    w: float,
    h: float,
) -> None:
    _panel(fig, x, y, w, h)
    fig.text(x + 6, y + 5, "(b) Coverage and controlled extrapolation", size=8.2, bold=True)

    # Compact quantitative claims.
    claims = (("7/7", "test unit types"), ("3/3", "terrain types"), ("3/4", "distances in range"))
    for index, (value, label) in enumerate(claims):
        cx = x + 9 + index * 54
        fig.text(cx, y + 20, value, size=8.0, bold=True, color=TRAIN if index < 2 else UNSEEN)
        fig.text(cx, y + 29, label, size=4.1, color=DARK_GRAY)

    # Keep the legend outside the data region so the panel remains readable
    # after reduction to a two-column paper width.
    legend_y = y + 41
    fig.circle(x + 10, legend_y + 1, 2.5, TRAIN, WHITE, 0.5)
    fig.text(x + 15, legend_y - 1.2, "train", size=4.5)
    fig.diamond(x + 43, legend_y + 1, 2.8, UNSEEN)
    fig.text(x + 49, legend_y - 1.2, "unseen", size=4.5)
    fig.text(x + 92, legend_y - 1.2, "marker size = # terrains", size=4.3, color=DARK_GRAY)

    plot_x0, plot_y0 = x + 26, y + 57
    plot_x1, plot_y1 = x + w - 9, y + h - 20
    x_min, x_max = 20.0, 86.0
    y_min, y_max = 2.0, 13.0
    to_x = lambda value: plot_x0 + (value - x_min) / (x_max - x_min) * (plot_x1 - plot_x0)
    to_y = lambda value: plot_y1 - (value - y_min) / (y_max - y_min) * (plot_y1 - plot_y0)

    train_records = [record for record in records if record["split"] == "train"]
    train_min = min(record["mean_distance"] for record in train_records)
    train_max = max(record["mean_distance"] for record in train_records)
    fig.rect(
        to_x(train_min),
        plot_y0,
        to_x(train_max) - to_x(train_min),
        plot_y1 - plot_y0,
        fill=TRAIN_LIGHT,
        stroke=None,
    )
    fig.text(
        (to_x(train_min) + to_x(train_max)) / 2,
        plot_y0 + 3,
        "training distance range",
        size=4.2,
        color=TRAIN,
        anchor="middle",
    )
    for tick in (20, 40, 60, 80):
        xx = to_x(tick)
        fig.line(xx, plot_y0, xx, plot_y1, GRID, 0.35)
        fig.text(xx, plot_y1 + 4, str(tick), size=4.4, color=DARK_GRAY, anchor="middle")
    for tick in (3, 6, 9, 12):
        yy = to_y(tick)
        fig.line(plot_x0, yy, plot_x1, yy, GRID, 0.35)
        fig.text(plot_x0 - 4, yy - 2, str(tick), size=4.4, color=DARK_GRAY, anchor="end")
    fig.line(plot_x0, plot_y1, plot_x1, plot_y1, DARK_GRAY, 0.6)
    fig.line(plot_x0, plot_y0, plot_x0, plot_y1, DARK_GRAY, 0.6)
    fig.text((plot_x0 + plot_x1) / 2, plot_y1 + 9, "Mean cross-team distance", size=5.0, anchor="middle")
    fig.text(plot_x0 - 17, plot_y0 - 1, "Units", size=4.8, color=DARK_GRAY)

    offsets = {
        "bypass": (-2, -8),
        "crossfire": (3, -8),
        "encirclement": (3, 3),
        "vsrangers": (-3, -8),
    }
    for record in records:
        px = to_x(record["mean_distance"])
        py = to_y(record["n_units"])
        radius = 2.2 + 0.65 * record["n_zones"]
        is_unseen = record["split"] == "unseen"
        if is_unseen:
            fig.diamond(px, py, radius + 0.8, UNSEEN)
        else:
            fig.circle(px, py, radius, TRAIN, WHITE, 0.55)
        # The feature matrix already identifies every training point. Labeling
        # only held-out markers prevents collisions in the compact paper panel.
        if is_unseen:
            dx, dy = offsets[record["name"]]
            anchor = "end" if dx < 0 else "start"
            fig.text(
                px + dx,
                py + dy,
                record["name"],
                size=4.3,
                color=UNSEEN,
                bold=True,
                anchor=anchor,
            )


def _world_transform(
    x: float, y: float, w: float, h: float, world_w: float, world_h: float
) -> tuple[float, float, float]:
    scale = min(w / world_w, h / world_h)
    return x + w / 2, y + h / 2, scale


def _draw_scene(
    fig: FigureCanvas,
    record: dict[str, Any],
    x: float,
    y: float,
    w: float,
    h: float,
) -> None:
    task = record["task"]
    world_w = float(task["grid_info"]["max_field_width"])
    world_h = float(task["grid_info"]["max_field_height"])
    fig.rect(x, y, w, h, fill=WHITE, stroke=LIGHT_GRAY, stroke_width=0.55)
    cx, cy, scale = _world_transform(x, y, w, h, world_w, world_h)
    fig.line(x, cy, x + w, cy, GRID, 0.35)
    fig.line(cx, y, cx, y + h, GRID, 0.35)

    zones = task["zone_scenario"]
    for index in range(int(zones["n_zone"])):
        zone_type = int(zones["zone_type"][index][0])
        zx, zy = zones["position"][index]
        ax, ay = zones["axes"][index]
        px, py = cx + float(zx) * scale, cy - float(zy) * scale
        fig.ellipse(
            px,
            py,
            float(ax) * scale,
            float(ay) * scale,
            ZONE_LIGHT[zone_type],
            ZONE_COLORS[zone_type],
            0.7,
        )
        fig.text(px, py - 2.0, {1: "L", 2: "B", 3: "S"}[zone_type], size=4.0, bold=True, color=ZONE_COLORS[zone_type], anchor="middle")

    scenario = task["scenario"]
    unit_ids = [int(value[0]) for value in scenario["unit_ids"]]
    teams = [int(value[0]) for value in scenario["teams"]]
    rotations = [float(value[0]) for value in scenario["rotations"]]
    for index, unit_id in enumerate(unit_ids):
        wx, wy = scenario["positions"][index]
        px, py = cx + float(wx) * scale, cy - float(wy) * scale
        angle = rotations[index]
        team_color = ALLY if teams[index] == 0 else ENEMY
        arrow_len = 5.0
        fig.line(px, py, px + math.cos(angle) * arrow_len, py - math.sin(angle) * arrow_len, team_color, 0.8)
        fig.circle(px, py, 3.0, team_color, WHITE, 0.45)
        fig.text(px, py - 1.8, UNIT_CODES[unit_id], size=3.5, bold=True, color=WHITE, anchor="middle")


def _draw_unseen_layouts(
    fig: FigureCanvas,
    records: list[dict[str, Any]],
    x: float,
    y: float,
    w: float,
    h: float,
) -> None:
    fig.text(x, y, "(c) Held-out layouts and intended generalization tests", size=8.2, bold=True)
    fig.circle(x + 267, y + 3.7, 2.5, ALLY, WHITE, 0.4)
    fig.text(x + 272, y + 1, "ally", size=4.7)
    fig.circle(x + 296, y + 3.7, 2.5, ENEMY, WHITE, 0.4)
    fig.text(x + 301, y + 1, "enemy", size=4.7)
    for idx, (label, color) in enumerate((("lava", LAVA), ("bush", BUSH), ("swamp", SWAMP))):
        lx = x + 336 + idx * 43
        fig.rect(lx, y + 1, 5, 5, fill=color, stroke=None)
        fig.text(lx + 8, y + 1, label, size=4.5)

    unseen = [record for record in records if record["split"] == "unseen"]
    notes = {
        "bypass": "distance extrapolation",
        "crossfire": "terrain-density extrapolation",
        "encirclement": "multi-axis coordination",
        "vsrangers": "terrain-free tactics",
    }
    gap = 5.0
    card_w = (w - 3 * gap) / 4
    card_y = y + 16
    map_h = h - 51
    for index, record in enumerate(unseen):
        xx = x + index * (card_w + gap)
        fig.rect(xx, card_y, card_w, h - 17, fill=PANEL_BG, stroke=LIGHT_GRAY, stroke_width=0.65, radius=2)
        fig.text(xx + 5, card_y + 5, record["name"], size=6.3, bold=True, color=UNSEEN)
        fig.text(xx + card_w - 5, card_y + 5.2, f"d={record['mean_distance']:.1f}", size=4.8, color=DARK_GRAY, anchor="end")
        _draw_scene(fig, record, xx + 4, card_y + 17, card_w - 8, map_h)
        fig.text(xx + card_w / 2, card_y + 21 + map_h, notes[record["name"]], size=4.8, italic=True, color=DARK_GRAY, anchor="middle")


def build_figure(records: list[dict[str, Any]]) -> Drawing:
    fig = FigureCanvas(PAGE_W, PAGE_H)
    fig.text(10, 6, "Data-driven construction of the TABX 9/4 generalization split", size=10.2, bold=True)
    fig.text(
        10,
        18,
        "All held-out primitives are observed during training; novelty is confined to geometry and composition.",
        size=5.8,
        color=DARK_GRAY,
    )
    fig.rect(PAGE_W - 92, 5, 36, 12, fill=TRAIN, stroke=None, radius=2)
    fig.text(PAGE_W - 74, 8, "TRAIN 9", size=5.5, bold=True, color=WHITE, anchor="middle")
    fig.rect(PAGE_W - 52, 5, 42, 12, fill=UNSEEN, stroke=None, radius=2)
    fig.text(PAGE_W - 31, 8, "UNSEEN 4", size=5.5, bold=True, color=WHITE, anchor="middle")

    records = sorted(records, key=lambda row: (row["split"] == "unseen", (TRAIN_SCENARIOS + UNSEEN_SCENARIOS).index(row["name"])))
    _draw_feature_matrix(fig, records, 10, 29, 326, 247)
    _draw_distribution(fig, records, 341, 29, 171, 247)
    _draw_unseen_layouts(fig, records, 10, 286, 502, 170)
    return fig.drawing


CAPTION = (
    "Data-driven construction of the 9/4 TABX challenge split. "
    "(a) Unit and terrain counts for the nine training and four held-out scenarios. "
    "Every primitive appearing in the held-out set is observed during training "
    "(7/7 unit types and 3/3 terrain types), isolating compositional and geometric "
    "generalization from primitive novelty. (b) Scenario distribution by team size and "
    "mean initial cross-team distance; marker size denotes the number of terrain regions. "
    "Three held-out scenarios interpolate within the training distance range, whereas "
    "bypass provides controlled distance extrapolation and crossfire provides terrain-density "
    "extrapolation (four lava regions versus a training maximum of three). "
    "(c) Initial layouts of the four held-out scenarios and their intended test roles."
)


def _render_png(pdf_path: Path, png_path: Path, dpi: int) -> bool:
    executable = shutil.which("pdftoppm")
    if executable is None:
        return False
    wrapper_path = Path(executable)
    if os.name == "nt" and wrapper_path.suffix.lower() in {".cmd", ".bat"}:
        bundled_executable = (
            wrapper_path.parents[2] / "native" / "poppler" / "Library" / "bin" / "pdftoppm.exe"
        )
        if bundled_executable.exists():
            executable = str(bundled_executable)
    command = [
        executable,
        "-png",
        "-singlefile",
        "-r",
        str(dpi),
        str(pdf_path),
        str(png_path.with_suffix("")),
    ]
    if os.name == "nt" and Path(executable).suffix.lower() in {".cmd", ".bat"}:
        command = ["cmd.exe", "/c", *command]
    subprocess.run(command, check=True)
    return png_path.exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "pdf" / "challenge_split_paper_figure",
    )
    parser.add_argument("--png-dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records()
    drawing = build_figure(records)
    pdf_path = args.output_dir / "challenge_split_paper_figure.pdf"
    svg_path = args.output_dir / "challenge_split_paper_figure.svg"
    png_path = args.output_dir / "challenge_split_paper_figure.png"
    renderPDF.drawToFile(drawing, str(pdf_path), title="TABX challenge split rationale")
    renderSVG.drawToFile(drawing, str(svg_path))
    png_written = _render_png(pdf_path, png_path, args.png_dpi)
    (args.output_dir / "caption.txt").write_text(CAPTION + "\n", encoding="utf-8")
    metadata = {
        "width_in": PAGE_W / 72,
        "height_in": PAGE_H / 72,
        "png_dpi": args.png_dpi if png_written else None,
        "train": list(TRAIN_SCENARIOS),
        "unseen": list(UNSEEN_SCENARIOS),
    }
    (args.output_dir / "figure_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "pdf": str(pdf_path),
                "svg": str(svg_path),
                "png": str(png_path) if png_written else None,
                **metadata,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
