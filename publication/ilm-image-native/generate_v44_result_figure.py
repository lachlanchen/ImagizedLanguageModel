#!/usr/bin/env python3
"""Render the measured V44 result from hash-pinned publication evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
DEFAULT_EVIDENCE = HERE / "evidence/v44"
DEFAULT_OUT = HERE / "figures/canonical_glyph_binding_v44_result.png"
EXPECTED_SHA256 = {
    "training_summary.json": (
        "e2d63d1660d70eb78fe4e28ba06fc4efb5297b177ebfa9c753dfd4f23c650f6a"
    ),
    "development_report.json": (
        "9ab60bb6648655609e0b0e999e4bf24c1b4d20f21fb8f65d54e3ff8a76752b7c"
    ),
    "diagnostic_report.json": (
        "b8a0926280e3d474d092f967b7a784ccaa6b7fd24735e7b59778bfe99523b1d0"
    ),
}

FONT_ROOT = Path("/usr/share/fonts/truetype/dejavu")
PAPER = "#f4f6f5"
WHITE = "#ffffff"
INK = "#173039"
MUTED = "#5f7075"
LINE = "#b8c4c5"
TEAL = "#177d83"
BLUE = "#426f8d"
GREEN = "#2f765a"
PALE_GREEN = "#e4efe9"
RED = "#9a403a"
PALE_RED = "#f3e5e2"
GOLD = "#a67822"
PALE_GOLD = "#f3ecd9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_evidence(root: Path) -> tuple[dict[str, Any], ...]:
    payloads = []
    for name, expected in EXPECTED_SHA256.items():
        path = root / name
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"V44 evidence hash mismatch for {name}: {actual}")
        with path.open(encoding="utf-8") as handle:
            payloads.append(json.load(handle))
    return tuple(payloads)


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = FONT_ROOT / name
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def centered_text(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    value: str,
    *,
    size: int,
    fill: str = INK,
    bold: bool = False,
    spacing: int = 4,
) -> None:
    face = font(size, bold=bold)
    box = draw.multiline_textbbox(
        (0, 0), value, font=face, spacing=spacing, align="center"
    )
    width = box[2] - box[0]
    height = box[3] - box[1]
    left, top, right, bottom = bounds
    draw.multiline_text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2),
        value,
        font=face,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def card(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    *,
    fill: str = WHITE,
    outline: str = LINE,
) -> None:
    draw.rounded_rectangle(bounds, radius=7, fill=fill, outline=outline, width=2)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    draw.line((*start, *end), fill=TEAL, width=4)
    draw.polygon(
        (
            (end[0], end[1]),
            (end[0] - 12, end[1] - 7),
            (end[0] - 12, end[1] + 7),
        ),
        fill=TEAL,
    )


def metric_bar(
    draw: ImageDraw.ImageDraw,
    *,
    y: int,
    label: str,
    value: float,
    maximum: float,
    color: str,
    suffix: str,
    left: int,
    width: int,
) -> None:
    draw.text((left, y), label, font=font(16, bold=True), fill=INK)
    bar_left = left + 265
    bar_top = y + 2
    bar_width = width - 385
    draw.rounded_rectangle(
        (bar_left, bar_top, bar_left + bar_width, bar_top + 20),
        radius=4,
        fill="#e6ebea",
    )
    active = max(2, round(bar_width * max(0.0, min(maximum, value)) / maximum))
    draw.rounded_rectangle(
        (bar_left, bar_top, bar_left + active, bar_top + 20),
        radius=4,
        fill=color,
    )
    draw.text(
        (left + width - 108, y - 1),
        f"{value}{suffix}",
        font=font(16, bold=True),
        fill=color,
    )


def render(
    training: dict[str, Any],
    development: dict[str, Any],
    diagnostic: dict[str, Any],
    output_path: Path,
) -> None:
    image = Image.new("RGB", (1800, 1325), PAPER)
    draw = ImageDraw.Draw(image)
    draw.text((62, 35), "CANONICAL GLYPH BINDING: V44", font=font(30, bold=True), fill=INK)
    draw.text(
        (62, 76),
        "Measured frozen-base residual test: nonrepeating data fixes memorization, not visual binding",
        font=font(16),
        fill=MUTED,
    )
    card(draw, (1480, 32, 1738, 100), fill=RED, outline=RED)
    centered_text(
        draw,
        (1490, 36, 1728, 96),
        "V44 REJECTED\n8 / 14 GATES",
        size=17,
        fill=WHITE,
        bold=True,
    )

    pipeline_y = 130
    boxes = (
        (62, 255, "64 x 32 x 32\nRASTER HISTORY"),
        (348, 545, "FROZEN V42\n24.35M PARAMETERS"),
        (638, 835, "PREFIX MEMORY\nEXCLUDES LAST 4"),
        (928, 1125, "TANGENT RESIDUAL\n1.736M TRAINABLE"),
        (1218, 1440, "CONTINUOUS\nIMAGE FIELD"),
    )
    for left, right, label in boxes:
        card(draw, (left, pipeline_y, right, pipeline_y + 92))
        centered_text(
            draw,
            (left + 8, pipeline_y + 8, right - 8, pipeline_y + 84),
            label,
            size=15,
            bold=True,
        )
    for first, second in zip(boxes, boxes[1:]):
        arrow(draw, (first[1] + 8, pipeline_y + 46), (second[0] - 12, pipeline_y + 46))
    card(draw, (1480, pipeline_y, 1738, pipeline_y + 92), fill=PALE_GREEN, outline=GREEN)
    centered_text(
        draw,
        (1490, pipeline_y + 7, 1728, pipeline_y + 85),
        "24,000 UNIQUE PAIRS\nONE PASS  |  0.198 GiB",
        size=15,
        fill=GREEN,
        bold=True,
    )

    left_panel = (62, 250, 882, 665)
    right_panel = (918, 250, 1738, 665)
    card(draw, left_panel)
    card(draw, right_panel)
    draw.text((88, 272), "NATURAL LANGUAGE RETENTION", font=font(20, bold=True), fill=INK)
    draw.text(
        (88, 303),
        "Matched 2,048-window development audit",
        font=font(14),
        fill=MUTED,
    )
    base_language = development["matched_base_language"]
    language = development["language"]
    metric_bar(
        draw,
        y=344,
        label="V42 full top-1",
        value=round(100 * base_language["full_top1"], 2),
        maximum=25,
        color=BLUE,
        suffix="%",
        left=88,
        width=758,
    )
    metric_bar(
        draw,
        y=386,
        label="V44 full top-1",
        value=round(100 * language["full_top1"], 2),
        maximum=25,
        color=RED,
        suffix="%",
        left=88,
        width=758,
    )
    metric_bar(
        draw,
        y=428,
        label="V42 mean-field cosine",
        value=round(
            diagnostic["natural_alpha_sweep"]["0.00"]["mean_field_cosine"], 3
        ),
        maximum=1.0,
        color=BLUE,
        suffix="",
        left=88,
        width=758,
    )
    metric_bar(
        draw,
        y=470,
        label="V44 mean-field cosine",
        value=round(
            diagnostic["natural_alpha_sweep"]["1.00"]["mean_field_cosine"], 3
        ),
        maximum=1.0,
        color=RED,
        suffix="",
        left=88,
        width=758,
    )
    card(draw, (88, 525, 856, 632), fill=PALE_RED, outline=RED)
    centered_text(
        draw,
        (104, 534, 840, 623),
        "RETENTION GATES FAIL\n"
        f"top-1 {100 * base_language['full_top1']:.2f}% -> {100 * language['full_top1']:.2f}%"
        "   |   log p -5.238 -> -5.777\n"
        "Target cosine rises while discriminative probability falls",
        size=16,
        fill=RED,
        bold=True,
    )

    draw.text((944, 272), "COUNTERFACTUAL BINDING", font=font(20, bold=True), fill=INK)
    draw.text(
        (944, 303),
        "Same final four glyph images; different earlier history and target",
        font=font(14),
        fill=MUTED,
    )
    base_pairs = development["matched_base_development_pairs"]
    pairs = development["development_pairs"]
    unseen = development["unseen_training_pairs"]
    consumed = development["consumed_training_pairs"]
    for y, label, value, color in (
        (344, "V42 development", 100 * base_pairs["full_arm_accuracy"], BLUE),
        (386, "V44 development", 100 * pairs["full_arm_accuracy"], RED),
        (428, "V44 shuffled prefix", 100 * pairs["shuffled_arm_accuracy"], GOLD),
        (470, "V44 unseen train", 100 * unseen["full_arm_accuracy"], TEAL),
    ):
        metric_bar(
            draw,
            y=y,
            label=label,
            value=round(value, 2),
            maximum=70,
            color=color,
            suffix="%",
            left=944,
            width=758,
        )
    card(draw, (944, 525, 1712, 632), fill=PALE_GOLD, outline=GOLD)
    centered_text(
        draw,
        (960, 534, 1696, 623),
        "BINDING GATES FAIL\n"
        f"development {100 * pairs['full_arm_accuracy']:.2f}%  |  gate >60%"
        f"  |  unseen {100 * unseen['full_arm_accuracy']:.2f}%\n"
        f"consumed {100 * consumed['full_arm_accuracy']:.2f}% vs unseen"
        f" {100 * unseen['full_arm_accuracy']:.2f}%: memorization gap removed",
        size=16,
        fill=GOLD,
        bold=True,
    )

    lower_left = (62, 695, 1015, 1208)
    lower_right = (1048, 695, 1738, 1208)
    card(draw, lower_left)
    card(draw, lower_right)
    draw.text((88, 720), "POST-RESULT RESIDUAL SCALE SWEEP", font=font(20, bold=True), fill=INK)
    draw.text(
        (88, 751),
        "Development pair arm accuracy; fixed decision unchanged",
        font=font(14),
        fill=MUTED,
    )
    chart = (120, 805, 970, 1110)
    x0, y0, x1, y1 = chart
    draw.line((x0, y1, x1, y1), fill=LINE, width=2)
    draw.line((x0, y0, x0, y1), fill=LINE, width=2)
    for percent in (45, 50, 55, 60, 65):
        y = y1 - (percent - 45) / 20 * (y1 - y0)
        draw.line((x0, y, x1, y), fill="#dce3e2", width=1)
        draw.text((78, y - 9), f"{percent}%", font=font(13), fill=MUTED)
    gate_y = y1 - (60 - 45) / 20 * (y1 - y0)
    draw.line((x0, gate_y, x1, gate_y), fill=RED, width=2)
    draw.text((x1 - 106, gate_y - 24), "60% gate", font=font(13, bold=True), fill=RED)
    sweep = diagnostic["development_pair_diagnostic"]["alpha_sweep"]
    points = []
    for index, alpha in enumerate(diagnostic["alphas"]):
        x = x0 + index * (x1 - x0) / (len(diagnostic["alphas"]) - 1)
        value = 100 * sweep[f"{alpha:.2f}"]["arm_accuracy"]
        y = y1 - (value - 45) / 20 * (y1 - y0)
        points.append((x, y))
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=TEAL)
        draw.text((x - 18, y1 + 13), str(alpha), font=font(12), fill=MUTED)
    draw.line(points, fill=TEAL, width=4)
    draw.text((475, 1150), "residual strength alpha", font=font(14), fill=MUTED)

    draw.text((1074, 720), "POST-RESULT GEOMETRY DIAGNOSIS", font=font(20, bold=True), fill=INK)
    draw.text(
        (1074, 751),
        "Measured direction, not an evaluator-selected rewrite",
        font=font(14),
        fill=MUTED,
    )
    delta = diagnostic["development_pair_diagnostic"]["delta_alignment"]
    card(draw, (1074, 805, 1712, 925), fill=PALE_RED, outline=RED)
    centered_text(
        draw,
        (1090, 814, 1696, 916),
        "LEARNED UPDATE POINTS THE WRONG WAY\n"
        f"target-displacement cosine {delta['learned_update_delta_cosine']:+.4f}\n"
        f"best alpha reaches only {100 * max(item['arm_accuracy'] for item in sweep.values()):.2f}%",
        size=17,
        fill=RED,
        bold=True,
    )
    card(draw, (1074, 950, 1712, 1080), fill=PALE_GOLD, outline=GOLD)
    centered_text(
        draw,
        (1090, 958, 1696, 1072),
        "COMMON IMAGE MODE ABSORBS THE UPDATE\n"
        "mean-field cosine 0.777 -> 0.882\n"
        "target cosine rises, but rank and probability decline",
        size=17,
        fill=GOLD,
        bold=True,
    )
    centered_text(
        draw,
        (1080, 1093, 1706, 1182),
        "Next bounded test: center and variance-balance the continuous raster field\nbefore training another language reader or reopening the writer",
        size=16,
        fill=INK,
        bold=True,
    )

    card(draw, (62, 1240, 1738, 1290), fill=INK, outline=INK)
    centered_text(
        draw,
        (78, 1242, 1722, 1288),
        "V42 LANGUAGE SIGNAL PRESERVED AS BASE  |  V44 BINDING NOT PASSED  |  WRITER CLOSED  |  FROZEN PARTITION SEALED",
        size=15,
        fill=WHITE,
        bold=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, optimize=True)


def main() -> None:
    args = parse_args()
    render(*load_evidence(args.evidence), args.out)
    print(args.out.resolve())


if __name__ == "__main__":
    main()
