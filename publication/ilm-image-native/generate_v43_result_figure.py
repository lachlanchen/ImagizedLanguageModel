#!/usr/bin/env python3
"""Render the measured V42/V43 result from hash-pinned local evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
DEFAULT_EVIDENCE = HERE / "evidence"
DEFAULT_OUT = HERE / "figures/canonical_glyph_flow_v43_result.png"
EXPECTED_SHA256 = {
    "v42/development_report.json": (
        "b376c3ae42c147c6e6ae9ed81f0294240bfb3c2da527b52312040e63a36584f7"
    ),
    "v42/training_summary.json": (
        "f49074d741b2a2ef2d9b831c39a39eaff05164fbbb319882bfaf02256758aa64"
    ),
    "v42/target_generated_pairs.png": (
        "ee34c2c880eb900ea43b35aff400170d0c84445e2f3f6618869a8ad3277efc49"
    ),
    "v43/binding_training_summary.json": (
        "f8fccacc8b8188d6b9029521d2e1f2b81042b6d50e2532178274b35799b47170"
    ),
    "v43/writer_training_summary.json": (
        "bd611d1d4d8101f761ef096a28bbc5973f2ca8a5af41ad6d2fc65347759401d9"
    ),
    "v43/development_report.json": (
        "2de1dd9da701735e1283dbbde23fc404ec8437e541f462c2d6cbb39d8ee9e0be"
    ),
    "v43/diagnostic_report.json": (
        "1b9d84282d29a2d1b07440548893da589631d89b25124e6134a63c995c0d3bab"
    ),
    "v43/target_generated_pairs.png": (
        "24c3dd090be232374df6fb9021245a48bbf2fdd3c5c21c715d2234f8d43dfb64"
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


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = FONT_ROOT / name
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def centered_text(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    text: str,
    *,
    size: int,
    fill: str = INK,
    bold: bool = False,
    spacing: int = 4,
) -> None:
    face = font(size, bold=bold)
    box = draw.multiline_textbbox(
        (0, 0), text, font=face, spacing=spacing, align="center"
    )
    width = box[2] - box[0]
    height = box[3] - box[1]
    left, top, right, bottom = bounds
    draw.multiline_text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2),
        text,
        font=face,
        fill=fill,
        spacing=spacing,
        align="center",
    )


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
) -> None:
    draw.text((82, y), label, font=font(16, bold=True), fill=INK)
    draw.rounded_rectangle((315, y + 2, 715, y + 24), radius=4, fill="#dfe6e5")
    width = round(400 * min(1.0, value / maximum))
    if width:
        draw.rounded_rectangle((315, y + 2, 315 + width, y + 24), radius=4, fill=color)
    draw.text((730, y - 1), suffix, font=font(16, bold=True), fill=color)


def load_evidence(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Image.Image, Image.Image]:
    for relative, expected in EXPECTED_SHA256.items():
        actual = file_sha256(root / relative)
        if actual != expected:
            raise ValueError(f"evidence hash changed for {relative}: {actual}")
    v42 = json.loads((root / "v42/development_report.json").read_text("utf-8"))
    v43 = json.loads((root / "v43/development_report.json").read_text("utf-8"))
    diagnostic = json.loads((root / "v43/diagnostic_report.json").read_text("utf-8"))
    if v42.get("experiment") != "canonical-glyph-language-v42-development-audit":
        raise ValueError("unexpected V42 evidence")
    if v43.get("experiment") != "canonical-glyph-flow-v43-development":
        raise ValueError("unexpected V43 evidence")
    if v43.get("claim_status") != "rejected-or-partial":
        raise ValueError("V43 decision changed")
    if v42.get("frozen_partition_opened") or v43.get("frozen_partition_opened"):
        raise ValueError("the frozen partition must remain sealed")
    with Image.open(root / "v42/target_generated_pairs.png") as source:
        v42_sheet = source.convert("L")
    with Image.open(root / "v43/target_generated_pairs.png") as source:
        v43_sheet = source.convert("L")
    return v42, v43, diagnostic, v42_sheet, v43_sheet


def render(
    v42: dict[str, Any],
    v43: dict[str, Any],
    diagnostic: dict[str, Any],
    v42_sheet: Image.Image,
    v43_sheet: Image.Image,
    output_path: Path,
) -> None:
    image = Image.new("RGB", (1800, 1340), PAPER)
    draw = ImageDraw.Draw(image)
    draw.text(
        (62, 38),
        "CANONICAL GLYPH LANGUAGE: V42 -> V43",
        font=font(34, bold=True),
        fill=INK,
    )
    draw.text(
        (62, 83),
        "Measured Chinese raster-stream language signal and bank-free spatial output",
        font=font(19),
        fill=MUTED,
    )
    draw.rounded_rectangle((1450, 38, 1738, 108), radius=7, fill=RED)
    centered_text(
        draw,
        (1450, 38, 1738, 108),
        "V43 PARTIAL\n8 / 10 GATES",
        size=18,
        fill=WHITE,
        bold=True,
    )

    pipeline_top = 138
    box_width = 280
    gap = 65
    boxes = []
    for index in range(5):
        left = 62 + index * (box_width + gap)
        boxes.append((left, pipeline_top, left + box_width, pipeline_top + 104))
    labels = (
        "64 ORDERED\nGLYPH RASTERS",
        "V42 CAUSAL\nVISUAL CORE",
        "1,024-D CONTINUOUS\nIMAGE FIELD",
        "V43 SPATIAL\nRECTIFIED FLOW",
        "32 x 32 RASTER\nTHRESHOLD + REREAD",
    )
    for bounds, label in zip(boxes, labels):
        draw.rounded_rectangle(bounds, radius=6, fill=WHITE, outline=LINE, width=2)
        centered_text(draw, bounds, label, size=16, bold=True)
    for left, right in zip(boxes[:-1], boxes[1:]):
        arrow(draw, (left[2], pipeline_top + 52), (right[0], pipeline_top + 52))
    draw.text(
        (62, 258),
        "Student boundary: floating-point images in; direct raster out; no token IDs, Unicode IDs, OCR, codebook, glyph lookup, or deployed bank",
        font=font(16),
        fill=TEAL,
    )

    left_panel = (58, 300, 865, 575)
    right_panel = (935, 300, 1742, 575)
    for panel in (left_panel, right_panel):
        draw.rounded_rectangle(panel, radius=7, fill=WHITE, outline=LINE, width=2)
    draw.text(
        (82, 322), "V42 BOUNDED LANGUAGE SIGNAL", font=font(20, bold=True), fill=INK
    )
    language42 = v42["language"]
    metric_bar(
        draw,
        y=365,
        label="ordered raster history",
        value=language42["full_top1"],
        maximum=0.25,
        color=GREEN,
        suffix=f"{100 * language42['full_top1']:.1f}%",
    )
    metric_bar(
        draw,
        y=410,
        label="shuffled history",
        value=language42["shuffled_top1"],
        maximum=0.25,
        color=BLUE,
        suffix=f"{100 * language42['shuffled_top1']:.1f}%",
    )
    metric_bar(
        draw,
        y=455,
        label="symbolic bigram control",
        value=language42["bigram_top1"],
        maximum=0.25,
        color=GOLD,
        suffix=f"{100 * language42['bigram_top1']:.1f}%",
    )
    metric_bar(
        draw,
        y=500,
        label="image unigram control",
        value=language42["unigram_top1"],
        maximum=0.25,
        color=MUTED,
        suffix=f"{100 * language42['unigram_top1']:.1f}%",
    )
    draw.text(
        (82, 544),
        "PASS: ordered pixels beat unigram, bigram, and shuffled controls",
        font=font(15, bold=True),
        fill=GREEN,
    )

    draw.text(
        (959, 322), "V43 FIXED DEVELOPMENT DECISION", font=font(20, bold=True), fill=INK
    )
    language43 = v43["language"]
    pairs43 = v43["counterfactual_pairs"]
    generated43 = v43["generated"]
    rows = (
        (
            "ordered - bigram top-1",
            100 * (language43["full_top1"] - language43["bigram_top1"]),
            "> 1.00 pt",
            True,
        ),
        (
            "ordered - shuffled top-1",
            100 * (language43["full_top1"] - language43["shuffled_top1"]),
            "> 1.50 pt",
            True,
        ),
        (
            "exact-suffix arm accuracy",
            100 * pairs43["full_arm_accuracy"],
            "> 60.00%",
            False,
        ),
        (
            "generated pixel F1",
            generated43["generated_pixel_f1"],
            "> 0.550",
            False,
        ),
    )
    for index, (label, value, requirement, passed) in enumerate(rows):
        y = 365 + index * 48
        color = GREEN if passed else RED
        display = (
            f"{value:.2f} pt"
            if index < 2
            else (f"{value:.2f}%" if index == 2 else f"{value:.3f}")
        )
        draw.text((959, y), label, font=font(16, bold=True), fill=INK)
        draw.text((1365, y), display, font=font(16, bold=True), fill=color)
        draw.text((1512, y), requirement, font=font(14), fill=MUTED)
        draw.text(
            (1668, y),
            "PASS" if passed else "FAIL",
            font=font(14, bold=True),
            fill=color,
        )
    draw.rounded_rectangle((959, 535, 1718, 560), radius=4, fill=PALE_RED)
    centered_text(
        draw,
        (959, 535, 1718, 560),
        "Decision: reader/writer advance; complete V43 rejected",
        size=14,
        fill=RED,
        bold=True,
    )

    draw.text((62, 605), "REAL DEVELOPMENT OUTPUTS", font=font(20, bold=True), fill=INK)
    draw.text(
        (62, 635),
        "Each sheet alternates target | autonomous generated glyph. The two sheets use independent fixed development samples.",
        font=font(15),
        fill=MUTED,
    )
    sheet_width = 775
    sheet_height = round(sheet_width * v42_sheet.height / v42_sheet.width)
    for x, title, sheet, border in (
        (62, "V42 field generator: fragmented marks", v42_sheet, BLUE),
        (963, "V43 spatial flow: coherent form, wrong binding", v43_sheet, TEAL),
    ):
        draw.text((x, 675), title, font=font(17, bold=True), fill=border)
        resized = sheet.resize((sheet_width, sheet_height), Image.Resampling.NEAREST)
        image.paste(resized.convert("RGB"), (x, 710))
        draw.rectangle(
            (x - 1, 709, x + sheet_width, 710 + sheet_height),
            outline=border,
            width=2,
        )

    diagnosis_top = 1120
    draw.rounded_rectangle(
        (62, diagnosis_top, 882, 1260), radius=7, fill=PALE_GOLD, outline=GOLD, width=2
    )
    draw.rounded_rectangle(
        (918, diagnosis_top, 1738, 1260),
        radius=7,
        fill=PALE_GREEN,
        outline=GREEN,
        width=2,
    )
    seen = diagnostic["counterfactual_pairs"]["seen_train"]["v43"]["full_arm_accuracy"]
    unseen = diagnostic["counterfactual_pairs"]["unseen_train"]["v43"][
        "full_arm_accuracy"
    ]
    development = diagnostic["counterfactual_pairs"]["development"]["v43"][
        "full_arm_accuracy"
    ]
    centered_text(
        draw,
        (82, diagnosis_top + 8, 862, diagnosis_top + 132),
        "POST-RESULT BINDING DIAGNOSIS\n"
        f"seen pairs {100 * seen:.1f}%  |  unseen train {100 * unseen:.1f}%  |  "
        f"development {100 * development:.1f}%\n"
        "Small pair pool was memorized; long-history binding did not generalize",
        size=16,
        fill=GOLD,
        bold=True,
    )
    predicted = diagnostic["writer"]["predicted_plan"]["runtime_anchor"]["pixel_f1"]
    exact = diagnostic["writer"]["evaluator_exact_target_plan"]["runtime_anchor"][
        "pixel_f1"
    ]
    centered_text(
        draw,
        (938, diagnosis_top + 8, 1718, diagnosis_top + 132),
        "POST-RESULT MOTOR DIAGNOSIS\n"
        f"predicted plan F1 {predicted:.3f}  ->  exact-plan writer F1 {exact:.3f}\n"
        "The spatial writer works; the autonomous visual language plan is limiting",
        size=16,
        fill=GREEN,
        bold=True,
    )

    draw.rounded_rectangle((62, 1280, 1738, 1323), radius=7, fill=INK)
    centered_text(
        draw,
        (80, 1280, 1720, 1323),
        "BOUNDED POSITIVE LANGUAGE EVIDENCE  |  COMPLETE V43 NOT PASSED  |  FROZEN PARTITION REMAINS SEALED",
        size=16,
        fill=WHITE,
        bold=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, optimize=True)


def main() -> None:
    args = parse_args()
    evidence = load_evidence(args.evidence)
    render(*evidence, args.out)
    print(args.out.resolve())


if __name__ == "__main__":
    main()
