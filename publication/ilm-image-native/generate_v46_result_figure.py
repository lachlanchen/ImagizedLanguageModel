#!/usr/bin/env python3
"""Render the measured V46 result from hash-pinned publication evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
DEFAULT_EVIDENCE = HERE / "evidence/v46"
DEFAULT_OUT = HERE / "figures/scaled_retinal_glyph_language_v46_result.png"
EXPECTED_SHA256 = {
    "development_report.json": (
        "553733028545cd40fcaedfccdfbc57acccdc0c2950251b26039a15090ba08a44"
    ),
    "development_summary.json": (
        "81081aff1af6f05253ea94227828ef5d576895c4ab54664457ba24e65fa1b437"
    ),
    "training_summary.json": (
        "8ff86342777d6687823d795581f1502ddd72a38a1ce468898693f3862f9960e6"
    ),
    "target_generated_pairs.png": (
        "111d07e788b11427fff5b36618213a6a15c8e0f1cf59fa8fe18b135447d4136c"
    ),
}

FONT_ROOT = Path("/usr/share/fonts/truetype/dejavu")
PAPER = "#f4f6f5"
WHITE = "#ffffff"
INK = "#173039"
MUTED = "#607176"
LINE = "#b9c5c6"
TEAL = "#177d83"
BLUE = "#426f8d"
PALE_BLUE = "#e6edf2"
GREEN = "#2f765a"
PALE_GREEN = "#e4efe9"
RED = "#9a403a"
PALE_RED = "#f3e5e2"
GOLD = "#9a721f"
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


def load_evidence(root: Path) -> tuple[dict[str, Any], dict[str, Any], Image.Image]:
    for name, expected in EXPECTED_SHA256.items():
        measured = file_sha256(root / name)
        if measured != expected:
            raise ValueError(f"V46 evidence hash mismatch for {name}: {measured}")

    with (root / "development_report.json").open(encoding="utf-8") as handle:
        report = json.load(handle)
    with (root / "training_summary.json").open(encoding="utf-8") as handle:
        training = json.load(handle)
    pairs = Image.open(root / "target_generated_pairs.png").convert("RGB")

    if report.get("claim_status") != "non-qualifying-development-result":
        raise ValueError("V46 report has an unexpected claim status")
    if report.get("all_gates_pass") or report.get("gates_passed") != 10:
        raise ValueError("V46 report does not record the frozen 10/14 decision")
    if report.get("gates_total") != 14 or sum(report["gates"].values()) != 10:
        raise ValueError("V46 gate count is inconsistent")
    if report.get("checkpoint_sha256") != training.get("checkpoint_sha256"):
        raise ValueError("V46 report and training summary checkpoints differ")
    if report.get("checkpoint_update") != 10_000:
        raise ValueError("V46 figure requires the production checkpoint")
    if not report.get("protocol_integrity_clean") or not report.get("boundary_clean"):
        raise ValueError("V46 report failed protocol or boundary integrity")
    return report, training, pairs


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = FONT_ROOT / name
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def card(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    *,
    fill: str = WHITE,
    outline: str = LINE,
) -> None:
    draw.rounded_rectangle(bounds, radius=7, fill=fill, outline=outline, width=2)


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


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    draw.line((*start, *end), fill=TEAL, width=4)
    draw.polygon(
        ((end[0], end[1]), (end[0] - 11, end[1] - 7), (end[0] - 11, end[1] + 7)),
        fill=TEAL,
    )


def metric_row(
    draw: ImageDraw.ImageDraw,
    *,
    y: int,
    label: str,
    baseline: str,
    measured: str,
    decision: str,
    passed: bool,
    left: int,
    right: int,
) -> None:
    draw.text((left, y), label, font=font(15, bold=True), fill=INK)
    base_left = right - 430
    card(draw, (base_left, y - 5, base_left + 105, y + 27), fill=PALE_BLUE)
    centered_text(
        draw,
        (base_left, y - 5, base_left + 105, y + 27),
        baseline,
        size=13,
        fill=BLUE,
        bold=True,
    )
    draw.text((base_left + 118, y - 1), "->", font=font(16, bold=True), fill=MUTED)
    color = GREEN if passed else RED
    fill = PALE_GREEN if passed else PALE_RED
    card(draw, (base_left + 155, y - 5, base_left + 270, y + 27), fill=fill, outline=color)
    centered_text(
        draw,
        (base_left + 155, y - 5, base_left + 270, y + 27),
        measured,
        size=13,
        fill=color,
        bold=True,
    )
    draw.text((right - 140, y), decision, font=font(13, bold=True), fill=color)


def render(
    report: dict[str, Any],
    training: dict[str, Any],
    pairs: Image.Image,
    output: Path,
) -> None:
    image = Image.new("RGB", (1800, 1390), PAPER)
    draw = ImageDraw.Draw(image)

    draw.text(
        (62, 31),
        "SCALED RETINAL GLYPH LANGUAGE: V46",
        font=font(30, bold=True),
        fill=INK,
    )
    draw.text(
        (62, 73),
        "Frozen from-scratch test of the full V45 field in the exact V42 causal architecture",
        font=font(16),
        fill=MUTED,
    )
    card(draw, (1450, 27, 1738, 102), fill=RED, outline=RED)
    centered_text(
        draw,
        (1462, 31, 1726, 98),
        "NON-QUALIFYING\n10 / 14 GATES",
        size=18,
        fill=WHITE,
        bold=True,
    )

    pipeline = (
        (62, 302, "ORDERED RASTERS\n64 x 32 x 32"),
        (340, 620, "FIXED V45 FIELD\nv = A(d - mu) / s"),
        (658, 938, "CAUSAL READER\n24.35M parameters"),
        (976, 1256, "ANCHOR + ENERGY\n4 full-field samples"),
        (1294, 1738, "EXACT INVERSE + REREAD\nbank-free raster feedback"),
    )
    for index, (left, right, label) in enumerate(pipeline):
        fill = PALE_GOLD if index == 3 else WHITE
        outline = GOLD if index == 3 else LINE
        card(draw, (left, 132, right, 265), fill=fill, outline=outline)
        centered_text(
            draw,
            (left + 8, 140, right - 8, 257),
            label,
            size=15,
            fill=GOLD if index == 3 else INK,
            bold=True,
        )
    for first, second in zip(pipeline, pipeline[1:]):
        arrow(draw, (first[1] + 7, 198), (second[0] - 11, 198))

    language_box = (62, 300, 880, 710)
    output_box = (920, 300, 1738, 710)
    card(draw, language_box)
    card(draw, output_box)
    draw.text((88, 324), "HELD-OUT LANGUAGE", font=font(20, bold=True), fill=INK)
    draw.text(
        (88, 356),
        "2,048 fixed windows; evaluator bank only identifies emitted fields",
        font=font(13),
        fill=MUTED,
    )
    language = report["language"]
    metric_row(
        draw,
        y=409,
        label="Full top-1",
        baseline="V42 19.97%",
        measured=f"{100 * language['full_top1']:.2f}%",
        decision="+0.78 pp  FAIL",
        passed=False,
        left=88,
        right=854,
    )
    metric_row(
        draw,
        y=466,
        label="Target log p",
        baseline="V42 -5.255",
        measured=f"{language['full_target_log_probability']:.3f}",
        decision="+0.320  PASS",
        passed=True,
        left=88,
        right=854,
    )
    metric_row(
        draw,
        y=523,
        label="Ordered vs shuffled",
        baseline=f"{100 * language['shuffled_top1']:.2f}%",
        measured=f"{100 * language['full_top1']:.2f}%",
        decision="+1.51 pp  PASS",
        passed=True,
        left=88,
        right=854,
    )
    pairs_report = report["counterfactual_pairs"]
    metric_row(
        draw,
        y=580,
        label="Pair arm accuracy",
        baseline="gate 60.00%",
        measured=f"{100 * pairs_report['full_arm_accuracy']:.2f}%",
        decision="-5.70 pp  FAIL",
        passed=False,
        left=88,
        right=854,
    )
    card(draw, (88, 632, 854, 685), fill=PALE_GREEN, outline=GREEN)
    centered_text(
        draw,
        (98, 635, 844, 682),
        "ORDERED PIXELS STILL BEAT UNIGRAM, BIGRAM, AND SHUFFLED CONTROLS",
        size=14,
        fill=GREEN,
        bold=True,
    )

    draw.text((946, 324), "AUTONOMOUS OUTPUT", font=font(20, bold=True), fill=INK)
    draw.text(
        (946, 356),
        "256 bank-free generated fields; exact inverse; generated pixels reread",
        font=font(13),
        fill=MUTED,
    )
    generated = report["generated"]
    metric_row(
        draw,
        y=409,
        label="Identity top-1",
        baseline="V42 8.20%",
        measured=f"{100 * generated['generated_identity_top1']:.2f}%",
        decision="+0.39 pp  FAIL",
        passed=False,
        left=946,
        right=1712,
    )
    metric_row(
        draw,
        y=466,
        label="Binary pixel F1",
        baseline="V42 0.373",
        measured=f"{generated['generated_pixel_f1']:.3f}",
        decision="gate 0.55  FAIL",
        passed=False,
        left=946,
        right=1712,
    )
    metric_row(
        draw,
        y=523,
        label="Radius MAE",
        baseline=f"anchor {generated['anchor_radius_mae']:.3f}",
        measured=f"sample {generated['generated_radius_mae']:.3f}",
        decision="diagnostic",
        passed=True,
        left=946,
        right=1712,
    )
    metric_row(
        draw,
        y=580,
        label="Blank rate",
        baseline="gate <2.00%",
        measured=f"{100 * generated['generated_blank_rate']:.2f}%",
        decision="PASS",
        passed=True,
        left=946,
        right=1712,
    )
    card(draw, (946, 632, 1712, 685), fill=PALE_RED, outline=RED)
    centered_text(
        draw,
        (956, 635, 1702, 682),
        "FIELD SCALING DOES NOT COUPLE IDENTITY, RADIUS, AND CLEAN PIXELS",
        size=14,
        fill=RED,
        bold=True,
    )

    raster_box = (62, 744, 1170, 1258)
    gate_box = (1210, 744, 1738, 1258)
    card(draw, raster_box)
    card(draw, gate_box)
    draw.text((88, 770), "REAL HELD-OUT TARGET / GENERATION PAIRS", font=font(20, bold=True), fill=INK)
    draw.text(
        (88, 803),
        "Alternating columns: target raster | selected V46 generation",
        font=font(13),
        fill=MUTED,
    )
    scale = min(3, (1030 // pairs.width), (365 // pairs.height))
    enlarged = pairs.resize(
        (pairs.width * scale, pairs.height * scale),
        Image.Resampling.NEAREST,
    )
    raster_left = 88 + (1056 - enlarged.width) // 2
    image.paste(enlarged, (raster_left, 850))
    draw.rectangle(
        (raster_left - 2, 848, raster_left + enlarged.width + 1, 850 + enlarged.height + 1),
        outline=LINE,
        width=2,
    )
    centered_text(
        draw,
        (88, 1221, 1144, 1247),
        "Recognizable structure survives in some samples; fragmentation remains common.",
        size=13,
        fill=MUTED,
    )

    draw.text((1236, 770), "FROZEN DECISION", font=font(20, bold=True), fill=INK)
    passed = [name for name, value in report["gates"].items() if value]
    failed = [name for name, value in report["gates"].items() if not value]
    if len(passed) != 10 or len(failed) != 4:
        raise ValueError("V46 gate partition changed")
    draw.text((1236, 819), "10 PASS", font=font(18, bold=True), fill=GREEN)
    pass_lines = (
        "ordered > unigram / bigram",
        "ordered > shuffled rank / log p",
        "generated > unigram; nonblank",
        "V42 log-p gain",
        "boundary, resources, integrity",
    )
    y = 856
    for line in pass_lines:
        draw.ellipse((1238, y + 5, 1248, y + 15), fill=GREEN)
        draw.text((1260, y), line, font=font(14), fill=INK)
        y += 34
    draw.line((1236, 1037, 1712, 1037), fill=LINE, width=2)
    draw.text((1236, 1060), "4 FAIL", font=font(18, bold=True), fill=RED)
    fail_lines = (
        "counterfactual binding > 0.60",
        "full top-1 > V42 + 0.01",
        "generated identity > V42 + 0.01",
        "generated pixel F1 > 0.55",
    )
    y = 1098
    for line in fail_lines:
        draw.rectangle((1238, y + 5, 1248, y + 15), fill=RED)
        draw.text((1260, y), line, font=font(14), fill=INK)
        y += 34

    footer_y = 1292
    draw.line((62, footer_y, 1738, footer_y), fill=LINE, width=2)
    elapsed = report["total_elapsed_seconds"]
    vram = report["language"]["peak_allocated_vram_gib"]
    checkpoint = report["checkpoint_sha256"]
    field_hash = report["retinal_field"]["field_state_sha256"]
    footer = (
        f"10,000 updates  |  train + audit {elapsed:.2f} s  |  peak {vram:.3f} GiB  |  "
        f"checkpoint {checkpoint[:12]}...  |  V45 field {field_hash[:12]}..."
    )
    centered_text(
        draw,
        (62, footer_y + 12, 1738, 1352),
        footer,
        size=14,
        fill=MUTED,
        bold=True,
    )
    centered_text(
        draw,
        (62, 1344, 1738, 1380),
        "Bounded result: continuous raster language survives; autonomous binding and writing do not qualify.",
        size=16,
        fill=RED,
        bold=True,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, optimize=True)


def main() -> None:
    args = parse_args()
    report, training, pairs = load_evidence(args.evidence)
    render(report, training, pairs, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
