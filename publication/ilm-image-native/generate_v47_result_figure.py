#!/usr/bin/env python3
"""Render the measured V47 result from hash-pinned publication evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
DEFAULT_EVIDENCE = HERE / "evidence/v47"
DEFAULT_OUT = HERE / "figures/codec_spherical_glyph_language_v47_result.png"
EXPECTED_SHA256 = {
    "development_report.json": (
        "984d3dc380dc4c7b54f93bfce2bd8e4cb9dcb45524348adeacb441749ecea49d"
    ),
    "development_summary.json": (
        "01b3a579774be93eb0260545ceb42bab792fe917269555c2f12877c8ce76237f"
    ),
    "training_summary.json": (
        "f474b2e5a572349cc8f26350906fd5ed2a431387ac1c0e80466d25739b7fa471"
    ),
    "target_generated_reread_triplets.png": (
        "0a3e51460c0abd4c2b43363c79fc2029db7a290004f8d2cc35cbfc8532e76c95"
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
            raise ValueError(f"V47 evidence hash mismatch for {name}: {measured}")

    with (root / "development_report.json").open(encoding="utf-8") as handle:
        report = json.load(handle)
    with (root / "training_summary.json").open(encoding="utf-8") as handle:
        training = json.load(handle)
    triplets = Image.open(root / "target_generated_reread_triplets.png").convert(
        "RGB"
    )

    if report.get("claim_status") != "non-qualifying-development-result":
        raise ValueError("V47 report has an unexpected claim status")
    if report.get("all_gates_pass") or report.get("gates_passed") != 6:
        raise ValueError("V47 report does not record the frozen 6/16 decision")
    if report.get("gates_total") != 16 or sum(report["gates"].values()) != 6:
        raise ValueError("V47 gate count is inconsistent")
    if report.get("checkpoint_sha256") != training.get("checkpoint_sha256"):
        raise ValueError("V47 report and training summary checkpoints differ")
    if report.get("checkpoint_update") != 10_000 or training.get("update") != 10_000:
        raise ValueError("V47 figure requires the production checkpoint")
    pair_sequence = report.get("pair_sequence", {})
    if pair_sequence.get("count") != 80_000:
        raise ValueError("V47 did not consume the frozen 80,000-row sequence")
    if not pair_sequence.get("consumed_without_replacement"):
        raise ValueError("V47 pair sequence was reused")
    protocol = report.get("protocol_integrity", {})
    required_integrity = (
        "all_runtime_fields_finite",
        "embedded_codec_matches",
        "field_preflight_passes",
        "fixed_evaluation_arguments",
        "fixed_training_arguments",
        "metrics_finite",
        "pair_sequence_matches",
        "parameter_budget",
        "production_checkpoint",
        "protocol_document_matches",
        "source_files_match",
    )
    if not all(protocol.get(key) for key in required_integrity):
        raise ValueError("V47 protocol integrity failed outside the runtime gate")
    if protocol.get("total_elapsed_below_35_minutes"):
        raise ValueError("V47 runtime gate unexpectedly passed")
    if report.get("frozen_partition_opened") or not report.get("boundary_clean"):
        raise ValueError("V47 partition or student boundary is not clean")
    return report, training, triplets


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
    reference: str,
    measured: str,
    decision: str,
    passed: bool,
    left: int,
    right: int,
) -> None:
    draw.text((left, y), label, font=font(15, bold=True), fill=INK)
    ref_left = right - 438
    card(draw, (ref_left, y - 5, ref_left + 118, y + 27), fill=PALE_BLUE)
    centered_text(
        draw,
        (ref_left, y - 5, ref_left + 118, y + 27),
        reference,
        size=12,
        fill=BLUE,
        bold=True,
    )
    draw.text((ref_left + 129, y - 1), "->", font=font(16, bold=True), fill=MUTED)
    color = GREEN if passed else RED
    fill = PALE_GREEN if passed else PALE_RED
    card(
        draw,
        (ref_left + 166, y - 5, ref_left + 284, y + 27),
        fill=fill,
        outline=color,
    )
    centered_text(
        draw,
        (ref_left + 166, y - 5, ref_left + 284, y + 27),
        measured,
        size=12,
        fill=color,
        bold=True,
    )
    draw.text((right - 140, y), decision, font=font(12, bold=True), fill=color)


def bullet_list(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    lines: tuple[str, ...],
    color: str,
    square: bool = False,
) -> None:
    for line in lines:
        if square:
            draw.rectangle((x, y + 5, x + 10, y + 15), fill=color)
        else:
            draw.ellipse((x, y + 5, x + 10, y + 15), fill=color)
        draw.text((x + 22, y), line, font=font(14), fill=INK)
        y += 33


def render(
    report: dict[str, Any],
    training: dict[str, Any],
    triplets: Image.Image,
    output: Path,
) -> None:
    image = Image.new("RGB", (1800, 1450), PAPER)
    draw = ImageDraw.Draw(image)

    draw.text(
        (62, 31),
        "CODEC-SPHERICAL GLYPH LANGUAGE: V47",
        font=font(30, bold=True),
        fill=INK,
    )
    draw.text(
        (62, 73),
        "Frozen visual retina; from-scratch causal prediction on a continuous unit sphere",
        font=font(16),
        fill=MUTED,
    )
    card(draw, (1450, 27, 1738, 102), fill=RED, outline=RED)
    centered_text(
        draw,
        (1462, 31, 1726, 98),
        "NON-QUALIFYING\n6 / 16 GATES",
        size=18,
        fill=WHITE,
        bold=True,
    )

    pipeline = (
        (62, 294, "ORDERED RASTERS\n64 x 32 x 32"),
        (330, 604, "FROZEN V34 CODEC\nunit visual sphere"),
        (640, 914, "CAUSAL READER\n23.95M trainable"),
        (950, 1224, "ANCHOR + ENERGY\n4 sphere proposals"),
        (1260, 1738, "DECODE + REREAD + SELECT\nvisible raster feedback"),
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
        "2,048 fixed windows; the bank is evaluator-only",
        font=font(13),
        fill=MUTED,
    )
    language = report["language"]
    metric_row(
        draw,
        y=409,
        label="Full top-1",
        reference="V46 20.75%",
        measured=f"{100 * language['full_top1']:.2f}%",
        decision="FAIL",
        passed=False,
        left=88,
        right=854,
    )
    metric_row(
        draw,
        y=466,
        label="Full vs bigram",
        reference=f"{100 * language['bigram_top1']:.2f}%",
        measured=f"{100 * language['full_top1']:.2f}%",
        decision="FAIL",
        passed=False,
        left=88,
        right=854,
    )
    metric_row(
        draw,
        y=523,
        label="Ordered vs shuffled",
        reference=f"{100 * language['shuffled_top1']:.2f}%",
        measured=f"{100 * language['full_top1']:.2f}%",
        decision="FAIL",
        passed=False,
        left=88,
        right=854,
    )
    pairs = report["counterfactual_pairs"]
    metric_row(
        draw,
        y=580,
        label="Pair arm accuracy",
        reference="gate 60.00%",
        measured=f"{100 * pairs['full_arm_accuracy']:.2f}%",
        decision="FAIL",
        passed=False,
        left=88,
        right=854,
    )
    card(draw, (88, 632, 854, 685), fill=PALE_RED, outline=RED)
    centered_text(
        draw,
        (98, 635, 844, 682),
        "ONE SPHERICAL GLYPH VECTOR LOSES THE V42/V46 LANGUAGE SIGNAL",
        size=14,
        fill=RED,
        bold=True,
    )

    draw.text((946, 324), "AUTONOMOUS OUTPUT", font=font(20, bold=True), fill=INK)
    draw.text(
        (946, 356),
        "256 outputs; four proposals decoded and reread from pixels",
        font=font(13),
        fill=MUTED,
    )
    generated = report["generated"]
    metric_row(
        draw,
        y=409,
        label="Identity top-1",
        reference="V46 8.59%",
        measured=f"{100 * generated['generated_identity_top1']:.2f}%",
        decision="FAIL",
        passed=False,
        left=946,
        right=1712,
    )
    metric_row(
        draw,
        y=466,
        label="Binary pixel F1",
        reference="gate 0.55",
        measured=f"{generated['generated_pixel_f1']:.3f}",
        decision="FAIL",
        passed=False,
        left=946,
        right=1712,
    )
    metric_row(
        draw,
        y=523,
        label="Proposal / reread",
        reference="gate 0.90",
        measured=f"{generated['mean_selected_proposal_to_visible_reread_cosine']:.3f}",
        decision="FAIL",
        passed=False,
        left=946,
        right=1712,
    )
    metric_row(
        draw,
        y=580,
        label="Blank rate",
        reference="gate <2.00%",
        measured=f"{100 * generated['generated_blank_rate']:.2f}%",
        decision="PASS",
        passed=True,
        left=946,
        right=1712,
    )
    card(draw, (946, 632, 1712, 685), fill=PALE_GREEN, outline=GREEN)
    centered_text(
        draw,
        (956, 635, 1702, 682),
        "THE FROZEN CODEC REMAINS FINITE, NONBLANK, AND VISUALLY REVERSIBLE",
        size=14,
        fill=GREEN,
        bold=True,
    )

    raster_box = (62, 744, 1190, 1305)
    gate_box = (1230, 744, 1738, 1305)
    card(draw, raster_box)
    card(draw, gate_box)
    draw.text(
        (88, 770),
        "REAL HELD-OUT TARGET / GENERATION / REREAD TRIPLETS",
        font=font(20, bold=True),
        fill=INK,
    )
    draw.text(
        (88, 803),
        "Columns repeat: target | selected V47 pixels | pixels reread by the frozen codec",
        font=font(13),
        fill=MUTED,
    )
    scale = min(3, 1060 // triplets.width, 410 // triplets.height)
    enlarged = triplets.resize(
        (triplets.width * scale, triplets.height * scale),
        Image.Resampling.NEAREST,
    )
    raster_left = 88 + (1076 - enlarged.width) // 2
    raster_top = 852 + (360 - enlarged.height) // 2
    image.paste(enlarged, (raster_left, raster_top))
    draw.rectangle(
        (
            raster_left - 2,
            raster_top - 2,
            raster_left + enlarged.width + 1,
            raster_top + enlarged.height + 1,
        ),
        outline=LINE,
        width=2,
    )
    centered_text(
        draw,
        (88, 1248, 1164, 1290),
        "Generated marks are nonblank and locally glyph-like, but identity is usually wrong.",
        size=13,
        fill=MUTED,
    )

    draw.text((1256, 770), "FROZEN DECISION", font=font(20, bold=True), fill=INK)
    passed = [name for name, value in report["gates"].items() if value]
    failed = [name for name, value in report["gates"].items() if not value]
    if len(passed) != 6 or len(failed) != 10:
        raise ValueError("V47 gate partition changed")
    draw.text((1256, 817), "6 PASS", font=font(18, bold=True), fill=GREEN)
    bullet_list(
        draw,
        x=1258,
        y=853,
        color=GREEN,
        lines=(
            "frozen field preflight",
            "finite, nonblank generation",
            "identity above unigram",
            "parameter budget",
            "clean image-only boundary",
            "10k updates / 80k unique pairs",
        ),
    )
    draw.line((1256, 1060, 1712, 1060), fill=LINE, width=2)
    draw.text((1256, 1082), "10 FAIL", font=font(18, bold=True), fill=RED)
    bullet_list(
        draw,
        x=1258,
        y=1118,
        color=RED,
        square=True,
        lines=(
            "natural rank and log-p controls",
            "counterfactual binding > 0.60",
            "V46 identity / top-1 gains",
            "pixel F1 and decode-reread",
            "35-minute total runtime",
        ),
    )

    footer_y = 1337
    draw.line((62, footer_y, 1738, footer_y), fill=LINE, width=2)
    checkpoint = report["checkpoint_sha256"]
    codec_state = report["embedded_codec_state_sha256"]
    footer = (
        f"10,000 updates | 80,000 unique pair rows | train + audit "
        f"{report['total_elapsed_seconds']:.2f} s | peak "
        f"{language['peak_allocated_vram_gib']:.3f} GiB | checkpoint "
        f"{checkpoint[:12]}... | codec {codec_state[:12]}..."
    )
    centered_text(
        draw,
        (62, footer_y + 10, 1738, 1385),
        footer,
        size=14,
        fill=MUTED,
        bold=True,
    )
    centered_text(
        draw,
        (62, 1385, 1738, 1436),
        "Supported conclusion: preserve the visual codec; reject one normalized glyph vector as the language state.",
        size=16,
        fill=RED,
        bold=True,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, optimize=True)


def main() -> None:
    args = parse_args()
    report, training, triplets = load_evidence(args.evidence)
    render(report, training, triplets, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
