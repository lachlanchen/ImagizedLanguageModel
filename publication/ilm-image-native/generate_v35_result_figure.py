#!/usr/bin/env python3
"""Compose the measured V35 result from hash-pinned development evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = (
    ROOT
    / "artifacts/causal_glyph_flow_v35_20260814/development/development_report.json"
)
DEFAULT_COPY = (
    ROOT
    / "artifacts/causal_glyph_flow_v35_20260814/development/galleries/ema/copy_anchor.png"
)
DEFAULT_INSTRUCTION = (
    ROOT
    / "artifacts/causal_glyph_flow_v35_20260814/development/galleries/ema/instruction_anchor.png"
)
DEFAULT_PROTOCOL = ROOT / "references/causal_glyph_flow_v35_protocol.md"
DEFAULT_OUT = Path(__file__).resolve().parent / "figures/causal_glyph_flow_v35_result.png"

EXPECTED_REPORT_SHA256 = (
    "3d14e15d6f4f8677de864a793cb76efcb35a9294e1c5f7d28a647b66aa6617ba"
)
EXPECTED_COPY_SHA256 = (
    "324f5a918ac4b1642b55f9906b846ee5b5a72b125907de52c211e05bdfcdc095"
)
EXPECTED_INSTRUCTION_SHA256 = (
    "4e9243b09411e341b936671148de6e423cbc5001f36fe51feb98ead47ff6355c"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "ca30872ffdc84d3719068d27ad456da9629428eed6a37ca9eaf62f40c3acb0b1"
)
EXPECTED_PROTOCOL_SHA256 = (
    "d7a4d49270676cd82c55e22ddd73466966e0b96723970f76fe66fa2381bd3718"
)

FONT_ROOT = Path("/usr/share/fonts/truetype/dejavu")
INK = "#17323c"
MUTED = "#587078"
LINE = "#a8b8bd"
PAPER = "#f3f6f7"
WHITE = "#ffffff"
TEAL = "#217f8f"
GREEN = "#34745d"
AMBER = "#a66c28"
RED = "#a44747"
BLUE = "#426b88"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--copy", type=Path, default=DEFAULT_COPY)
    parser.add_argument("--instruction", type=Path, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_ROOT / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate(args: argparse.Namespace) -> dict[str, Any]:
    expected = {
        args.report: EXPECTED_REPORT_SHA256,
        args.copy: EXPECTED_COPY_SHA256,
        args.instruction: EXPECTED_INSTRUCTION_SHA256,
        args.protocol: EXPECTED_PROTOCOL_SHA256,
    }
    for path, digest in expected.items():
        if file_sha256(path) != digest:
            raise ValueError(f"V35 evidence SHA-256 changed: {path}")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("architecture") != "causal-glyph-flow-v35-development-audit":
        raise ValueError("unexpected V35 report architecture")
    if report.get("label") != "evidence" or not report.get("evidence_eligible"):
        raise ValueError("V35 report is not production evidence")
    if report.get("decision", {}).get("status") != "not-qualified":
        raise ValueError("V35 decision changed")
    if report.get("decision", {}).get("selected_writer") != "anchor":
        raise ValueError("V35 selected writer changed")
    if report.get("checkpoint", {}).get("ema", {}).get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("V35 checkpoint receipt changed")
    checkpoint_path = report.get("checkpoint", {}).get("ema", {}).get("checkpoint")
    if not checkpoint_path:
        raise ValueError("V35 checkpoint path receipt is absent")
    run_protocol = report.get("checkpoint_audit", {}).get("checks", {})
    if not report.get("checkpoint_audit", {}).get("passed") or not all(run_protocol.values()):
        raise ValueError("V35 checkpoint audit changed")
    if not report.get("closed_loop_receipt", {}).get("passed"):
        raise ValueError("V35 closed-loop receipt changed")
    return report


def centered(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    value: str,
    *,
    size: int,
    fill: str = INK,
    bold: bool = False,
) -> None:
    left, top, right, bottom = bounds
    face = font(size, bold=bold)
    box = draw.multiline_textbbox((0, 0), value, font=face, align="center", spacing=5)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.multiline_text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2),
        value,
        font=face,
        fill=fill,
        align="center",
        spacing=5,
    )


def badge(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    value: str,
    *,
    fill: str,
) -> None:
    draw.rounded_rectangle(bounds, radius=6, fill=fill)
    centered(draw, bounds, value, size=15, fill=WHITE, bold=True)


def summary_panel(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    title: str,
    lines: tuple[str, ...],
    *,
    accent: str,
) -> None:
    left, top, right, bottom = bounds
    draw.rounded_rectangle(bounds, radius=7, fill=WHITE, outline=LINE, width=2)
    draw.rectangle((left, top, left + 10, bottom), fill=accent)
    draw.text((left + 28, top + 18), title, font=font(21, bold=True), fill=INK)
    for index, line in enumerate(lines):
        draw.text(
            (left + 28, top + 60 + index * 31),
            line,
            font=font(16),
            fill=MUTED,
        )


def metric_tile(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    label: str,
    value: str,
    gate: str,
    *,
    passed: bool,
) -> None:
    left, top, right, bottom = bounds
    accent = GREEN if passed else RED
    fill = "#edf6f2" if passed else "#f8e9e8"
    draw.rounded_rectangle(bounds, radius=7, fill=fill, outline=LINE, width=2)
    draw.rectangle((left, top, left + 8, bottom), fill=accent)
    draw.text((left + 23, top + 12), label, font=font(14, bold=True), fill=INK)
    draw.text((left + 23, top + 44), value, font=font(23, bold=True), fill=accent)
    wrapped = "\n".join(textwrap.wrap(gate, width=35))
    draw.multiline_text(
        (left + 23, top + 81),
        wrapped,
        font=font(12),
        fill=MUTED,
        spacing=4,
    )
    badge(
        draw,
        (right - 76, top + 12, right - 14, top + 38),
        "PASS" if passed else "FAIL",
        fill=accent,
    )


def evidence_crop(path: Path) -> Image.Image:
    with Image.open(path) as source:
        crop = source.convert("RGB").crop((0, 45, source.width, 510))
    return crop.resize((940, 837), Image.Resampling.LANCZOS)


def evidence_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    evidence: Image.Image,
) -> None:
    left, top, right, bottom = bounds
    draw.rounded_rectangle(bounds, radius=7, fill=WHITE, outline=LINE, width=2)
    draw.text((left + 20, top + 15), title, font=font(18, bold=True), fill=INK)
    draw.text((left + 20, top + 48), subtitle, font=font(13), fill=RED)
    image_top = top + 80
    available_height = bottom - image_top - 16
    resized = evidence.resize(
        (right - left - 32, available_height), Image.Resampling.LANCZOS
    )
    canvas.paste(resized, (left + 16, image_top))


def main() -> None:
    args = parse_args()
    report = load_and_validate(args)
    decision = report["decision"]
    writer = decision["selected_writer"]
    ema = report["states"]["ema"]
    copy = ema["autonomous"]["copy"][writer]
    instruction = ema["autonomous"]["instruction"][writer]
    counterfactual = ema["autonomous"]["copy_counterfactual"][writer]
    public_teacher = ema["teacher_forced"]["public"]

    copy_correct = copy["conditions"]["correct"]
    copy_shuffled = copy["conditions"]["shuffled"]
    instruction_correct = instruction["conditions"]["correct"]
    instruction_shuffled = instruction["conditions"]["shuffled"]
    instruction_blank = instruction["conditions"]["blank"]
    copy_ceiling = copy["target_ocr"]["codec_character_accuracy"]
    instruction_ceiling = instruction["target_ocr"]["codec_character_accuracy"]

    visual_checks = decision["visual_causal"]["checks"]
    semantic_checks = decision["semantic_raster"]["checks"]
    visual_passes = sum(bool(value) for value in visual_checks.values())
    semantic_passes = sum(bool(value) for value in semantic_checks.values())

    canvas = Image.new("RGB", (2400, 1700), PAPER)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (58, 34),
        "V35 closes the raster loop, but prompt-to-answer binding fails",
        font=font(36, bold=True),
        fill=INK,
    )
    draw.text(
        (60, 89),
        "129.09M parameters | 22,000 BF16 updates | one RTX 4090 | fixed EMA development audit",
        font=font(18),
        fill=MUTED,
    )
    badge(draw, (2032, 42, 2340, 94), "NOT QUALIFIED", fill=RED)

    summary_panel(
        draw,
        (58, 140, 765, 330),
        "Training integrity",
        (
            "22,000 / 22,000 finite updates",
            "Stage A alignment passed",
            "2.899 GiB peak allocated VRAM",
            "checkpoint ca30872f...b0b1",
        ),
        accent=GREEN,
    )
    summary_panel(
        draw,
        (822, 140, 1582, 330),
        "Independent visual runtime",
        (
            "inputs: pixels + patch mask only",
            "output: generated binary raster",
            "feedback: decode -> threshold -> re-encode",
            "no token, OCR, lookup, or runtime teacher",
        ),
        accent=TEAL,
    )
    summary_panel(
        draw,
        (1639, 140, 2340, 330),
        "Evidence decision",
        (
            f"visual-causal gates: {visual_passes}/12",
            f"semantic-raster gates: {semantic_passes}/9",
            "selected writer: EMA anchor",
            "sealed split: NOT OPENED",
        ),
        accent=RED,
    )

    draw.text((58, 370), "Fixed primary gates", font=font(24, bold=True), fill=INK)
    draw.text(
        (360, 376),
        "Nonblank, prompt-responsive pixels are insufficient without correct target binding",
        font=font(16),
        fill=MUTED,
    )

    tiles = (
        (
            "Copy target OCR ceiling",
            f"{100 * copy_ceiling:.2f}%",
            "required >= 70%",
            bool(visual_checks["copy_target_ocr_ceiling"]),
        ),
        (
            "Copy output / ceiling",
            f"{100 * copy_correct['ocr_character_accuracy'] / copy_ceiling:.2f}%",
            "required >= 60% retention",
            bool(visual_checks["copy_ocr_retention"]),
        ),
        (
            "Copy correct - shuffled",
            f"{100 * (copy_correct['ocr_character_accuracy'] - copy_shuffled['ocr_character_accuracy']):+.3f} pp",
            "required >= +20 percentage points",
            bool(visual_checks["copy_correct_minus_shuffled"]),
        ),
        (
            "Copy counterfactual preference",
            f"{100 * counterfactual['target_preference_rate']:.1f}%",
            "required >= 75%",
            bool(visual_checks["copy_counterfactual_target_preference"]),
        ),
        (
            "Public teacher ink / edge F1",
            f"{public_teacher['decoded_ink_f1']:.3f} / {public_teacher['decoded_edge_f1']:.3f}",
            "both required >= 0.70",
            bool(visual_checks["public_teacher_ink_f1"])
            and bool(visual_checks["public_teacher_edge_f1"]),
        ),
        (
            "Instruction target ceiling",
            f"{100 * instruction_ceiling:.2f}%",
            "required >= 60%",
            bool(semantic_checks["instruction_target_ocr_ceiling"]),
        ),
        (
            "Instruction output accuracy",
            f"{100 * instruction_correct['ocr_character_accuracy']:.3f}%",
            "required >= 8%",
            bool(semantic_checks["instruction_correct_accuracy"]),
        ),
        (
            "Instruction correct - shuffled",
            f"{100 * (instruction_correct['ocr_character_accuracy'] - instruction_shuffled['ocr_character_accuracy']):+.3f} pp",
            "required >= +2 percentage points",
            bool(semantic_checks["instruction_correct_minus_shuffled"]),
        ),
        (
            "Instruction correct - blank",
            f"{100 * (instruction_correct['ocr_character_accuracy'] - instruction_blank['ocr_character_accuracy']):+.3f} pp",
            "required >= +3 percentage points",
            bool(semantic_checks["instruction_correct_minus_blank"]),
        ),
        (
            "Prompt pixel response",
            "present",
            "shuffled + blank response checks pass",
            bool(semantic_checks["instruction_pixel_response_shuffled"])
            and bool(semantic_checks["instruction_pixel_response_blank"]),
        ),
    )
    tile_width = 444
    tile_height = 135
    for index, (label, value, gate, passed) in enumerate(tiles):
        column = index % 5
        row = index // 5
        left = 58 + column * 462
        top = 415 + row * 151
        metric_tile(
            draw,
            (left, top, left + tile_width, top + tile_height),
            label,
            value,
            gate,
            passed=passed,
        )

    draw.text(
        (58, 735),
        "Direct pixel evidence from the fixed EMA development gallery",
        font=font(23, bold=True),
        fill=INK,
    )
    draw.text(
        (900, 741),
        "Targets are readable; generated condition rows are corrupted and mostly OCR-incorrect",
        font=font(15),
        fill=RED,
    )
    evidence_panel(
        canvas,
        draw,
        (58, 778, 1170, 1636),
        "Copy controls",
        "correct, shuffled, blank, and final-quarter outputs",
        evidence_crop(args.copy),
    )
    evidence_panel(
        canvas,
        draw,
        (1228, 778, 2340, 1636),
        "Instruction controls",
        "correct prompts do not produce the requested answers",
        evidence_crop(args.instruction),
    )
    draw.text(
        (60, 1654),
        "Report 3d14e15d...17ba | checkpoint ca30872f...b0b1 | sealed unopened | external PIXAR initialization credited",
        font=font(13),
        fill=MUTED,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)


if __name__ == "__main__":
    main()
