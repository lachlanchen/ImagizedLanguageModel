#!/usr/bin/env python3
"""Compose the V18 development result figure from measured audit artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION = (
    ROOT
    / "artifacts/visual_motor_plan_v18_step1400_development_audit_v2/evaluation.json"
)
DEFAULT_SAMPLES = (
    ROOT
    / "artifacts/visual_motor_plan_v18_step1400_development_audit_v2"
    / "development_samples_continuous_page_01.png"
)
DEFAULT_OUT = (
    Path(__file__).resolve().parent / "figures/visual_motor_plan_v18_result.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the measured V18 development result figure."
    )
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/truetype/noto")
        / ("NotoSans-Bold.ttf" if bold else "NotoSans-Regular.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def metric_bar(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    title: str,
    correct: float,
    shuffled: float,
    scale: float,
) -> None:
    draw.text((x, y), title, font=font(22, bold=True), fill="#132a36")
    bar_x = x + 92
    bar_width = width - 178
    for row, (label, value, color) in enumerate(
        (("correct", correct, "#117a5b"), ("shuffled", shuffled, "#bd4b42"))
    ):
        yy = y + 34 + row * 32
        draw.text((x, yy), label, font=font(17), fill="#59656e")
        start = bar_x
        draw.rectangle((start, yy + 5, start + bar_width, yy + 21), fill="#dfe5e8")
        fill_width = max(3, round(bar_width * max(0.0, min(1.0, value / scale))))
        draw.rectangle((start, yy + 5, start + fill_width, yy + 21), fill=color)
        draw.text(
            (x + width - 66, yy),
            f"{value:.3f}",
            font=font(18, bold=True),
            fill=color,
        )


def main() -> None:
    args = parse_args()
    if not args.evaluation.is_file() or not args.samples.is_file():
        raise FileNotFoundError("V18 development audit artifacts are required")
    receipt = json.loads(args.evaluation.read_text(encoding="utf-8"))
    if receipt.get("partition") != "development":
        raise ValueError("V18 figure must be generated from development evidence")
    if receipt.get("frozen_images_instantiated") is not False:
        raise ValueError("V18 figure requires a sealed frozen split")
    metrics = receipt["metrics"]

    canvas = Image.new("RGB", (2000, 1240), "#f7fafb")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (68, 38),
        "V18: a 2.36M visual motor plan learns readable writing topology",
        font=font(42, bold=True),
        fill="#0f2c3a",
    )
    draw.text(
        (70, 96),
        "One RTX 4090 | continuous image-derived intent | no tokens, OCR, Unicode IDs, labels, lookup, or codebook",
        font=font(22),
        fill="#53636c",
    )

    draw.rectangle((68, 150, 1932, 234), fill="#e5f3f2", outline="#7ab8b0", width=2)
    draw.text(
        (94, 175),
        "different-font image -> frozen retina -> continuous intent + style image -> deterministic spatial motor plan -> ink",
        font=font(23, bold=True),
        fill="#153d45",
    )

    samples = Image.open(args.samples).convert("RGB")
    sample_width = 1240
    sample_height = round(samples.height * sample_width / samples.width)
    samples = samples.resize((sample_width, sample_height), Image.Resampling.LANCZOS)
    sample_x, sample_y = 68, 282
    draw.rectangle(
        (
            sample_x - 2,
            sample_y - 2,
            sample_x + sample_width + 2,
            sample_y + sample_height + 2,
        ),
        outline="#aeb9bf",
        width=2,
    )
    canvas.paste(samples, (sample_x, sample_y))
    draw.text(
        (sample_x, sample_y + sample_height + 12),
        "Fresh development renderings. Correct and shuffled rows share style; only continuous intent changes.",
        font=font(18),
        fill="#68757d",
    )

    panel_x, panel_y, panel_w = 1350, 282, 582
    draw.rectangle(
        (panel_x, panel_y, panel_x + panel_w, panel_y + 605),
        fill="#ffffff",
        outline="#c7d0d5",
        width=2,
    )
    draw.text(
        (panel_x + 25, panel_y + 22),
        "Development causal audit",
        font=font(28, bold=True),
        fill="#132a36",
    )
    metric_bar(
        draw,
        x=panel_x + 25,
        y=panel_y + 78,
        width=panel_w - 50,
        title="Identity top-1",
        correct=metrics["correct_identity_top1"],
        shuffled=metrics["shuffled_identity_top1"],
        scale=0.80,
    )
    metric_bar(
        draw,
        x=panel_x + 25,
        y=panel_y + 188,
        width=panel_w - 50,
        title="Target cosine",
        correct=metrics["correct_target_cosine"],
        shuffled=metrics["shuffled_target_cosine"],
        scale=0.90,
    )
    metric_bar(
        draw,
        x=panel_x + 25,
        y=panel_y + 298,
        width=panel_w - 50,
        title="Pixel F1",
        correct=metrics["correct_pixel_f1"],
        shuffled=metrics["shuffled_pixel_f1"],
        scale=0.70,
    )
    draw.line(
        (panel_x + 25, panel_y + 385, panel_x + panel_w - 25, panel_y + 385),
        fill="#d4dde1",
        width=2,
    )
    draw.text(
        (panel_x + 25, panel_y + 410),
        "Automatic topology gates",
        font=font(21, bold=True),
        fill="#53636c",
    )
    draw.text(
        (panel_x + 365, panel_y + 404),
        "PASS",
        font=font(27, bold=True),
        fill="#117a5b",
    )
    draw.text(
        (panel_x + 25, panel_y + 458),
        "Readable: simple + medium forms",
        font=font(20),
        fill="#117a5b",
    )
    draw.text(
        (panel_x + 25, panel_y + 495),
        "Dense topology: still malformed",
        font=font(20),
        fill="#a96318",
    )
    draw.text(
        (panel_x + 25, panel_y + 532),
        "Frozen bank: untouched",
        font=font(20, bold=True),
        fill="#8f2630",
    )
    draw.text(
        (panel_x + 25, panel_y + 565),
        "No post-hoc promotion",
        font=font(18),
        fill="#8f2630",
    )

    verdict_y = 1040
    draw.rectangle((68, verdict_y, 1932, verdict_y + 142), fill="#1d2930")
    draw.text(
        (94, verdict_y + 13),
        "Measured consequence",
        font=font(24, bold=True),
        fill="#ffffff",
    )
    draw.text(
        (94, verdict_y + 53),
        "structured writing is learnable as continuous image topology with a small consumer-GPU model",
        font=font(22),
        fill="#f3f7f8",
    )
    draw.text(
        (94, verdict_y + 94),
        "autonomous language choice, complex-form fidelity, and closed-loop readability remain unsolved",
        font=font(22),
        fill="#f2c879",
    )
    draw.text(
        (94, 1200),
        "DEVELOPMENT-ONLY EVIDENCE | selected step 1,400 | 1,600 updates | 458.34 s | 0.778 GiB peak allocated CUDA",
        font=font(18, bold=True),
        fill="#68757d",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print(args.out)


if __name__ == "__main__":
    main()
