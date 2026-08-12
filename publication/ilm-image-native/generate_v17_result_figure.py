#!/usr/bin/env python3
"""Compose the measured V17 actuator figure from frozen evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION = (
    ROOT / "artifacts/visual_state_actuator_v17_frozen_eval/evaluation.json"
)
DEFAULT_SAMPLES = (
    ROOT / "artifacts/visual_state_actuator_v17_frozen_eval/frozen_samples.png"
)
DEFAULT_OUT = (
    Path(__file__).resolve().parent / "figures/visual_state_actuator_v17_result.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the V17 frozen visual-state actuator result figure."
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
    draw.text((x, y), title, font=font(23, bold=True), fill="#14212b")
    bar_x = x + 195
    bar_width = width - 195
    for index, (label, value, color) in enumerate(
        (("correct", correct, "#16835b"), ("shuffled", shuffled, "#b4473f"))
    ):
        yy = y + 2 + index * 40
        draw.text((bar_x, yy), label, font=font(18), fill="#52606d")
        start = bar_x + 88
        draw.rounded_rectangle(
            (start, yy + 4, start + bar_width - 170, yy + 22),
            radius=7,
            fill="#dfe5ea",
        )
        fraction = max(0.0, min(1.0, value / scale))
        fill_width = max(4, round((bar_width - 170) * fraction))
        draw.rounded_rectangle(
            (start, yy + 4, start + fill_width, yy + 22),
            radius=7,
            fill=color,
        )
        draw.text(
            (x + width - 72, yy),
            f"{value:.3f}",
            font=font(19, bold=True),
            fill=color,
        )


def main() -> None:
    args = parse_args()
    if not args.evaluation.is_file() or not args.samples.is_file():
        raise FileNotFoundError("V17 frozen evaluation artifacts are required")
    receipt = json.loads(args.evaluation.read_text(encoding="utf-8"))
    metrics = receipt["metrics"]

    canvas = Image.new("RGB", (2000, 1180), "#f8fafb")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (72, 42),
        "V17: continuous visual state controls pixels, not yet readable strokes",
        font=font(43, bold=True),
        fill="#102a43",
    )
    draw.text(
        (74, 101),
        "One RTX 4090 | 5.73M trainable actuator parameters | no token IDs, OCR, Unicode IDs, labels, or glyph lookup",
        font=font(23),
        fill="#52606d",
    )

    draw.rounded_rectangle(
        (72, 158, 1928, 240), radius=8, fill="#e8f4f8", outline="#8ec9d6", width=2
    )
    draw.text(
        (98, 178),
        "different-font image -> frozen retina -> continuous intended state + style image -> pixel flow -> generated ink -> reread",
        font=font(24, bold=True),
        fill="#12344d",
    )

    samples = Image.open(args.samples).convert("RGB")
    sample_width = 1210
    sample_height = round(samples.height * sample_width / samples.width)
    samples = samples.resize((sample_width, sample_height), Image.Resampling.LANCZOS)
    sample_x, sample_y = 72, 285
    draw.rectangle(
        (sample_x - 2, sample_y - 2, sample_x + sample_width + 2, sample_y + sample_height + 2),
        outline="#aeb7c0",
        width=2,
    )
    canvas.paste(samples, (sample_x, sample_y))
    draw.text(
        (sample_x, sample_y + sample_height + 12),
        "Untouched frozen examples. Correct and shuffled rows share style and initial noise.",
        font=font(19),
        fill="#69737d",
    )

    panel_x, panel_y, panel_w = 1340, 286, 588
    draw.rounded_rectangle(
        (panel_x, panel_y, panel_x + panel_w, panel_y + 540),
        radius=8,
        fill="#ffffff",
        outline="#c8d1d9",
        width=2,
    )
    draw.text(
        (panel_x + 28, panel_y + 25),
        "Frozen causal intervention",
        font=font(29, bold=True),
        fill="#14212b",
    )
    metric_bar(
        draw,
        x=panel_x + 28,
        y=panel_y + 88,
        width=panel_w - 56,
        title="Identity top-1",
        correct=metrics["correct_identity_top1"],
        shuffled=metrics["shuffled_identity_top1"],
        scale=0.65,
    )
    metric_bar(
        draw,
        x=panel_x + 28,
        y=panel_y + 190,
        width=panel_w - 56,
        title="Target cosine",
        correct=metrics["correct_target_cosine"],
        shuffled=metrics["shuffled_target_cosine"],
        scale=0.80,
    )
    metric_bar(
        draw,
        x=panel_x + 28,
        y=panel_y + 292,
        width=panel_w - 56,
        title="Pixel F1",
        correct=metrics["correct_pixel_f1"],
        shuffled=metrics["shuffled_pixel_f1"],
        scale=0.55,
    )
    draw.line(
        (panel_x + 28, panel_y + 398, panel_x + panel_w - 28, panel_y + 398),
        fill="#d7dee4",
        width=2,
    )
    draw.text(
        (panel_x + 28, panel_y + 420),
        "Automatic gate",
        font=font(21, bold=True),
        fill="#52606d",
    )
    draw.text(
        (panel_x + 230, panel_y + 416),
        "FAIL",
        font=font(28, bold=True),
        fill="#b42318",
    )
    draw.text(
        (panel_x + 28, panel_y + 465),
        "Pixel F1 0.438 < 0.500",
        font=font(22),
        fill="#b42318",
    )
    draw.text(
        (panel_x + 28, panel_y + 500),
        "Human readability: rejected",
        font=font(22),
        fill="#b42318",
    )

    verdict_y = 1020
    draw.rectangle((72, verdict_y, 1928, verdict_y + 102), fill="#202830")
    draw.text((99, verdict_y + 16), "Measured verdict", font=font(25, bold=True), fill="#ffffff")
    draw.text(
        (350, verdict_y + 15),
        "visual identity is causally controlled; exact stroke topology and readability are not solved",
        font=font(24),
        fill="#f1f4f6",
    )
    draw.text(
        (99, verdict_y + 61),
        "Next: decode the continuous state into a supervised spatial motor plan, then use stochastic flow only for refinement",
        font=font(20),
        fill="#b8c5cf",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print(args.out)


if __name__ == "__main__":
    main()
