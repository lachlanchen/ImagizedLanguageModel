#!/usr/bin/env python3
"""Compose the measured V19 development result figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT = ROOT / "artifacts/spatial_motor_plan_v19_step1600_development_audit"
DEFAULT_EVALUATION = DEFAULT_AUDIT / "evaluation.json"
DEFAULT_SAMPLES = DEFAULT_AUDIT / "development_samples_continuous_page_01.png"
DEFAULT_OUT = Path(__file__).resolve().parent / "figures/spatial_motor_plan_v19_result.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the measured V19 result figure.")
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


def gate_row(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    label: str,
    value: str,
    target: str,
    passed: bool,
) -> None:
    color = "#14785d" if passed else "#b43d3d"
    draw.line((x, y + 42, x + width, y + 42), fill="#dde4e7", width=1)
    draw.text((x, y + 7), label, font=font(17, bold=True), fill="#22353e")
    draw.text((x + 310, y + 7), value, font=font(18, bold=True), fill=color)
    draw.text((x + 435, y + 8), target, font=font(15), fill="#68767d")
    draw.text(
        (x + width - 58, y + 6),
        "PASS" if passed else "FAIL",
        font=font(16, bold=True),
        fill=color,
    )


def main() -> None:
    args = parse_args()
    if not args.evaluation.is_file() or not args.samples.is_file():
        raise FileNotFoundError("V19 development audit artifacts are required")
    receipt = json.loads(args.evaluation.read_text(encoding="utf-8"))
    if receipt.get("partition") != "development":
        raise ValueError("V19 figure must use development evidence")
    if receipt.get("frozen_images_instantiated") is not False:
        raise ValueError("V19 figure requires a sealed frozen split")
    if receipt.get("automatic_development_gate_passed") is not False:
        raise ValueError("This figure records the rejected V19 checkpoint")

    metrics = receipt["metrics"]
    gates = receipt["automatic_gate_report"]
    dense_spatial_gain = (
        metrics["correct_pixel_f1_dense"]
        - metrics["spatial_shuffled_pixel_f1_dense"]
    )
    dense_zero_gain = (
        metrics["correct_pixel_f1_dense"] - metrics["zero_field_pixel_f1_dense"]
    )

    canvas = Image.new("RGB", (2000, 1320), "#f7fafb")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (62, 36),
        "V19: spatial retinal residual fails its causal topology test",
        font=font(42, bold=True),
        fill="#17303b",
    )
    draw.text(
        (64, 96),
        "One RTX 4090 | 764,545 trainable parameters | fixed protocol | frozen split sealed",
        font=font(22),
        fill="#596970",
    )

    draw.rectangle((62, 148, 1938, 232), fill="#edf1f3", outline="#aab6bc", width=2)
    draw.text(
        (88, 173),
        "global visual plan + gated 4x4 retinal-field residual -> ink | correct / shuffled / zero-field interventions",
        font=font(23, bold=True),
        fill="#27404b",
    )

    samples = Image.open(args.samples).convert("RGB")
    sample_width = 1120
    sample_height = round(samples.height * sample_width / samples.width)
    samples = samples.resize((sample_width, sample_height), Image.Resampling.LANCZOS)
    sample_x, sample_y = 62, 272
    draw.rectangle(
        (
            sample_x - 2,
            sample_y - 2,
            sample_x + sample_width + 2,
            sample_y + sample_height + 2,
        ),
        fill="#ffffff",
        outline="#aeb9bf",
        width=2,
    )
    canvas.paste(samples, (sample_x, sample_y))

    panel_x, panel_y, panel_w, panel_h = 1220, 272, 718, 932
    draw.rectangle(
        (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h),
        fill="#ffffff",
        outline="#b9c3c8",
        width=2,
    )
    draw.text(
        (panel_x + 26, panel_y + 22),
        "Fresh 512-candidate audit",
        font=font(28, bold=True),
        fill="#17303b",
    )
    draw.text(
        (panel_x + 26, panel_y + 63),
        "value     fixed target",
        font=font(16),
        fill="#7a878d",
    )

    rows = (
        (
            "overall pixel F1",
            f"{metrics['correct_pixel_f1']:.3f}",
            "> 0.680",
            gates["overall_pixel_f1"],
        ),
        (
            "dense pixel F1",
            f"{metrics['correct_pixel_f1_dense']:.3f}",
            "> 0.580",
            gates["dense_pixel_f1"],
        ),
        (
            "dense field gain",
            f"+{dense_spatial_gain:.3f}",
            "> 0.120",
            gates["dense_spatial_shuffle_margin"],
        ),
        (
            "dense zero gain",
            f"+{dense_zero_gain:.3f}",
            "> 0.030",
            gates["dense_zero_field_margin"],
        ),
        (
            "identity top-1",
            f"{metrics['correct_identity_top1']:.3f}",
            "> 0.750",
            gates["identity_top1"],
        ),
        (
            "target cosine",
            f"{metrics['correct_target_cosine']:.3f}",
            "> 0.840",
            gates["target_cosine"],
        ),
    )
    for index, (label, value, target, passed) in enumerate(rows):
        gate_row(
            draw,
            x=panel_x + 26,
            y=panel_y + 90 + index * 53,
            width=panel_w - 52,
            label=label,
            value=value,
            target=target,
            passed=passed,
        )

    explanation_y = panel_y + 438
    draw.rectangle(
        (panel_x + 24, explanation_y, panel_x + panel_w - 24, explanation_y + 188),
        fill="#fff0ef",
        outline="#d39b96",
        width=2,
    )
    draw.text(
        (panel_x + 46, explanation_y + 18),
        "CAUSAL HYPOTHESIS REJECTED",
        font=font(23, bold=True),
        fill="#a73030",
    )
    draw.text(
        (panel_x + 46, explanation_y + 62),
        "Correct, shuffled-field, and zero-field",
        font=font(19),
        fill="#4a585f",
    )
    draw.text(
        (panel_x + 46, explanation_y + 94),
        "outputs are nearly identical. The residual",
        font=font(19),
        fill="#4a585f",
    )
    draw.text(
        (panel_x + 46, explanation_y + 126),
        "polishes the global plan; it does not carry",
        font=font(19),
        fill="#4a585f",
    )
    draw.text(
        (panel_x + 46, explanation_y + 158),
        "necessary local topology.",
        font=font(19, bold=True),
        fill="#a73030",
    )

    next_y = explanation_y + 220
    draw.text(
        (panel_x + 26, next_y),
        "Enhanced next proof: V20",
        font=font(25, bold=True),
        fill="#17303b",
    )
    next_lines = (
        "Make the retinal field the primary topology path.",
        "Restrict global state to coarse semantics/style.",
        "Project local features onto unavailable detail.",
        "Require shuffle, zero, and occlusion causality.",
        "Do not open frozen data after this failed gate.",
    )
    for index, line in enumerate(next_lines):
        y = next_y + 47 + index * 38
        draw.ellipse((panel_x + 29, y + 8, panel_x + 39, y + 18), fill="#1a7565")
        draw.text((panel_x + 51, y), line, font=font(18), fill="#3e5058")

    draw.text(
        (62, 1232),
        "DEVELOPMENT-ONLY NEGATIVE RESULT | 1,600 updates | 329.51 s | 0.899 GiB peak CUDA | human audit not authorized",
        font=font(19, bold=True),
        fill="#67757c",
    )
    draw.text(
        (62, 1272),
        "Measured consequence: adding spatial features is insufficient when a complete global writer can ignore them.",
        font=font(22, bold=True),
        fill="#17303b",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print(args.out)


if __name__ == "__main__":
    main()
