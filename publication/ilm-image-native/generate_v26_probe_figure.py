#!/usr/bin/env python3
"""Compose the measured frozen-V26 compatibility diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from generate_v20_result_figure import font


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = ROOT / "artifacts/v26_visual_compatibility_probe/diagnostic.json"
DEFAULT_OUT = (
    Path(__file__).resolve().parent
    / "figures/v26_frozen_visual_compatibility_probe.png"
)
EXPECTED_ARCHITECTURE = "v26-frozen-visual-compatibility-probe"
EXPECTED_CHECKPOINT_SHA256 = (
    "065f84e1a7dc44ca8c304018c4eb9b29bfbcaef8f24b9e99ca0c84a3d6db6e1d"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the V26 probe figure.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def require_close(value: object, expected: float, name: str) -> None:
    if not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"unexpected {name}: {value!r}")


def validate(report: dict[str, Any]) -> None:
    if report.get("architecture") != EXPECTED_ARCHITECTURE:
        raise ValueError("unexpected probe architecture")
    if report.get("is_preregistered_evidence"):
        raise ValueError("post-hoc probe is incorrectly labeled as evidence")
    if report.get("authorizes_frozen_evaluation"):
        raise ValueError("probe cannot authorize frozen evaluation")
    if report.get("checkpoint", {}).get("sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("probe used a different V26 checkpoint")
    if int(report.get("train_pairs", {}).get("count", -1)) != 16_384:
        raise ValueError("probe did not use all fixed train pairs")
    if int(report.get("development_pairs", {}).get("count", -1)) != 512:
        raise ValueError("probe did not use the fixed development pairs")
    if report.get("frozen_images_instantiated"):
        raise ValueError("probe opened frozen images")
    final = report.get("final_development", {})
    require_close(final.get("suffix_pixel_equality"), 1.0, "suffix equality")
    require_close(
        final.get("states", {}).get("appearance_state", {}).get("arm_accuracy"),
        0.5,
        "appearance control",
    )


def box(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    title: str,
    lines: tuple[str, ...],
    *,
    accent: str,
) -> None:
    left, top, right, bottom = bounds
    draw.rounded_rectangle(
        bounds, radius=7, fill="white", outline="#afbec3", width=2
    )
    draw.rectangle((left, top, left + 9, bottom), fill=accent)
    draw.text(
        (left + 22, top + 15),
        title,
        font=font(19, bold=True),
        fill="#19343d",
    )
    for index, line in enumerate(lines):
        draw.text(
            (left + 22, top + 52 + index * 27),
            line,
            font=font(15),
            fill="#51666e",
        )


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    draw.line((*start, *end), fill="#5d7077", width=4)
    x, y = end
    draw.polygon(
        ((x, y), (x - 14, y - 9), (x - 14, y + 9)), fill="#5d7077"
    )


def metric(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    label: str,
    value: float,
    color: str,
) -> None:
    draw.text((x, y), label, font=font(18), fill="#334b54")
    left = x + 360
    width = 880
    draw.rectangle((left, y + 2, left + width, y + 28), fill="#dce5e7")
    draw.rectangle(
        (left, y + 2, left + int(width * value), y + 28), fill=color
    )
    draw.line(
        (left + width // 2, y - 5, left + width // 2, y + 35),
        fill="#263f48",
        width=2,
    )
    draw.text(
        (left + width + 18, y - 2),
        f"{100 * value:.3f}%",
        font=font(18, bold=True),
        fill=color,
    )


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    validate(report)
    final = report["final_development"]
    states = final["states"]
    identity = final["retina_identity_control"]

    canvas = Image.new("RGB", (2100, 1450), "#f1f5f5")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (58, 34),
        "Frozen V26 state does not expose next-glyph binding",
        font=font(38, bold=True),
        fill="#17323c",
    )
    draw.text(
        (60, 91),
        "Post-hoc localization diagnostic | development only | frozen split sealed | not model evidence",
        font=font(19),
        fill="#5b6e75",
    )

    y = 148
    box(
        draw,
        (60, y, 405, y + 185),
        "Matched contexts",
        ("same 4 glyph pixels", "different earlier history", "different next images"),
        accent="#277f8f",
    )
    box(
        draw,
        (535, y, 900, y + 185),
        "Frozen V26",
        ("8,000-step checkpoint", "appearance / history / fused", "all 19.14M parameters fixed"),
        accent="#725f91",
    )
    box(
        draw,
        (1030, y, 1405, y + 185),
        "Visual scorer",
        ("1.11M trainable parameters", "2 x 2 symmetric assignment", "candidate images, no IDs"),
        accent="#aa7336",
    )
    box(
        draw,
        (1535, y, 2040, y + 185),
        "Held-out decision",
        ("512 disjoint-record pairs", "2 unseen-font assignments", "2,048 scored arms"),
        accent="#3e765d",
    )
    arrow(draw, (405, y + 92), (535, y + 92))
    arrow(draw, (900, y + 92), (1030, y + 92))
    arrow(draw, (1405, y + 92), (1535, y + 92))

    draw.rounded_rectangle(
        (60, 380, 2040, 900),
        radius=7,
        fill="white",
        outline="#afbec3",
        width=2,
    )
    draw.text(
        (88, 408),
        "Cross-font paired accuracy",
        font=font(25, bold=True),
        fill="#213c45",
    )
    draw.text(
        (88, 449),
        "Vertical marker: chance. The positive control uses the same candidate images and frozen retina.",
        font=font(16),
        fill="#61737a",
    )
    metric(
        draw,
        x=90,
        y=510,
        label="Appearance-only control",
        value=states["appearance_state"]["arm_accuracy"],
        color="#78868b",
    )
    metric(
        draw,
        x=90,
        y=595,
        label="History residual",
        value=states["history_residual"]["arm_accuracy"],
        color="#9e4e4e",
    )
    metric(
        draw,
        x=90,
        y=680,
        label="Fused V26 state",
        value=states["fused_state"]["arm_accuracy"],
        color="#9e4e4e",
    )
    metric(
        draw,
        x=90,
        y=765,
        label="Retina identity control",
        value=identity["arm_accuracy"],
        color="#19715c",
    )

    draw.rounded_rectangle(
        (60, 950, 2040, 1375),
        radius=7,
        fill="#17323c",
        outline="#17323c",
    )
    draw.text((92, 980), "Diagnosis", font=font(25, bold=True), fill="#d9eff1")
    draw.text(
        (92, 1032),
        "History changes numerically, but a deterministic nonlinear scorer cannot decode",
        font=font(22, bold=True),
        fill="white",
    )
    draw.text(
        (92, 1069),
        "a transferable next-glyph relation from it. Candidate visibility is not the bottleneck.",
        font=font(22, bold=True),
        fill="white",
    )
    draw.text(
        (92, 1143),
        f"History margin {states['history_residual']['mean_margin']:.6f}   |   Fused margin {states['fused_state']['mean_margin']:.6f}   |   Retina margin {identity['mean_cosine_margin']:.6f}",
        font=font(18),
        fill="#b9d6da",
    )
    draw.text(
        (92, 1210),
        "V27 decision: jointly train the image context encoder and candidate-conditioned",
        font=font(21, bold=True),
        fill="#f0cc83",
    )
    draw.text(
        (92, 1247),
        "compatibility objective before attaching a stochastic visual writer.",
        font=font(21, bold=True),
        fill="#f0cc83",
    )
    draw.text(
        (92, 1312),
        f"One train-pair pass | {report['runtime_seconds']:.2f} s | {report['peak_allocated_vram_gib']:.3f} GiB peak on one RTX 4090",
        font=font(17),
        fill="#bfd3d7",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print(args.out)


if __name__ == "__main__":
    main()
