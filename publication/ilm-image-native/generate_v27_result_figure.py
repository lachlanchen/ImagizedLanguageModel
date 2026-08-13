#!/usr/bin/env python3
"""Compose the measured V27 development result from fixed evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from generate_v20_result_figure import font


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = (
    ROOT
    / "artifacts/joint_visual_compatibility_v27_evidence"
    / "development_audit.json"
)
DEFAULT_OUT = (
    Path(__file__).resolve().parent
    / "figures/joint_visual_compatibility_v27_result.png"
)
EXPECTED_ARCHITECTURE = "joint-visual-compatibility-v27-development-audit"
EXPECTED_PROTOCOL_SHA256 = (
    "0c386dd2bb5198dd358613040e22297b16fbc5950f6b0bacda676f63cb223310"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "49dc07f2f9bf369cd19c563be48145d93bd723bd4450bb09e76ce02c1e77e539"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the measured V27 figure.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_close(value: object, expected: float, name: str) -> None:
    if not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Unexpected V27 {name}: {value!r}")


def validate_evidence(report: dict[str, Any]) -> None:
    if report.get("architecture") != EXPECTED_ARCHITECTURE:
        raise ValueError("Unexpected V27 report architecture")
    if int(report.get("checkpoint_step", -1)) != 8_000:
        raise ValueError("V27 report is not the fixed 8,000-step endpoint")
    if report.get("checkpoint_smoke_only"):
        raise ValueError("V27 report is smoke-only")
    if int(report.get("natural_windows", -1)) != 2_048:
        raise ValueError("V27 report does not use 2,048 natural windows")
    if int(report.get("suffix4_pairs", -1)) != 512:
        raise ValueError("V27 report does not use 512 suffix-4 pairs")
    if report.get("frozen_images_instantiated"):
        raise ValueError("V27 report opened frozen images")
    if report.get("mechanism_selected") or report.get("language_selected"):
        raise ValueError("V27 report unexpectedly selected a model")
    if report.get("frozen_evaluation_authorized"):
        raise ValueError("V27 report unexpectedly authorized frozen evaluation")
    if report.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("V27 protocol SHA-256 changed")
    if report.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("V27 checkpoint SHA-256 changed")

    natural = report.get("natural")
    suffix4 = report.get("suffix4")
    if not isinstance(natural, dict) or not isinstance(suffix4, dict):
        raise ValueError("V27 report is missing fixed audit sections")
    require_close(natural.get("student_boundary_clean"), 1.0, "student boundary")
    require_close(suffix4.get("suffix_pixel_equality"), 1.0, "suffix equality")
    require_close(
        suffix4.get("candidate_permutation_max_score_error"),
        0.0,
        "candidate permutation",
    )
    require_close(suffix4.get("last_arm_accuracy"), 0.5, "last-only control")
    require_close(suffix4.get("suffix4_arm_accuracy"), 0.5, "suffix-4 control")


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: str = "#536971",
    width: int = 4,
) -> None:
    draw.line((*start, *end), fill=color, width=width)
    x, y = end
    draw.polygon(((x, y), (x - 14, y - 9), (x - 14, y + 9)), fill=color)


def architecture_box(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    title: str,
    lines: tuple[str, ...],
    *,
    accent: str,
) -> None:
    left, top, right, bottom = bounds
    draw.rounded_rectangle(
        bounds,
        radius=7,
        fill="white",
        outline="#aebcc1",
        width=2,
    )
    draw.rectangle((left, top, left + 9, bottom), fill=accent)
    draw.text((left + 22, top + 17), title, font=font(18, bold=True), fill="#19343d")
    for index, line in enumerate(lines):
        draw.text(
            (left + 22, top + 54 + index * 25),
            line,
            font=font(14),
            fill="#53666d",
        )


def metric_bar(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    label: str,
    value: float,
    maximum: float,
    color: str,
) -> None:
    draw.text((x, y), label, font=font(16), fill="#2b424a")
    bar_left = x + 218
    bar_width = width - 358
    draw.rectangle((bar_left, y + 2, bar_left + bar_width, y + 24), fill="#e0e7e9")
    filled = int(bar_width * value / maximum)
    if value > 0:
        filled = max(2, filled)
    draw.rectangle((bar_left, y + 2, bar_left + filled, y + 24), fill=color)
    draw.text(
        (bar_left + bar_width + 12, y - 1),
        f"{100.0 * value:7.3f}%",
        font=font(15, bold=True),
        fill=color,
    )


def stat_row(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    label: str,
    value: str,
    value_x: int,
    color: str = "#263f48",
) -> None:
    draw.text((x, y), label, font=font(16), fill="#4b6068")
    draw.text((value_x, y - 2), value, font=font(18, bold=True), fill=color)


def verdict_tile(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    label: str,
    value: str,
    passed: bool | None,
) -> None:
    color = "#36576b" if passed is None else ("#19725d" if passed else "#a94747")
    draw.rounded_rectangle(
        (x, y, x + width, y + 88),
        radius=7,
        fill="white",
        outline="#b5c2c7",
        width=2,
    )
    draw.text((x + 14, y + 12), label, font=font(13, bold=True), fill="#40565e")
    draw.text((x + 14, y + 42), value, font=font(21, bold=True), fill=color)


def main() -> None:
    args = parse_args()
    report = read_json(args.report)
    validate_evidence(report)
    natural = report["natural"]
    suffix4 = report["suffix4"]

    full_gain_suffix = (
        natural["full_target_log_probability"]
        - natural["suffix4_target_log_probability"]
    )
    full_gain_shuffled = (
        natural["full_target_log_probability"]
        - natural["shuffled_target_log_probability"]
    )

    canvas = Image.new("RGB", (2400, 1700), "#f2f5f6")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (62, 34),
        "V27: candidate images remain visible, but visual compatibility is not language",
        font=font(38, bold=True),
        fill="#17323c",
    )
    draw.text(
        (64, 94),
        "18.60M total parameters | 8,000 fixed BF16 updates | one RTX 4090 | 2.268 GiB peak | frozen split sealed",
        font=font(20),
        fill="#586b73",
    )

    y = 148
    architecture_box(
        draw,
        (62, y, 405, y + 190),
        "64 image cells",
        ("ordinary Chinese", "1 x 32 x 32 each", "no IDs, OCR, or text"),
        accent="#277f8f",
    )
    architecture_box(
        draw,
        (500, y, 870, y + 190),
        "Online visual context",
        ("V16-initialized retina", "8 causal rotary blocks", "normalized query q(X)"),
        accent="#6f6597",
    )
    architecture_box(
        draw,
        (965, y, 1327, y + 190),
        "Arbitrary image Y",
        ("EMA retina + projector", "normalized key k(Y)", "no persistent bank"),
        accent="#a8753e",
    )
    architecture_box(
        draw,
        (1422, y, 1772, y + 190),
        "Compatibility",
        ("scaled q(X) dot k(Y)", "candidate permutation", "deterministic score"),
        accent="#46755d",
    )
    architecture_box(
        draw,
        (1867, y, 2338, y + 190),
        "Evaluator only",
        ("1,024 candidate images", "unigram + bigram controls", "never enters checkpoint"),
        accent="#496f8e",
    )
    arrow(draw, (405, 243), (500, 243))
    arrow(draw, (870, 243), (965, 243))
    arrow(draw, (1327, 243), (1422, 243))
    arrow(draw, (1772, 243), (1867, 243))
    draw.text(
        (63, 355),
        "Training sees image-to-image assignments; candidate order is randomized independently, so row position cannot encode identity.",
        font=font(16),
        fill="#53666d",
    )

    left = (62, 414, 1125, 1120)
    draw.rounded_rectangle(left, radius=8, fill="white", outline="#aebcc1", width=2)
    draw.text((88, 438), "Natural next-image audit", font=font(25, bold=True), fill="#1a3741")
    draw.text(
        (88, 477),
        "2,048 development windows; top-1 over the shared 1,024-image bank",
        font=font(15),
        fill="#61737a",
    )
    bars = (
        ("Full 64-cell context", natural["full_top1"], "#277f8f"),
        ("Last visible cell", natural["last_top1"], "#89989e"),
        ("Four-cell suffix", natural["suffix4_top1"], "#6f6597"),
        ("Shuffled prefix", natural["shuffled_top1"], "#89989e"),
        ("Image unigram", natural["unigram_top1"], "#a8753e"),
        ("Symbolic bigram", natural["bigram_top1"], "#a55252"),
    )
    for index, (label, value, color) in enumerate(bars):
        metric_bar(
            draw,
            x=90,
            y=526 + index * 53,
            width=990,
            label=label,
            value=float(value),
            maximum=0.14,
            color=color,
        )

    draw.line((88, 863, 1098, 863), fill="#d1dadd", width=2)
    stat_row(
        draw,
        x=90,
        y=887,
        label="Full minus four-cell suffix log p",
        value=f"{full_gain_suffix:+.3f} nat  pass",
        value_x=775,
        color="#19725d",
    )
    stat_row(
        draw,
        x=90,
        y=927,
        label="Full minus shuffled-prefix log p",
        value=f"{full_gain_shuffled:+.3f} nat  fail",
        value_x=775,
        color="#a94747",
    )
    stat_row(
        draw,
        x=90,
        y=967,
        label="Raw pair / learned 1,024-bank identity",
        value=(
            f"{100.0 * suffix4['raw_retina_cross_font_identity_accuracy']:.2f}%"
            f" / {100.0 * natural['learned_candidate_cross_font_identity_top1']:.2f}%"
        ),
        value_x=730,
        color="#a94747",
    )
    stat_row(
        draw,
        x=90,
        y=1007,
        label="Language gates passed",
        value="1 / 5",
        value_x=942,
        color="#a94747",
    )
    draw.rounded_rectangle((87, 1052, 1100, 1097), radius=6, fill="#f8e7e5")
    draw.text(
        (106, 1062),
        "LANGUAGE REJECTED: below unigram and bigram; order gain is negligible.",
        font=font(17, bold=True),
        fill="#8d3838",
    )

    right = (1165, 414, 2338, 1120)
    draw.rounded_rectangle(right, radius=8, fill="white", outline="#aebcc1", width=2)
    draw.text((1191, 438), "Pixel-identical suffix-4 audit", font=font(25, bold=True), fill="#1a3741")
    draw.text(
        (1191, 477),
        "512 cross-record pairs; same final four glyph pixels, different histories and targets",
        font=font(15),
        fill="#61737a",
    )

    draw.rounded_rectangle(
        (1200, 530, 2304, 710),
        radius=7,
        fill="#eef3f4",
        outline="#bac7cb",
        width=2,
    )
    draw.text((1225, 552), "Context A", font=font(18, bold=True), fill="#334c55")
    draw.text((1225, 588), "ordered prefix A", font=font(16), fill="#6f6597")
    draw.text((1489, 588), "+", font=font(19, bold=True), fill="#51666e")
    draw.text((1530, 588), "same 4-glyph suffix", font=font(16), fill="#a8753e")
    draw.text((1910, 588), "score image A", font=font(17, bold=True), fill="#334c55")
    draw.text((1225, 641), "Context B", font=font(18, bold=True), fill="#334c55")
    draw.text((1225, 677), "ordered prefix B", font=font(16), fill="#6f6597")
    draw.text((1489, 677), "+", font=font(19, bold=True), fill="#51666e")
    draw.text((1530, 677), "same 4-glyph suffix", font=font(16), fill="#a8753e")
    draw.text((1910, 677), "score image B", font=font(17, bold=True), fill="#334c55")

    verdict_tile(
        draw,
        x=1192,
        y=746,
        width=260,
        label="Suffix pixels equal",
        value="100% pass",
        passed=True,
    )
    verdict_tile(
        draw,
        x=1470,
        y=746,
        width=260,
        label="Permutation error",
        value="0.0 pass",
        passed=True,
    )
    verdict_tile(
        draw,
        x=1748,
        y=746,
        width=262,
        label="Raw retina identity",
        value=f"{100.0 * suffix4['raw_retina_cross_font_identity_accuracy']:.2f}% pass",
        passed=True,
    )
    verdict_tile(
        draw,
        x=2028,
        y=746,
        width=282,
        label="Full pair assignment",
        value=f"{100.0 * suffix4['full_arm_accuracy']:.2f}% fail",
        passed=False,
    )
    stat_row(
        draw,
        x=1200,
        y=868,
        label="Last / suffix-4 controls",
        value="50.00% / 50.00%  pass",
        value_x=1910,
        color="#19725d",
    )
    stat_row(
        draw,
        x=1200,
        y=912,
        label="Shuffled-prefix assignment",
        value=f"{100.0 * suffix4['shuffled_arm_accuracy']:.2f}%",
        value_x=2080,
    )
    stat_row(
        draw,
        x=1200,
        y=956,
        label="Full minus shuffled assignment",
        value=f"{100.0 * suffix4['full_minus_shuffled_arm_accuracy']:+.3f} point",
        value_x=1960,
        color="#a94747",
    )
    stat_row(
        draw,
        x=1200,
        y=1000,
        label="Mechanism gates passed",
        value="7 / 13",
        value_x=2160,
        color="#a94747",
    )
    draw.rounded_rectangle((1192, 1052, 2310, 1097), radius=6, fill="#f8e7e5")
    draw.text(
        (1212, 1062),
        "MECHANISM REJECTED: full context and shuffled context remain near chance.",
        font=font(17, bold=True),
        fill="#8d3838",
    )

    bottom = (62, 1160, 2338, 1634)
    draw.rounded_rectangle(bottom, radius=8, fill="#172f39", outline="#172f39")
    draw.text((92, 1190), "What the fixed experiment establishes", font=font(24, bold=True), fill="white")
    draw.text(
        (92, 1234),
        "The image-only scorer is permutation-equivariant, the exact suffix intervention holds, and raw candidate visibility is 99.95%.",
        font=font(18),
        fill="#d9e4e7",
    )
    draw.text(
        (92, 1270),
        "The complete audit runs in 39.1 minutes with 2.268 GiB peak allocated VRAM; frozen writing stays sealed.",
        font=font(18),
        fill="#d9e4e7",
    )
    draw.text((92, 1330), "What V27 falsifies", font=font(24, bold=True), fill="white")
    draw.text(
        (92, 1374),
        "Jointly adapting one global context query and one candidate-image key does not learn useful ordered next-writing compatibility.",
        font=font(18),
        fill="#d9e4e7",
    )
    draw.text(
        (92, 1410),
        "The separate learned 1,024-bank identity gate is 94.87%; full context is only 0.15 point above a prefix shuffle.",
        font=font(18),
        fill="#d9e4e7",
    )
    draw.text((92, 1470), "Next controlled question", font=font(24, bold=True), fill="white")
    draw.text(
        (92, 1514),
        "Preserve the raw retina exactly and train dense, explicitly ordered future-field prediction over the full glyph-image stream.",
        font=font(18),
        fill="#d9e4e7",
    )
    draw.text(
        (92, 1550),
        "Require a causal advantage over suffix and shuffled history before adding a writer, longer lattice, depth, or motion.",
        font=font(18),
        fill="#d9e4e7",
    )
    draw.text((1782, 1590), "DEVELOPMENT FALSIFICATION", font=font(15, bold=True), fill="#f0b7ad")
    draw.text((2110, 1590), "FROZEN SEALED", font=font(15, bold=True), fill="#a9d7c9")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print(args.out)


if __name__ == "__main__":
    main()
