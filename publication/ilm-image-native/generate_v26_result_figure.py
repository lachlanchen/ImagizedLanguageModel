#!/usr/bin/env python3
"""Compose the measured V26 development result from fixed evidence."""

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
    / "artifacts/factorized_visual_context_v26_evidence"
    / "development_audit.json"
)
DEFAULT_OUT = (
    Path(__file__).resolve().parent
    / "figures/factorized_visual_context_v26_result.png"
)
EXPECTED_ARCHITECTURE = "factorized-visual-context-v26-development-audit"
EXPECTED_PROTOCOL_SHA256 = (
    "dcfa5b974e617be8a7995dd0f4bb123094837d74c587e48b2ce987785b899df1"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "065f84e1a7dc44ca8c304018c4eb9b29bfbcaef8f24b9e99ca0c84a3d6db6e1d"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the measured V26 figure.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_close(value: object, expected: float, name: str) -> None:
    if not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Unexpected V26 {name}: {value!r}")


def validate_evidence(report: dict[str, Any]) -> None:
    if report.get("architecture") != EXPECTED_ARCHITECTURE:
        raise ValueError("Unexpected V26 report architecture")
    if int(report.get("checkpoint_step", -1)) != 8_000:
        raise ValueError("V26 report is not the fixed 8,000-step endpoint")
    if report.get("checkpoint_smoke_only"):
        raise ValueError("V26 report is smoke-only")
    if int(report.get("natural_windows", -1)) != 2_048:
        raise ValueError("V26 report does not use 2,048 natural windows")
    if int(report.get("suffix4_pairs", -1)) != 512:
        raise ValueError("V26 report does not use 512 suffix-4 pairs")
    if report.get("frozen_images_instantiated"):
        raise ValueError("V26 report opened frozen images")
    if report.get("mechanism_selected") or report.get("language_selected"):
        raise ValueError("V26 report unexpectedly selected a model")
    if report.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("V26 protocol SHA-256 changed")
    if report.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("V26 checkpoint SHA-256 changed")

    natural = report.get("natural")
    suffix4 = report.get("suffix4")
    if not isinstance(natural, dict) or not isinstance(suffix4, dict):
        raise ValueError("V26 report is missing fixed audit sections")
    require_close(natural.get("retina_bank_oracle_top1"), 1.0, "retina oracle")
    require_close(natural.get("student_boundary_clean"), 1.0, "student boundary")
    require_close(suffix4.get("suffix_pixel_equality"), 1.0, "suffix equality")
    require_close(
        suffix4.get("mean_appearance_state_difference"),
        0.0,
        "appearance-state equality",
    )


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
    bar_left = x + 220
    bar_width = width - 365
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
    color: str = "#263f48",
) -> None:
    draw.text((x, y), label, font=font(16), fill="#4b6068")
    draw.text((x + 725, y - 2), value, font=font(18, bold=True), fill=color)


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
    if passed is None:
        color = "#36576b"
    else:
        color = "#19725d" if passed else "#a94747"
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

    full_logp_gain_last = (
        natural["full_target_log_probability"]
        - natural["last_target_log_probability"]
    )
    full_logp_gain_suffix4 = (
        natural["full_target_log_probability"]
        - natural["suffix_4_target_log_probability"]
    )
    ordered_logp_gain = (
        natural["full_target_log_probability"]
        - natural["shuffled_prefix_target_log_probability"]
    )

    canvas = Image.new("RGB", (2400, 1700), "#f2f5f6")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (62, 34),
        "V26: visual history changes state, but does not bind the next glyph",
        font=font(40, bold=True),
        fill="#17323c",
    )
    draw.text(
        (64, 94),
        "19.14M total parameters | 8,000 fixed BF16 updates | one RTX 4090 | 0.888 GiB peak | frozen split sealed",
        font=font(20),
        fill="#586b73",
    )

    architecture_y = 148
    architecture_box(
        draw,
        (62, architecture_y, 382, architecture_y + 190),
        "64 image cells",
        ("ordinary Chinese", "1 x 32 x 32 each", "no IDs or OCR"),
        accent="#277f8f",
    )
    architecture_box(
        draw,
        (500, architecture_y, 835, architecture_y + 88),
        "Appearance",
        ("last glyph -> exact visible state",),
        accent="#a8753e",
    )
    architecture_box(
        draw,
        (500, architecture_y + 102, 835, architecture_y + 190),
        "History residual",
        ("earlier 63 -> 8 causal layers",),
        accent="#6f6597",
    )
    architecture_box(
        draw,
        (953, architecture_y, 1295, architecture_y + 190),
        "Fused context",
        ("appearance + residual", "last-only intervention", "state-swap access"),
        accent="#46755d",
    )
    architecture_box(
        draw,
        (1413, architecture_y, 1774, architecture_y + 190),
        "Visual particles",
        ("8 samples per horizon", "future 1, 2, 4, 8", "192-D, continuous"),
        accent="#a55252",
    )
    architecture_box(
        draw,
        (1892, architecture_y, 2338, architecture_y + 190),
        "Evaluator only",
        ("1,024 image forms", "energy-based ranking", "not a student input"),
        accent="#496f8e",
    )
    draw.line((382, 243, 448, 243), fill="#536971", width=4)
    draw.line((448, 192, 448, 294), fill="#536971", width=4)
    arrow(draw, (448, 192), (500, 192))
    arrow(draw, (448, 294), (500, 294))
    draw.line((835, 192, 893, 192), fill="#536971", width=4)
    draw.line((835, 294, 893, 294), fill="#536971", width=4)
    draw.line((893, 192, 893, 294), fill="#536971", width=4)
    arrow(draw, (893, 243), (953, 243))
    arrow(draw, (1295, 243), (1413, 243))
    arrow(draw, (1774, 243), (1892, 243))
    draw.text(
        (63, 355),
        "The appearance and history routes are separable. Suffix pairs keep the final four glyph images exactly equal while changing earlier text.",
        font=font(16),
        fill="#53666d",
    )

    left = (62, 414, 1125, 1120)
    draw.rounded_rectangle(left, radius=8, fill="white", outline="#aebcc1", width=2)
    draw.text((88, 438), "Natural next-glyph audit", font=font(25, bold=True), fill="#1a3741")
    draw.text((88, 477), "2,048 fixed development windows; top-1 over one shared 1,024-image bank", font=font(15), fill="#61737a")
    bars = (
        ("Full 64-cell history", natural["full_top1"], "#277f8f"),
        ("Last visible cell", natural["last_top1"], "#89989e"),
        ("Four-cell suffix", natural["suffix_4_top1"], "#6f6597"),
        ("Shuffled prefix", natural["shuffled_prefix_top1"], "#89989e"),
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
        label="Full minus last target log probability",
        value=f"{full_logp_gain_last:+.3f} nat  pass",
        color="#19725d",
    )
    stat_row(
        draw,
        x=90,
        y=927,
        label="Full minus four-cell suffix",
        value=f"{full_logp_gain_suffix4:+.3f} nat  fail",
        color="#a94747",
    )
    stat_row(
        draw,
        x=90,
        y=967,
        label="Full minus shuffled prefix",
        value=f"{ordered_logp_gain:+.3f} nat  fail",
        color="#a94747",
    )
    stat_row(
        draw,
        x=90,
        y=1007,
        label="Retina-bank oracle",
        value=f"{100.0 * natural['retina_bank_oracle_top1']:.1f}%  pass",
        color="#19725d",
    )
    draw.rounded_rectangle((87, 1052, 1100, 1097), radius=6, fill="#f8e7e5")
    draw.text(
        (106, 1062),
        "LANGUAGE REJECTED: the visual unigram and symbolic bigram are much stronger.",
        font=font(17, bold=True),
        fill="#8d3838",
    )

    right = (1165, 414, 2338, 1120)
    draw.rounded_rectangle(right, radius=8, fill="white", outline="#aebcc1", width=2)
    draw.text((1191, 438), "Causal suffix-4 audit", font=font(25, bold=True), fill="#1a3741")
    draw.text((1191, 477), "512 cross-record pairs: same final four glyph pixels, different earlier history and target", font=font(15), fill="#61737a")

    draw.rounded_rectangle((1200, 530, 2304, 710), radius=7, fill="#eef3f4", outline="#bac7cb", width=2)
    draw.text((1225, 552), "Context A", font=font(18, bold=True), fill="#334c55")
    draw.text((1225, 588), "earlier visual history A", font=font(16), fill="#6f6597")
    draw.text((1489, 588), "+", font=font(19, bold=True), fill="#51666e")
    draw.text((1530, 588), "same 4-glyph suffix", font=font(16), fill="#a8753e")
    draw.text((1910, 588), "-> target A", font=font(17, bold=True), fill="#334c55")
    draw.text((1225, 641), "Context B", font=font(18, bold=True), fill="#334c55")
    draw.text((1225, 677), "earlier visual history B", font=font(16), fill="#6f6597")
    draw.text((1489, 677), "+", font=font(19, bold=True), fill="#51666e")
    draw.text((1530, 677), "same 4-glyph suffix", font=font(16), fill="#a8753e")
    draw.text((1910, 677), "-> target B", font=font(17, bold=True), fill="#334c55")

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
        label="Appearance difference",
        value="0.000 pass",
        passed=True,
    )
    verdict_tile(
        draw,
        x=1748,
        y=746,
        width=262,
        label="History-state difference",
        value=f"{suffix4['mean_history_residual_difference']:.3f} changed",
        passed=None,
    )
    verdict_tile(
        draw,
        x=2028,
        y=746,
        width=282,
        label="Correct pair ranking",
        value=f"{100.0 * suffix4['pair_ranking_accuracy']:.1f}% fail",
        passed=False,
    )
    stat_row(
        draw,
        x=1200,
        y=868,
        label="Swapped-residual target accuracy",
        value=f"{100.0 * suffix4['swapped_residual_target_accuracy']:.1f}%  fail",
        color="#a94747",
    )
    stat_row(
        draw,
        x=1200,
        y=912,
        label="Mean correct-versus-other margin",
        value=f"{suffix4['mean_pair_margin']:.6f}",
        color="#a94747",
    )
    stat_row(
        draw,
        x=1200,
        y=956,
        label="Top-1 output switch rate",
        value=f"{100.0 * suffix4['top1_switch_rate']:.2f}%",
    )
    stat_row(
        draw,
        x=1200,
        y=1000,
        label="Conditional particle spread",
        value=f"{natural['full_particle_spread']:.3f}",
    )
    draw.rounded_rectangle((1192, 1052, 2310, 1097), radius=6, fill="#f8e7e5")
    draw.text(
        (1212, 1062),
        "MECHANISM REJECTED: changed history state yields chance target preference.",
        font=font(17, bold=True),
        fill="#8d3838",
    )

    bottom = (62, 1160, 2338, 1634)
    draw.rounded_rectangle(bottom, radius=8, fill="#172f39", outline="#172f39")
    draw.text((92, 1190), "What the fixed experiment establishes", font=font(24, bold=True), fill="white")
    draw.text((92, 1234), "The full factorized image-only system trains in 31.1 minutes with 0.888 GiB peak allocated VRAM.", font=font(18), fill="#d9e4e7")
    draw.text((92, 1270), "The retina/evaluator is intact, the exact suffix intervention holds, and earlier history changes the residual state.", font=font(18), fill="#d9e4e7")
    draw.text((92, 1330), "What V26 falsifies", font=font(24, bold=True), fill="white")
    draw.text((92, 1374), "Appearance/residual factorization plus energy, queue contrast, and pair ranking does not learn useful next-glyph binding.", font=font(18), fill="#d9e4e7")
    draw.text((92, 1410), "Internal-state sensitivity and broad visual particles are not conditional language behavior.", font=font(18), fill="#d9e4e7")
    draw.text((92, 1470), "Next controlled question", font=font(24, bold=True), fill="white")
    draw.text((92, 1514), "Make target choice conditionally identifiable before adding a writer, longer context, page geometry, depth, or motion.", font=font(18), fill="#d9e4e7")
    draw.text((92, 1550), "A next model must beat suffix, shuffled-history, unigram, and bigram controls under paired causal intervention.", font=font(18), fill="#d9e4e7")
    draw.text((1782, 1590), "DEVELOPMENT FALSIFICATION", font=font(15, bold=True), fill="#f0b7ad")
    draw.text((2110, 1590), "FROZEN SEALED", font=font(15, bold=True), fill="#a9d7c9")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print(args.out)


if __name__ == "__main__":
    main()
