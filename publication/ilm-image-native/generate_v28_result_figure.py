#!/usr/bin/env python3
"""Compose the measured V28 development result from fixed evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from generate_v20_result_figure import font


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = (
    ROOT
    / "artifacts/dense_visual_future_energy_v28_evidence"
    / "development_audit.json"
)
DEFAULT_OUT = (
    Path(__file__).resolve().parent
    / "figures/dense_visual_future_energy_v28_result.png"
)
EXPECTED_REPORT_SHA256 = (
    "2cb73707a01fccb5bef750690014a2729b6235941a4de33ccdfdb600e8f0fb3d"
)
EXPECTED_PROTOCOL_SHA256 = (
    "b8e515d27f619033f53a04d1afd3ff8d71ba0dd68484728f2f4c1b68a7780f7f"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "22503464cf5f5e8ed2d6adebbd6c794f6bc9b2836f978872027cb51712c7f64f"
)
EXPECTED_ARCHITECTURE = "dense-visual-future-energy-v28-development-audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the measured V28 figure.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if file_sha256(path) != EXPECTED_REPORT_SHA256:
        raise ValueError("V28 report SHA-256 does not match fixed evidence")
    return json.loads(path.read_text(encoding="utf-8"))


def require_close(value: object, expected: float, name: str) -> None:
    if not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Unexpected V28 {name}: {value!r}")


def validate_evidence(report: dict[str, Any]) -> None:
    if report.get("architecture") != EXPECTED_ARCHITECTURE:
        raise ValueError("Unexpected V28 report architecture")
    if report.get("checkpoint_architecture") != "dense-visual-future-energy-v28":
        raise ValueError("Unexpected V28 checkpoint architecture")
    if report.get("smoke_only"):
        raise ValueError("V28 report is smoke-only")
    if int(report.get("natural_windows", -1)) != 2_048:
        raise ValueError("V28 report does not use 2,048 natural windows")
    if int(report.get("suffix4_pairs", -1)) != 512:
        raise ValueError("V28 report does not use 512 suffix-4 pairs")
    if report.get("frozen_images_instantiated"):
        raise ValueError("V28 report opened frozen images")
    if report.get("mechanism_selected") or report.get("language_selected"):
        raise ValueError("V28 report unexpectedly selected a model")
    if report.get("frozen_evaluation_authorized"):
        raise ValueError("V28 report unexpectedly authorized frozen evaluation")
    if report.get("writer_training_authorized"):
        raise ValueError("V28 report unexpectedly authorized writer training")
    if report.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("V28 protocol SHA-256 changed")
    if report.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("V28 checkpoint SHA-256 changed")
    if int(report.get("total_parameters", -1)) != 17_859_142:
        raise ValueError("V28 total parameter count changed")
    if int(report.get("trainable_parameters", -1)) != 16_377_990:
        raise ValueError("V28 trainable parameter count changed")
    if int(report.get("training_metrics", {}).get("step", -1)) != 10_000:
        raise ValueError("V28 report is not the fixed 10,000-step endpoint")

    mechanism = report.get("mechanism_gates")
    language = report.get("language_gates")
    if not isinstance(mechanism, dict) or sum(mechanism.values()) != 10:
        raise ValueError("V28 mechanism gate receipt changed")
    if not isinstance(language, dict) or sum(language.values()) != 2:
        raise ValueError("V28 language gate receipt changed")

    natural = report.get("natural")
    suffix4 = report.get("suffix4")
    if not isinstance(natural, dict) or not isinstance(suffix4, dict):
        raise ValueError("V28 report is missing fixed audit sections")
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
    color: str = "#526970",
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
        outline="#afbdc2",
        width=2,
    )
    draw.rectangle((left, top, left + 9, bottom), fill=accent)
    draw.text((left + 23, top + 18), title, font=font(18, bold=True), fill="#17343e")
    for index, line in enumerate(lines):
        draw.text(
            (left + 23, top + 57 + index * 27),
            line,
            font=font(14),
            fill="#53676e",
        )


def percent_bar(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    label: str,
    value: float,
    maximum: float,
    color: str,
    target: float | None = None,
) -> None:
    draw.text((x, y), label, font=font(16), fill="#2b424a")
    bar_left = x + 250
    bar_width = width - 395
    draw.rectangle((bar_left, y + 2, bar_left + bar_width, y + 25), fill="#e1e8ea")
    filled = max(2, int(bar_width * max(0.0, value) / maximum))
    draw.rectangle((bar_left, y + 2, bar_left + filled, y + 25), fill=color)
    if target is not None:
        marker = bar_left + int(bar_width * target / maximum)
        draw.line((marker, y - 3, marker, y + 30), fill="#792e2e", width=3)
    draw.text(
        (bar_left + bar_width + 14, y - 1),
        f"{100.0 * value:7.3f}%",
        font=font(15, bold=True),
        fill=color,
    )


def stat_tile(
    draw: ImageDraw.ImageDraw,
    *,
    bounds: tuple[int, int, int, int],
    label: str,
    value: str,
    detail: str,
    passed: bool | None,
) -> None:
    left, top, right, bottom = bounds
    color = "#36576b" if passed is None else ("#19725d" if passed else "#a94747")
    fill = "#eef5f2" if passed else ("#f8e9e7" if passed is False else "#edf2f4")
    draw.rounded_rectangle(bounds, radius=7, fill=fill, outline="#b6c3c7", width=2)
    draw.text((left + 16, top + 13), label, font=font(14, bold=True), fill="#435860")
    draw.text((left + 16, top + 45), value, font=font(24, bold=True), fill=color)
    draw.text((left + 16, bottom - 30), detail, font=font(13), fill="#5d7077")


def main() -> None:
    args = parse_args()
    report = read_json(args.report)
    validate_evidence(report)
    natural = report["natural"]
    suffix4 = report["suffix4"]

    full_suffix_logp = (
        natural["full_target_log_probability"]
        - natural["suffix4_target_log_probability"]
    )
    full_shuffle_logp = (
        natural["full_target_log_probability"]
        - natural["shuffled_target_log_probability"]
    )
    identity_gain = (
        natural["ema_semantic_cross_font_identity_top1"]
        - natural["raw_retina_cross_font_identity_top1"]
    )

    canvas = Image.new("RGB", (2400, 1800), "#f2f5f6")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (62, 34),
        "V28: dense visual futures learn form and order, but fail language choice",
        font=font(37, bold=True),
        fill="#17323c",
    )
    draw.text(
        (64, 94),
        "17.86M total / 16.38M trainable | 10,000 BF16 updates | one RTX 4090 | 1.144 GiB peak | frozen split sealed",
        font=font(19),
        fill="#586b73",
    )

    y = 148
    architecture_box(
        draw,
        (62, y, 405, y + 210),
        "68 writing images",
        (
            "B x 68 x 1 x 32 x 32",
            "64 context + 4 future",
            "no strings, IDs, or OCR",
        ),
        accent="#277f8f",
    )
    architecture_box(
        draw,
        (492, y, 835, y + 210),
        "Frozen geometry",
        (
            "V16 image retina",
            "normalized raw state r(x)",
            "weights never updated",
        ),
        accent="#a8753e",
    )
    architecture_box(
        draw,
        (922, y, 1265, y + 210),
        "Semantic image route",
        (
            "identity-init residual",
            "online + EMA target",
            "still continuous visual state",
        ),
        accent="#6f6597",
    )
    architecture_box(
        draw,
        (1352, y, 1740, y + 210),
        "Causal future field",
        (
            "8 rotary layers, width 384",
            "horizons 1 / 2 / 4",
            "four hypotheses per horizon",
        ),
        accent="#46755d",
    )
    architecture_box(
        draw,
        (1827, y, 2338, y + 210),
        "Arbitrary image energy",
        (
            "raw + semantic similarity",
            "continuous mixture score",
            "no deployed candidate bank",
        ),
        accent="#496f8e",
    )
    arrow(draw, (405, y + 105), (492, y + 105))
    arrow(draw, (835, y + 105), (922, y + 105))
    arrow(draw, (1265, y + 105), (1352, y + 105))
    arrow(draw, (1740, y + 105), (1827, y + 105))
    draw.text(
        (64, 380),
        "The student predicts and scores continuous writing images. Pixel-derived equality groups and symbolic controls exist only in training loss or evaluation.",
        font=font(16),
        fill="#52666d",
    )

    left = (62, 432, 1160, 1170)
    draw.rounded_rectangle(left, radius=8, fill="white", outline="#aebcc1", width=2)
    draw.text((88, 458), "Natural next-image audit", font=font(25, bold=True), fill="#1a3741")
    draw.text(
        (88, 498),
        "2,048 development windows; top-1 over an evaluator-only 1,024-image bank",
        font=font(15),
        fill="#61737a",
    )
    natural_bars = (
        ("Full 64-image context", natural["full_top1"], "#277f8f"),
        ("Four-image suffix", natural["suffix4_top1"], "#6f6597"),
        ("Shuffled prefix", natural["shuffled_top1"], "#89989e"),
        ("Image unigram", natural["unigram_top1"], "#a8753e"),
        ("Symbolic bigram", natural["bigram_top1"], "#a55252"),
        ("Symbolic trigram", natural["trigram_top1"], "#7b3f69"),
    )
    for index, (label, value, color) in enumerate(natural_bars):
        percent_bar(
            draw,
            x=90,
            y=548 + index * 54,
            width=1020,
            label=label,
            value=float(value),
            maximum=0.21,
            color=color,
        )

    draw.line((88, 895, 1132, 895), fill="#d2dcdf", width=2)
    stat_tile(
        draw,
        bounds=(88, 925, 414, 1082),
        label="Full vs suffix log p",
        value=f"{full_suffix_logp:+.3f} nat",
        detail="> +0.030 fixed gate",
        passed=True,
    )
    stat_tile(
        draw,
        bounds=(431, 925, 757, 1082),
        label="Full vs shuffle log p",
        value=f"{full_shuffle_logp:+.3f} nat",
        detail="> +0.030 fixed gate",
        passed=True,
    )
    stat_tile(
        draw,
        bounds=(774, 925, 1132, 1082),
        label="Full vs bigram log p",
        value="-1.801 nat",
        detail="fails transferable language gate",
        passed=False,
    )
    draw.rounded_rectangle((88, 1100, 1132, 1144), radius=6, fill="#f8e7e5")
    draw.text(
        (108, 1110),
        "LANGUAGE REJECTED: full top-1 is below unigram and over 9x below bigram.",
        font=font(16, bold=True),
        fill="#8d3838",
    )

    right = (1200, 432, 2338, 1170)
    draw.rounded_rectangle(right, radius=8, fill="white", outline="#aebcc1", width=2)
    draw.text((1226, 458), "Pixel-identical suffix-4 intervention", font=font(25, bold=True), fill="#1a3741")
    draw.text(
        (1226, 498),
        "512 cross-record pairs; same final four image cells, different histories and next images",
        font=font(15),
        fill="#61737a",
    )
    pair_bars = (
        ("Full context", suffix4["full_arm_accuracy"], "#277f8f", 0.65),
        ("Four-image suffix", suffix4["suffix4_arm_accuracy"], "#6f6597", None),
        ("Shuffled prefix", suffix4["shuffled_arm_accuracy"], "#89989e", None),
        ("Both contexts correct", suffix4["full_both_correct_rate"], "#a55252", 0.40),
    )
    for index, (label, value, color, target) in enumerate(pair_bars):
        percent_bar(
            draw,
            x=1228,
            y=558 + index * 61,
            width=1055,
            label=label,
            value=float(value),
            maximum=0.70,
            color=color,
            target=target,
        )
    draw.text(
        (1480, 810),
        "red markers: fixed 65% arm / 40% both-correct gates",
        font=font(14),
        fill="#792e2e",
    )

    stat_tile(
        draw,
        bounds=(1226, 850, 1565, 1010),
        label="Suffix pixels",
        value="100% equal",
        detail="exact intervention",
        passed=True,
    )
    stat_tile(
        draw,
        bounds=(1582, 850, 1921, 1010),
        label="Permutation error",
        value="0.0",
        detail="candidate equivariance",
        passed=True,
    )
    stat_tile(
        draw,
        bounds=(1938, 850, 2310, 1010),
        label="Full vs shuffle margin",
        value=f"{suffix4['full_minus_shuffled_mean_margin']:+.3f}",
        detail="> +0.020, but accuracy fails",
        passed=True,
    )
    draw.rounded_rectangle((1226, 1032, 2310, 1144), radius=7, fill="#f8e7e5")
    draw.text(
        (1248, 1051),
        "BINDING REJECTED",
        font=font(20, bold=True),
        fill="#8d3838",
    )
    draw.text(
        (1248, 1087),
        "49.56% full arms; -0.44 point vs suffix and -0.39 point vs shuffle.",
        font=font(16),
        fill="#704b4b",
    )

    identity_panel = (62, 1210, 1160, 1522)
    draw.rounded_rectangle(identity_panel, radius=8, fill="white", outline="#aebcc1", width=2)
    draw.text((88, 1236), "Same-scope visual form invariance", font=font(24, bold=True), fill="#1a3741")
    draw.text(
        (88, 1275),
        "Cross-font top-1 on the same 1,024-image development bank",
        font=font(15),
        fill="#61737a",
    )
    percent_bar(
        draw,
        x=90,
        y=1330,
        width=1020,
        label="Frozen raw retina",
        value=float(natural["raw_retina_cross_font_identity_top1"]),
        maximum=1.0,
        color="#a8753e",
    )
    percent_bar(
        draw,
        x=90,
        y=1392,
        width=1020,
        label="EMA semantic route",
        value=float(natural["ema_semantic_cross_font_identity_top1"]),
        maximum=1.0,
        color="#19725d",
        target=0.95,
    )
    draw.text(
        (90, 1455),
        f"Gain: {100.0 * identity_gain:+.3f} points. This is form invariance, not language understanding.",
        font=font(16, bold=True),
        fill="#435b63",
    )

    verdict = (1200, 1210, 2338, 1522)
    draw.rounded_rectangle(verdict, radius=8, fill="#172f39", outline="#172f39")
    draw.text((1228, 1237), "Preregistered decision", font=font(24, bold=True), fill="white")
    stat_tile(
        draw,
        bounds=(1228, 1290, 1550, 1435),
        label="Mechanism gates",
        value="10 / 14",
        detail="four decisive failures",
        passed=False,
    )
    stat_tile(
        draw,
        bounds=(1568, 1290, 1890, 1435),
        label="Language gates",
        value="2 / 6",
        detail="below frequency controls",
        passed=False,
    )
    stat_tile(
        draw,
        bounds=(1908, 1290, 2310, 1435),
        label="Resource result",
        value="1.144 GiB",
        detail="not capability efficiency",
        passed=None,
    )
    draw.text(
        (1228, 1467),
        "REJECTED ON DEVELOPMENT | FROZEN SEALED | WRITER NOT AUTHORIZED",
        font=font(17, bold=True),
        fill="#f0b7ad",
    )

    bottom = (62, 1562, 2338, 1742)
    draw.rounded_rectangle(bottom, radius=8, fill="#e8eef0", outline="#aebcc1", width=2)
    draw.text((88, 1588), "Localized next question", font=font(22, bold=True), fill="#1a3741")
    draw.text(
        (88, 1629),
        "Train candidate-conditioned prefix evidence directly: score what images 1-60 add beyond the identical four-image suffix.",
        font=font(18),
        fill="#41575f",
    )
    draw.text(
        (88, 1667),
        "Require transferable pair accuracy and frequency-baseline wins before adding a writer, page fold, geometric depth, motion, or scale.",
        font=font(18),
        fill="#41575f",
    )
    draw.text(
        (1984, 1710),
        "FIXED DEVELOPMENT EVIDENCE",
        font=font(14, bold=True),
        fill="#536970",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print(args.out)


if __name__ == "__main__":
    main()
