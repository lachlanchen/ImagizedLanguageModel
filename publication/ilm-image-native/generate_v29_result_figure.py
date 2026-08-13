#!/usr/bin/env python3
"""Compose the measured V29 development result from fixed evidence."""

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
    / "artifacts/conditional_visual_density_ratio_v29_evidence"
    / "development_audit.json"
)
DEFAULT_OUT = (
    Path(__file__).resolve().parent
    / "figures/conditional_visual_density_ratio_v29_result.png"
)
EXPECTED_REPORT_SHA256 = (
    "16645844dd0b9dd4fb1e5157edbdd20d6e13b34b201c1147c177ac52464a5108"
)
EXPECTED_PROTOCOL_SHA256 = (
    "4cb17b793de051d858418d3ec0a4cb08b2928308c2d8e56530fc5655d9ffad0f"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "a8ec991968b577518d801090f5953406de13c688552107f26ac400fc2d508b8a"
)
EXPECTED_CANONICAL_SHA256 = (
    "fbcce3b2661b6eca7697d631772a84043f42495afd289d067d5ea2d7384d50ce"
)
EXPECTED_BANK_IMAGES_SHA256 = (
    "3d536fdb06b795080e7e0c8814b8b155b37221dd3c7986a025e96badb003bc31"
)
EXPECTED_ARCHITECTURE = "conditional-visual-density-ratio-v29-development-audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the measured V29 figure.")
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
        raise ValueError("V29 report SHA-256 does not match fixed evidence")
    return json.loads(path.read_text(encoding="utf-8"))


def require_close(value: object, expected: float, name: str) -> None:
    if not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Unexpected V29 {name}: {value!r}")


def validate_evidence(report: dict[str, Any]) -> None:
    if report.get("architecture") != EXPECTED_ARCHITECTURE:
        raise ValueError("Unexpected V29 report architecture")
    if report.get("checkpoint_architecture") != "conditional-visual-density-ratio-v29":
        raise ValueError("Unexpected V29 checkpoint architecture")
    if report.get("smoke_only"):
        raise ValueError("V29 report is smoke-only")
    if int(report.get("natural_windows", -1)) != 2_048:
        raise ValueError("V29 report does not use 2,048 natural windows")
    if int(report.get("suffix4_pairs", -1)) != 512:
        raise ValueError("V29 report does not use 512 suffix-4 pairs")
    if report.get("frozen_images_instantiated"):
        raise ValueError("V29 report opened frozen images")
    if report.get("mechanism_selected") or report.get("language_selected"):
        raise ValueError("V29 report unexpectedly selected a model")
    if report.get("frozen_evaluation_authorized"):
        raise ValueError("V29 report unexpectedly authorized frozen evaluation")
    if report.get("writer_training_authorized"):
        raise ValueError("V29 report unexpectedly authorized writer training")
    if report.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("V29 protocol SHA-256 changed")
    if report.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("V29 checkpoint SHA-256 changed")
    if int(report.get("total_parameters", -1)) != 20_080_961:
        raise ValueError("V29 total parameter count changed")
    if int(report.get("trainable_parameters", -1)) != 18_451_585:
        raise ValueError("V29 trainable parameter count changed")
    if int(report.get("training_metrics", {}).get("step", -1)) != 8_000:
        raise ValueError("V29 report is not the fixed 8,000-step endpoint")

    mechanism = report.get("mechanism_gates")
    language = report.get("language_gates")
    if not isinstance(mechanism, dict) or sum(mechanism.values()) != 8:
        raise ValueError("V29 mechanism gate receipt changed")
    if not isinstance(language, dict) or sum(language.values()) != 2:
        raise ValueError("V29 language gate receipt changed")

    natural = report.get("natural")
    suffix4 = report.get("suffix4")
    bank = report.get("candidate_bank_manifest")
    if not isinstance(natural, dict) or not isinstance(suffix4, dict):
        raise ValueError("V29 report is missing fixed audit sections")
    if not isinstance(bank, dict) or int(bank.get("bank_size", -1)) != 1_024:
        raise ValueError("V29 candidate bank receipt changed")
    if bank.get("canonical_sha256") != EXPECTED_CANONICAL_SHA256:
        raise ValueError("V29 canonical candidate pixels changed")
    if bank.get("images_sha256") != EXPECTED_BANK_IMAGES_SHA256:
        raise ValueError("V29 candidate image views changed")
    if bank.get("images_in_checkpoint") or bank.get("forms_in_checkpoint"):
        raise ValueError("V29 checkpoint unexpectedly contains the bank")

    require_close(natural.get("student_boundary_clean"), 1.0, "student boundary")
    require_close(
        natural.get("training_bank_absent_from_checkpoint"),
        1.0,
        "bank exclusion",
    )
    require_close(suffix4.get("suffix_pixel_equality"), 1.0, "suffix pixels")
    require_close(
        suffix4.get("suffix_score_row_equality"), 1.0, "suffix score rows"
    )
    require_close(suffix4.get("suffix4_arm_accuracy"), 0.5, "suffix control")
    for score_name in ("full", "suffix4", "increment"):
        require_close(
            suffix4.get(f"candidate_permutation_{score_name}_max_score_error"),
            0.0,
            f"{score_name} permutation error",
        )
        require_close(
            suffix4.get(f"candidate_permutation_{score_name}_accuracy_agreement"),
            1.0,
            f"{score_name} permutation agreement",
        )
    require_close(
        suffix4.get("full_mean_margin"),
        suffix4.get("increment_mean_margin"),
        "full/increment aggregate margin identity",
    )


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
    left, top, _, bottom = bounds
    draw.rounded_rectangle(bounds, radius=7, fill="white", outline="#afbdc2", width=2)
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
    bar_left = x + 258
    bar_width = width - 408
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
    left, top, _, bottom = bounds
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
    full_bigram_logp = (
        natural["full_target_log_probability"]
        - natural["bigram_target_log_probability"]
    )
    increment_accuracy_gain = (
        suffix4["increment_arm_accuracy"]
        - suffix4["shuffled_increment_arm_accuracy"]
    )
    increment_margin_gain = (
        suffix4["increment_mean_margin"]
        - suffix4["shuffled_increment_mean_margin"]
    )

    canvas = Image.new("RGB", (2400, 1800), "#f2f5f6")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (62, 34),
        "V29: visual order signal grows, but candidate binding remains at chance",
        font=font(37, bold=True),
        fill="#17323c",
    )
    draw.text(
        (64, 94),
        "20.08M total / 18.45M trainable | 8,000 BF16 updates | one RTX 4090 | 3.356 GiB peak | frozen split sealed",
        font=font(19),
        fill="#586b73",
    )

    y = 148
    architecture_box(
        draw,
        (62, y, 405, y + 210),
        "Visual-time stream",
        ("B x N x 1 x 32 x 32", "64 context images", "no strings, IDs, or OCR"),
        accent="#277f8f",
    )
    architecture_box(
        draw,
        (492, y, 835, y + 210),
        "Frozen perception",
        ("V16 raw retina", "V28 semantic adapters", "continuous image features"),
        accent="#a8753e",
    )
    architecture_box(
        draw,
        (922, y, 1265, y + 210),
        "Causal context field",
        ("8 layers, width 384", "one retained state per cell", "full and suffix contexts"),
        accent="#46755d",
    )
    architecture_box(
        draw,
        (1352, y, 1740, y + 210),
        "Candidate visual query",
        ("arbitrary 32 x 32 image", "2 cross-attention layers", "query-to-history evidence"),
        accent="#6f6597",
    )
    architecture_box(
        draw,
        (1827, y, 2338, y + 210),
        "Conditional energies",
        ("F = full-context score", "B = exact-suffix score", "G = row-center(F - B)"),
        accent="#496f8e",
    )
    arrow(draw, (405, y + 105), (492, y + 105))
    arrow(draw, (835, y + 105), (922, y + 105))
    arrow(draw, (1265, y + 105), (1352, y + 105))
    arrow(draw, (1740, y + 105), (1827, y + 105))
    draw.text(
        (64, 380),
        "Candidate images are scored in chunks. The 1,024-image bank and symbolic controls stay outside model state, checkpoints, and inference.",
        font=font(16),
        fill="#52666d",
    )

    left = (62, 432, 1160, 1160)
    draw.rounded_rectangle(left, radius=8, fill="white", outline="#aebcc1", width=2)
    draw.text((88, 458), "Natural next-image audit", font=font(25, bold=True), fill="#1a3741")
    draw.text(
        (88, 498),
        "2,048 windows; top-1 over an evaluator-only 1,024-image bank",
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
            y=548 + index * 52,
            width=1020,
            label=label,
            value=float(value),
            maximum=0.21,
            color=color,
        )

    draw.line((88, 875, 1132, 875), fill="#d2dcdf", width=2)
    stat_tile(
        draw,
        bounds=(88, 902, 414, 1059),
        label="Full vs suffix log p",
        value=f"{full_suffix_logp:+.3f} nat",
        detail="> +0.030 fixed gate",
        passed=True,
    )
    stat_tile(
        draw,
        bounds=(431, 902, 757, 1059),
        label="Full vs shuffle log p",
        value=f"{full_shuffle_logp:+.3f} nat",
        detail="large order effect",
        passed=True,
    )
    stat_tile(
        draw,
        bounds=(774, 902, 1132, 1059),
        label="Full vs bigram log p",
        value=f"{full_bigram_logp:+.3f} nat",
        detail="fails language gate",
        passed=False,
    )
    draw.rounded_rectangle((88, 1078, 1132, 1134), radius=6, fill="#f8e7e5")
    draw.text(
        (108, 1094),
        "LANGUAGE REJECTED: 2.34% full top-1 versus 13.87% bigram.",
        font=font(16, bold=True),
        fill="#8d3838",
    )

    right = (1200, 432, 2338, 1160)
    draw.rounded_rectangle(right, radius=8, fill="white", outline="#aebcc1", width=2)
    draw.text((1226, 458), "Pixel-identical suffix-4 intervention", font=font(25, bold=True), fill="#1a3741")
    draw.text(
        (1226, 498),
        "512 cross-record pairs; same suffix images, different histories and targets",
        font=font(15),
        fill="#61737a",
    )
    pair_bars = (
        ("Full context", suffix4["full_arm_accuracy"], "#277f8f", 0.65),
        ("Exact suffix", suffix4["suffix4_arm_accuracy"], "#6f6597", None),
        ("Shuffled prefix", suffix4["shuffled_arm_accuracy"], "#89989e", None),
        ("Full - suffix", suffix4["increment_arm_accuracy"], "#46755d", 0.65),
        ("Shuffled - suffix", suffix4["shuffled_increment_arm_accuracy"], "#a8753e", None),
    )
    for index, (label, value, color, target) in enumerate(pair_bars):
        percent_bar(
            draw,
            x=1228,
            y=548 + index * 54,
            width=1055,
            label=label,
            value=float(value),
            maximum=0.70,
            color=color,
            target=target,
        )
    draw.text(
        (1484, 822),
        "red marker: fixed >65% arm-accuracy gate",
        font=font(14),
        fill="#792e2e",
    )

    stat_tile(
        draw,
        bounds=(1226, 856, 1565, 1013),
        label="Increment gain",
        value=f"{100 * increment_accuracy_gain:+.2f} pt",
        detail="> +10 points required",
        passed=False,
    )
    stat_tile(
        draw,
        bounds=(1582, 856, 1921, 1013),
        label="Both correct",
        value=f"{100 * suffix4['increment_both_correct_rate']:.2f}%",
        detail="> 40% required",
        passed=False,
    )
    stat_tile(
        draw,
        bounds=(1938, 856, 2310, 1013),
        label="Margin gain",
        value=f"{increment_margin_gain:+.4f}",
        detail="> +0.050 required",
        passed=False,
    )
    draw.rounded_rectangle((1226, 1035, 2310, 1134), radius=7, fill="#f8e7e5")
    draw.text((1248, 1052), "BINDING REJECTED", font=font(20, bold=True), fill="#8d3838")
    draw.text(
        (1248, 1087),
        "50.71% increment arms; exact suffix and permutation controls pass.",
        font=font(16),
        fill="#704b4b",
    )

    identity_panel = (62, 1200, 1160, 1508)
    draw.rounded_rectangle(identity_panel, radius=8, fill="white", outline="#aebcc1", width=2)
    draw.text((88, 1226), "Perception is visible and stable", font=font(24, bold=True), fill="#1a3741")
    draw.text(
        (88, 1265),
        "Cross-font identity on the same 1,024-image scope",
        font=font(15),
        fill="#61737a",
    )
    percent_bar(
        draw,
        x=90,
        y=1320,
        width=1020,
        label="Frozen raw retina",
        value=float(natural["raw_retina_cross_font_identity_top1"]),
        maximum=1.0,
        color="#a8753e",
    )
    percent_bar(
        draw,
        x=90,
        y=1380,
        width=1020,
        label="Frozen semantic route",
        value=float(natural["frozen_semantic_cross_font_identity_top1"]),
        maximum=1.0,
        color="#19725d",
        target=0.95,
    )
    draw.text(
        (90, 1442),
        f"Two-candidate raw identity: {100 * suffix4['raw_retina_two_candidate_identity_accuracy']:.3f}%. Perception passes; binding does not.",
        font=font(16, bold=True),
        fill="#435b63",
    )

    verdict = (1200, 1200, 2338, 1508)
    draw.rounded_rectangle(verdict, radius=8, fill="#172f39", outline="#172f39")
    draw.text((1228, 1227), "Preregistered decision", font=font(24, bold=True), fill="white")
    stat_tile(
        draw,
        bounds=(1228, 1280, 1550, 1425),
        label="Mechanism gates",
        value="8 / 14",
        detail="binding gates fail",
        passed=False,
    )
    stat_tile(
        draw,
        bounds=(1568, 1280, 1890, 1425),
        label="Language gates",
        value="2 / 6",
        detail="bigram remains stronger",
        passed=False,
    )
    stat_tile(
        draw,
        bounds=(1908, 1280, 2310, 1425),
        label="Resource result",
        value="3.356 GiB",
        detail="not capability efficiency",
        passed=None,
    )
    draw.text(
        (1228, 1457),
        "REJECTED ON DEVELOPMENT | FROZEN SEALED | WRITER NOT AUTHORIZED",
        font=font(17, bold=True),
        fill="#f0b7ad",
    )

    bottom = (62, 1548, 2338, 1742)
    draw.rounded_rectangle(bottom, radius=8, fill="#e8eef0", outline="#aebcc1", width=2)
    draw.text((88, 1574), "Exact diagnosis and bounded next question", font=font(22, bold=True), fill="#1a3741")
    draw.text(
        (88, 1614),
        "For exact shared suffix B: mean assignment margin of G = F - B equals mean assignment margin of F; baseline terms cancel.",
        font=font(18, bold=True),
        fill="#41575f",
    )
    draw.text(
        (88, 1654),
        "Next test: predict a spatial next-image field, compare candidate patches before scalar reduction, and retain the same held-out controls.",
        font=font(18),
        fill="#41575f",
    )
    draw.text(
        (88, 1692),
        "Do not add a writer, page fold, geometric depth, historical-glyph task, or scale until conditional visual binding passes.",
        font=font(18),
        fill="#41575f",
    )
    draw.text((1984, 1710), "FIXED DEVELOPMENT EVIDENCE", font=font(14, bold=True), fill="#536970")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print(args.out)


if __name__ == "__main__":
    main()
