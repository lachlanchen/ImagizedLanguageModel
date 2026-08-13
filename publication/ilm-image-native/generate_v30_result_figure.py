#!/usr/bin/env python3
"""Compose the measured V30 matched-arm result from fixed evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from generate_v20_result_figure import font


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = (
    ROOT / "artifacts/spatial_visual_next_field_v30_evidence" / "development_audit.json"
)
DEFAULT_OUT = (
    Path(__file__).resolve().parent / "figures/spatial_visual_next_field_v30_result.png"
)
EXPECTED_REPORT_SHA256 = (
    "2d0a3a08e2f5d4b267e276b695448cc8311687a822776a33c0a883dc0a74fd8f"
)
EXPECTED_PROTOCOL_SHA256 = (
    "81d2b2af1eb3a305b4acd1028c004ddddc607e826eea1d50b6d137d32ed180a5"
)
EXPECTED_SPATIAL_CHECKPOINT_SHA256 = (
    "11a3a7e9f13f1db932dcc913e1c79b0e0db49b95bd49f98a878897028bf86130"
)
EXPECTED_GLOBAL_CHECKPOINT_SHA256 = (
    "66378d4b972702490f6819d87d95c2576546e15e6fc74d10542307aaf4483411"
)
EXPECTED_ARCHITECTURE = "spatial-visual-next-field-v30-development-audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the measured V30 figure.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_report(path: Path) -> dict[str, Any]:
    if file_sha256(path) != EXPECTED_REPORT_SHA256:
        raise ValueError("V30 report SHA-256 does not match fixed evidence")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_evidence(report: dict[str, Any]) -> None:
    if report.get("architecture") != EXPECTED_ARCHITECTURE:
        raise ValueError("Unexpected V30 report architecture")
    if report.get("checkpoint_architecture") != "spatial-visual-next-field-v30":
        raise ValueError("Unexpected V30 checkpoint architecture")
    if report.get("smoke_only"):
        raise ValueError("V30 report is smoke-only")
    if report.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("V30 protocol receipt changed")
    if report.get("spatial_checkpoint_sha256") != EXPECTED_SPATIAL_CHECKPOINT_SHA256:
        raise ValueError("V30 spatial checkpoint receipt changed")
    if report.get("global_checkpoint_sha256") != EXPECTED_GLOBAL_CHECKPOINT_SHA256:
        raise ValueError("V30 global checkpoint receipt changed")
    if report.get("frozen_images_instantiated"):
        raise ValueError("V30 report opened frozen images")
    if report.get("spatial_mechanism_selected"):
        raise ValueError("V30 report unexpectedly selected the spatial mechanism")
    if report.get("frozen_evaluation_authorized"):
        raise ValueError("V30 report unexpectedly authorized frozen evaluation")
    if report.get("writer_training_authorized"):
        raise ValueError("V30 report unexpectedly authorized writer training")
    if int(report.get("natural_windows", {}).get("count", -1)) != 2_048:
        raise ValueError("V30 report does not use 2,048 natural windows")
    if int(report.get("suffix4_pairs", {}).get("count", -1)) != 512:
        raise ValueError("V30 report does not use 512 suffix-4 pairs")

    expected_gates = {
        "spatial_common_gates": (12, 18),
        "global_integrity_gates": (12, 12),
        "matched_arm_gates": (5, 9),
        "spatial_language_gates": (0, 8),
    }
    for key, (passed, total) in expected_gates.items():
        gates = report.get(key)
        if not isinstance(gates, dict):
            raise ValueError(f"V30 report is missing {key}")
        if len(gates) != total or sum(gates.values()) != passed:
            raise ValueError(f"V30 {key} receipt changed")

    for route in ("spatial", "global_control"):
        integrity = report.get(route, {}).get("integrity", {})
        if int(integrity.get("step", -1)) != 8_000:
            raise ValueError(f"V30 {route} endpoint changed")
        if int(integrity.get("finite_updates_verified", -1)) != 8_000:
            raise ValueError(f"V30 {route} finite-update receipt changed")
        if int(integrity.get("total_parameters", -1)) != 18_641_153:
            raise ValueError(f"V30 {route} parameter count changed")
        if int(integrity.get("trainable_parameters", -1)) != 17_011_777:
            raise ValueError(f"V30 {route} trainable count changed")
        if not integrity.get("final_checkpoint_clean"):
            raise ValueError(f"V30 {route} final checkpoint is not clean")

    matched = report.get("matched", {})
    initialization = matched.get("initialization", {})
    if not initialization.get("values_equal"):
        raise ValueError("V30 initialized arm tensors differ")
    if not matched.get("rendered_audit_pixels_equal"):
        raise ValueError("V30 arm audit pixels differ")


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: str = "#506970",
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
    draw.text((left + 23, top + 14), title, font=font(17, bold=True), fill="#17343e")
    for index, line in enumerate(lines):
        draw.text(
            (left + 23, top + 48 + index * 25),
            line,
            font=font(13),
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
    draw.text((x, y), label, font=font(15), fill="#2b424a")
    bar_left = x + 278
    bar_width = width - 440
    draw.rectangle((bar_left, y + 2, bar_left + bar_width, y + 25), fill="#e1e8ea")
    filled = max(2, int(bar_width * max(0.0, value) / maximum))
    draw.rectangle((bar_left, y + 2, bar_left + filled, y + 25), fill=color)
    if target is not None:
        marker = bar_left + int(bar_width * target / maximum)
        draw.line((marker, y - 3, marker, y + 30), fill="#792e2e", width=3)
    draw.text(
        (bar_left + bar_width + 13, y - 1),
        f"{100.0 * value:7.3f}%",
        font=font(14, bold=True),
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
    fill = "#edf2f4" if passed is None else ("#eef5f2" if passed else "#f8e9e7")
    draw.rounded_rectangle(bounds, radius=7, fill=fill, outline="#b6c3c7", width=2)
    draw.text((left + 15, top + 12), label, font=font(13, bold=True), fill="#435860")
    draw.text((left + 15, top + 43), value, font=font(22, bold=True), fill=color)
    draw.text((left + 15, bottom - 29), detail, font=font(12), fill="#5d7077")


def main() -> None:
    args = parse_args()
    report = read_report(args.report)
    validate_evidence(report)
    spatial = report["spatial"]
    global_control = report["global_control"]
    sn = spatial["natural"]
    sp = spatial["suffix4"]
    gn = global_control["natural"]
    gp = global_control["suffix4"]

    natural_top1_delta = sn["full_top1"] - gn["full_top1"]
    natural_logp_delta = (
        sn["full_target_log_probability"] - gn["full_target_log_probability"]
    )
    pair_accuracy_delta = sp["full_arm_accuracy"] - gp["full_arm_accuracy"]
    pair_both_delta = sp["full_both_correct_rate"] - gp["full_both_correct_rate"]
    spatial_logp_gain = (
        sn["full_target_log_probability"]
        - sn["spatial_permuted_target_log_probability"]
    )

    canvas = Image.new("RGB", (2400, 1800), "#f2f5f6")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (62, 32),
        "V30: aligned spatial fields change scores, but do not bind next writing",
        font=font(35, bold=True),
        fill="#17323c",
    )
    draw.text(
        (64, 88),
        "Two parameter-identical 18.64M models | 8,000 BF16 updates each | one RTX 4090 | 1.60 GiB peak per arm | frozen sealed",
        font=font(18),
        fill="#586b73",
    )

    y = 138
    architecture_box(
        draw,
        (62, y, 372, y + 222),
        "Visual-time stream",
        ("B x 64 x 1 x 32 x 32", "ordinary Chinese images", "no strings, IDs, or OCR"),
        accent="#277f8f",
    )
    architecture_box(
        draw,
        (436, y, 792, y + 222),
        "Shared visual context",
        ("frozen V16 retina", "8-layer causal field", "byte-identical start"),
        accent="#a8753e",
    )
    architecture_box(
        draw,
        (856, y, 1194, y + 222),
        "Next-image field",
        ("candidate-independent", "4 x 4 x 192 output", "same decoder in both arms"),
        accent="#46755d",
    )
    architecture_box(
        draw,
        (1270, y, 1690, y + 100),
        "Spatial arm",
        ("aligned retinal cells", "local interaction then mean"),
        accent="#277f8f",
    )
    architecture_box(
        draw,
        (1270, y + 122, 1690, y + 222),
        "Global control",
        ("semantic vector tiled 16x", "same capacity and score path"),
        accent="#6f6597",
    )
    architecture_box(
        draw,
        (1770, y, 2338, y + 222),
        "Fixed matched audit",
        (
            "2,048 natural windows",
            "512 exact-suffix pairs",
            "patch and column permutations",
        ),
        accent="#496f8e",
    )
    arrow(draw, (372, y + 111), (436, y + 111))
    arrow(draw, (792, y + 111), (856, y + 111))
    draw.line((1194, y + 111, 1232, y + 111), fill="#506970", width=4)
    draw.line((1232, y + 50, 1232, y + 172), fill="#506970", width=4)
    arrow(draw, (1232, y + 50), (1270, y + 50))
    arrow(draw, (1232, y + 172), (1270, y + 172))
    draw.line((1690, y + 50, 1730, y + 50), fill="#506970", width=4)
    draw.line((1690, y + 172, 1730, y + 172), fill="#506970", width=4)
    draw.line((1730, y + 50, 1730, y + 172), fill="#506970", width=4)
    arrow(draw, (1730, y + 111), (1770, y + 111))
    draw.text(
        (64, 380),
        "Candidates are arbitrary images. The 1,024-image bank and symbolic controls remain outside model state, checkpoints, and inference.",
        font=font(15),
        fill="#52666d",
    )

    left = (62, 425, 1160, 1158)
    draw.rounded_rectangle(left, radius=8, fill="white", outline="#aebcc1", width=2)
    draw.text(
        (88, 450), "Natural next-image audit", font=font(24, bold=True), fill="#1a3741"
    )
    draw.text(
        (88, 489),
        "Top-1 over the same evaluator-only 1,024-image bank",
        font=font(14),
        fill="#61737a",
    )
    natural_bars = (
        ("Spatial full context", sn["full_top1"], "#277f8f", 0.15),
        ("Global full context", gn["full_top1"], "#6f6597", None),
        ("Spatial suffix-4", sn["suffix4_top1"], "#46755d", None),
        ("Spatial shuffled prefix", sn["shuffled_top1"], "#89989e", None),
        ("Spatial patch reversal", sn["spatial_permuted_top1"], "#a8753e", None),
        ("Image unigram", sn["unigram_top1"], "#b58c42", None),
        ("Symbolic bigram", sn["bigram_top1"], "#a55252", None),
        ("Symbolic trigram", sn["trigram_top1"], "#7b3f69", None),
    )
    for index, (label, value, color, target) in enumerate(natural_bars):
        percent_bar(
            draw,
            x=90,
            y=535 + index * 48,
            width=1020,
            label=label,
            value=float(value),
            maximum=0.205,
            color=color,
            target=target,
        )
    draw.text(
        (365, 923),
        "red marker: fixed 15% spatial top-1 gate",
        font=font(13),
        fill="#792e2e",
    )
    stat_tile(
        draw,
        bounds=(88, 960, 414, 1115),
        label="Spatial vs global top-1",
        value=f"{100 * natural_top1_delta:+.2f} pt",
        detail="> +1 point required",
        passed=False,
    )
    stat_tile(
        draw,
        bounds=(431, 960, 757, 1115),
        label="Spatial vs global log p",
        value=f"{natural_logp_delta:+.3f} nat",
        detail="> +0.050 required",
        passed=False,
    )
    stat_tile(
        draw,
        bounds=(774, 960, 1132, 1115),
        label="Full vs patch reversal",
        value=f"{spatial_logp_gain:+.3f} nat",
        detail="> +0.050 required",
        passed=False,
    )

    right = (1200, 425, 2338, 1158)
    draw.rounded_rectangle(right, radius=8, fill="white", outline="#aebcc1", width=2)
    draw.text(
        (1226, 450),
        "Pixel-identical suffix-4 binding audit",
        font=font(24, bold=True),
        fill="#1a3741",
    )
    draw.text(
        (1226, 489),
        "512 pairs in two unseen fonts; 2,048 scored context arms",
        font=font(14),
        fill="#61737a",
    )
    pair_bars = (
        ("Spatial full context", sp["full_arm_accuracy"], "#277f8f", 0.65),
        ("Global full context", gp["full_arm_accuracy"], "#6f6597", None),
        ("Spatial exact suffix", sp["suffix4_arm_accuracy"], "#46755d", None),
        ("Spatial shuffled prefix", sp["shuffled_arm_accuracy"], "#89989e", None),
        (
            "Spatial patch reversal",
            sp["spatial_permuted_arm_accuracy"],
            "#a8753e",
            None,
        ),
    )
    for index, (label, value, color, target) in enumerate(pair_bars):
        percent_bar(
            draw,
            x=1228,
            y=548 + index * 58,
            width=1055,
            label=label,
            value=float(value),
            maximum=0.70,
            color=color,
            target=target,
        )
    draw.text(
        (1502, 833),
        "red marker: fixed >65% spatial arm-accuracy gate",
        font=font(13),
        fill="#792e2e",
    )
    stat_tile(
        draw,
        bounds=(1226, 879, 1565, 1034),
        label="Spatial vs global",
        value=f"{100 * pair_accuracy_delta:+.2f} pt",
        detail="> +5 points required",
        passed=False,
    )
    stat_tile(
        draw,
        bounds=(1582, 879, 1921, 1034),
        label="Both-correct delta",
        value=f"{100 * pair_both_delta:+.2f} pt",
        detail="> +5 points required",
        passed=False,
    )
    stat_tile(
        draw,
        bounds=(1938, 879, 2310, 1034),
        label="Spatial vs reversal",
        value=f"{100 * sp['full_minus_spatial_permuted_arm_accuracy']:+.2f} pt",
        detail="> +5 points required",
        passed=False,
    )
    draw.rounded_rectangle((1226, 1055, 2310, 1128), radius=7, fill="#f8e7e5")
    draw.text(
        (1248, 1070), "BINDING REJECTED", font=font(18, bold=True), fill="#8d3838"
    )
    draw.text(
        (1248, 1100),
        "Exact suffix rows and candidate-column equivariance pass; assignment stays at chance.",
        font=font(14),
        fill="#704b4b",
    )

    evidence = (62, 1194, 1160, 1513)
    draw.rounded_rectangle(evidence, radius=8, fill="white", outline="#aebcc1", width=2)
    draw.text(
        (88, 1220),
        "What the spatial route did learn",
        font=font(23, bold=True),
        fill="#1a3741",
    )
    draw.text(
        (88, 1261),
        "Aligned frozen-field cross-font identity",
        font=font(15),
        fill="#52666d",
    )
    percent_bar(
        draw,
        x=90,
        y=1300,
        width=1020,
        label="Spatial candidate visibility",
        value=float(sn["candidate_cross_font_identity_top1"]),
        maximum=1.0,
        color="#19725d",
        target=0.95,
    )
    draw.text(
        (90, 1363),
        f"Patch reversal changes natural scores by up to {sn['spatial_permutation_max_score_error']:.3f},",
        font=font(16, bold=True),
        fill="#435b63",
    )
    draw.text(
        (90, 1398),
        "but target probability and paired correctness move too little.",
        font=font(16),
        fill="#435b63",
    )
    draw.text(
        (90, 1453),
        "Sensitivity to local geometry is not conditional visual language.",
        font=font(17, bold=True),
        fill="#8d3838",
    )

    verdict = (1200, 1194, 2338, 1513)
    draw.rounded_rectangle(verdict, radius=8, fill="#172f39", outline="#172f39")
    draw.text(
        (1228, 1220), "Preregistered decision", font=font(23, bold=True), fill="white"
    )
    stat_tile(
        draw,
        bounds=(1228, 1270, 1478, 1418),
        label="Spatial common",
        value="12 / 18",
        detail="six causal gates fail",
        passed=False,
    )
    stat_tile(
        draw,
        bounds=(1493, 1270, 1743, 1418),
        label="Control integrity",
        value="12 / 12",
        detail="matched control valid",
        passed=True,
    )
    stat_tile(
        draw,
        bounds=(1758, 1270, 2008, 1418),
        label="Matched arms",
        value="5 / 9",
        detail="all gain gates fail",
        passed=False,
    )
    stat_tile(
        draw,
        bounds=(2023, 1270, 2310, 1418),
        label="Spatial language",
        value="0 / 8",
        detail="no language gate passes",
        passed=False,
    )
    draw.text(
        (1228, 1458),
        "REJECTED | FROZEN SEALED | WRITER NOT AUTHORIZED",
        font=font(17, bold=True),
        fill="#f0b7ad",
    )

    bottom = (62, 1548, 2338, 1742)
    draw.rounded_rectangle(bottom, radius=8, fill="#e8eef0", outline="#aebcc1", width=2)
    draw.text((88, 1573), "Bounded diagnosis", font=font(21, bold=True), fill="#1a3741")
    draw.text(
        (88, 1612),
        "A single deterministic bilinear field learns average spatial appearance and natural order, but not a context-specific next-image distribution.",
        font=font(17, bold=True),
        fill="#41575f",
    )
    draw.text(
        (88, 1651),
        "Any successor must model multimodal continuous field density and beat this global control plus unigram and symbolic-bigram baselines.",
        font=font(17),
        fill="#41575f",
    )
    draw.text(
        (88, 1690),
        "Do not add a writer, page fold, depth axis, movie, historical-glyph task, or scale until conditional visual binding passes.",
        font=font(17),
        fill="#41575f",
    )
    draw.text(
        (1980, 1711),
        "FIXED DEVELOPMENT EVIDENCE",
        font=font(13, bold=True),
        fill="#536970",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print(args.out)


if __name__ == "__main__":
    main()
