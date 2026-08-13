#!/usr/bin/env python3
"""Compose the measured V31 result from fixed development evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from generate_v20_result_figure import font


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = (
    ROOT
    / "artifacts/conditional_visual_field_flow_v31_evidence"
    / "development_audit.json"
)
DEFAULT_COMPARISON = (
    ROOT
    / "artifacts/conditional_visual_field_flow_v31_evidence"
    / "comparison_receipt.json"
)
DEFAULT_SAMPLE = (
    ROOT
    / "artifacts/conditional_visual_field_flow_v31_evidence"
    / "autonomous_sample_nearest_images.png"
)
FIGURE_DIR = Path(__file__).resolve().parent / "figures"
DEFAULT_OUT = FIGURE_DIR / "conditional_visual_field_flow_v31_result.png"
DEFAULT_SAMPLE_OUT = (
    FIGURE_DIR / "conditional_visual_field_flow_v31_autonomous_nearest.png"
)

EXPECTED_REPORT_SHA256 = (
    "530c21d3d0f14e67e6616780a63d061c02f0a3ca22cf78849c914a65b8630985"
)
EXPECTED_COMPARISON_SHA256 = (
    "2e97b7324c6dcbbf004737b3d5b6d832145b90791ce1310fc2bcbbbc870f1859"
)
EXPECTED_SAMPLE_SHA256 = (
    "d57656cbb66c33a8f6c3f07b8eecdee7a460d5d675836d0a1eeb645d1b8384d0"
)
EXPECTED_PROTOCOL_SHA256 = (
    "92b6f70975dffe25723e332268b8929fa547b9d848a296f9ed80968cf798f8f7"
)
EXPECTED_SPATIAL_CHECKPOINT_SHA256 = (
    "9808e9966b02c2f200cc91d8c63e611c9dd52692903d7b51ab5131d0ee05859f"
)
EXPECTED_GLOBAL_CHECKPOINT_SHA256 = (
    "485011a4db854626b468bf7dba93962ce0ed22a98020282aa19c4ba06d999b7e"
)
EXPECTED_ARCHITECTURE = "conditional-visual-field-flow-v31-development-audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the measured V31 figure.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample-out", type=Path, default=DEFAULT_SAMPLE_OUT)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate(args: argparse.Namespace) -> dict[str, Any]:
    if file_sha256(args.report) != EXPECTED_REPORT_SHA256:
        raise ValueError("V31 report SHA-256 does not match fixed evidence")
    if file_sha256(args.comparison) != EXPECTED_COMPARISON_SHA256:
        raise ValueError("V31 comparison SHA-256 does not match fixed evidence")
    if file_sha256(args.sample) != EXPECTED_SAMPLE_SHA256:
        raise ValueError("V31 sample sheet SHA-256 does not match fixed evidence")

    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("architecture") != EXPECTED_ARCHITECTURE:
        raise ValueError("Unexpected V31 report architecture")
    if report.get("checkpoint_architecture") != "conditional-visual-field-flow-v31":
        raise ValueError("Unexpected V31 checkpoint architecture")
    if report.get("smoke_only"):
        raise ValueError("V31 report is smoke-only")
    if report.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("V31 protocol receipt changed")
    if report.get("spatial_checkpoint_sha256") != EXPECTED_SPATIAL_CHECKPOINT_SHA256:
        raise ValueError("V31 spatial checkpoint receipt changed")
    if report.get("global_checkpoint_sha256") != EXPECTED_GLOBAL_CHECKPOINT_SHA256:
        raise ValueError("V31 global checkpoint receipt changed")
    if report.get("sample_contact_sheet_sha256") != EXPECTED_SAMPLE_SHA256:
        raise ValueError("V31 embedded sample receipt changed")
    if report.get("frozen_images_instantiated"):
        raise ValueError("V31 report opened frozen images")
    if report.get("spatial_mechanism_selected"):
        raise ValueError("V31 report unexpectedly selected the spatial mechanism")
    if report.get("frozen_evaluation_authorized"):
        raise ValueError("V31 report unexpectedly authorized frozen evaluation")
    if report.get("writer_training_authorized"):
        raise ValueError("V31 report unexpectedly authorized writer training")
    if int(report.get("natural_windows", {}).get("count", -1)) != 2_048:
        raise ValueError("V31 report does not use 2,048 natural windows")
    if int(report.get("suffix4_pairs", {}).get("count", -1)) != 512:
        raise ValueError("V31 report does not use 512 suffix-4 pairs")

    expected_gates = {
        "spatial_common_gates": (14, 19),
        "global_integrity_gates": (6, 6),
        "matched_arm_gates": (4, 8),
        "spatial_language_and_generation_gates": (0, 10),
    }
    for key, (passed, total) in expected_gates.items():
        gates = report.get(key)
        if not isinstance(gates, dict):
            raise ValueError(f"V31 report is missing {key}")
        if len(gates) != total or sum(gates.values()) != passed:
            raise ValueError(f"V31 {key} receipt changed")

    for route in ("spatial", "global_control"):
        integrity = report.get(route, {}).get("integrity", {})
        if int(integrity.get("step", -1)) != 10_000:
            raise ValueError(f"V31 {route} endpoint changed")
        if int(integrity.get("finite_updates_verified", -1)) != 10_000:
            raise ValueError(f"V31 {route} finite-update receipt changed")
        if int(integrity.get("total_parameters", -1)) != 18_736_577:
            raise ValueError(f"V31 {route} parameter count changed")
        if int(integrity.get("trainable_parameters", -1)) != 17_107_201:
            raise ValueError(f"V31 {route} trainable count changed")
        if not integrity.get("final_checkpoint_clean"):
            raise ValueError(f"V31 {route} final checkpoint is not clean")

    matched = report.get("matched", {})
    initialization = matched.get("initialization", {})
    if not initialization.get("values_equal"):
        raise ValueError("V31 initialized arm tensors differ")
    if not matched.get("rendered_audit_pixels_equal"):
        raise ValueError("V31 arm audit pixels differ")
    return report


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: str = "#526a72",
    width: int = 4,
) -> None:
    draw.line((*start, *end), fill=color, width=width)
    x, y = end
    draw.polygon(((x, y), (x - 14, y - 9), (x - 14, y + 9)), fill=color)


def box(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    title: str,
    lines: tuple[str, ...],
    *,
    accent: str,
) -> None:
    left, top, _, bottom = bounds
    draw.rounded_rectangle(bounds, radius=7, fill="white", outline="#adbdc2", width=2)
    draw.rectangle((left, top, left + 9, bottom), fill=accent)
    draw.text((left + 24, top + 14), title, font=font(17, bold=True), fill="#17343e")
    for index, line in enumerate(lines):
        draw.text(
            (left + 24, top + 49 + index * 25),
            line,
            font=font(13),
            fill="#53676e",
        )


def bar(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    label: str,
    value: float,
    maximum: float,
    color: str,
    value_text: str | None = None,
) -> None:
    draw.text((x, y), label, font=font(14), fill="#2b424a")
    bar_left = x + 255
    bar_width = width - 407
    draw.rectangle((bar_left, y + 2, bar_left + bar_width, y + 24), fill="#e1e8ea")
    filled = max(2, int(bar_width * max(0.0, value) / maximum))
    draw.rectangle((bar_left, y + 2, bar_left + filled, y + 24), fill=color)
    draw.text(
        (bar_left + bar_width + 12, y - 1),
        value_text if value_text is not None else f"{100 * value:.3f}%",
        font=font(13, bold=True),
        fill=color,
    )


def tile(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    label: str,
    value: str,
    detail: str,
    *,
    passed: bool,
) -> None:
    left, top, _, bottom = bounds
    fill = "#edf6f2" if passed else "#f8e9e7"
    color = "#19725d" if passed else "#a94747"
    draw.rounded_rectangle(bounds, radius=7, fill=fill, outline="#b6c3c7", width=2)
    draw.text((left + 14, top + 11), label, font=font(12, bold=True), fill="#435860")
    draw.text((left + 14, top + 40), value, font=font(21, bold=True), fill=color)
    draw.text((left + 14, bottom - 28), detail, font=font(11), fill="#5d7077")


def main() -> None:
    args = parse_args()
    report = load_and_validate(args)
    spatial = report["spatial"]
    global_control = report["global_control"]
    sn = spatial["natural"]
    sp = spatial["suffix4"]
    gn = global_control["natural"]
    gp = global_control["suffix4"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.sample_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.sample, args.sample_out)

    canvas = Image.new("RGB", (2400, 1800), "#f2f5f6")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (62, 32),
        "V31: conditional visual flow reacts to context, but does not learn language",
        font=font(34, bold=True),
        fill="#17323c",
    )
    draw.text(
        (64, 87),
        "Matched 18.74M models | 10,000 BF16 updates each | one RTX 4090 | 0.997 GiB peak | fixed development audit",
        font=font(18),
        fill="#586b73",
    )

    y = 136
    box(
        draw,
        (62, y, 360, y + 210),
        "Visual context",
        ("64 glyph images", "1 x 32 x 32 cells", "no strings or OCR"),
        accent="#277f8f",
    )
    box(
        draw,
        (420, y, 752, y + 210),
        "Causal QKV reader",
        ("frozen retina", "eight causal blocks", "visual predictive state"),
        accent="#a8753e",
    )
    box(
        draw,
        (812, y, 1172, y + 210),
        "Conditional field flow",
        ("Zt = (1-t)E + tY", "predict velocity Y-E", "coherent base noise"),
        accent="#46755d",
    )
    box(
        draw,
        (1232, y, 1558, y + 98),
        "Spatial arm",
        ("16 x 192 retinal field", "local geometry preserved"),
        accent="#277f8f",
    )
    box(
        draw,
        (1232, y + 112, 1558, y + 210),
        "Global control",
        ("one vector tiled 16x", "same parameters and data"),
        accent="#6f6597",
    )
    box(
        draw,
        (1630, y, 1965, y + 210),
        "Path and sample",
        ("eight fixed path probes", "eight-step Heun", "candidate-independent draw"),
        accent="#496f8e",
    )
    box(
        draw,
        (2030, y, 2338, y + 210),
        "Fixed audit",
        ("2,048 natural", "512 suffix pairs", "1,024 evaluator bank"),
        accent="#8b6253",
    )
    arrow(draw, (360, y + 105), (420, y + 105))
    arrow(draw, (752, y + 105), (812, y + 105))
    draw.line((1172, y + 105, 1200, y + 105), fill="#526a72", width=4)
    draw.line((1200, y + 49, 1200, y + 161), fill="#526a72", width=4)
    arrow(draw, (1200, y + 49), (1232, y + 49))
    arrow(draw, (1200, y + 161), (1232, y + 161))
    draw.line((1558, y + 49, 1594, y + 49), fill="#526a72", width=4)
    draw.line((1558, y + 161, 1594, y + 161), fill="#526a72", width=4)
    draw.line((1594, y + 49, 1594, y + 161), fill="#526a72", width=4)
    arrow(draw, (1594, y + 105), (1630, y + 105))
    arrow(draw, (1965, y + 105), (2030, y + 105))

    left = (62, 384, 1125, 1128)
    draw.rounded_rectangle(left, radius=8, fill="white", outline="#aebcc1", width=2)
    draw.text(
        (88, 408), "Natural continuation", font=font(24, bold=True), fill="#1a3741"
    )
    draw.text(
        (88, 447),
        "Top-1 over the evaluator-only 1,024-image bank",
        font=font(14),
        fill="#61737a",
    )
    natural = (
        ("Spatial path, full", sn["path_full_top1"], "#277f8f"),
        ("Global path, full", gn["path_full_top1"], "#6f6597"),
        ("Spatial autonomous", sn["sample_full_top1"], "#46755d"),
        ("Global autonomous", gn["sample_full_top1"], "#776d9f"),
        ("Image unigram", sn["unigram_top1"], "#b58c42"),
        ("Symbolic bigram", sn["bigram_top1"], "#a55252"),
        ("Symbolic trigram", sn["trigram_top1"], "#7b3f69"),
    )
    for index, (label, value, color) in enumerate(natural):
        bar(
            draw,
            x=90,
            y=495 + index * 48,
            width=990,
            label=label,
            value=float(value),
            maximum=0.22,
            color=color,
        )

    tile(
        draw,
        (88, 860, 396, 1018),
        "Full vs shuffle log p",
        f"{sn['path_full_target_log_probability'] - sn['path_shuffled_target_log_probability']:+.3f} nat",
        "order intervention passes",
        passed=True,
    )
    tile(
        draw,
        (412, 860, 720, 1018),
        "Full vs spatial perm.",
        f"{sn['path_full_target_log_probability'] - sn['path_spatial_permuted_target_log_probability']:+.3f} nat",
        "local-layout gate passes",
        passed=True,
    )
    tile(
        draw,
        (736, 860, 1042, 1018),
        "Natural path top-1",
        f"{100 * sn['path_full_top1']:.3f}%",
        "15% required",
        passed=False,
    )
    draw.text(
        (89, 1054),
        "The flow detects context changes but ranks the correct next glyph below even an image unigram.",
        font=font(15, bold=True),
        fill="#8d3838",
    )

    middle = (1162, 384, 1724, 1128)
    draw.rounded_rectangle(middle, radius=8, fill="white", outline="#aebcc1", width=2)
    draw.text(
        (1188, 408), "Exact-suffix binding", font=font(23, bold=True), fill="#1a3741"
    )
    draw.text(
        (1188, 447), "512 pixel-identical suffix pairs", font=font(14), fill="#61737a"
    )
    pair = (
        ("Spatial path", sp["path_full_arm_accuracy"], "#277f8f"),
        ("Global path", gp["path_full_arm_accuracy"], "#6f6597"),
        ("Spatial sample", sp["sample_full_arm_accuracy"], "#46755d"),
        ("Global sample", gp["sample_full_arm_accuracy"], "#776d9f"),
    )
    for index, (label, value, color) in enumerate(pair):
        bar(
            draw,
            x=1188,
            y=505 + index * 63,
            width=510,
            label=label,
            value=float(value),
            maximum=0.70,
            color=color,
        )
    draw.line((1300, 790, 1684, 790), fill="#d5dddf", width=2)
    draw.text((1188, 820), "Spatial path both correct", font=font(14), fill="#435860")
    draw.text(
        (1586, 818),
        f"{100 * sp['path_full_both_correct_rate']:.2f}%",
        font=font(15, bold=True),
        fill="#a94747",
    )
    draw.text((1188, 863), "Spatial sample both correct", font=font(14), fill="#435860")
    draw.text(
        (1586, 861),
        f"{100 * sp['sample_full_both_correct_rate']:.2f}%",
        font=font(15, bold=True),
        fill="#a94747",
    )
    draw.text((1188, 906), "Suffix row max error", font=font(14), fill="#435860")
    draw.text((1586, 904), "0.0", font=font(15, bold=True), fill="#19725d")
    draw.rounded_rectangle((1188, 964, 1698, 1088), radius=7, fill="#f8e9e7")
    draw.text((1210, 987), "BINDING REJECTED", font=font(18, bold=True), fill="#8d3838")
    draw.text((1210, 1025), "50.49% path", font=font(15), fill="#704b4b")
    draw.text((1210, 1052), "50.20% autonomous", font=font(15), fill="#704b4b")

    sample_panel = (1760, 384, 2338, 1128)
    draw.rounded_rectangle(
        sample_panel, radius=8, fill="white", outline="#aebcc1", width=2
    )
    draw.text(
        (1786, 408),
        "Autonomous latent-field diagnostic",
        font=font(21, bold=True),
        fill="#1a3741",
    )
    draw.text(
        (1786, 447),
        "Target | nearest to full | nearest to shuffle",
        font=font(13),
        fill="#61737a",
    )
    sample = Image.open(args.sample).convert("RGB")
    sample = ImageOps.crop(sample, border=(0, 0, 0, 0))
    sample.thumbnail((330, 575), Image.Resampling.LANCZOS)
    sample_x = 1884
    sample_y = 486
    canvas.paste(sample, (sample_x, sample_y))
    draw.rectangle(
        (sample_x - 1, sample_y - 1, sample_x + sample.width, sample_y + sample.height),
        outline="#8fa0a6",
        width=1,
    )
    draw.text(
        (1786, 1072),
        "Nearest bank images, not generated pixels",
        font=font(13, bold=True),
        fill="#8d3838",
    )

    gates = (62, 1165, 2338, 1480)
    draw.rounded_rectangle(gates, radius=8, fill="#172f39", outline="#172f39")
    draw.text(
        (88, 1191), "Preregistered decision", font=font(23, bold=True), fill="white"
    )
    tile(
        draw,
        (88, 1242, 430, 1395),
        "Spatial common",
        "14 / 19",
        "binding gates fail",
        passed=False,
    )
    tile(
        draw,
        (448, 1242, 790, 1395),
        "Control integrity",
        "6 / 6",
        "matched control valid",
        passed=True,
    )
    tile(
        draw,
        (808, 1242, 1150, 1395),
        "Matched arms",
        "4 / 8",
        "all capability gains fail",
        passed=False,
    )
    tile(
        draw,
        (1168, 1242, 1560, 1395),
        "Language + generation",
        "0 / 10",
        "no language gate passes",
        passed=False,
    )
    tile(
        draw,
        (1578, 1242, 1935, 1395),
        "Peak per arm",
        "0.997 GiB",
        "resource receipt only",
        passed=True,
    )
    tile(
        draw,
        (1953, 1242, 2310, 1395),
        "Finite updates",
        "10,000 x 2",
        "byte-identical start",
        passed=True,
    )
    draw.text(
        (88, 1430),
        "REJECTED | FROZEN IMAGES SEALED | DIRECT PIXEL WRITER NOT AUTHORIZED",
        font=font(18, bold=True),
        fill="#f0b7ad",
    )

    bottom = (62, 1516, 2338, 1744)
    draw.rounded_rectangle(bottom, radius=8, fill="#e8eef0", outline="#aebcc1", width=2)
    draw.text((88, 1542), "What changes next", font=font(21, bold=True), fill="#1a3741")
    draw.text(
        (88, 1585),
        "QKV attention must plan a multi-glyph visual answer; a separate continuous renderer must generate its pixels directly.",
        font=font(17, bold=True),
        fill="#41575f",
    )
    draw.text(
        (88, 1626),
        "Train on rendered prompt-to-answer blocks and page/line continuations; test held-out semantics, counterfactual binding, readability, and continuation.",
        font=font(16),
        fill="#41575f",
    )
    draw.text(
        (88, 1666),
        "Diffusion or rectified flow may render the answer, but it cannot substitute for the visual-language state or its causal tests.",
        font=font(16),
        fill="#41575f",
    )
    draw.text(
        (1998, 1710),
        "FIXED DEVELOPMENT EVIDENCE",
        font=font(13, bold=True),
        fill="#536970",
    )

    canvas.save(args.out, optimize=True)
    print(args.out)
    print(args.sample_out)


if __name__ == "__main__":
    main()
