#!/usr/bin/env python3
"""Compose the measured V25 development result from checked evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

from generate_v20_result_figure import font


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LANGUAGE = (
    ROOT
    / "artifacts/visual_cell_stream_v25_evidence"
    / "development_language.json"
)
DEFAULT_WRITER = (
    ROOT
    / "artifacts/visual_cell_stream_v25_exploratory_writer"
    / "development_writer.json"
)
DEFAULT_SAMPLE = (
    ROOT
    / "artifacts/visual_cell_stream_v25_exploratory_writer"
    / "writer_sample.png"
)
DEFAULT_OUT = (
    Path(__file__).resolve().parent
    / "figures/visual_cell_stream_v25_result.png"
)
EXPECTED_ARCHITECTURE = "visual-cell-stream-v25-development-audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the measured V25 figure.")
    parser.add_argument("--language-report", type=Path, default=DEFAULT_LANGUAGE)
    parser.add_argument("--writer-report", type=Path, default=DEFAULT_WRITER)
    parser.add_argument("--writer-sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_evidence(
    language_report: dict[str, object],
    writer_report: dict[str, object],
    sample: Image.Image,
) -> None:
    for name, report, stage in (
        ("language", language_report, "language"),
        ("writer", writer_report, "writer"),
    ):
        if report.get("architecture") != EXPECTED_ARCHITECTURE:
            raise ValueError(f"Unexpected V25 {name} report architecture")
        if report.get("checkpoint_stage") != stage:
            raise ValueError(f"Unexpected V25 {name} checkpoint stage")
        if report.get("checkpoint_smoke_only"):
            raise ValueError(f"V25 {name} report is smoke-only")
        if int(report.get("audit_windows", -1)) != 2_048:
            raise ValueError(f"V25 {name} report does not use 2,048 windows")
        if report.get("language_selected"):
            raise ValueError(f"V25 {name} report unexpectedly selects language")
        if int(report.get("frozen_images_instantiated", -1)) != 0:
            raise ValueError(f"V25 {name} report opened frozen images")

    if language_report.get("writer") is not None:
        raise ValueError("Fixed V25 evidence report unexpectedly evaluated writer")
    if writer_report.get("writer_selected"):
        raise ValueError("Exploratory V25 writer unexpectedly selected")
    if writer_report.get("writer_trained_after_language_selected") is not False:
        raise ValueError("V25 writer report lost its exploratory provenance")
    if language_report.get("protocol_sha256") != writer_report.get("protocol_sha256"):
        raise ValueError("V25 reports disagree on protocol SHA-256")

    evidence_metrics = language_report.get("language")
    writer_language = writer_report.get("language")
    if not isinstance(evidence_metrics, dict) or not isinstance(writer_language, dict):
        raise ValueError("V25 reports contain no language metrics")
    for key in (
        "full_top1",
        "last_top1",
        "shuffled_top1",
        "unigram_top1",
        "bigram_top1",
        "counterfactual_switch_accuracy",
    ):
        if not math.isclose(
            float(evidence_metrics[key]),
            float(writer_language[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"V25 writer rerun changed language metric {key}")

    writer = writer_report.get("writer")
    if not isinstance(writer, dict) or int(writer.get("examples", -1)) != 256:
        raise ValueError("V25 writer report does not contain 256 examples")
    if int(writer.get("autonomous_examples", -1)) != 16:
        raise ValueError("V25 writer report does not contain 16 rollouts")
    if sample.size != (527, 131):
        raise ValueError(f"Unexpected V25 writer sample size: {sample.size}")


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: str = "#506872",
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
    draw.text((x, y), label, font=font(17), fill="#2b424a")
    bar_left = x + 225
    bar_width = width - 360
    draw.rectangle((bar_left, y + 2, bar_left + bar_width, y + 24), fill="#e0e7e9")
    draw.rectangle(
        (
            bar_left,
            y + 2,
            bar_left + max(3, int(bar_width * value / maximum)),
            y + 24,
        ),
        fill=color,
    )
    draw.text(
        (bar_left + bar_width + 14, y - 1),
        f"{100.0 * value:5.2f}%",
        font=font(16, bold=True),
        fill=color,
    )


def metric_tile(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    label: str,
    value: str,
    result: str,
    passed: bool,
) -> None:
    color = "#14745d" if passed else "#a94747"
    draw.rounded_rectangle(
        (x, y, x + width, y + 94),
        radius=7,
        fill="white",
        outline="#b5c2c7",
        width=2,
    )
    draw.text((x + 16, y + 13), label, font=font(14, bold=True), fill="#344b53")
    draw.text((x + 16, y + 43), value, font=font(23, bold=True), fill=color)
    draw.text((x + width - 70, y + 57), result, font=font(13, bold=True), fill=color)


def main() -> None:
    args = parse_args()
    language_report = read_json(args.language_report)
    writer_report = read_json(args.writer_report)
    sample = Image.open(args.writer_sample).convert("RGB")
    validate_evidence(language_report, writer_report, sample)

    language = language_report["language"]
    writer = writer_report["writer"]

    canvas = Image.new("RGB", (2400, 1700), "#f2f5f6")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (62, 34),
        "V25: natural Chinese visual history is measurable, but the model is rejected",
        font=font(41, bold=True),
        fill="#17323c",
    )
    draw.text(
        (64, 94),
        "64 visible 32 x 32 cells | 25.55M total parameters | 2,400 fixed language updates | one RTX 4090 | frozen split sealed",
        font=font(20),
        fill="#586b73",
    )

    architecture_y = 148
    architecture_box(
        draw,
        (62, architecture_y, 380, architecture_y + 190),
        "Visual-time stream",
        ("64 clean image cells", "B x T x 1 x 32 x 32", "no token or Unicode ID"),
        accent="#227c8c",
    )
    architecture_box(
        draw,
        (438, architecture_y, 716, architecture_y + 190),
        "Frozen retina",
        ("pixels -> unit state", "cross-font visual view", "192 continuous values"),
        accent="#687d9b",
    )
    architecture_box(
        draw,
        (774, architecture_y, 1114, architecture_y + 190),
        "Causal visual field",
        ("8 layers, width 384", "ordered history", "15.81M trained params"),
        accent="#7b6ca8",
    )
    architecture_box(
        draw,
        (1172, architecture_y, 1487, architecture_y + 190),
        "Next visual state",
        ("continuous proposal", "evaluator compares images", "no output vocabulary"),
        accent="#9b7654",
    )
    architecture_box(
        draw,
        (1545, architecture_y, 1853, architecture_y + 190),
        "Flow writer",
        ("noise -> 32 x 32 ink", "7.07M writer params", "exploratory after fail"),
        accent="#ad654d",
    )
    architecture_box(
        draw,
        (1911, architecture_y, 2338, architecture_y + 190),
        "Generated-pixel rereading",
        ("append actual output image", "reread before next cell", "stack or fold exactly"),
        accent="#2c806a",
    )
    for start, end in (
        ((380, 243), (438, 243)),
        ((716, 243), (774, 243)),
        ((1114, 243), (1172, 243)),
        ((1487, 243), (1545, 243)),
        ((1853, 243), (1911, 243)),
    ):
        arrow(draw, start, end)
    draw.text(
        (63, 355),
        "Exact long-context view: N x 1 x 32 x 32 visual stream <-> serpentine C x rows x columns retinal lattice; the lattice was not used to obtain V25 metrics.",
        font=font(16),
        fill="#53666d",
    )

    left = (62, 414, 1125, 1120)
    draw.rounded_rectangle(left, radius=8, fill="white", outline="#aebcc1", width=2)
    draw.text((88, 438), "Fixed language audit: next-cell top-1", font=font(25, bold=True), fill="#1a3741")
    draw.text((88, 477), "2,048 development windows; all controls share the same 1,024-form evaluator bank", font=font(15), fill="#61737a")
    bar_values = (
        ("Full 64-cell history", float(language["full_top1"]), "#287f95"),
        ("Last visible cell", float(language["last_top1"]), "#89989e"),
        ("Shuffled prior history", float(language["shuffled_top1"]), "#89989e"),
        ("Image unigram", float(language["unigram_top1"]), "#b07a3e"),
        ("Symbolic bigram", float(language["bigram_top1"]), "#754d86"),
    )
    for index, (label, value, color) in enumerate(bar_values):
        metric_bar(
            draw,
            x=90,
            y=530 + index * 60,
            width=990,
            label=label,
            value=value,
            maximum=0.13,
            color=color,
        )
    draw.line((88, 852, 1098, 852), fill="#d1dadd", width=2)
    draw.text((90, 878), "Ordered-history gain over last-only", font=font(16), fill="#354c54")
    draw.text((785, 874), "+0.98 pp", font=font(20, bold=True), fill="#287f95")
    draw.text((90, 920), "Required full-history gain", font=font(16), fill="#354c54")
    draw.text((785, 916), ">3.00 pp", font=font(20, bold=True), fill="#a94747")
    draw.text((90, 962), "Correct counterfactual switches", font=font(16), fill="#354c54")
    draw.text((785, 958), "12.89%", font=font(20, bold=True), fill="#a94747")
    draw.text((90, 1004), "Target cosine", font=font(16), fill="#354c54")
    draw.text((785, 1000), "0.275 / >0.550", font=font(20, bold=True), fill="#a94747")
    draw.rounded_rectangle((87, 1052, 1100, 1097), radius=6, fill="#f8e7e5")
    draw.text((106, 1062), "LANGUAGE REJECTED: six fixed semantic/causal gates fail; no frozen evaluation.", font=font(17, bold=True), fill="#8d3838")

    right = (1165, 414, 2338, 1120)
    draw.rounded_rectangle(right, radius=8, fill="white", outline="#aebcc1", width=2)
    draw.text((1191, 438), "Exploratory writer diagnostic: actual output", font=font(25, bold=True), fill="#1a3741")
    draw.text((1191, 477), "This stage was run only after language rejection and cannot change the evidence verdict.", font=font(15), fill="#61737a")
    scaled_sample = sample.resize((1058, 262), Image.Resampling.NEAREST)
    draw.rectangle((1219, 528, 2291, 804), fill="#eef2f3", outline="#b2c0c5", width=2)
    canvas.paste(scaled_sample, (1226, 535))
    labels = (
        "row 1: observed 16-cell context",
        "row 2: held-out target",
        "row 3: sampled next cell",
        "row 4: autonomous 16-cell continuation",
    )
    for index, label in enumerate(labels):
        draw.text((1200 + (index % 2) * 560, 821 + (index // 2) * 27), label, font=font(14), fill="#53666d")

    metric_tile(draw, x=1193, y=900, width=256, label="Identity top-1", value="0.000", result="fail", passed=False)
    metric_tile(draw, x=1468, y=900, width=256, label="Reread cosine", value="0.080", result="fail", passed=False)
    metric_tile(draw, x=1743, y=900, width=256, label="Pixel F1", value="0.322", result="fail", passed=False)
    metric_tile(draw, x=2018, y=900, width=292, label="Position-16 density", value="0.977 x", result="pass", passed=True)
    draw.rounded_rectangle((1192, 1021, 2310, 1097), radius=6, fill="#f8e7e5")
    draw.text((1212, 1033), "WRITER REJECTED: ink stays nonblank, but generated images do not encode the intended next form.", font=font(16, bold=True), fill="#8d3838")
    draw.text((1212, 1063), "Actual output is glyph-like texture, not readable Chinese continuation.", font=font(15), fill="#8d3838")

    bottom = (62, 1160, 2338, 1634)
    draw.rounded_rectangle(bottom, radius=8, fill="#172f39", outline="#172f39")
    draw.text((92, 1190), "What V25 establishes", font=font(24, bold=True), fill="white")
    draw.text((92, 1234), "A compact image-only causal field can detect a weak ordered-writing dependency.", font=font(18), fill="#d9e4e7")
    draw.text((92, 1270), "Its 0.598 GiB peak memory is a resource result, not proof of language efficiency.", font=font(18), fill="#d9e4e7")
    draw.text((92, 1330), "What V25 falsifies", font=font(24, bold=True), fill="white")
    draw.text((92, 1374), "Next-state cosine plus in-batch visual contrast does not bind 64-cell context to the correct next glyph.", font=font(18), fill="#d9e4e7")
    draw.text((92, 1410), "Stable ink density and frequent output changes are not semantic continuation.", font=font(18), fill="#d9e4e7")
    draw.text((92, 1470), "Next controlled hypothesis", font=font(24, bold=True), fill="white")
    draw.text((92, 1514), "Separate exact appearance from a context-predictive residual; require both through state-shuffle controls.", font=font(18), fill="#d9e4e7")
    draw.text((92, 1550), "Repair the 64-cell objective before spending compute on the 65,536-cell serpentine lattice.", font=font(18), fill="#d9e4e7")
    draw.text((1810, 1590), "DEVELOPMENT RESULT", font=font(15, bold=True), fill="#f0b7ad")
    draw.text((2050, 1590), "FROZEN SEALED", font=font(15, bold=True), fill="#a9d7c9")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print(args.out)


if __name__ == "__main__":
    main()
