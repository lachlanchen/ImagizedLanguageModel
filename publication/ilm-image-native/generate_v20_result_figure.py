#!/usr/bin/env python3
"""Compose the measured V20 development result figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE = (
    ROOT / "artifacts/retinal_topology_router_v20_field_evidence_20260812"
)
DEFAULT_CONTROL = (
    ROOT / "artifacts/retinal_topology_router_v20_control_evidence_20260812"
)
DEFAULT_OUT = (
    Path(__file__).resolve().parent
    / "figures/retinal_topology_router_v20_result.png"
)
PARAMETERS_PER_ARM = 506_448


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the measured V20 figure.")
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
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


def read_reports(root: Path) -> tuple[dict[str, float], dict[str, object]]:
    path = root / "training.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    validation: dict[str, float] | None = None
    complete: dict[str, object] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("stage") == "validation":
            validation = record
        elif record.get("stage") == "complete":
            complete = record
    if validation is None or complete is None:
        raise ValueError(f"Missing final reports in {path}")
    return validation, complete


def sample_rows(root: Path, row_indices: tuple[int, ...]) -> Image.Image:
    path = root / "development_samples/step_0001600.png"
    source = Image.open(path).convert("RGB")
    if source.size != (1118, 1040):
        raise ValueError(f"Unexpected V20 grid size: {source.size}")
    row_height = 112
    sample_count = 6
    crop_width = 16 + 190 + sample_count * row_height + 16
    rows = [
        source.crop((0, 16 + row * row_height, crop_width, 16 + (row + 1) * row_height))
        for row in row_indices
    ]
    strip = Image.new("RGB", (crop_width, len(rows) * row_height), "white")
    for index, row in enumerate(rows):
        strip.paste(row, (0, index * row_height))
    return strip


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
    color = "#11735a" if passed else "#b33b3b"
    draw.line((x, y + 49, x + width, y + 49), fill="#dce3e6", width=1)
    draw.text((x, y + 10), label, font=font(17, bold=True), fill="#20353e")
    draw.text((x + 360, y + 9), value, font=font(19, bold=True), fill=color)
    draw.text((x + 505, y + 12), target, font=font(15), fill="#64747b")
    draw.text(
        (x + width - 58, y + 10),
        "PASS" if passed else "FAIL",
        font=font(15, bold=True),
        fill=color,
    )


def main() -> None:
    args = parse_args()
    candidate, candidate_complete = read_reports(args.candidate)
    control, control_complete = read_reports(args.control)
    if candidate.get("route_mode") != "field":
        raise ValueError("Candidate evidence must use the field route")
    if control.get("route_mode") != "global_control":
        raise ValueError("Control evidence must use the global-control route")
    for report, complete in (
        (candidate, candidate_complete),
        (control, control_complete),
    ):
        if report.get("step") != 1600 or complete.get("step") != 1600:
            raise ValueError("V20 result requires the fixed 1,600-update endpoint")
        if report.get("frozen_images_instantiated") != 0.0:
            raise ValueError("V20 result requires the sealed frozen partition")
        if complete.get("best_development") is not None:
            raise ValueError("This figure records two unselected V20 arms")

    dense_shuffle_gain = (
        candidate["correct_pixel_f1_dense"]
        - candidate["field_shuffled_pixel_f1_dense"]
    )
    dense_zero_gain = (
        candidate["correct_pixel_f1_dense"]
        - candidate["zero_field_pixel_f1_dense"]
    )
    dense_control_gain = (
        candidate["correct_pixel_f1_dense"] - control["correct_pixel_f1_dense"]
    )
    overall_control_gain = (
        candidate["correct_pixel_f1"] - control["correct_pixel_f1"]
    )

    canvas = Image.new("RGB", (2200, 1500), "#f5f8f9")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (62, 34),
        "V20: local retinal topology becomes necessary, writer gate still fails",
        font=font(42, bold=True),
        fill="#17313b",
    )
    draw.text(
        (64, 94),
        "One RTX 4090 | 506,448 parameters per arm | fixed development protocol | frozen split sealed",
        font=font(21),
        fill="#586a72",
    )

    draw.rectangle((62, 145, 2138, 245), fill="#e9eef0", outline="#a7b5bb", width=2)
    draw.text((88, 167), "GLOBAL z", font=font(18, bold=True), fill="#8b4a21")
    draw.text((205, 167), "-> coarse 4x4 block logits", font=font(18), fill="#344950")
    draw.text((655, 167), "+", font=font(27, bold=True), fill="#344950")
    draw.text((710, 167), "LOCAL F", font=font(18, bold=True), fill="#136d70")
    draw.text(
        (825, 167),
        "-> pointwise 8x8 zero-mean detail per cell",
        font=font(18),
        fill="#344950",
    )
    draw.text((1450, 167), "->", font=font(24, bold=True), fill="#344950")
    draw.text((1500, 167), "continuous 32x32 ink", font=font(19, bold=True), fill="#17313b")
    draw.text(
        (88, 210),
        "Exact-capacity control repeats global z across the local grid; only information routing changes.",
        font=font(17),
        fill="#53666e",
    )

    left_x, top_y = 62, 282
    draw.text(
        (left_x, top_y),
        "Field candidate: measured interventions",
        font=font(26, bold=True),
        fill="#17313b",
    )
    candidate_rows = sample_rows(args.candidate, (0, 3, 4, 5, 8))
    draw.rectangle(
        (left_x - 2, top_y + 43, left_x + candidate_rows.width + 2, top_y + 607),
        fill="white",
        outline="#aebbc0",
        width=2,
    )
    canvas.paste(candidate_rows, (left_x, top_y + 45))

    control_y = top_y + 650
    draw.text(
        (left_x, control_y),
        "Matched global-repeat control: field edits have no effect",
        font=font(23, bold=True),
        fill="#17313b",
    )
    control_rows = sample_rows(args.control, (3, 4, 5))
    draw.rectangle(
        (
            left_x - 2,
            control_y + 41,
            left_x + control_rows.width + 2,
            control_y + 381,
        ),
        fill="white",
        outline="#aebbc0",
        width=2,
    )
    canvas.paste(control_rows, (left_x, control_y + 43))

    panel_x, panel_y, panel_w, panel_h = 1010, 282, 1128, 1048
    draw.rectangle(
        (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h),
        fill="white",
        outline="#b4c0c5",
        width=2,
    )
    draw.text(
        (panel_x + 28, panel_y + 22),
        "Final 512-candidate development endpoint",
        font=font(27, bold=True),
        fill="#17313b",
    )
    draw.text(
        (panel_x + 28, panel_y + 63),
        "candidate value       fixed requirement",
        font=font(16),
        fill="#748188",
    )

    rows = (
        ("overall pixel F1", f"{candidate['correct_pixel_f1']:.3f}", "> 0.660", False),
        ("dense pixel F1", f"{candidate['correct_pixel_f1_dense']:.3f}", "> 0.680", True),
        ("dense gain: shuffled F", f"+{dense_shuffle_gain:.3f}", "> 0.120", dense_shuffle_gain > 0.12),
        ("dense gain: zero F", f"+{dense_zero_gain:.3f}", "> 0.100", dense_zero_gain > 0.10),
        ("identity top-1", f"{candidate['correct_identity_top1']:.3f}", "> 0.700", True),
        ("target cosine", f"{candidate['correct_target_cosine']:.3f}", "> 0.820", False),
        ("occlusion locality", f"{candidate['occlusion_locality']:.3f}", "> 0.400", True),
        ("detail block mean", f"{candidate['detail_block_mean_abs_max']:.2e}", "< 1e-6", False),
    )
    for index, (label, value, target, passed) in enumerate(rows):
        gate_row(
            draw,
            x=panel_x + 28,
            y=panel_y + 91 + index * 53,
            width=panel_w - 56,
            label=label,
            value=value,
            target=target,
            passed=passed,
        )

    paired_y = panel_y + 545
    draw.rectangle(
        (panel_x + 26, paired_y, panel_x + panel_w - 26, paired_y + 184),
        fill="#fff7e6",
        outline="#d2ac57",
        width=2,
    )
    draw.text(
        (panel_x + 48, paired_y + 18),
        "Equal-capacity endpoint comparison",
        font=font(22, bold=True),
        fill="#72511b",
    )
    draw.text(
        (panel_x + 48, paired_y + 60),
        f"Dense F1: candidate {candidate['correct_pixel_f1_dense']:.3f} | control {control['correct_pixel_f1_dense']:.3f} | gain +{dense_control_gain:.3f}",
        font=font(18),
        fill="#3f5057",
    )
    draw.text(
        (panel_x + 48, paired_y + 96),
        "Fixed paired requirement: > +0.030  |  FAIL",
        font=font(18, bold=True),
        fill="#b33b3b",
    )
    draw.text(
        (panel_x + 48, paired_y + 132),
        f"Overall candidate-control gain: {overall_control_gain:+.3f} | parameters exactly equal",
        font=font(18),
        fill="#3f5057",
    )

    verdict_y = paired_y + 215
    draw.rectangle(
        (panel_x + 26, verdict_y, panel_x + panel_w - 26, verdict_y + 250),
        fill="#ffefee",
        outline="#d18e89",
        width=2,
    )
    draw.text(
        (panel_x + 48, verdict_y + 18),
        "REJECTED AS A WRITER; ACCEPTED AS CAUSAL EVIDENCE",
        font=font(21, bold=True),
        fill="#a53131",
    )
    verdict_lines = (
        "The field now carries necessary high-frequency structure.",
        "Local occlusion changes only the matching output quadrant.",
        "Quality, reread similarity, and exact decomposition still fail.",
        "Neither arm selected; human and frozen evaluation stay closed.",
        "Next: let the field drive coarse occupancy and detail, with z as FiLM only.",
    )
    for index, line in enumerate(verdict_lines):
        y = verdict_y + 62 + index * 35
        draw.ellipse((panel_x + 51, y + 8, panel_x + 61, y + 18), fill="#17706b")
        draw.text((panel_x + 74, y), line, font=font(17), fill="#3d5058")

    draw.text(
        (62, 1392),
        "DEVELOPMENT-ONLY RESULT | candidate 325.90 s, 0.323 GiB peak | control 327.11 s, 0.397 GiB peak",
        font=font(18, bold=True),
        fill="#65757c",
    )
    draw.text(
        (62, 1434),
        "Measured advance: topology is routed through a local continuous field. Remaining proof: readable field-complete writing, then causal visual language rollout.",
        font=font(20, bold=True),
        fill="#17313b",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print(args.out)


if __name__ == "__main__":
    main()
