#!/usr/bin/env python3
"""Compose the measured V21 development result figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from generate_v20_result_figure import font, gate_row


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE = ROOT / "artifacts/field_complete_writer_v21_field_evidence_20260813"
DEFAULT_CONTROL = ROOT / "artifacts/field_complete_writer_v21_control_evidence_20260813"
DEFAULT_OUT = (
    Path(__file__).resolve().parent / "figures/field_complete_writer_v21_result.png"
)
PARAMETERS_PER_ARM = 582_336


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the measured V21 figure.")
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_reports(root: Path) -> tuple[list[dict[str, float]], dict[str, object]]:
    path = root / "training.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    validations: list[dict[str, float]] = []
    complete: dict[str, object] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("stage") == "validation":
            validations.append(record)
        elif record.get("stage") == "complete":
            complete = record
    if not validations or complete is None:
        raise ValueError(f"Missing V21 reports in {path}")
    return validations, complete


def ranked_best(reports: list[dict[str, float]]) -> dict[str, float]:
    return max(
        reports,
        key=lambda row: (
            row["correct_pixel_f1"],
            row["correct_pixel_f1_dense"],
            -row["step"],
        ),
    )


def sample_rows(
    root: Path,
    step: int,
    row_indices: tuple[int, ...],
    *,
    sample_count: int = 6,
) -> Image.Image:
    path = root / f"development_samples/step_{step:07d}.png"
    source = Image.open(path).convert("RGB")
    row_height = 112
    label_width = 190
    margin = 16
    expected_height = 9 * row_height + 2 * margin
    if source.height != expected_height:
        raise ValueError(f"Unexpected V21 grid height: {source.size}")
    crop_width = margin + label_width + sample_count * row_height + margin
    rows = [
        source.crop(
            (
                0,
                margin + row * row_height,
                crop_width,
                margin + (row + 1) * row_height,
            )
        )
        for row in row_indices
    ]
    strip = Image.new("RGB", (crop_width, len(rows) * row_height), "white")
    for index, row in enumerate(rows):
        strip.paste(row, (0, index * row_height))
    return strip


def main() -> None:
    args = parse_args()
    candidate_reports, candidate_complete = read_reports(args.candidate)
    control_reports, control_complete = read_reports(args.control)
    candidate = ranked_best(candidate_reports)
    control = ranked_best(control_reports)
    if candidate.get("route_mode") != "field_complete":
        raise ValueError("Candidate evidence must use the field-complete route")
    if control.get("route_mode") != "tiled_global_control":
        raise ValueError("Control evidence must use the tiled-global route")
    if candidate_complete.get("step") != 1600 or control_complete.get("step") != 1600:
        raise ValueError("V21 result requires both fixed 1,600-update runs")
    if candidate_complete.get("best_development") is not None:
        raise ValueError("V21 candidate unexpectedly selected a checkpoint")
    if control_complete.get("best_development") is None:
        raise ValueError("V21 control must select its structural checkpoint")
    if (
        candidate.get("frozen_images_instantiated") != 0.0
        or control.get("frozen_images_instantiated") != 0.0
    ):
        raise ValueError("V21 result requires the sealed frozen partition")

    dense_shuffle_gain = (
        candidate["correct_pixel_f1_dense"]
        - candidate["field_shuffled_pixel_f1_dense"]
    )
    dense_zero_gain = (
        candidate["correct_pixel_f1_dense"] - candidate["zero_field_pixel_f1_dense"]
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
        "V21: a complete local field writes Chinese structure, broad writer gate still fails",
        font=font(38, bold=True),
        fill="#17313b",
    )
    draw.text(
        (64, 92),
        "One RTX 4090 | 582,336 parameters per arm | image-only student | frozen split sealed",
        font=font(21),
        fill="#586a72",
    )

    draw.rectangle((62, 140, 2138, 260), fill="#e9eef0", outline="#a7b5bb", width=2)
    draw.text((88, 159), "LOCAL F[i,j]", font=font(18, bold=True), fill="#136d70")
    draw.text(
        (245, 159),
        "-> shared pointwise writer -> coarse scalar + 63 zero-DC basis coefficients -> complete 8x8 patch",
        font=font(18),
        fill="#344950",
    )
    draw.text(
        (88, 207),
        "Global z + style provide one spatially uniform modulation. No coordinate input or cell mixing can draw around the local field.",
        font=font(17),
        fill="#53666e",
    )

    left_x, top_y = 62, 296
    draw.text(
        (left_x, top_y),
        f"Field-complete candidate: best diagnostic step {int(candidate['step'])}",
        font=font(25, bold=True),
        fill="#17313b",
    )
    candidate_rows = sample_rows(
        args.candidate,
        int(candidate["step"]),
        (0, 1, 3, 4, 5, 8),
    )
    draw.rectangle(
        (left_x - 2, top_y + 42, left_x + candidate_rows.width + 2, top_y + 718),
        fill="white",
        outline="#aebbc0",
        width=2,
    )
    canvas.paste(candidate_rows, (left_x, top_y + 44))

    control_y = top_y + 760
    draw.text(
        (left_x, control_y),
        f"Tiled-global control: selected structural step {int(control['step'])}",
        font=font(23, bold=True),
        fill="#17313b",
    )
    control_rows = sample_rows(args.control, int(control["step"]), (0, 1, 3))
    draw.rectangle(
        (left_x - 2, control_y + 40, left_x + control_rows.width + 2, control_y + 380),
        fill="white",
        outline="#aebbc0",
        width=2,
    )
    canvas.paste(control_rows, (left_x, control_y + 42))

    panel_x, panel_y, panel_w, panel_h = 1010, 296, 1128, 1114
    draw.rectangle(
        (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h),
        fill="white",
        outline="#b4c0c5",
        width=2,
    )
    draw.text(
        (panel_x + 28, panel_y + 22),
        "Best 512-candidate development snapshot",
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
        ("simple pixel F1", f"{candidate['correct_pixel_f1_simple']:.3f}", "> 0.580", False),
        ("medium pixel F1", f"{candidate['correct_pixel_f1_medium']:.3f}", "> 0.600", False),
        ("dense pixel F1", f"{candidate['correct_pixel_f1_dense']:.3f}", "> 0.700", True),
        ("dense gain: shuffled F", f"+{dense_shuffle_gain:.3f}", "> 0.150", True),
        ("dense gain: zero F", f"+{dense_zero_gain:.3f}", "> 0.200", True),
        ("identity top-1", f"{candidate['correct_identity_top1']:.3f}", "> 0.740", True),
        ("target cosine", f"{candidate['correct_target_cosine']:.3f}", "> 0.820", True),
        ("occlusion locality", f"{candidate['occlusion_locality']:.3f}", "> 0.950", True),
        ("detail block mean", f"{candidate['detail_block_mean_abs_max']:.2e}", "< 5e-6", True),
    )
    for index, (label, value, target, passed) in enumerate(rows):
        gate_row(
            draw,
            x=panel_x + 28,
            y=panel_y + 91 + index * 50,
            width=panel_w - 56,
            label=label,
            value=value,
            target=target,
            passed=passed,
        )

    compare_y = panel_y + 610
    draw.rectangle(
        (panel_x + 26, compare_y, panel_x + panel_w - 26, compare_y + 190),
        fill="#eef7f4",
        outline="#71a99b",
        width=2,
    )
    draw.text(
        (panel_x + 48, compare_y + 18),
        "Descriptive equal-capacity comparison",
        font=font(22, bold=True),
        fill="#185e53",
    )
    draw.text(
        (panel_x + 48, compare_y + 62),
        f"Overall F1: {candidate['correct_pixel_f1']:.3f} vs {control['correct_pixel_f1']:.3f} ({overall_control_gain:+.3f})",
        font=font(18),
        fill="#3f5057",
    )
    draw.text(
        (panel_x + 48, compare_y + 98),
        f"Dense F1: {candidate['correct_pixel_f1_dense']:.3f} vs {control['correct_pixel_f1_dense']:.3f} ({dense_control_gain:+.3f})",
        font=font(18),
        fill="#3f5057",
    )
    draw.text(
        (panel_x + 48, compare_y + 136),
        "Not a formal paired audit: the candidate did not select.",
        font=font(17, bold=True),
        fill="#8a4b23",
    )

    verdict_y = compare_y + 220
    draw.rectangle(
        (panel_x + 26, verdict_y, panel_x + panel_w - 26, verdict_y + 268),
        fill="#fff1ef",
        outline="#d18e89",
        width=2,
    )
    draw.text(
        (panel_x + 48, verdict_y + 18),
        "WRITER REJECTED; FIELD-COMPLETE CAUSAL ROUTE SUPPORTED",
        font=font(20, bold=True),
        fill="#a53131",
    )
    verdict_lines = (
        "Local state carries coarse occupancy and fine detail.",
        "The exact basis and every intervention invariant pass.",
        "The matched global-only source collapses to repeated patches.",
        "Simple and medium fidelity remain below fixed thresholds.",
        "No paired audit, human review, or frozen query is permitted.",
        "Next: improve local raster continuity, then predict visual answer streams.",
    )
    for index, line in enumerate(verdict_lines):
        y = verdict_y + 62 + index * 32
        draw.ellipse((panel_x + 51, y + 8, panel_x + 61, y + 18), fill="#17706b")
        draw.text((panel_x + 74, y), line, font=font(16), fill="#3d5058")

    draw.text(
        (62, 1440),
        "DEVELOPMENT ONLY | candidate 327.04 s / 0.325 GiB | control 333.41 s / 0.400 GiB | frozen images: 0",
        font=font(18, bold=True),
        fill="#65757c",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print(args.out)


if __name__ == "__main__":
    main()
