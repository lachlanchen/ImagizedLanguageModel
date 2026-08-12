#!/usr/bin/env python3
"""Compose the measured V22 visual-binding development result figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from generate_v20_result_figure import font, gate_row


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE = (
    ROOT / "artifacts/visual_binding_stream_v22_candidate_20260813"
)
DEFAULT_CONTROL = ROOT / "artifacts/visual_binding_stream_v22_control_20260813"
DEFAULT_AUDIT = (
    ROOT
    / "artifacts/visual_binding_stream_v22_attention_audit_20260813"
    / "attention_roles.json"
)
DEFAULT_OUT = (
    Path(__file__).resolve().parent
    / "figures/visual_binding_stream_v22_result.png"
)
PARAMETERS_PER_ARM = 3_410_128
ROLE_LABELS = (
    "label 1",
    "glyph 1",
    "label 2",
    "glyph 2",
    "operation",
    "query label",
)
ROLE_KEYS = (
    "label_1",
    "glyph_1",
    "label_2",
    "glyph_2",
    "operation",
    "query_label",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the measured V22 figure.")
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--attention-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_reports(
    root: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    path = root / "training.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    validations: list[dict[str, object]] = []
    complete: dict[str, object] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("stage") == "validation":
            validations.append(record)
        elif record.get("stage") == "complete":
            complete = record
    if len(validations) != 8 or complete is None:
        raise ValueError(f"Incomplete V22 report in {path}")
    return validations, complete


def sample_pair(root: Path) -> Image.Image:
    path = root / "development_samples/step_0001600.png"
    source = Image.open(path).convert("RGB")
    if source.size != (1584, 508):
        raise ValueError(f"Unexpected V22 sample sheet size: {source.size}")
    first = source.crop((8, 8, 392, 246))
    second = source.crop((400, 8, 784, 246))
    strip = Image.new("RGB", (768, 238), "white")
    strip.paste(first, (0, 0))
    strip.paste(second, (384, 0))
    return strip


def draw_attention_roles(
    draw: ImageDraw.ImageDraw,
    audit: dict[str, object],
    *,
    x: int,
    y: int,
    width: int,
) -> None:
    candidate = audit["candidate"]
    means = candidate["mean_attention"]
    counts = candidate["argmax_counts"]
    gap = 12
    cell_width = (width - gap * 5) // 6
    for index, (key, label) in enumerate(zip(ROLE_KEYS, ROLE_LABELS)):
        left = x + index * (cell_width + gap)
        value = float(means[key])
        selected = key == "operation"
        fill = "#f9e7e5" if selected else "#edf2f4"
        outline = "#b44238" if selected else "#9eafb6"
        draw.rounded_rectangle(
            (left, y, left + cell_width, y + 128),
            radius=5,
            fill=fill,
            outline=outline,
            width=3 if selected else 1,
        )
        draw.text(
            (left + 12, y + 12),
            f"frame {index + 1}",
            font=font(14, bold=True),
            fill="#66777e",
        )
        draw.text(
            (left + 12, y + 42),
            label,
            font=font(17, bold=True),
            fill="#213840",
        )
        display = f"{value:.1%}" if value >= 0.001 else f"{value:.1e}"
        draw.text(
            (left + 12, y + 75),
            f"mean {display}",
            font=font(15),
            fill="#a23531" if selected else "#5c6f76",
        )
        draw.text(
            (left + 12, y + 99),
            f"argmax {int(counts[key])}/1024",
            font=font(14, bold=selected),
            fill="#a23531" if selected else "#66777e",
        )


def draw_trajectory(
    draw: ImageDraw.ImageDraw,
    reports: list[dict[str, object]],
    *,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    draw.rectangle((x, y, x + width, y + height), fill="white", outline="#aebbc0")
    draw.text(
        (x + 22, y + 15),
        "Candidate development trajectory",
        font=font(21, bold=True),
        fill="#1d363f",
    )
    left, top = x + 62, y + 62
    plot_width, plot_height = width - 92, height - 112
    for tick in range(6):
        value = tick / 5
        yy = top + plot_height - int(value * plot_height)
        draw.line((left, yy, left + plot_width, yy), fill="#e1e7e9", width=1)
        draw.text((x + 14, yy - 8), f"{value:.1f}", font=font(12), fill="#6c7c82")
    draw.line((left, top, left, top + plot_height), fill="#82939a", width=2)
    draw.line(
        (left, top + plot_height, left + plot_width, top + plot_height),
        fill="#82939a",
        width=2,
    )
    series = (
        ("counterfactual switch", "counterfactual_switch_accuracy", "#b64037"),
        ("identity top-1", "identity_top1", "#15728a"),
        ("oracle writer F1", "oracle_pixel_f1", "#2b7a58"),
    )
    for label_index, (label, key, color) in enumerate(series):
        points = []
        for index, report in enumerate(reports):
            px = left + int(index * plot_width / (len(reports) - 1))
            py = top + plot_height - int(float(report[key]) * plot_height)
            points.append((px, py))
        draw.line(points, fill=color, width=4)
        for point in points:
            draw.ellipse(
                (point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4),
                fill=color,
            )
        legend_x = left + label_index * 245
        draw.line((legend_x, y + height - 24, legend_x + 30, y + height - 24), fill=color, width=4)
        draw.text(
            (legend_x + 38, y + height - 33),
            label,
            font=font(13, bold=True),
            fill="#42565e",
        )
    for index, report in enumerate(reports):
        px = left + int(index * plot_width / (len(reports) - 1))
        draw.text(
            (px - 18, top + plot_height + 10),
            str(int(report["step"])),
            font=font(11),
            fill="#687980",
        )


def main() -> None:
    args = parse_args()
    candidate_reports, candidate_complete = read_reports(args.candidate)
    control_reports, control_complete = read_reports(args.control)
    candidate = candidate_reports[-1]
    control = control_reports[-1]
    audit = json.loads(args.attention_audit.read_text(encoding="utf-8"))
    if candidate.get("route_mode") != "query_aware":
        raise ValueError("V22 candidate must use the query-aware route")
    if control.get("route_mode") != "query_blind":
        raise ValueError("V22 control must use the query-blind route")
    if candidate_complete.get("best_development") is not None:
        raise ValueError("V22 candidate unexpectedly selected")
    if control_complete.get("best_development") is None:
        raise ValueError("V22 control did not select its structural checkpoint")
    if int(audit.get("frozen_images_instantiated", -1)) != 0:
        raise ValueError("V22 result requires a sealed frozen split")
    if int(audit["candidate"]["argmax_counts"]["operation"]) != 1024:
        raise ValueError("V22 recorded attention diagnosis differs")

    canvas = Image.new("RGB", (2200, 1560), "#f4f7f8")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (62, 34),
        "V22: visible prompt binding fails through an operation-frame shortcut",
        font=font(38, bold=True),
        fill="#17313b",
    )
    draw.text(
        (64, 92),
        "One RTX 4090 | 3,410,128 trainable parameters per arm | image-only input/output | frozen identities sealed",
        font=font(20),
        fill="#586a72",
    )
    draw.text(
        (64, 139),
        "Six-frame visual prompt and measured candidate selector behavior",
        font=font(22, bold=True),
        fill="#264650",
    )
    draw_attention_roles(draw, audit, x=62, y=179, width=2076)

    left_x, sample_y = 62, 346
    draw.text(
        (left_x, sample_y),
        "Actual endpoint samples",
        font=font(24, bold=True),
        fill="#17313b",
    )
    draw.text(
        (left_x, sample_y + 38),
        "Only the final query image changes inside each pair; targets switch, predictions barely do.",
        font=font(16),
        fill="#5a6d74",
    )
    candidate_strip = sample_pair(args.candidate).resize((960, 298), Image.Resampling.LANCZOS)
    control_strip = sample_pair(args.control).resize((960, 298), Image.Resampling.LANCZOS)
    draw.text((left_x, sample_y + 72), "query-aware candidate", font=font(16, bold=True), fill="#226474")
    canvas.paste(candidate_strip, (left_x, sample_y + 98))
    draw.text((left_x, sample_y + 413), "query-blind control", font=font(16, bold=True), fill="#6a5f75")
    canvas.paste(control_strip, (left_x, sample_y + 439))

    panel_x, panel_y, panel_w = 1060, 346, 1078
    draw.rectangle(
        (panel_x, panel_y, panel_x + panel_w, panel_y + 770),
        fill="white",
        outline="#b4c0c5",
        width=2,
    )
    draw.text(
        (panel_x + 28, panel_y + 22),
        "Fixed candidate gates at step 1,600",
        font=font(25, bold=True),
        fill="#17313b",
    )
    draw.text(
        (panel_x + 28, panel_y + 62),
        "measured candidate       fixed requirement",
        font=font(15),
        fill="#748188",
    )
    identity_gain = float(candidate["identity_top1"]) - float(
        candidate["query_shuffled_identity_top1"]
    )
    rows = (
        ("binary choice", f"{float(candidate['binary_choice_accuracy']):.3f}", "> 0.850", False),
        ("counterfactual switch", f"{float(candidate['counterfactual_switch_accuracy']):.3f}", "> 0.800", False),
        ("held-out switch", f"{float(candidate['heldout_combination_switch_accuracy']):.3f}", "> 0.750", False),
        ("identity top-1", f"{float(candidate['identity_top1']):.3f}", "> 0.450", False),
        ("identity gain: query", f"{identity_gain:+.3f}", "> 0.200", False),
        ("pixel F1", f"{float(candidate['pixel_f1']):.3f}", "> 0.580", False),
        ("oracle-writer F1", f"{float(candidate['oracle_pixel_f1']):.3f}", "> 0.640", False),
        ("paired output L1", f"{float(candidate['paired_output_pixel_l1']):.4f}", "> 0.080", False),
        ("target vs op. margin", f"{float(candidate['target_margin_over_operation']):+.3f}", "> 0.150", True),
        ("frozen images", "0", "= 0", True),
    )
    for index, (label, value, target, passed) in enumerate(rows):
        gate_row(
            draw,
            x=panel_x + 28,
            y=panel_y + 88 + index * 54,
            width=panel_w - 56,
            label=label,
            value=value,
            target=target,
            passed=passed,
        )

    compare_y = panel_y + 641
    draw.rectangle(
        (panel_x + 26, compare_y, panel_x + panel_w - 26, compare_y + 101),
        fill="#fff0ee",
        outline="#cb8a84",
        width=2,
    )
    draw.text(
        (panel_x + 46, compare_y + 15),
        "Candidate vs query-blind control endpoint",
        font=font(18, bold=True),
        fill="#8f322f",
    )
    draw.text(
        (panel_x + 46, compare_y + 50),
        f"switch {float(candidate['counterfactual_switch_accuracy']):.4f} vs 0.0000 | "
        f"identity {float(candidate['identity_top1']):.4f} vs {float(control['identity_top1']):.4f} | "
        f"F1 {float(candidate['pixel_f1']):.4f} vs {float(control['pixel_f1']):.4f}",
        font=font(15),
        fill="#465a62",
    )

    draw_trajectory(draw, candidate_reports, x=62, y=1147, width=1110, height=344)
    verdict_x, verdict_y = 1204, 1147
    draw.rectangle(
        (verdict_x, verdict_y, 2138, 1491),
        fill="#fff0ee",
        outline="#c77d76",
        width=2,
    )
    draw.text(
        (verdict_x + 28, verdict_y + 22),
        "BINDING MECHANISM REJECTED",
        font=font(24, bold=True),
        fill="#a43131",
    )
    verdict_lines = (
        "The writer acquires weak unseen-glyph structure.",
        "The candidate does not bind query to either glyph.",
        "Entropy minimization makes the fixed operation marker",
        "the easiest single-frame shortcut: 1,024/1,024 argmaxes.",
        "No candidate checkpoint selects; paired audit refuses.",
        "No human review or frozen identity image is permitted.",
        "Next: explicit multi-frame visual relation composition.",
    )
    for index, line in enumerate(verdict_lines):
        yy = verdict_y + 72 + index * 35
        draw.ellipse((verdict_x + 32, yy + 8, verdict_x + 42, yy + 18), fill="#a6413b")
        draw.text((verdict_x + 55, yy), line, font=font(16), fill="#3e525a")

    draw.text(
        (62, 1520),
        "DEVELOPMENT ONLY | candidate 258.70 s / 0.315 GiB | control 255.11 s / 0.316 GiB | no selected candidate",
        font=font(17, bold=True),
        fill="#65757c",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print(args.out)


if __name__ == "__main__":
    main()
