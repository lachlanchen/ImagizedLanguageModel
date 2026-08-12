#!/usr/bin/env python3
"""Compose the measured V24 visual-packet stream result figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from generate_v20_result_figure import font


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRED = (
    ROOT
    / "artifacts/visual_packet_stream_v24_paired_audit"
    / "paired_development_audit.json"
)
DEFAULT_REVIEW = (
    ROOT
    / "artifacts/visual_packet_stream_v24_opaque_review"
    / "review_result.json"
)
DEFAULT_CARD = (
    ROOT
    / "artifacts/visual_packet_stream_v24_opaque_review"
    / "cards/01_P-1968B07440.png"
)
DEFAULT_FROZEN = (
    ROOT
    / "artifacts/visual_packet_stream_v24_frozen"
    / "frozen_evaluation.json"
)
DEFAULT_OUT = (
    Path(__file__).resolve().parent
    / "figures/visual_packet_stream_v24_result.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the measured V24 figure.")
    parser.add_argument("--paired-audit", type=Path, default=DEFAULT_PAIRED)
    parser.add_argument("--review-result", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--review-card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--frozen-result", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_evidence(
    paired: dict[str, object],
    review: dict[str, object],
    frozen: dict[str, object],
) -> None:
    if paired.get("architecture") != "visual-packet-reread-stream-v24-paired-audit":
        raise ValueError("Unexpected V24 paired-audit architecture")
    if not paired.get("paired_gate_passed"):
        raise ValueError("V24 paired gate did not pass")
    if int(paired.get("frozen_images_instantiated", -1)) != 0:
        raise ValueError("V24 paired audit did not preserve the frozen seal")
    expected_parameters = {"packet_aware", "header_blind", "query_blind",
                           "operation_blind", "history_blind"}
    arm_parameters = paired.get("arm_parameters")
    if not isinstance(arm_parameters, dict) or set(arm_parameters) != expected_parameters:
        raise ValueError("V24 paired audit does not contain all matched arms")
    if set(int(value) for value in arm_parameters.values()) != {1_347}:
        raise ValueError("V24 paired arms are not parameter matched")

    if review.get("architecture") != (
        "visual-packet-reread-stream-v24-opaque-review-result"
    ):
        raise ValueError("Unexpected V24 review-result architecture")
    if not review.get("opaque_review_passed"):
        raise ValueError("V24 opaque review did not pass")

    if frozen.get("architecture") != (
        "visual-packet-reread-stream-v24-frozen-evaluation"
    ):
        raise ValueError("Unexpected V24 frozen-result architecture")
    if not frozen.get("frozen_gate_passed"):
        raise ValueError("V24 frozen gate did not pass")
    if frozen.get("frozen_evaluation_repeated"):
        raise ValueError("V24 frozen evaluation was repeated")
    if frozen.get("model_selection_performed"):
        raise ValueError("V24 frozen evaluation performed model selection")
    if frozen.get("thresholds_changed"):
        raise ValueError("V24 frozen evaluation changed thresholds")
    gates = frozen.get("performance_gates")
    if not isinstance(gates, dict) or not all(bool(value) for value in gates.values()):
        raise ValueError("V24 frozen performance gates are incomplete")


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: str = "#49616b",
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
    left, top, right, bottom = bounds
    draw.rounded_rectangle(
        bounds,
        radius=8,
        fill="white",
        outline="#aebdc3",
        width=2,
    )
    draw.rectangle((left, top, left + 9, bottom), fill=accent)
    draw.text((left + 25, top + 19), title, font=font(20, bold=True), fill="#19333d")
    for index, line in enumerate(lines):
        draw.text(
            (left + 25, top + 61 + index * 28),
            line,
            font=font(15),
            fill="#53676f",
        )


def causal_bar(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    label: str,
    candidate: float,
    control: float,
) -> None:
    draw.text((x, y), label, font=font(17, bold=True), fill="#213a44")
    bar_left = x + 235
    bar_width = width - 340
    for offset, value, color, name in (
        (0, candidate, "#087459", "candidate"),
        (30, control, "#b24b63", "blind"),
    ):
        yy = y + offset
        draw.rectangle(
            (bar_left, yy + 3, bar_left + bar_width, yy + 20),
            fill="#e1e8ea",
        )
        fill_width = max(2, int(bar_width * value))
        draw.rectangle(
            (bar_left, yy + 3, bar_left + fill_width, yy + 20),
            fill=color,
        )
        draw.text(
            (bar_left + bar_width + 16, yy),
            f"{name} {value:.3f}",
            font=font(15, bold=True),
            fill=color,
        )


def metric_tile(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    label: str,
    value: float,
    requirement: str,
) -> None:
    draw.rounded_rectangle(
        (x, y, x + width, y + 90),
        radius=7,
        fill="white",
        outline="#b9c5c9",
        width=2,
    )
    draw.text((x + 17, y + 13), label, font=font(14, bold=True), fill="#314950")
    draw.text(
        (x + 17, y + 41),
        f"{value:.4f}",
        font=font(23, bold=True),
        fill="#087459",
    )
    draw.text(
        (x + width - 86, y + 54),
        requirement,
        font=font(12),
        fill="#66777e",
    )


def main() -> None:
    args = parse_args()
    paired = read_json(args.paired_audit)
    review = read_json(args.review_result)
    frozen = read_json(args.frozen_result)
    validate_evidence(paired, review, frozen)

    review_card = Image.open(args.review_card).convert("RGB")
    if review_card.size != (694, 175):
        raise ValueError(f"Unexpected V24 review-card size: {review_card.size}")

    metrics = paired["metrics"]
    candidate = metrics["packet_aware"]
    header_blind = metrics["header_blind"]
    query_blind = metrics["query_blind"]
    operation_blind = metrics["operation_blind"]
    history_blind = metrics["history_blind"]
    frozen_metrics = frozen["metrics"]

    canvas = Image.new("RGB", (2400, 1840), "#f3f6f7")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (64, 34),
        "V24: visible packets become a causal two-frame image stream",
        font=font(43, bold=True),
        fill="#17323c",
    )
    draw.text(
        (66, 94),
        "Variable 15-24 frame prompt | generated glyph then generated label | 1,347 trainable parameters | one RTX 4090",
        font=font(20),
        fill="#586c74",
    )

    architecture_y = 148
    draw.rounded_rectangle(
        (62, architecture_y, 2338, architecture_y + 300),
        radius=9,
        fill="#e9f0f2",
        outline="#a9b9bf",
        width=2,
    )
    draw.text(
        (86, architecture_y + 18),
        "Measured image-only computation",
        font=font(23, bold=True),
        fill="#1a3944",
    )
    box(
        draw,
        (86, architecture_y + 68, 430, architecture_y + 258),
        "Visible packet stream",
        ("5-8 shuffled packets", "3 raster frames / packet", "headers + contents + stop"),
        accent="#287b8a",
    )
    arrow(draw, (448, architecture_y + 163), (505, architecture_y + 163))
    box(
        draw,
        (520, architecture_y + 68, 850, architecture_y + 258),
        "Learned role routing",
        ("header-image prototypes", "locate pair / operation / query", "no length or padding mask"),
        accent="#745a9d",
    )
    arrow(draw, (868, architecture_y + 163), (925, architecture_y + 163))
    box(
        draw,
        (940, architecture_y + 68, 1260, architecture_y + 258),
        "Visual relation",
        ("query-label comparison", "same / other image gate", "route visible glyph pixels"),
        accent="#a56435",
    )
    arrow(draw, (1278, architecture_y + 163), (1335, architecture_y + 163))
    box(
        draw,
        (1350, architecture_y + 68, 1632, architecture_y + 258),
        "Frame 1",
        ("render selected glyph", "continuous 32 x 32 ink", "unseen identity allowed"),
        accent="#087459",
    )
    arrow(draw, (1650, architecture_y + 163), (1707, architecture_y + 163))
    box(
        draw,
        (1722, architecture_y + 68, 2020, architecture_y + 258),
        "Reread pixels",
        ("frozen visual retina", "match output to source glyphs", "history intervention tested"),
        accent="#287b8a",
    )
    arrow(draw, (2038, architecture_y + 163), (2095, architecture_y + 163))
    box(
        draw,
        (2110, architecture_y + 68, 2314, architecture_y + 258),
        "Frame 2",
        ("render bound label", "from frame-1 pixels", "not hidden state"),
        accent="#087459",
    )

    sample_y = 486
    draw.rounded_rectangle(
        (62, sample_y, 1592, sample_y + 455),
        radius=9,
        fill="white",
        outline="#aebdc3",
        width=2,
    )
    draw.text(
        (88, sample_y + 20),
        "Actual opaque development card",
        font=font(24, bold=True),
        fill="#17323c",
    )
    draw.text(
        (88, sample_y + 58),
        "Top: prompt packets. Bottom left: visible A/B bindings. Bottom right: autonomous output frames 1 and 2.",
        font=font(15),
        fill="#5d7077",
    )
    large_card = review_card.resize((1388, 350), Image.Resampling.NEAREST)
    canvas.paste(large_card, (88, sample_y + 95))

    review_x = 1628
    draw.rounded_rectangle(
        (review_x, sample_y, 2338, sample_y + 455),
        radius=9,
        fill="#edf7f2",
        outline="#7eae9e",
        width=2,
    )
    draw.text(
        (review_x + 28, sample_y + 24),
        "Opaque visual audit",
        font=font(24, bold=True),
        fill="#0c6149",
    )
    draw.text(
        (review_x + 28, sample_y + 69),
        f"{int(review['frame1_correct'])}/{int(review['review_cases'])}",
        font=font(52, bold=True),
        fill="#087459",
    )
    draw.text(
        (review_x + 215, sample_y + 90),
        "frame-1 source choices",
        font=font(17, bold=True),
        fill="#325b50",
    )
    draw.text(
        (review_x + 28, sample_y + 145),
        f"{int(review['frame2_correct'])}/{int(review['review_cases'])}",
        font=font(52, bold=True),
        fill="#087459",
    )
    draw.text(
        (review_x + 215, sample_y + 166),
        "frame-2 label choices",
        font=font(17, bold=True),
        fill="#325b50",
    )
    draw.line(
        (review_x + 28, sample_y + 225, review_x + 682, sample_y + 225),
        fill="#b8d1c8",
        width=2,
    )
    draw.text(
        (review_x + 28, sample_y + 247),
        "Held-out T=24: 12/12 for both frames",
        font=font(19, bold=True),
        fill="#244c40",
    )
    draw.text(
        (review_x + 28, sample_y + 287),
        "The agent saw no targets, transcriptions,",
        font=font(15),
        fill="#536d64",
    )
    draw.text(
        (review_x + 28, sample_y + 315),
        "correctness labels, or frozen identities.",
        font=font(15),
        fill="#536d64",
    )
    draw.text(
        (review_x + 28, sample_y + 356),
        "Visual audit, not a human-subject study",
        font=font(14, bold=True),
        fill="#776142",
    )

    paired_y = 976
    draw.rounded_rectangle(
        (62, paired_y, 1272, paired_y + 420),
        radius=9,
        fill="white",
        outline="#aebdc3",
        width=2,
    )
    draw.text(
        (88, paired_y + 20),
        "Fresh paired causal audit",
        font=font(24, bold=True),
        fill="#17323c",
    )
    draw.text(
        (88, paired_y + 58),
        "1,024 new episodes; every arm has exactly 1,347 trainable parameters",
        font=font(15),
        fill="#5d7077",
    )
    causal_bar(
        draw,
        x=88,
        y=paired_y + 102,
        width=1148,
        label="query switch",
        candidate=float(candidate["query_switch_accuracy"]),
        control=float(query_blind["query_switch_accuracy"]),
    )
    causal_bar(
        draw,
        x=88,
        y=paired_y + 176,
        width=1148,
        label="operation switch",
        candidate=float(candidate["operation_switch_accuracy"]),
        control=float(operation_blind["operation_switch_accuracy"]),
    )
    causal_bar(
        draw,
        x=88,
        y=paired_y + 250,
        width=1148,
        label="history switch",
        candidate=float(candidate["history_switch_accuracy"]),
        control=float(history_blind["history_switch_accuracy"]),
    )
    candidate_role = min(
        float(candidate["pair_header_localization_accuracy"]),
        float(candidate["operation_header_localization_accuracy"]),
        float(candidate["query_header_localization_accuracy"]),
    )
    blind_role = min(
        float(header_blind["pair_header_localization_accuracy"]),
        float(header_blind["operation_header_localization_accuracy"]),
        float(header_blind["query_header_localization_accuracy"]),
    )
    causal_bar(
        draw,
        x=88,
        y=paired_y + 324,
        width=1148,
        label="minimum role location",
        candidate=candidate_role,
        control=blind_role,
    )

    frozen_x = 1308
    draw.rounded_rectangle(
        (frozen_x, paired_y, 2338, paired_y + 630),
        radius=9,
        fill="#eef5f3",
        outline="#84a89d",
        width=2,
    )
    draw.text(
        (frozen_x + 28, paired_y + 20),
        "Single authorized frozen evaluation",
        font=font(24, bold=True),
        fill="#173d34",
    )
    draw.text(
        (frozen_x + 28, paired_y + 58),
        "1,024 episodes | 107 unseen identities | no selection, retuning, or repeat",
        font=font(15),
        fill="#587168",
    )
    metric_rows = (
        ("binary choice", "frame1_binary_choice_accuracy", "> .95"),
        ("frame-1 identity", "frame1_identity_top1", "> .75"),
        ("frame-2 label", "frame2_label_top1", "> .95"),
        ("query switch", "query_switch_accuracy", "> .90"),
        ("operation switch", "operation_switch_accuracy", "> .90"),
        ("history switch", "history_switch_accuracy", "> .90"),
        ("held-out minimum", "heldout_combination_minimum_switch_accuracy", "> .85"),
        ("T=24 frame 1", "heldout_length_frame1_identity_top1", "> .90"),
        ("T=24 frame 2", "heldout_length_frame2_label_top1", "> .90"),
        ("frame-1 pixel F1", "frame1_pixel_f1", "> .68"),
        ("frame-2 pixel F1", "frame2_pixel_f1", "> .58"),
        ("history consistency", "frame2_generated_history_consistency", "> .92"),
    )
    tile_width = 304
    for index, (label, key, requirement) in enumerate(metric_rows):
        column = index % 3
        row = index // 3
        metric_tile(
            draw,
            x=frozen_x + 28 + column * 326,
            y=paired_y + 96 + row * 112,
            width=tile_width,
            label=label,
            value=float(frozen_metrics[key]),
            requirement=requirement,
        )
    draw.text(
        (frozen_x + 28, paired_y + 558),
        f"all fixed gates passed | {float(frozen['peak_cuda_gib']):.3f} GiB peak evaluation CUDA",
        font=font(17, bold=True),
        fill="#0b664d",
    )

    boundary_y = 1436
    draw.rounded_rectangle(
        (62, boundary_y, 1272, boundary_y + 170),
        radius=9,
        fill="#fff6e8",
        outline="#d4a564",
        width=2,
    )
    draw.text(
        (88, boundary_y + 22),
        "ACCEPTED",
        font=font(23, bold=True),
        fill="#8a541e",
    )
    draw.text(
        (250, boundary_y + 24),
        "fixed visual packet grammar + causal two-frame output",
        font=font(20, bold=True),
        fill="#543d29",
    )
    draw.text(
        (88, boundary_y + 70),
        "The prompt is a variable image stream. Frame 2 changes when the model rereads a",
        font=font(16),
        fill="#675442",
    )
    draw.text(
        (88, boundary_y + 99),
        "counterfactual frame 1; a history-blind matched control is exactly invariant.",
        font=font(16),
        fill="#675442",
    )
    draw.text(
        (88, boundary_y + 132),
        "The primary output is pixels; no deployed token, Unicode, OCR, or lookup path.",
        font=font(16, bold=True),
        fill="#675442",
    )

    draw.rounded_rectangle(
        (62, 1634, 2338, 1790),
        radius=9,
        fill="#f7e9ec",
        outline="#c18492",
        width=2,
    )
    draw.text(
        (88, 1656),
        "NOT YET ESTABLISHED",
        font=font(21, bold=True),
        fill="#8d354c",
    )
    draw.text(
        (88, 1698),
        "Arbitrary sentence/page understanding, factual etymology, open-ended writing, and unrestricted stream or movie generation.",
        font=font(17, bold=True),
        fill="#5f3b45",
    )
    draw.text(
        (88, 1740),
        "V24 is the bridge to that experiment, not evidence that the final language model already exists.",
        font=font(16),
        fill="#6a4b53",
    )

    draw.text(
        (64, 1807),
        "V24 FROZEN RESULT | generated glyph -> visual reread -> generated label | output [B,2,1,32,32] | fixed protocol dated 2026-08-13",
        font=font(15, bold=True),
        fill="#66787f",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print(args.out)


if __name__ == "__main__":
    main()
