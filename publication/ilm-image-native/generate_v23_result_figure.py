#!/usr/bin/env python3
"""Compose the measured V23 visual-relation result figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from generate_v20_result_figure import font


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRED = (
    ROOT
    / "artifacts/visual_relation_circuit_v23_paired_audit"
    / "paired_development_audit.json"
)
DEFAULT_REVIEW = (
    ROOT / "artifacts/visual_relation_circuit_v23_blinded_review" / "review_result.json"
)
DEFAULT_FROZEN = (
    ROOT / "artifacts/visual_relation_circuit_v23_frozen" / "frozen_evaluation.json"
)
DEFAULT_SAMPLES = (
    ROOT / "artifacts/visual_relation_circuit_v23_frozen" / "frozen_samples.png"
)
DEFAULT_OUT = (
    Path(__file__).resolve().parent / "figures/visual_relation_circuit_v23_result.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the measured V23 figure.")
    parser.add_argument("--paired-audit", type=Path, default=DEFAULT_PAIRED)
    parser.add_argument("--review-result", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--frozen-result", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_evidence(
    paired: dict[str, object],
    review: dict[str, object],
    frozen: dict[str, object],
) -> None:
    if paired.get("architecture") != "visual-relation-circuit-v23-paired-audit":
        raise ValueError("Unexpected V23 paired-audit architecture")
    if not paired.get("paired_gate_passed"):
        raise ValueError("V23 paired gate did not pass")
    if int(paired.get("frozen_images_instantiated", -1)) != 0:
        raise ValueError("V23 paired audit did not preserve the frozen seal")
    if review.get("architecture") != (
        "visual-relation-circuit-v23-blinded-review-result"
    ):
        raise ValueError("Unexpected V23 review-result architecture")
    if not review.get("blinded_review_passed"):
        raise ValueError("V23 blinded review did not pass")
    if frozen.get("architecture") != ("visual-relation-circuit-v23-frozen-evaluation"):
        raise ValueError("Unexpected V23 frozen-result architecture")
    if not frozen.get("frozen_gate_passed"):
        raise ValueError("V23 frozen gate did not pass")
    if frozen.get("frozen_evaluation_repeated"):
        raise ValueError("V23 frozen evaluation was repeated")
    if frozen.get("model_selection_performed"):
        raise ValueError("V23 frozen evaluation performed model selection")
    if frozen.get("thresholds_changed"):
        raise ValueError("V23 frozen evaluation changed thresholds")
    if not all(bool(value) for value in frozen["performance_gates"].values()):
        raise ValueError("V23 frozen performance gates are incomplete")


def prompt_and_output(samples: Image.Image) -> tuple[list[Image.Image], Image.Image]:
    if samples.size != (1616, 952):
        raise ValueError(f"Unexpected V23 sample sheet size: {samples.size}")
    card_x, card_y, tile = 8, 8, 64
    prompts = [
        samples.crop(
            (
                card_x + frame * tile,
                card_y + 18,
                card_x + (frame + 1) * tile,
                card_y + 18 + tile,
            )
        )
        for frame in range(6)
    ]
    generated = samples.crop((card_x, card_y + 92, card_x + tile, card_y + 92 + tile))
    return prompts, generated


def sample_cards(samples: Image.Image) -> tuple[Image.Image, Image.Image]:
    card_width = 400
    first = samples.crop((8, 8, 8 + 392, 8 + 226))
    heldout = samples.crop((8 + 2 * card_width, 8, 8 + 2 * card_width + 392, 8 + 226))
    if first.size != (392, 226) or heldout.size != (392, 226):
        raise ValueError("Could not extract V23 sample cards")
    return first, heldout


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: str = "#50656d",
    width: int = 4,
) -> None:
    draw.line((*start, *end), fill=color, width=width)
    x, y = end
    draw.polygon(((x, y), (x - 13, y - 8), (x - 13, y + 8)), fill=color)


def draw_metric_row(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    label: str,
    value: str,
    target: str,
) -> None:
    draw.line((x, y + 47, x + width, y + 47), fill="#dce4e7", width=1)
    draw.text((x, y + 10), label, font=font(16, bold=True), fill="#263b43")
    draw.text((x + 345, y + 8), value, font=font(19, bold=True), fill="#087459")
    draw.text((x + 485, y + 11), target, font=font(14), fill="#687980")
    draw.text(
        (x + width - 48, y + 10), "PASS", font=font(14, bold=True), fill="#087459"
    )


def draw_control_box(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    title: str,
    switch: float,
    identity: float,
    output_l1: float,
    accent: str,
) -> None:
    draw.rectangle((x, y, x + width, y + 118), fill="white", outline="#b6c2c6", width=2)
    draw.rectangle((x, y, x + 8, y + 118), fill=accent)
    draw.text((x + 24, y + 14), title, font=font(18, bold=True), fill="#233a43")
    draw.text(
        (x + 24, y + 51),
        f"switch {switch:.3f}   identity {identity:.3f}   changed pixels {output_l1:.3f}",
        font=font(15),
        fill="#51656d",
    )
    bar_left, bar_top, bar_width = x + 24, y + 83, width - 48
    draw.rectangle(
        (bar_left, bar_top, bar_left + bar_width, bar_top + 13),
        fill="#e4eaec",
    )
    draw.rectangle(
        (bar_left, bar_top, bar_left + int(bar_width * switch), bar_top + 13),
        fill=accent,
    )


def main() -> None:
    args = parse_args()
    paired = read_json(args.paired_audit)
    review = read_json(args.review_result)
    frozen = read_json(args.frozen_result)
    validate_evidence(paired, review, frozen)
    samples = Image.open(args.samples).convert("RGB")
    prompt_images, generated = prompt_and_output(samples)
    first_card, heldout_card = sample_cards(samples)

    paired_metrics = paired["metrics"]
    candidate = paired_metrics["relation_aware"]
    query_blind = paired_metrics["query_blind"]
    operation_blind = paired_metrics["operation_blind"]
    frozen_metrics = frozen["metrics"]

    canvas = Image.new("RGB", (2200, 1600), "#f4f7f8")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (62, 34),
        "V23: an image-only relation circuit follows a visible prompt",
        font=font(40, bold=True),
        fill="#17313b",
    )
    draw.text(
        (64, 94),
        "One RTX 4090 | six raster frames in, one raster answer out | 98 unseen frozen identities | no token or Unicode path",
        font=font(20),
        fill="#586a72",
    )

    architecture_y = 145
    draw.rectangle(
        (62, architecture_y, 2138, architecture_y + 205),
        fill="white",
        outline="#aab9bf",
        width=2,
    )
    draw.text(
        (84, architecture_y + 15),
        "Measured image-native computation",
        font=font(22, bold=True),
        fill="#1c3a44",
    )
    labels = ("label 1", "glyph 1", "label 2", "glyph 2", "operation", "query")
    for index, (label, image) in enumerate(zip(labels, prompt_images)):
        left = 84 + index * 112
        resized = image.resize((76, 76), Image.Resampling.NEAREST)
        draw.rectangle(
            (left - 2, architecture_y + 58, left + 78, architecture_y + 138),
            fill="white",
            outline="#9eb0b7",
            width=2,
        )
        canvas.paste(resized, (left, architecture_y + 60))
        draw.text(
            (left, architecture_y + 146),
            label,
            font=font(12, bold=True),
            fill="#546870",
        )
    draw_arrow(draw, (760, architecture_y + 99), (820, architecture_y + 99))
    draw.text((838, architecture_y + 60), "frozen", font=font(15), fill="#687980")
    draw.text(
        (838, architecture_y + 86),
        "visual retina",
        font=font(19, bold=True),
        fill="#146a7b",
    )
    draw.text(
        (838, architecture_y + 118), "continuous state", font=font(14), fill="#687980"
    )
    draw_arrow(draw, (1010, architecture_y + 99), (1070, architecture_y + 99))
    draw.text(
        (1088, architecture_y + 55),
        "query-label",
        font=font(16, bold=True),
        fill="#253f48",
    )
    draw.text(
        (1088, architecture_y + 81),
        "visual match",
        font=font(16, bold=True),
        fill="#253f48",
    )
    draw.text(
        (1088, architecture_y + 112), "+ operation gate", font=font(14), fill="#687980"
    )
    draw_arrow(draw, (1290, architecture_y + 99), (1350, architecture_y + 99))
    draw.text(
        (1368, architecture_y + 66),
        "route source",
        font=font(18, bold=True),
        fill="#8a5323",
    )
    draw.text(
        (1368, architecture_y + 99), "pixels", font=font(18, bold=True), fill="#8a5323"
    )
    draw_arrow(draw, (1520, architecture_y + 99), (1580, architecture_y + 99))
    draw.text(
        (1598, architecture_y + 60), "frozen visual", font=font(15), fill="#687980"
    )
    draw.text(
        (1598, architecture_y + 86),
        "canonicalizer",
        font=font(19, bold=True),
        fill="#146a7b",
    )
    draw_arrow(draw, (1790, architecture_y + 99), (1850, architecture_y + 99))
    answer = generated.resize((92, 92), Image.Resampling.NEAREST)
    draw.rectangle(
        (1874, architecture_y + 48, 1970, architecture_y + 144),
        fill="white",
        outline="#087459",
        width=3,
    )
    canvas.paste(answer, (1876, architecture_y + 50))
    draw.text(
        (1990, architecture_y + 73), "answer", font=font(19, bold=True), fill="#087459"
    )
    draw.text(
        (1990, architecture_y + 106), "image", font=font(19, bold=True), fill="#087459"
    )

    sample_x, sample_y, sample_w = 62, 385, 1260
    draw.rectangle(
        (sample_x, sample_y, sample_x + sample_w, sample_y + 620),
        fill="white",
        outline="#b4c0c5",
        width=2,
    )
    draw.text(
        (sample_x + 25, sample_y + 20),
        "Actual frozen prompts and generated answers",
        font=font(24, bold=True),
        fill="#17313b",
    )
    draw.text(
        (sample_x + 25, sample_y + 58),
        "Each card: six prompt images; then generated row over target row for base, query change, operation change, and pair swap.",
        font=font(15),
        fill="#5c6e75",
    )
    card_size = (580, 334)
    first_large = first_card.resize(card_size, Image.Resampling.LANCZOS)
    heldout_large = heldout_card.resize(card_size, Image.Resampling.LANCZOS)
    canvas.paste(first_large, (sample_x + 25, sample_y + 105))
    canvas.paste(heldout_large, (sample_x + 630, sample_y + 105))
    draw.text(
        (sample_x + 25, sample_y + 458),
        "The correct identity switches when query or operation changes and remains invariant when the two bound pairs swap.",
        font=font(16, bold=True),
        fill="#31515b",
    )
    draw.rectangle(
        (sample_x + 25, sample_y + 500, sample_x + sample_w - 25, sample_y + 584),
        fill="#e8f4ef",
        outline="#7eb09e",
        width=2,
    )
    draw.text(
        (sample_x + 47, sample_y + 516),
        f"Opaque visual review: {int(review['correct'])}/{int(review['review_cases'])} overall and "
        f"{int(review['heldout_correct'])}/{int(review['heldout_cases'])} held-out",
        font=font(20, bold=True),
        fill="#116348",
    )
    draw.text(
        (sample_x + 47, sample_y + 548),
        "Agent reviewer saw only the image cards; the scorer opened the sealed answer key afterward.",
        font=font(14),
        fill="#4f6861",
    )

    metric_x, metric_y, metric_w = 1350, 385, 788
    draw.rectangle(
        (metric_x, metric_y, metric_x + metric_w, metric_y + 620),
        fill="white",
        outline="#b4c0c5",
        width=2,
    )
    draw.text(
        (metric_x + 25, metric_y + 20),
        "Single frozen evaluation",
        font=font(24, bold=True),
        fill="#17313b",
    )
    draw.text(
        (metric_x + 25, metric_y + 58),
        "measured                 fixed requirement",
        font=font(14),
        fill="#748188",
    )
    rows = (
        (
            "binary choice",
            f"{float(frozen_metrics['binary_choice_accuracy']):.4f}",
            "> 0.95",
        ),
        (
            "query switch",
            f"{float(frozen_metrics['query_switch_accuracy']):.4f}",
            "> 0.90",
        ),
        (
            "operation switch",
            f"{float(frozen_metrics['operation_switch_accuracy']):.4f}",
            "> 0.90",
        ),
        (
            "held-out minimum",
            f"{float(frozen_metrics['heldout_combination_minimum_switch_accuracy']):.4f}",
            "> 0.85",
        ),
        ("identity top-1", f"{float(frozen_metrics['identity_top1']):.4f}", "> 0.75"),
        ("pixel F1", f"{float(frozen_metrics['pixel_f1']):.4f}", "> 0.68"),
        ("target cosine", f"{float(frozen_metrics['target_cosine']):.4f}", "> 0.82"),
        (
            "query-label match",
            f"{float(frozen_metrics['query_label_match_accuracy']):.4f}",
            "> 0.98",
        ),
        (
            "operation gate",
            f"{float(frozen_metrics['operation_gate_accuracy']):.4f}",
            "> 0.98",
        ),
        (
            "pair-swap invariant",
            f"{float(frozen_metrics['pair_swap_identity_consistency']):.4f}",
            "> 0.99",
        ),
    )
    for index, (label, value, target) in enumerate(rows):
        draw_metric_row(
            draw,
            x=metric_x + 25,
            y=metric_y + 82 + index * 49,
            width=metric_w - 50,
            label=label,
            value=value,
            target=target,
        )

    controls_x, controls_y, controls_w = 62, 1040, 1070
    draw.text(
        (controls_x, controls_y),
        "Fresh paired causal audit",
        font=font(24, bold=True),
        fill="#17313b",
    )
    draw.text(
        (controls_x, controls_y + 37),
        "1,024 new development episodes; matched parameter names and shapes",
        font=font(15),
        fill="#5b6e75",
    )
    draw_control_box(
        draw,
        x=controls_x,
        y=controls_y + 72,
        width=controls_w,
        title="relation-aware candidate",
        switch=min(
            float(candidate["query_switch_accuracy"]),
            float(candidate["operation_switch_accuracy"]),
        ),
        identity=float(candidate["identity_top1"]),
        output_l1=min(
            float(candidate["query_output_pixel_l1"]),
            float(candidate["operation_output_pixel_l1"]),
        ),
        accent="#087459",
    )
    draw_control_box(
        draw,
        x=controls_x,
        y=controls_y + 204,
        width=controls_w,
        title="query-blind control",
        switch=float(query_blind["query_switch_accuracy"]),
        identity=float(query_blind["identity_top1"]),
        output_l1=float(query_blind["query_output_pixel_l1"]),
        accent="#9a606f",
    )
    draw_control_box(
        draw,
        x=controls_x,
        y=controls_y + 336,
        width=controls_w,
        title="operation-blind control",
        switch=float(operation_blind["operation_switch_accuracy"]),
        identity=float(operation_blind["identity_top1"]),
        output_l1=float(operation_blind["operation_output_pixel_l1"]),
        accent="#8a6a33",
    )

    chain_x, chain_y, chain_w = 1162, 1040, 976
    draw.rectangle(
        (chain_x, chain_y, chain_x + chain_w, chain_y + 454),
        fill="#eaf5f1",
        outline="#74a997",
        width=2,
    )
    draw.text(
        (chain_x + 28, chain_y + 22),
        "EVIDENCE CHAIN ACCEPTED",
        font=font(25, bold=True),
        fill="#0b684c",
    )
    chain = (
        ("1", "Canonicalizer selected", "0.758 F1, 99.51% unseen identity"),
        ("2", "Relation candidate selected", "99.02% minimum switch on development"),
        ("3", "Matched controls passed", "blind factor gives exactly zero switch"),
        ("4", "Opaque visual review passed", "48/48 overall; 12/12 held-out"),
        ("5", "Frozen evaluation passed once", "no selection or threshold change"),
    )
    for index, (number, title, detail) in enumerate(chain):
        yy = chain_y + 78 + index * 69
        draw.ellipse((chain_x + 30, yy, chain_x + 66, yy + 36), fill="#0b7555")
        draw.text(
            (chain_x + 42, yy + 7), number, font=font(15, bold=True), fill="white"
        )
        draw.text(
            (chain_x + 84, yy - 1), title, font=font(18, bold=True), fill="#24443c"
        )
        draw.text((chain_x + 84, yy + 27), detail, font=font(14), fill="#526b64")
    draw.text(
        (chain_x + 28, chain_y + 420),
        "Scope: fixed two-pair same/other grammar, not open-ended language.",
        font=font(15, bold=True),
        fill="#4b665e",
    )

    draw.text(
        (62, 1541),
        "FROZEN RESULT | 1,024 episodes / 4,096 variants | 25,602 relation parameters | 0.498 GiB peak eval CUDA | output is pixels",
        font=font(17, bold=True),
        fill="#63757c",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print(args.out)


if __name__ == "__main__":
    main()
