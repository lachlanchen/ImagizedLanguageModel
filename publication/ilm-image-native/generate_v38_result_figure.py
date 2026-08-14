#!/usr/bin/env python3
"""Render the measured V38 decision from tracked, hash-pinned evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).resolve().parent / "evidence/v38"
DEFAULT_RECEIPT = EVIDENCE / "run_receipt.json"
DEFAULT_SUMMARY = EVIDENCE / "training_summary.json"
DEFAULT_EMA = EVIDENCE / "development_report_ema_v38.json"
DEFAULT_RAW = EVIDENCE / "development_report_raw_v38.json"
DEFAULT_PROTOCOL = ROOT / "references/visual_path_alignment_v38_protocol.md"
DEFAULT_OUT = (
    Path(__file__).resolve().parent / "figures/visual_path_alignment_v38_result.png"
)

EXPECTED_SHA256 = {
    "receipt": "8b2a0ea2b5c786d7945e9d60b303d48c9da195d90fee9957b3edb585458df3c8",
    "summary": "59d2e3b9dac3c4fc7d2b03bc3197fa8588ba098703d9c16c67aae01393c74fbe",
    "ema": "835cba957dcaa735889ce4408ea60d81d1fe3ac35a42954d31d4ae543ffadfa4",
    "raw": "931809bc29b9f7fe8c1432a1488a08e732a51f096a99ef760df42ef122e02eda",
    "protocol": "cc7112e3f04e7fab622652d7da21955a5b632c5008e51f30182f3bcf2094c0c5",
}

FONT_ROOT = Path("/usr/share/fonts/truetype/dejavu")
INK = "#172d35"
MUTED = "#526970"
LINE = "#a8b7bc"
PAPER = "#f3f5f4"
WHITE = "#ffffff"
TEAL = "#247b86"
GREEN = "#32745d"
AMBER = "#a86f2b"
RED = "#a74848"
BLUE = "#416c88"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--ema", type=Path, default=DEFAULT_EMA)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_ROOT / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def load_evidence(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "receipt": args.receipt,
        "summary": args.summary,
        "ema": args.ema,
        "raw": args.raw,
        "protocol": args.protocol,
    }
    for name, path in paths.items():
        actual = file_sha256(path)
        if actual != EXPECTED_SHA256[name]:
            raise ValueError(
                f"V38 evidence SHA-256 changed for {name}: "
                f"expected {EXPECTED_SHA256[name]}, got {actual}"
            )

    receipt = load_json(args.receipt)
    summary = load_json(args.summary)
    ema = load_json(args.ema)
    raw = load_json(args.raw)
    if receipt.get("experiment") != "visual-path-alignment-v38":
        raise ValueError("unexpected V38 run receipt")
    if summary.get("global_update") != 8_000 or not summary.get("complete"):
        raise ValueError("V38 training coverage changed")
    if ema.get("weight_route") != "all-parameter-ema":
        raise ValueError("V38 EMA route changed")
    if raw.get("weight_route") != "raw":
        raise ValueError("V38 raw route changed")
    for report in (ema, raw):
        if report.get("split") != "development":
            raise ValueError("unexpected V38 report split")
        if report.get("gate", {}).get("decision") != "not-qualified":
            raise ValueError("V38 gate decision changed")
        if report.get("data", {}).get("sealed_rows_rendered") != 0:
            raise ValueError("V38 sealed-data boundary changed")
        boundary = report.get("checkpoint", {}).get("boundary", {})
        if boundary.get("generates_raster") is not False:
            raise ValueError("V38 renderer boundary changed")
        if not report.get("finite"):
            raise ValueError("V38 report is non-finite")
    checkpoint_hash = summary.get("checkpoint_sha256")
    if checkpoint_hash != ema["checkpoint"]["checkpoint_sha256"]:
        raise ValueError("V38 checkpoint/EMA report hash mismatch")
    if checkpoint_hash != raw["checkpoint"]["checkpoint_sha256"]:
        raise ValueError("V38 checkpoint/raw report hash mismatch")
    return {"receipt": receipt, "summary": summary, "ema": ema, "raw": raw}


def centered(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    value: str,
    *,
    size: int,
    fill: str = INK,
    bold: bool = False,
) -> None:
    left, top, right, bottom = bounds
    face = font(size, bold=bold)
    box = draw.multiline_textbbox((0, 0), value, font=face, spacing=5, align="center")
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.multiline_text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2),
        value,
        font=face,
        fill=fill,
        spacing=5,
        align="center",
    )


def panel(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    title: str,
    *,
    accent: str,
) -> None:
    left, top, right, bottom = bounds
    draw.rounded_rectangle(bounds, radius=7, fill=WHITE, outline=LINE, width=2)
    draw.rectangle((left, top, left + 9, bottom), fill=accent)
    draw.text((left + 27, top + 17), title, font=font(21, bold=True), fill=INK)


def metric_bar(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    label: str,
    value: float,
    threshold: float,
    maximum: float = 1.0,
) -> None:
    passed = value >= threshold
    draw.text((x, y), label, font=font(15), fill=INK)
    draw.text(
        (x + width - 152, y),
        f"{value:.3f} / {threshold:.2f}",
        font=font(14, bold=True),
        fill=GREEN if passed else RED,
    )
    bar_y = y + 27
    draw.rectangle((x, bar_y, x + width, bar_y + 14), fill="#dce4e6")
    draw.rectangle(
        (x, bar_y, x + int(width * min(value / maximum, 1.0)), bar_y + 14),
        fill=GREEN if passed else RED,
    )
    threshold_x = x + int(width * threshold / maximum)
    draw.line((threshold_x, bar_y - 4, threshold_x, bar_y + 18), fill=INK, width=2)


def lower_is_better_row(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    label: str,
    value: float,
    threshold: float,
) -> None:
    passed = value <= threshold
    draw.text((x, y), label, font=font(15), fill=INK)
    draw.text(
        (x + 235, y),
        f"{value:.3f} / <= {threshold:.1f}",
        font=font(14, bold=True),
        fill=GREEN if passed else RED,
    )


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    draw.line((*start, *end), fill=TEAL, width=4)
    draw.polygon(
        [(end[0], end[1]), (end[0] - 13, end[1] - 8), (end[0] - 13, end[1] + 8)],
        fill=TEAL,
    )


def render(evidence: Mapping[str, Any], out: Path) -> None:
    receipt = evidence["receipt"]
    summary = evidence["summary"]
    ema = evidence["ema"]
    raw = evidence["raw"]
    correct = ema["correct"]
    thresholds = ema["gate"]["thresholds"]

    image = Image.new("RGB", (1800, 1120), PAPER)
    draw = ImageDraw.Draw(image)
    draw.text(
        (70, 48),
        "V38 VISUAL PATH ALIGNMENT",
        font=font(34, bold=True),
        fill=INK,
    )
    draw.text(
        (70, 96),
        "Reading and font invariance improve; the frozen answer-semantic gate still rejects the model",
        font=font(19),
        fill=MUTED,
    )
    draw.rounded_rectangle((1430, 48, 1730, 112), radius=7, fill=RED)
    centered(
        draw,
        (1430, 48, 1730, 112),
        "NOT QUALIFIED",
        size=22,
        fill=WHITE,
        bold=True,
    )

    pipeline_y = 155
    boxes = (
        (80, pipeline_y, 430, pipeline_y + 130),
        (550, pipeline_y, 1190, pipeline_y + 130),
        (1310, pipeline_y, 1720, pipeline_y + 130),
    )
    for bounds in boxes:
        draw.rounded_rectangle(bounds, radius=7, fill=WHITE, outline=LINE, width=2)
    centered(
        draw,
        boxes[0],
        "PAIRED PROMPT RASTERS\n5 fonts + paraphrase view",
        size=19,
        bold=True,
    )
    centered(
        draw,
        boxes[1],
        "90.75M IMAGE-ONLY READER + ANSWER MAP\nshared visual path; no angle cap",
        size=19,
        bold=True,
    )
    centered(
        draw,
        boxes[2],
        "CONTINUOUS OUTPUT\nprompt state + answer state + length",
        size=19,
        bold=True,
    )
    arrow(draw, (430, pipeline_y + 65), (550, pipeline_y + 65))
    arrow(draw, (1190, pipeline_y + 65), (1310, pipeline_y + 65))
    draw.rounded_rectangle((445, 300, 1355, 338), radius=7, fill="#f5eadc")
    centered(
        draw,
        (455, 300, 1345, 338),
        "TRAINING ONLY: BGE targets, Qwen-prepared paraphrases, and detached hard negatives",
        size=14,
        fill=AMBER,
        bold=True,
    )

    left = (70, 370, 575, 870)
    middle = (635, 370, 1200, 870)
    right = (1260, 370, 1730, 870)
    panel(draw, left, "Measured training", accent=GREEN)
    panel(draw, middle, "Reading versus answering", accent=BLUE)
    panel(draw, right, "Invariance and controls", accent=AMBER)

    final_rank = summary["stage_summaries"]["full-path-adaptation"]["rank_probe"][
        "answer_effective_rank"
    ]
    training_rows = (
        ("Updates", f"{summary['global_update']:,} / 8,000"),
        ("Elapsed", f"{summary['training_elapsed_seconds'] / 60:.2f} min"),
        ("Peak CUDA", f"{summary['peak_allocated_vram_bytes'] / 1024**3:.3f} GiB"),
        ("Parameters", f"{receipt['model_boundary']['total_parameters'] / 1e6:.2f}M"),
        ("Final answer rank", f"{final_rank:.2f}"),
        ("EMA gate", f"{ema['gate']['passed_conditions']} / {ema['gate']['total_conditions']}"),
    )
    for index, (label, value) in enumerate(training_rows):
        y = left[1] + 72 + index * 55
        draw.text((left[0] + 28, y), label, font=font(15), fill=MUTED)
        draw.text((left[0] + 220, y), value, font=font(17, bold=True), fill=INK)
    draw.rounded_rectangle(
        (left[0] + 28, left[3] - 82, left[2] - 28, left[3] - 28),
        radius=6,
        fill="#e6f0eb",
    )
    centered(
        draw,
        (left[0] + 28, left[3] - 82, left[2] - 28, left[3] - 28),
        "Finite, hash-pinned, one-4090 run",
        size=15,
        fill=GREEN,
        bold=True,
    )

    metric_bar(
        draw,
        x=middle[0] + 28,
        y=middle[1] + 70,
        width=middle[2] - middle[0] - 56,
        label="Prompt top-1",
        value=correct["prompt_state"]["top1"],
        threshold=thresholds["prompt_top1"],
    )
    metric_bar(
        draw,
        x=middle[0] + 28,
        y=middle[1] + 142,
        width=middle[2] - middle[0] - 56,
        label="Prompt paired cosine",
        value=correct["prompt_state"]["correct_cosine"],
        threshold=thresholds["prompt_cosine"],
    )
    metric_bar(
        draw,
        x=middle[0] + 28,
        y=middle[1] + 214,
        width=middle[2] - middle[0] - 56,
        label="Answer-state top-1",
        value=correct["answer_state"]["top1"],
        threshold=thresholds["answer_top1"],
    )
    metric_bar(
        draw,
        x=middle[0] + 28,
        y=middle[1] + 286,
        width=middle[2] - middle[0] - 56,
        label="Answer-state top-5",
        value=correct["answer_state"]["top5"],
        threshold=thresholds["answer_top5"],
    )
    metric_bar(
        draw,
        x=middle[0] + 28,
        y=middle[1] + 358,
        width=middle[2] - middle[0] - 56,
        label="Transition direction",
        value=ema["transition"]["transition_direction_cosine"],
        threshold=thresholds["transition_direction_cosine"],
    )
    draw.text(
        (middle[0] + 28, middle[3] - 50),
        f"Raw answer top-1: {100 * raw['correct']['answer_state']['top1']:.2f}% (also rejected)",
        font=font(14, bold=True),
        fill=MUTED,
    )

    metric_bar(
        draw,
        x=right[0] + 28,
        y=right[1] + 70,
        width=right[2] - right[0] - 56,
        label="Held-font prompt cosine",
        value=ema["font"]["prompt_state_cosine"],
        threshold=thresholds["held_prompt_cosine"],
    )
    metric_bar(
        draw,
        x=right[0] + 28,
        y=right[1] + 142,
        width=right[2] - right[0] - 56,
        label="Held-font answer cosine",
        value=ema["font"]["answer_state_cosine"],
        threshold=thresholds["held_answer_cosine"],
    )
    metric_bar(
        draw,
        x=right[0] + 28,
        y=right[1] + 214,
        width=right[2] - right[0] - 56,
        label="Paraphrase top-5",
        value=ema["paraphrase"]["top5"],
        threshold=thresholds["paraphrase_top5"],
    )
    metric_bar(
        draw,
        x=right[0] + 28,
        y=right[1] + 286,
        width=right[2] - right[0] - 56,
        label="Paraphrase answer cosine",
        value=ema["paraphrase"]["original_answer_cosine"],
        threshold=thresholds["paraphrase_answer_cosine"],
    )
    lower_is_better_row(
        draw,
        x=right[0] + 28,
        y=right[1] + 372,
        label="Visual-length MAE",
        value=correct["answer_state"]["length_mae"],
        threshold=thresholds["maximum_length_mae"],
    )
    draw.text(
        (right[0] + 28, right[1] + 420),
        "Font invariance passes for prompts; answer",
        font=font(14),
        fill=MUTED,
    )
    draw.text(
        (right[0] + 28, right[1] + 445),
        "and wording transfer remain below gate.",
        font=font(14),
        fill=MUTED,
    )

    draw.rounded_rectangle((70, 915, 1730, 1055), radius=7, fill=INK)
    centered(
        draw,
        (90, 930, 560, 1040),
        "SEALED SPLIT\n0 ROWS OPENED",
        size=23,
        fill=WHITE,
        bold=True,
    )
    centered(
        draw,
        (665, 930, 1135, 1040),
        "RASTER RENDERER\nNOT AUTHORIZED",
        size=23,
        fill=WHITE,
        bold=True,
    )
    centered(
        draw,
        (1240, 930, 1710, 1040),
        "NEXT TEST\nMORE DATA + ANSWER DYNAMICS",
        size=21,
        fill=WHITE,
        bold=True,
    )
    draw.text(
        (70, 1080),
        "Evaluator retrieval probes continuous states only. V38 emits no answer pixels and no generated language.",
        font=font(14),
        fill=MUTED,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, optimize=True)


def main() -> None:
    args = parse_args()
    evidence = load_evidence(args)
    render(evidence, args.out)
    with Image.open(args.out) as generated:
        size = generated.size
    print(
        json.dumps(
            {
                "out": str(args.out),
                "sha256": file_sha256(args.out),
                "size": size,
                "decision": evidence["ema"]["gate"]["decision"],
                "sealed_rows_rendered": evidence["ema"]["data"][
                    "sealed_rows_rendered"
                ],
                "generates_raster": evidence["ema"]["checkpoint"]["boundary"][
                    "generates_raster"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
