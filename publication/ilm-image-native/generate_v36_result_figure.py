#!/usr/bin/env python3
"""Render the measured V36 decision from hash-pinned evidence artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = (
    ROOT / "artifacts/visual_semantic_plan_v36_20260814/training_summary.json"
)
DEFAULT_EMA = (
    ROOT / "artifacts/visual_semantic_plan_v36_20260814/development_report_ema.json"
)
DEFAULT_RAW = (
    ROOT / "artifacts/visual_semantic_plan_v36_20260814/development_report_raw.json"
)
DEFAULT_TRAIN_BANK = ROOT / "artifacts/visual_semantic_plan_v36_targets/train.pt"
DEFAULT_DEVELOPMENT_BANK = (
    ROOT / "artifacts/visual_semantic_plan_v36_targets/development.pt"
)
DEFAULT_PROTOCOL = ROOT / "references/visual_semantic_plan_v36_protocol.md"
DEFAULT_OUT = (
    Path(__file__).resolve().parent
    / "figures/visual_semantic_plan_v36_result.png"
)

EXPECTED_SHA256 = {
    "summary": "38471c96ec5c684d2c5e34ea0769fab1c21e5615ee3a7f90371b2d136913c1f8",
    "ema": "004ee7af3da0a7f2f7fe9ebf276402774827823af46af6f25d6494e01cd75ad5",
    "raw": "7638e3c173bc5e25c1927e3da2ed97f31d3d0fa0bdb7b55a0ef45c97d92819da",
    "train_bank": "b0d8b9acde6b8b2e12a4660680610eddc8814b5f9f7d9dcf3ef67476037a4161",
    "development_bank": "099c8555864cb9980dcf52427821e0cdbd1d9f1608555e5a568dcd9694af23e5",
    "protocol": "7e637698af08803c4ef509c564160ea63e5a952398a1e50cd924ec888167d6fb",
}

FONT_ROOT = Path("/usr/share/fonts/truetype/dejavu")
INK = "#152f38"
MUTED = "#526970"
LINE = "#a8b7bc"
PAPER = "#f2f5f5"
WHITE = "#ffffff"
TEAL = "#247b86"
GREEN = "#32745d"
AMBER = "#a86f2b"
RED = "#a74848"
BLUE = "#416c88"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--ema", type=Path, default=DEFAULT_EMA)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--train-bank", type=Path, default=DEFAULT_TRAIN_BANK)
    parser.add_argument(
        "--development-bank", type=Path, default=DEFAULT_DEVELOPMENT_BANK
    )
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


def validate_hashes(args: argparse.Namespace) -> None:
    paths = {
        "summary": args.summary,
        "ema": args.ema,
        "raw": args.raw,
        "train_bank": args.train_bank,
        "development_bank": args.development_bank,
        "protocol": args.protocol,
    }
    for name, path in paths.items():
        actual = file_sha256(path)
        if actual != EXPECTED_SHA256[name]:
            raise ValueError(
                f"V36 evidence SHA-256 changed for {name}: "
                f"expected {EXPECTED_SHA256[name]}, got {actual}"
            )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def load_bank(path: Path) -> Mapping[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a target-bank mapping in {path}")
    if value.get("architecture") != "visual-semantic-plan-target-bank-v36":
        raise ValueError(f"unexpected target-bank architecture in {path}")
    return value


def effective_rank(values: torch.Tensor) -> float:
    values = torch.nn.functional.normalize(values.float(), dim=-1)
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / values.shape[0]
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    return float(eigenvalues.sum().square() / eigenvalues.square().sum())


def load_evidence(args: argparse.Namespace) -> dict[str, Any]:
    validate_hashes(args)
    summary = load_json(args.summary)
    ema = load_json(args.ema)
    raw = load_json(args.raw)
    train_bank = load_bank(args.train_bank)
    development_bank = load_bank(args.development_bank)

    if summary.get("global_update") != 6_000 or not summary.get("complete"):
        raise ValueError("V36 training coverage changed")
    if (
        ema.get("experiment") != "visual-semantic-plan-v36"
        or ema.get("split") != "development"
    ):
        raise ValueError("unexpected V36 EMA report")
    if ema.get("weight_route") != "selective-ema":
        raise ValueError("V36 EMA route changed")
    if raw.get("weight_route") != "raw":
        raise ValueError("V36 raw route changed")
    if ema.get("gate", {}).get("decision") != "not-qualified":
        raise ValueError("V36 gate decision changed")
    if ema.get("sealed_opened") or raw.get("sealed_opened"):
        raise ValueError("V36 sealed split was unexpectedly opened")
    if not ema.get("finite") or not raw.get("finite"):
        raise ValueError("V36 report is non-finite")

    train_lengths = train_bank["lengths"].float()
    development_lengths = development_bank["lengths"].float()
    train_plans = train_bank["global_plans"]
    return {
        "summary": summary,
        "ema": ema,
        "raw": raw,
        "train_length_mean": float(train_lengths.mean()),
        "development_length_mean": float(development_lengths.mean()),
        "train_target_effective_rank": effective_rank(train_plans),
    }


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
    maximum: float,
) -> None:
    draw.text((x, y), label, font=font(15), fill=INK)
    draw.text(
        (x + width - 155, y),
        f"{100 * value:5.2f}% / {100 * threshold:5.2f}%",
        font=font(14, bold=True),
        fill=GREEN if value >= threshold else RED,
    )
    bar_y = y + 27
    draw.rectangle((x, bar_y, x + width, bar_y + 14), fill="#dce4e6")
    draw.rectangle(
        (x, bar_y, x + int(width * min(value / maximum, 1.0)), bar_y + 14),
        fill=GREEN if value >= threshold else RED,
    )
    threshold_x = x + int(width * threshold / maximum)
    draw.line((threshold_x, bar_y - 4, threshold_x, bar_y + 18), fill=INK, width=2)


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
    summary = evidence["summary"]
    ema = evidence["ema"]
    raw = evidence["raw"]
    image = Image.new("RGB", (1800, 1120), PAPER)
    draw = ImageDraw.Draw(image)

    draw.text((70, 48), "V36 VISUAL SEMANTIC PLAN", font=font(34, bold=True), fill=INK)
    draw.text(
        (70, 96),
        "A complete one-GPU semantic-planning run, rejected by its frozen development gate",
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

    pipeline_y = 160
    boxes = (
        (90, pipeline_y, 450, pipeline_y + 130),
        (570, pipeline_y, 1170, pipeline_y + 130),
        (1290, pipeline_y, 1710, pipeline_y + 130),
    )
    for bounds in boxes:
        draw.rounded_rectangle(bounds, radius=7, fill=WHITE, outline=LINE, width=2)
    centered(draw, boxes[0], "PROMPT RASTER\n3 x 16 x 1024 + mask", size=20, bold=True)
    centered(
        draw,
        boxes[1],
        "93.47M IMAGE-ONLY PLANNER\nPixel-Linguist reader + causal plan queries",
        size=20,
        bold=True,
    )
    centered(
        draw,
        boxes[2],
        "CONTINUOUS OUTPUT\n5 x 768 plans + visual length",
        size=20,
        bold=True,
    )
    arrow(draw, (450, pipeline_y + 65), (570, pipeline_y + 65))
    arrow(draw, (1170, pipeline_y + 65), (1290, pipeline_y + 65))
    draw.text(
        (570, pipeline_y + 142),
        "No strings, token/Unicode IDs, OCR, candidates, answer teacher, codebook, or renderer at runtime",
        font=font(15),
        fill=MUTED,
    )

    left = (70, 355, 580, 865)
    middle = (645, 355, 1190, 865)
    right = (1255, 355, 1730, 865)
    panel(draw, left, "Measured training", accent=GREEN)
    panel(draw, middle, "Held-out semantic gate", accent=RED)
    panel(draw, right, "Post-result diagnosis", accent=AMBER)

    training_rows = (
        ("Updates", f"{summary['global_update']:,} / 6,000"),
        ("Elapsed", f"{summary['training_elapsed_seconds'] / 60:.2f} min"),
        (
            "Peak CUDA",
            f"{summary['peak_allocated_vram_bytes'] / 1024**3:.3f} GiB",
        ),
        ("EMA checkpoint", "finite; no target tensors"),
        ("Development", f"{ema['correct']['samples']} unseen pairs"),
        ("Gate checks", f"{ema['gate']['passed_conditions']} / {ema['gate']['total_conditions']}"),
    )
    for index, (label, value) in enumerate(training_rows):
        y = left[1] + 70 + index * 58
        draw.text((left[0] + 28, y), label, font=font(15), fill=MUTED)
        draw.text((left[0] + 220, y), value, font=font(17, bold=True), fill=INK)
    draw.rounded_rectangle(
        (left[0] + 28, left[3] - 92, left[2] - 28, left[3] - 30),
        radius=6,
        fill="#e6f0eb",
    )
    centered(
        draw,
        (left[0] + 28, left[3] - 92, left[2] - 28, left[3] - 30),
        "Integrity and one-4090 resource bounds passed",
        size=15,
        fill=GREEN,
        bold=True,
    )

    metric_bar(
        draw,
        x=middle[0] + 28,
        y=middle[1] + 72,
        width=middle[2] - middle[0] - 56,
        label="EMA top-1",
        value=ema["correct"]["top1"],
        threshold=0.08,
        maximum=0.30,
    )
    metric_bar(
        draw,
        x=middle[0] + 28,
        y=middle[1] + 145,
        width=middle[2] - middle[0] - 56,
        label="EMA top-5",
        value=ema["correct"]["top5"],
        threshold=0.25,
        maximum=0.50,
    )
    metric_bar(
        draw,
        x=middle[0] + 28,
        y=middle[1] + 218,
        width=middle[2] - middle[0] - 56,
        label="EMA MRR",
        value=ema["correct"]["mrr"],
        threshold=0.15,
        maximum=0.35,
    )
    metric_bar(
        draw,
        x=middle[0] + 28,
        y=middle[1] + 291,
        width=middle[2] - middle[0] - 56,
        label="Counterfactual assignment",
        value=ema["counterfactual"]["assignment_rate"],
        threshold=0.70,
        maximum=1.0,
    )
    draw.text(
        (middle[0] + 28, middle[1] + 374),
        f"Raw top-1: {100 * raw['correct']['top1']:.2f}%",
        font=font(16, bold=True),
        fill=INK,
    )
    draw.text(
        (middle[0] + 28, middle[1] + 408),
        f"Length MAE: {ema['correct']['length_mae']:.2f} patches (gate <= 4)",
        font=font(16, bold=True),
        fill=RED,
    )
    draw.text(
        (middle[0] + 28, middle[1] + 442),
        f"Held-font plan cosine: {ema['font']['prompt_plan_cosine']:.3f} (gate >= .85)",
        font=font(16),
        fill=RED,
    )

    train_length = evidence["train_length_mean"]
    development_length = evidence["development_length_mean"]
    rank = evidence["train_target_effective_rank"]
    draw.text(
        (right[0] + 28, right[1] + 74),
        "Occupancy-mask distribution shift",
        font=font(17, bold=True),
        fill=INK,
    )
    scale = 300 / 64
    for index, (label, value, color) in enumerate(
        (
            ("train", train_length, RED),
            ("development", development_length, BLUE),
        )
    ):
        y = right[1] + 120 + index * 64
        draw.text((right[0] + 28, y), label, font=font(15), fill=MUTED)
        draw.rectangle((right[0] + 140, y, right[0] + 440, y + 20), fill="#dce4e6")
        draw.rectangle(
            (right[0] + 140, y, right[0] + 140 + int(value * scale), y + 20),
            fill=color,
        )
        draw.text(
            (right[0] + 140, y + 26),
            f"mean active patches: {value:.2f}",
            font=font(14),
            fill=INK,
        )
    diagnosis = (
        "Mask was measured after contrast/noise.",
        "Shifted white background became active.",
        f"Train target effective rank: {rank:.2f} / 768.",
        "Frozen visual semantics also remained weak.",
        "The defect is contributory, not a sole cause.",
    )
    for index, value in enumerate(diagnosis):
        draw.text(
            (right[0] + 28, right[1] + 280 + index * 36),
            value,
            font=font(15),
            fill=INK if index != 2 else RED,
        )

    draw.rounded_rectangle((70, 915, 1730, 1055), radius=7, fill=INK)
    centered(
        draw,
        (90, 930, 560, 1040),
        "SEALED SPLIT\nCLOSED",
        size=23,
        fill=WHITE,
        bold=True,
    )
    centered(
        draw,
        (665, 930, 1135, 1040),
        "V36-R RENDERER\nNOT OPENED",
        size=23,
        fill=WHITE,
        bold=True,
    )
    centered(
        draw,
        (1240, 930, 1710, 1040),
        "NEXT TEST\nSEMANTIC DISTILLATION",
        size=23,
        fill=WHITE,
        bold=True,
    )
    draw.text(
        (70, 1080),
        "Candidate retrieval is evaluator-only. V36 predicts plans, not answer pixels or generated language.",
        font=font(14),
        fill=MUTED,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, optimize=True)


def main() -> None:
    args = parse_args()
    evidence = load_evidence(args)
    render(evidence, args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "sha256": file_sha256(args.out),
                "size": Image.open(args.out).size,
                "decision": evidence["ema"]["gate"]["decision"],
                "sealed_opened": evidence["ema"]["sealed_opened"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
