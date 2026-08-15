#!/usr/bin/env python3
"""Render the measured V48 result from hash-pinned publication evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
DEFAULT_EVIDENCE = HERE / "evidence/v48"
DEFAULT_OUT = HERE / "figures/visual_future_block_language_v48_result.png"
EXPECTED_SHA256 = {
    "development_report.json": (
        "2d3e5857ce7c50c8b3b85f5056f7c9f26b3ed3b6eb162c68c3ad7ca5a94742fa"
    ),
    "development_summary.json": (
        "a1371bd7be804100e351e83e04b1df6d56dcd03fdb13cc89cdfc64d461f59d9d"
    ),
    "training_summary.json": (
        "fe658885bed9ebc10cb8ff4cf61a487c2b0334b6feabf3e2108059f101ed0c7b"
    ),
    "target_proposal_visible_rollout.png": (
        "bce170e2e5342c0d7d61735e98bd14f8668c00583ac63f3c218bd3a8326aeeea"
    ),
}

FONT_ROOT = Path("/usr/share/fonts/truetype/dejavu")
PAPER = "#f3f6f5"
WHITE = "#ffffff"
INK = "#183039"
MUTED = "#617176"
LINE = "#b9c7c7"
TEAL = "#147d83"
PALE_TEAL = "#e4f0ef"
BLUE = "#426f8d"
PALE_BLUE = "#e6edf2"
GREEN = "#2f765a"
PALE_GREEN = "#e4efe9"
RED = "#a3433c"
PALE_RED = "#f3e5e2"
GOLD = "#9a721f"
PALE_GOLD = "#f3ecd9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_evidence(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Image.Image]:
    for name, expected in EXPECTED_SHA256.items():
        measured = file_sha256(root / name)
        if measured != expected:
            raise ValueError(f"V48 evidence hash mismatch for {name}: {measured}")

    report = load_json(root / "development_report.json")
    summary = load_json(root / "development_summary.json")
    training = load_json(root / "training_summary.json")
    raster = Image.open(root / "target_proposal_visible_rollout.png").convert("RGB")

    gates = report.get("gates", {})
    integrity = report.get("integrity", {})
    if report.get("all_gates_pass") or len(gates) != 16 or sum(gates.values()) != 11:
        raise ValueError("V48 report does not record the frozen 11/16 decision")
    if not report.get("all_integrity_checks_pass"):
        raise ValueError("V48 report does not pass the integrity aggregate")
    if len(integrity) != 23 or not all(integrity.values()):
        raise ValueError("V48 report does not pass all 23 positive integrity checks")
    if report.get("frozen_partition_opened"):
        raise ValueError("V48 frozen partition was opened")
    if not report.get("source_receipt", {}).get("valid"):
        raise ValueError("V48 amended source receipt is invalid")
    if not report.get("source_receipt", {}).get("documented_evaluator_amendment"):
        raise ValueError("V48 evaluator amendment is not documented")
    if report.get("checkpoint_update") != 10_000 or training.get("update") != 10_000:
        raise ValueError("V48 figure requires the production checkpoint")
    if report.get("checkpoint_sha256") != training.get("checkpoint_sha256"):
        raise ValueError("V48 report and training summary checkpoints differ")
    if training.get("segments_consumed") != 160_000:
        raise ValueError("V48 training summary has an unexpected segment count")
    if report.get("audit_receipts", {}).get("raster_sheet_sha256") != EXPECTED_SHA256[
        "target_proposal_visible_rollout.png"
    ]:
        raise ValueError("V48 report does not bind the displayed raster sheet")
    if summary.get("v48_full_top1") != report["language"]["full_top1"]:
        raise ValueError("V48 summary and report language metrics differ")
    if summary.get("matched_v42_full_top1") != report["matched_v42"]["full_top1"]:
        raise ValueError("V48 summary and report V42 controls differ")
    return report, summary, training, raster


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = FONT_ROOT / name
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def card(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    *,
    fill: str = WHITE,
    outline: str = LINE,
    width: int = 2,
) -> None:
    draw.rounded_rectangle(bounds, radius=8, fill=fill, outline=outline, width=width)


def centered_text(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    value: str,
    *,
    size: int,
    fill: str = INK,
    bold: bool = False,
    spacing: int = 4,
) -> None:
    face = font(size, bold=bold)
    box = draw.multiline_textbbox(
        (0, 0), value, font=face, spacing=spacing, align="center"
    )
    width = box[2] - box[0]
    height = box[3] - box[1]
    left, top, right, bottom = bounds
    draw.multiline_text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2),
        value,
        font=face,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    draw.line((*start, *end), fill=TEAL, width=4)
    draw.polygon(
        ((end[0], end[1]), (end[0] - 12, end[1] - 7), (end[0] - 12, end[1] + 7)),
        fill=TEAL,
    )


def metric_row(
    draw: ImageDraw.ImageDraw,
    *,
    y: int,
    label: str,
    reference: str,
    measured: str,
    decision: str,
    passed: bool,
    left: int,
    right: int,
) -> None:
    draw.text((left, y), label, font=font(15, bold=True), fill=INK)
    ref_left = right - 438
    card(draw, (ref_left, y - 5, ref_left + 122, y + 28), fill=PALE_BLUE)
    centered_text(
        draw,
        (ref_left, y - 5, ref_left + 122, y + 28),
        reference,
        size=12,
        fill=BLUE,
        bold=True,
    )
    draw.text((ref_left + 136, y), "->", font=font(15, bold=True), fill=MUTED)
    color = GREEN if passed else RED
    pale = PALE_GREEN if passed else PALE_RED
    card(
        draw,
        (ref_left + 178, y - 5, ref_left + 303, y + 28),
        fill=pale,
        outline=color,
    )
    centered_text(
        draw,
        (ref_left + 178, y - 5, ref_left + 303, y + 28),
        measured,
        size=12,
        fill=color,
        bold=True,
    )
    draw.text((right - 116, y), decision, font=font(12, bold=True), fill=color)


def horizon_bar(
    draw: ImageDraw.ImageDraw,
    *,
    y: int,
    horizon: str,
    measured: float,
    control: float,
    x: int,
    width: int,
    scale_max: float = 0.18,
) -> None:
    draw.text((x, y - 3), f"h{horizon}", font=font(13, bold=True), fill=INK)
    bar_left = x + 42
    bar_width = width - 145
    measured_width = int(bar_width * min(measured / scale_max, 1.0))
    control_width = int(bar_width * min(control / scale_max, 1.0))
    draw.rounded_rectangle(
        (bar_left, y, bar_left + bar_width, y + 11),
        radius=5,
        fill=PALE_BLUE,
    )
    draw.rounded_rectangle(
        (bar_left, y, bar_left + measured_width, y + 11),
        radius=5,
        fill=TEAL,
    )
    draw.rounded_rectangle(
        (bar_left, y + 17, bar_left + bar_width, y + 28),
        radius=5,
        fill="#eeeeea",
    )
    draw.rounded_rectangle(
        (bar_left, y + 17, bar_left + control_width, y + 28),
        radius=5,
        fill=GOLD,
    )
    gain = measured - control
    color = GREEN if gain > 0.01 else RED
    draw.text(
        (bar_left + bar_width + 14, y + 2),
        f"{100 * measured:.2f}%",
        font=font(12, bold=True),
        fill=TEAL,
    )
    draw.text(
        (bar_left + bar_width + 14, y + 18),
        f"{gain:+.2%}",
        font=font(11, bold=True),
        fill=color,
    )


def render(
    report: dict[str, Any],
    summary: dict[str, Any],
    training: dict[str, Any],
    raster: Image.Image,
    output: Path,
) -> None:
    image = Image.new("RGB", (1800, 1500), PAPER)
    draw = ImageDraw.Draw(image)

    draw.text(
        (62, 30),
        "VISUAL FUTURE BLOCK LANGUAGE: V48",
        font=font(30, bold=True),
        fill=INK,
    )
    draw.text(
        (62, 72),
        "Ordered raster language survives; deterministic point writing does not",
        font=font(16),
        fill=MUTED,
    )
    card(draw, (1450, 26, 1738, 102), fill=RED, outline=RED)
    centered_text(
        draw,
        (1462, 30, 1726, 98),
        "NON-QUALIFYING\n11 / 16 GATES",
        size=18,
        fill=WHITE,
        bold=True,
    )

    pipeline = (
        (62, 300, "ORDERED RASTERS\n64 x 32 x 32"),
        (336, 615, "FIXED DCT FIELD\nexact image carrier"),
        (651, 930, "CAUSAL READER\n16.28M from scratch"),
        (966, 1245, "FOUR POINT FUTURES\nshared visual head"),
        (1281, 1738, "INVERSE DCT + REREAD\nvisible raster feedback"),
    )
    for index, (left, right, label) in enumerate(pipeline):
        highlight = index == 3
        card(
            draw,
            (left, 132, right, 264),
            fill=PALE_GOLD if highlight else WHITE,
            outline=GOLD if highlight else LINE,
        )
        centered_text(
            draw,
            (left + 8, 140, right - 8, 256),
            label,
            size=15,
            fill=GOLD if highlight else INK,
            bold=True,
        )
    for first, second in zip(pipeline, pipeline[1:]):
        arrow(draw, (first[1] + 7, 198), (second[0] - 11, 198))

    language_box = (62, 300, 880, 710)
    writer_box = (920, 300, 1738, 710)
    card(draw, language_box)
    card(draw, writer_box)

    draw.text((88, 324), "HELD-OUT LANGUAGE", font=font(20, bold=True), fill=INK)
    draw.text(
        (88, 356),
        "2,048 fixed windows; candidate bank used only after prediction",
        font=font(13),
        fill=MUTED,
    )
    language = report["language"]
    v42 = report["matched_v42"]
    pairs = report["counterfactual_pairs"]
    metric_row(
        draw,
        y=408,
        label="Full top-1 vs V42",
        reference=f"{100 * v42['full_top1']:.2f}%",
        measured=f"{100 * language['full_top1']:.2f}%",
        decision="FAIL",
        passed=False,
        left=88,
        right=854,
    )
    metric_row(
        draw,
        y=466,
        label="Full vs bigram",
        reference=f"{100 * language['bigram_top1']:.2f}%",
        measured=f"{100 * language['full_top1']:.2f}%",
        decision="PASS",
        passed=True,
        left=88,
        right=854,
    )
    metric_row(
        draw,
        y=524,
        label="Ordered vs shuffled",
        reference=f"{100 * language['shuffled_top1']:.2f}%",
        measured=f"{100 * language['full_top1']:.2f}%",
        decision="PASS",
        passed=True,
        left=88,
        right=854,
    )
    metric_row(
        draw,
        y=582,
        label="Pair arm accuracy",
        reference="gate >55.00%",
        measured=f"{100 * pairs['full_arm_accuracy']:.2f}%",
        decision="FAIL",
        passed=False,
        left=88,
        right=854,
    )
    card(draw, (88, 636, 854, 685), fill=PALE_TEAL, outline=TEAL)
    centered_text(
        draw,
        (98, 640, 844, 681),
        "ORDERED HISTORY HELPS, BUT V48 DOES NOT BEAT THE V42 READER",
        size=13,
        fill=TEAL,
        bold=True,
    )

    draw.text((946, 324), "VISIBLE WRITER LOOP", font=font(20, bold=True), fill=INK)
    draw.text(
        (946, 356),
        "Direct thresholded output and four-step pixel-only feedback",
        font=font(13),
        fill=MUTED,
    )
    direct = report["direct_raster"]
    metric_row(
        draw,
        y=408,
        label="Visible identity",
        reference="gate >15.00%",
        measured=f"{100 * direct['visible_identity_top1']:.2f}%",
        decision="FAIL",
        passed=False,
        left=946,
        right=1712,
    )
    metric_row(
        draw,
        y=466,
        label="Binary pixel F1",
        reference="gate >0.460",
        measured=f"{direct['visible_pixel_f1']:.3f}",
        decision="PASS",
        passed=True,
        left=946,
        right=1712,
    )
    retention = direct["visible_identity_top1"] / direct["anchor_identity_top1"]
    metric_row(
        draw,
        y=524,
        label="Identity retention",
        reference="gate >=85.0%",
        measured=f"{100 * retention:.2f}%",
        decision="FAIL",
        passed=False,
        left=946,
        right=1712,
    )
    loop = report["closed_loop"]
    metric_row(
        draw,
        y=582,
        label="Closed-loop mean id.",
        reference="unigram 1.42%",
        measured=f"{100 * loop['mean_identity_top1']:.2f}%",
        decision="PASS",
        passed=True,
        left=946,
        right=1712,
    )
    step_values = [loop["steps"][str(index)]["identity_top1"] for index in range(1, 5)]
    card(draw, (946, 636, 1712, 685), fill=PALE_RED, outline=RED)
    centered_text(
        draw,
        (956, 640, 1702, 681),
        "VISIBLE STEP IDENTITY: "
        + " -> ".join(f"{100 * value:.2f}%" for value in step_values),
        size=13,
        fill=RED,
        bold=True,
    )

    evidence_box = (62, 750, 1180, 1327)
    card(draw, evidence_box)
    draw.text(
        (88, 776),
        "ACTUAL TARGET / POINT FIELD / VISIBLE / ROLLOUT RASTERS",
        font=font(19, bold=True),
        fill=INK,
    )
    draw.text(
        (88, 808),
        "Four held-out rows; the three generated groups are dense conditional averages",
        font=font(13),
        fill=MUTED,
    )

    target_width = 1030
    target_height = round(target_width * raster.height / raster.width)
    enlarged = raster.resize(
        (target_width, target_height),
        resample=Image.Resampling.NEAREST,
    )
    raster_left = 106
    raster_top = 892
    group_width = target_width / 4
    group_labels = (
        ("TARGET", GREEN, PALE_GREEN),
        ("POINT FIELD", GOLD, PALE_GOLD),
        ("VISIBLE", RED, PALE_RED),
        ("ROLLOUT", RED, PALE_RED),
    )
    for index, (label, color, fill) in enumerate(group_labels):
        left = round(raster_left + index * group_width)
        right = round(raster_left + (index + 1) * group_width - 8)
        card(draw, (left, 846, right, 880), fill=fill, outline=color, width=1)
        centered_text(
            draw,
            (left, 847, right, 879),
            label,
            size=11,
            fill=color,
            bold=True,
        )
    draw.rectangle(
        (
            raster_left - 2,
            raster_top - 2,
            raster_left + target_width + 2,
            raster_top + target_height + 2,
        ),
        fill=WHITE,
        outline=LINE,
        width=2,
    )
    image.paste(enlarged, (raster_left, raster_top))
    for index in range(1, 4):
        x = round(raster_left + index * group_width)
        draw.line((x, raster_top - 2, x, raster_top + target_height + 2), fill=RED, width=2)

    card(draw, (88, 1204, 1154, 1295), fill=PALE_RED, outline=RED)
    centered_text(
        draw,
        (108, 1212, 1134, 1287),
        "PIXEL F1 PASSES, YET THE OUTPUT IS NOT A CLEAN GLYPH SAMPLE.\n"
        "REREADING ITS OWN SPECKLE CAUSES IMMEDIATE IDENTITY DECAY.",
        size=14,
        fill=RED,
        bold=True,
    )

    horizon_box = (1220, 750, 1738, 1104)
    card(draw, horizon_box)
    draw.text((1246, 776), "FOUR FUTURE HORIZONS", font=font(19, bold=True), fill=INK)
    draw.text(
        (1246, 808),
        "teal: V48 identity  |  gold: offset control",
        font=font(12),
        fill=MUTED,
    )
    horizons = report["future"]["horizons"]
    controls = report["future"]["offset_conditional_control"]
    for offset, horizon in enumerate(("1", "2", "3", "4")):
        horizon_bar(
            draw,
            y=850 + 58 * offset,
            horizon=horizon,
            measured=horizons[horizon]["top1"],
            control=controls[horizon]["top1"],
            x=1246,
            width=466,
        )
    draw.text(
        (1246, 1070),
        "Only horizon 1 clears the fixed +1 point margin.",
        font=font(12, bold=True),
        fill=RED,
    )

    decision_box = (1220, 1132, 1738, 1327)
    card(draw, decision_box)
    draw.text((1246, 1156), "FROZEN DECISION", font=font(19, bold=True), fill=INK)
    draw.text((1246, 1194), "11 PASS", font=font(15, bold=True), fill=GREEN)
    draw.text(
        (1338, 1194),
        "order | diversity | terminal stability | finite loop",
        font=font(11),
        fill=INK,
    )
    draw.text((1246, 1228), "5 FAIL", font=font(15, bold=True), fill=RED)
    draw.text(
        (1330, 1228),
        "V42 gain | all horizons | binding | visible id. | retention",
        font=font(11),
        fill=INK,
    )
    draw.line((1246, 1260, 1712, 1260), fill=LINE, width=2)
    draw.text(
        (1246, 1274),
        "23/23 integrity checks | frozen partition closed",
        font=font(12, bold=True),
        fill=TEAL,
    )

    draw.line((62, 1363, 1738, 1363), fill=LINE, width=2)
    elapsed = training["elapsed_seconds"]
    peak = report["peak_allocated_vram_gib"]
    draw.text(
        (162, 1384),
        (
            f"10,000 updates | 160,000 image segments | {elapsed / 60:.1f} min train | "
            f"peak train+audit {peak:.3f} GiB | checkpoint "
            f"{training['checkpoint_sha256'][:12]}..."
        ),
        font=font(14, bold=True),
        fill=MUTED,
    )
    if summary["v48_full_minus_v42_top1"] >= 0:
        raise ValueError("V48 unexpectedly matches or exceeds V42 in frozen evidence")
    centered_text(
        draw,
        (120, 1420, 1680, 1480),
        "Supported conclusion: retain the ordered visual reader; replace point-average writing "
        "with sampled raster density and visible-error recovery.",
        size=15,
        fill=RED,
        bold=True,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)


def main() -> None:
    args = parse_args()
    report, summary, training, raster = load_evidence(args.evidence)
    render(report, summary, training, raster, args.out)
    print(f"saved {args.out} sha256={file_sha256(args.out)}")


if __name__ == "__main__":
    main()
