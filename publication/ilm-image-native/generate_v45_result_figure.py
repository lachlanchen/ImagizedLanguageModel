#!/usr/bin/env python3
"""Render the measured V45 result from hash-pinned publication evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
DEFAULT_EVIDENCE = HERE / "evidence/v45"
DEFAULT_OUT = HERE / "figures/noise_limited_retinal_field_v45_result.png"
EXPECTED_SHA256 = {
    "report.json": "1d7c0867870b89a86b3da91913e0877cf9ef6b420a2eddcd0118f56c10357e7d",
    "checkpoint_receipt.json": (
        "fde8bc331ab13b86c5485cbc7aede2a24b686343bddb176ccc0392fe7980be2f"
    ),
}

FONT_ROOT = Path("/usr/share/fonts/truetype/dejavu")
PAPER = "#f4f6f5"
WHITE = "#ffffff"
INK = "#173039"
MUTED = "#5f7075"
LINE = "#b8c4c5"
TEAL = "#177d83"
BLUE = "#426f8d"
GREEN = "#2f765a"
PALE_GREEN = "#e4efe9"
RED = "#9a403a"
PALE_RED = "#f3e5e2"
GOLD = "#a67822"
PALE_GOLD = "#f3ecd9"
PALE_BLUE = "#e5edf2"


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


def load_evidence(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payloads = []
    for name, expected in EXPECTED_SHA256.items():
        path = root / name
        measured = file_sha256(path)
        if measured != expected:
            raise ValueError(f"V45 evidence hash mismatch for {name}: {measured}")
        with path.open(encoding="utf-8") as handle:
            payloads.append(json.load(handle))
    report, receipt = payloads
    if report.get("claim_status") != "qualified-retinal-field":
        raise ValueError("V45 report is not a qualified retinal field")
    gates = report.get("gates", {})
    if len(gates) != 13 or not all(gates.values()):
        raise ValueError("V45 report does not pass all 13 gates")
    if report.get("smoke_only") or not report.get("checkpoint_reload_verified"):
        raise ValueError("V45 report is smoke-only or failed checkpoint reload")
    for key in ("checkpoint_sha256", "field_state_sha256"):
        if report.get(key) != receipt.get(key):
            raise ValueError(f"V45 report and checkpoint receipt differ at {key}")
    if report["manifest"]["sha256"] != receipt["manifest"]["sha256"]:
        raise ValueError("V45 report and checkpoint receipt use different corpora")
    if (
        report["holdout_pair_receipt"]["sha256"]
        != receipt["holdout_pair_receipt"]["sha256"]
    ):
        raise ValueError("V45 report and checkpoint receipt use different pairs")
    return report, receipt


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
) -> None:
    draw.rounded_rectangle(bounds, radius=7, fill=fill, outline=outline, width=2)


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
    text_box = draw.multiline_textbbox(
        (0, 0), value, font=face, spacing=spacing, align="center"
    )
    width = text_box[2] - text_box[0]
    height = text_box[3] - text_box[1]
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
    *,
    color: str = TEAL,
) -> None:
    draw.line((*start, *end), fill=color, width=4)
    draw.polygon(
        (
            (end[0], end[1]),
            (end[0] - 12, end[1] - 7),
            (end[0] - 12, end[1] + 7),
        ),
        fill=color,
    )


def comparison_row(
    draw: ImageDraw.ImageDraw,
    *,
    y: int,
    label: str,
    before: str,
    after: str,
    note: str,
    left: int,
    right: int,
    good: bool = True,
) -> None:
    draw.text((left, y), label, font=font(15, bold=True), fill=INK)
    value_left = right - 390
    draw.rounded_rectangle(
        (value_left, y - 3, value_left + 112, y + 25),
        radius=4,
        fill=PALE_BLUE,
    )
    centered_text(
        draw,
        (value_left, y - 3, value_left + 112, y + 25),
        before,
        size=14,
        fill=BLUE,
        bold=True,
    )
    draw.text((value_left + 126, y - 1), "->", font=font(16, bold=True), fill=MUTED)
    color = GREEN if good else RED
    fill = PALE_GREEN if good else PALE_RED
    draw.rounded_rectangle(
        (value_left + 165, y - 3, value_left + 277, y + 25),
        radius=4,
        fill=fill,
    )
    centered_text(
        draw,
        (value_left + 165, y - 3, value_left + 277, y + 25),
        after,
        size=14,
        fill=color,
        bold=True,
    )
    draw.text((right - 100, y), note, font=font(13, bold=True), fill=color)


def render(report: dict[str, Any], receipt: dict[str, Any], output: Path) -> None:
    image = Image.new("RGB", (1800, 1325), PAPER)
    draw = ImageDraw.Draw(image)

    draw.text(
        (62, 32),
        "NOISE-LIMITED RETINAL FIELD: V45",
        font=font(30, bold=True),
        fill=INK,
    )
    draw.text(
        (62, 74),
        "Preregistered fixed representation audit: exact raster path, conditioned image geometry",
        font=font(16),
        fill=MUTED,
    )
    card(draw, (1460, 28, 1738, 101), fill=GREEN, outline=GREEN)
    centered_text(
        draw,
        (1472, 32, 1726, 97),
        "REPRESENTATION QUALIFIED\n13 / 13 GATES",
        size=17,
        fill=WHITE,
        bold=True,
    )

    pipeline_y0, pipeline_y1 = 132, 268
    stages = (
        (62, 302, "32 x 32 RASTER\nx"),
        (340, 580, "ORTHONORMAL DCT\nd"),
        (618, 858, "TRAIN-ONLY CENTER\nd - mu"),
        (896, 1136, "MATRIX POWER\nA, p = 0.10"),
        (1174, 1414, "DIRECTION + RADIUS\nu, q"),
        (1452, 1738, "EXACT INVERSE\nx reconstructed"),
    )
    for index, (left, right, label) in enumerate(stages):
        fill = PALE_GREEN if index == len(stages) - 1 else WHITE
        outline = GREEN if index == len(stages) - 1 else LINE
        card(draw, (left, pipeline_y0, right, pipeline_y1), fill=fill, outline=outline)
        centered_text(
            draw,
            (left + 8, pipeline_y0 + 8, right - 8, pipeline_y1 - 8),
            label,
            size=15,
            fill=GREEN if index == len(stages) - 1 else INK,
            bold=True,
        )
    for first, second in zip(stages, stages[1:]):
        arrow(draw, (first[1] + 8, 200), (second[0] - 12, 200))
    draw.text(
        (646, 240),
        "8,000 training-only forms",
        font=font(12, bold=True),
        fill=TEAL,
    )
    draw.text(
        (1191, 240),
        "z / ||z||,  log ||z||",
        font=font(12, bold=True),
        fill=TEAL,
    )

    left_top = (62, 298, 882, 716)
    right_top = (918, 298, 1738, 716)
    left_bottom = (62, 746, 882, 1197)
    right_bottom = (918, 746, 1738, 1197)
    for bounds in (left_top, right_top, left_bottom, right_bottom):
        card(draw, bounds)

    draw.text((88, 322), "COMMON-MODE CONDITIONING", font=font(20, bold=True), fill=INK)
    draw.text(
        (88, 353),
        "Weighted 8,000-form training bank; raw field is the V42 target geometry",
        font=font(13),
        fill=MUTED,
    )
    raw_geometry = report["raw_geometry"]
    geometry = report["field_geometry"]
    comparison_row(
        draw,
        y=404,
        label="Common resultant",
        before=f"{raw_geometry['weighted_resultant_length']:.3f}",
        after=f"{geometry['weighted_resultant_length']:.3f}",
        note="-97.9%",
        left=88,
        right=856,
    )
    comparison_row(
        draw,
        y=456,
        label="Effective rank",
        before=f"{raw_geometry['effective_rank']:.1f}",
        after=f"{geometry['effective_rank']:.1f}",
        note="1.278x",
        left=88,
        right=856,
    )
    comparison_row(
        draw,
        y=508,
        label="Stable rank",
        before=f"{raw_geometry['stable_rank']:.1f}",
        after=f"{geometry['stable_rank']:.1f}",
        note="1.331x",
        left=88,
        right=856,
    )
    card(draw, (88, 568, 856, 681), fill=PALE_GREEN, outline=GREEN)
    centered_text(
        draw,
        (104, 576, 840, 673),
        "DOMINANT BACKGROUND / COMMON-STROKE DIRECTION REMOVED\n"
        "Selected: ridge = 0.50 mean variance  |  matrix power = 0.10\n"
        "Full whitening retained only as a descriptive control",
        size=15,
        fill=GREEN,
        bold=True,
    )

    draw.text((944, 322), "PINNED V44 PAIR HOLDOUT", font=font(20, bold=True), fill=INK)
    draw.text(
        (944, 353),
        "1,024 unseen training-partition pairs; identical suffix images, different target",
        font=font(13),
        fill=MUTED,
    )
    raw_pairs = report["raw_pair_geometry"]
    pairs = report["field_pair_geometry"]
    comparison_row(
        draw,
        y=404,
        label="Candidate cosine",
        before=f"{raw_pairs['candidate_pair_cosine']:.3f}",
        after=f"{pairs['candidate_pair_cosine']:.3f}",
        note="-0.499",
        left=944,
        right=1712,
    )
    comparison_row(
        draw,
        y=456,
        label="Delta norm p05",
        before=f"{raw_pairs['delta_norm_p05']:.3f}",
        after=f"{pairs['delta_norm_p05']:.3f}",
        note="2.102x",
        left=944,
        right=1712,
    )
    comparison_row(
        draw,
        y=508,
        label="Delta effective rank",
        before=f"{raw_pairs['delta_effective_rank']:.1f}",
        after=f"{pairs['delta_effective_rank']:.1f}",
        note="1.180x",
        left=944,
        right=1712,
    )
    comparison_row(
        draw,
        y=560,
        label="Delta stable rank",
        before=f"{raw_pairs['delta_stable_rank']:.1f}",
        after=f"{pairs['delta_stable_rank']:.1f}",
        note="1.115x",
        left=944,
        right=1712,
    )
    card(draw, (944, 616, 1712, 681), fill=PALE_GREEN, outline=GREEN)
    centered_text(
        draw,
        (960, 622, 1696, 675),
        "ALL FOUR DISPLACEMENT GATES PASS  |  HOLDOUT RECEIPT EXACT",
        size=15,
        fill=GREEN,
        bold=True,
    )

    draw.text((88, 770), "EXACTNESS + NUISANCE CONTINUITY", font=font(20, bold=True), fill=INK)
    draw.text(
        (88, 801),
        "Seven audited banks: fit, two held fonts, four cardinal one-pixel shifts",
        font=font(13),
        fill=MUTED,
    )
    roundtrip = report["roundtrip"]
    small_cards = (
        (88, 842, 320, 948, f"{int(roundtrip['examples']):,}\nRASTERS", BLUE),
        (
            338,
            842,
            570,
            948,
            f"{roundtrip['maximum_dct_absolute_error']:.2e}\nMAX DCT ERROR",
            TEAL,
        ),
        (588, 842, 856, 948, "100% / 1.000\nPIXEL ACC. / INK F1", GREEN),
    )
    for left, top, right, bottom, label, color in small_cards:
        card(draw, (left, top, right, bottom), fill=WHITE, outline=color)
        centered_text(
            draw,
            (left + 8, top + 8, right - 8, bottom - 8),
            label,
            size=15,
            fill=color,
            bold=True,
        )
    fonts = report["held_fonts"]
    sans = fonts["NotoSansCJK-Bold.ttc"]
    serif = fonts["NotoSerifCJK-Medium.ttc"]
    shift_raw = sum(item["raw"]["top1"] for item in report["one_pixel_shifts"].values()) / 4
    shift_v45 = sum(item["v45"]["top1"] for item in report["one_pixel_shifts"].values()) / 4
    comparison_row(
        draw,
        y=986,
        label="Bold held font top-1",
        before=f"{100 * sans['raw']['top1']:.2f}%",
        after=f"{100 * sans['v45']['top1']:.2f}%",
        note="same",
        left=88,
        right=856,
    )
    comparison_row(
        draw,
        y=1038,
        label="Serif held font top-1",
        before=f"{100 * serif['raw']['top1']:.2f}%",
        after=f"{100 * serif['v45']['top1']:.2f}%",
        note="+1.86",
        left=88,
        right=856,
    )
    comparison_row(
        draw,
        y=1090,
        label="Mean shift top-1",
        before=f"{100 * shift_raw:.2f}%",
        after=f"{100 * shift_v45:.2f}%",
        note="+0.88",
        left=88,
        right=856,
    )
    centered_text(
        draw,
        (90, 1135, 854, 1182),
        "FINITE  |  ZERO BLANKS  |  FONT GATE PASS  |  SHIFT GATE PASS",
        size=14,
        fill=GREEN,
        bold=True,
    )

    draw.text((944, 770), "BOUNDARY + NEXT CAUSAL TEST", font=font(20, bold=True), fill=INK)
    draw.text(
        (944, 801),
        "V45 qualifies target geometry; it is not itself a learned language model",
        font=font(13),
        fill=MUTED,
    )
    card(draw, (944, 842, 1712, 935), fill=PALE_BLUE, outline=BLUE)
    centered_text(
        draw,
        (960, 850, 1696, 927),
        "ZERO TRAINABLE PARAMETERS  |  NO TOKENS / IDS / OCR / CODEBOOK\n"
        f"{report['elapsed_seconds']:.2f} s  |  {report['peak_allocated_vram_gib']:.3f} GiB  |  checkpoint reload verified",
        size=15,
        fill=BLUE,
        bold=True,
    )
    retrofit = report["frozen_v42_retrofit_report_only"]
    card(draw, (944, 960, 1712, 1057), fill=PALE_GOLD, outline=GOLD)
    centered_text(
        draw,
        (960, 968, 1696, 1049),
        "REPORT-ONLY V42 RETROFIT DECLINES\n"
        f"natural top-1 {100 * retrofit['raw']['top1']:.2f}% -> {100 * retrofit['v45']['top1']:.2f}%\n"
        "Do not retrofit a reader trained in raw-DCT coordinates",
        size=15,
        fill=GOLD,
        bold=True,
    )
    card(draw, (944, 1082, 1712, 1167), fill=PALE_GREEN, outline=GREEN)
    centered_text(
        draw,
        (960, 1090, 1696, 1159),
        "AUTHORIZED NEXT STEP\n"
        "Preregister, then train a causal raster reader from scratch in V45 geometry",
        size=15,
        fill=GREEN,
        bold=True,
    )
    draw.text(
        (974, 1174),
        f"field state {receipt['field_state_sha256'][:12]}...  |  frozen split sealed  |  writer closed",
        font=font(12, bold=True),
        fill=MUTED,
    )

    card(draw, (62, 1231, 1738, 1292), fill=INK, outline=INK)
    centered_text(
        draw,
        (78, 1237, 1722, 1286),
        "V45 REPRESENTATION QUALIFIED  |  LANGUAGE CLAIM UNCHANGED  |  NEXT READER MUST LEARN IN V45 COORDINATES",
        size=15,
        fill=WHITE,
        bold=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, optimize=True)


def main() -> None:
    args = parse_args()
    report, receipt = load_evidence(args.evidence)
    render(report, receipt, args.out)
    print(args.out.resolve())


if __name__ == "__main__":
    main()
