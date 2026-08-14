#!/usr/bin/env python3
"""Render the measured V41 glyph-motor bridge from hash-pinned evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
DEFAULT_RECEIPT = HERE / "evidence/v41/glyph_motor_bridge_receipt.json"
DEFAULT_CONTACT = HERE / "evidence/v41/glyph_motor_bridge_contact_sheet.png"
DEFAULT_OUT = HERE / "figures/glyph_motor_bridge_v41_result.png"
EXPECTED_RECEIPT_SHA256 = (
    "4980dc8f1922e76f9f68153c83abd240738e76f2b4d13b55d445f7026865873f"
)
EXPECTED_CONTACT_SHA256 = (
    "6d8c610fac65806c28253f574f397aa5e62cc8665da633cf364246916aeb93da"
)

FONT_ROOT = Path("/usr/share/fonts/truetype/dejavu")
PAPER = "#f4f6f5"
WHITE = "#ffffff"
INK = "#173039"
MUTED = "#5b7077"
LINE = "#b7c4c7"
TEAL = "#177d83"
GREEN = "#2f765a"
PALE_GREEN = "#e2f0e9"
BLUE = "#416f8d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--contact", type=Path, default=DEFAULT_CONTACT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_ROOT / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def load_evidence(receipt_path: Path, contact_path: Path) -> tuple[dict[str, Any], Image.Image]:
    receipt_hash = file_sha256(receipt_path)
    contact_hash = file_sha256(contact_path)
    if receipt_hash != EXPECTED_RECEIPT_SHA256:
        raise ValueError(f"V41 receipt hash changed: {receipt_hash}")
    if contact_hash != EXPECTED_CONTACT_SHA256:
        raise ValueError(f"V41 contact-sheet hash changed: {contact_hash}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("experiment") != "image-conditioned-glyph-motor-bridge-v41-audit":
        raise ValueError("unexpected V41 evidence experiment")
    if not receipt.get("finite"):
        raise ValueError("V41 evidence is non-finite")
    if not receipt.get("motor_gate", {}).get("passed"):
        raise ValueError("V41 evidence does not pass its conjunctive motor gate")
    if any(
        receipt.get(name)
        for name in ("uses_token_ids", "uses_unicode_ids_in_model", "uses_ocr", "uses_retrieval")
    ):
        raise ValueError("V41 runtime boundary changed")
    with Image.open(contact_path) as source:
        contact = source.convert("L")
    expected = receipt["contact_sheet"]
    if contact.size != (
        expected["columns"] * expected["cell_pixels"],
        expected["rows"] * expected["cell_pixels"],
    ):
        raise ValueError("V41 contact-sheet geometry changed")
    return receipt, contact


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    draw.line((*start, *end), fill=TEAL, width=4)
    draw.polygon(
        ((end[0], end[1]), (end[0] - 13, end[1] - 8), (end[0] - 13, end[1] + 8)),
        fill=TEAL,
    )


def centered_text(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    text: str,
    *,
    size: int,
    fill: str = INK,
    bold: bool = False,
) -> None:
    face = font(size, bold=bold)
    box = draw.multiline_textbbox((0, 0), text, font=face, spacing=4, align="center")
    width = box[2] - box[0]
    height = box[3] - box[1]
    left, top, right, bottom = bounds
    draw.multiline_text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2),
        text,
        font=face,
        fill=fill,
        spacing=4,
        align="center",
    )


def glyph_row(contact: Image.Image, row: int, columns: int) -> Image.Image:
    cell = 128
    strip = contact.crop((0, row * cell, columns * cell, (row + 1) * cell))
    return strip.resize((columns * 92, 92), Image.Resampling.LANCZOS)


def render(receipt: dict[str, Any], contact: Image.Image, output_path: Path) -> None:
    image = Image.new("RGB", (1800, 1160), PAPER)
    draw = ImageDraw.Draw(image)
    draw.text((64, 42), "V41 IMAGE-CONDITIONED GLYPH MOTOR", font=font(34, bold=True), fill=INK)
    draw.text(
        (64, 88),
        "A real pretrained calligraphy generator cleans imperfect continuous ILM glyph projections",
        font=font(19),
        fill=MUTED,
    )
    draw.rounded_rectangle((1445, 42, 1735, 108), radius=7, fill=GREEN)
    centered_text(draw, (1445, 42, 1735, 108), "MOTOR GATE PASSED", size=19, fill=WHITE, bold=True)

    pipeline_top = 138
    boxes = (
        (70, pipeline_top, 380, pipeline_top + 100),
        (500, pipeline_top, 830, pipeline_top + 100),
        (950, pipeline_top, 1280, pipeline_top + 100),
        (1400, pipeline_top, 1730, pipeline_top + 100),
    )
    labels = (
        "VISIBLE GLYPH\nRASTER",
        "V34 CONTINUOUS\nPROJECTION",
        "MX-FONT MOTOR\n+ STYLE RASTERS",
        "VISIBLE OUTPUT\nRASTER",
    )
    for bounds, label in zip(boxes, labels):
        draw.rounded_rectangle(bounds, radius=6, fill=WHITE, outline=LINE, width=2)
        centered_text(draw, bounds, label, size=17, bold=True)
    for left, right in zip(boxes[:-1], boxes[1:]):
        arrow(draw, (left[2], pipeline_top + 50), (right[0], pipeline_top + 50))

    columns = receipt["contact_sheet"]["columns"]
    metric = receipt["metrics"]
    rows = (
        ("Canonical source raster", 0, None, None),
        ("Held target style", 1, None, None),
        ("Exact source + motor", 2, "exact_source", "baseline"),
        ("32 px coarse + motor", 3, "coarse_32", "coarse"),
        ("V34 clean + motor", 4, "v34_sigma_0p000", "projected"),
        ("V34 sigma 0.05 + motor", 6, "v34_sigma_0p050", "noise"),
        ("V34 sigma 0.10 + motor", 7, "v34_sigma_0p100", "noise"),
    )
    row_top = 275
    row_height = 112
    for index, (label, contact_row, key, role) in enumerate(rows):
        y = row_top + index * row_height
        if index % 2 == 0:
            draw.rectangle((50, y - 5, 1750, y + 101), fill=WHITE)
        draw.text((70, y + 30), label, font=font(16, bold=True), fill=INK)
        strip = glyph_row(contact, contact_row, columns)
        image.paste(strip.convert("RGB"), (315, y))
        if key is not None:
            values = metric[key]
            f1 = values["target_ink_f1_mean"]
            gain = values["motor_delta_ink_f1"]
            draw.text((1270, y + 19), f"ink F1  {f1:.3f}", font=font(17, bold=True), fill=GREEN)
            draw.text((1270, y + 48), f"motor gain  +{gain:.3f}", font=font(15), fill=BLUE)
            if role == "noise":
                clean = metric["v34_sigma_0p000"]["target_ink_f1_mean"]
                draw.text(
                    (1485, y + 48),
                    f"retains {100 * f1 / clean:.1f}%",
                    font=font(15),
                    fill=MUTED,
                )

    footer_top = 1080
    draw.rounded_rectangle((70, footer_top, 1730, 1132), radius=7, fill=PALE_GREEN, outline=GREEN, width=2)
    clean = metric["v34_sigma_0p000"]["target_ink_f1_mean"]
    noisy = metric["v34_sigma_0p050"]["target_ink_f1_mean"]
    gain = metric["v34_sigma_0p050"]["motor_delta_ink_f1"]
    centered_text(
        draw,
        (90, footer_top, 1710, 1132),
        f"PASS  |  sigma 0.05 retains {100 * noisy / clean:.2f}% of clean motor F1  |  "
        f"+{gain:.3f} F1 over projected input  |  100% nonblank",
        size=17,
        fill=GREEN,
        bold=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, optimize=True)


def main() -> None:
    args = parse_args()
    receipt, contact = load_evidence(args.receipt, args.contact)
    render(receipt, contact, args.out)
    print(args.out.resolve())


if __name__ == "__main__":
    main()
