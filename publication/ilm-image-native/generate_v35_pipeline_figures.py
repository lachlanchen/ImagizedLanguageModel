#!/usr/bin/env python3
"""Generate deterministic diagrams for the implemented V35 training and runtime."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FIGURE_DIR = Path(__file__).resolve().parent / "figures"
DEFAULT_TRAINING_OUT = FIGURE_DIR / "causal_glyph_flow_v35_training.png"
DEFAULT_INFERENCE_OUT = FIGURE_DIR / "causal_glyph_flow_v35_inference.png"
FONT_ROOT = Path("/usr/share/fonts/truetype/dejavu")
CJK_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

INK = "#17323c"
MUTED = "#587078"
LINE = "#9fb1b7"
PAPER = "#f4f7f8"
WHITE = "#ffffff"
TEAL = "#217f8f"
GREEN = "#38745e"
AMBER = "#a76d2a"
BLUE = "#3d6687"
VIOLET = "#6f6295"
RED = "#a34b4b"
NAVY = "#163342"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-out", type=Path, default=DEFAULT_TRAINING_OUT)
    parser.add_argument("--inference-out", type=Path, default=DEFAULT_INFERENCE_OUT)
    return parser.parse_args()


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_ROOT / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def cjk_font(size: int) -> ImageFont.FreeTypeFont:
    if CJK_FONT.is_file():
        return ImageFont.truetype(str(CJK_FONT), size=size)
    return font(size)


def wrapped(draw: ImageDraw.ImageDraw, position: tuple[int, int], value: str, *,
            size: int, width: int, fill: str = MUTED, bold: bool = False,
            spacing: int = 7) -> None:
    draw.multiline_text(
        position,
        "\n".join(textwrap.wrap(value, width=width)),
        font=font(size, bold=bold),
        fill=fill,
        spacing=spacing,
    )


def centered(draw: ImageDraw.ImageDraw, bounds: tuple[int, int, int, int], value: str,
             *, size: int, fill: str = INK, bold: bool = False) -> None:
    left, top, right, bottom = bounds
    face = font(size, bold=bold)
    box = draw.multiline_textbbox((0, 0), value, font=face, align="center", spacing=5)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.multiline_text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2),
        value,
        font=face,
        fill=fill,
        align="center",
        spacing=5,
    )


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], *,
          color: str = "#536a72", width: int = 5, dashed: bool = False) -> None:
    if dashed:
        x1, y1 = start
        x2, y2 = end
        segments = 12
        for index in range(0, segments, 2):
            begin = index / segments
            finish = min(1.0, (index + 1) / segments)
            draw.line(
                (
                    x1 + (x2 - x1) * begin,
                    y1 + (y2 - y1) * begin,
                    x1 + (x2 - x1) * finish,
                    y1 + (y2 - y1) * finish,
                ),
                fill=color,
                width=width,
            )
    else:
        draw.line((*start, *end), fill=color, width=width)
    x2, y2 = end
    x1, y1 = start
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 > x1 else -1
        draw.polygon(
            ((x2, y2), (x2 - 16 * direction, y2 - 10), (x2 - 16 * direction, y2 + 10)),
            fill=color,
        )
    else:
        direction = 1 if y2 > y1 else -1
        draw.polygon(
            ((x2, y2), (x2 - 10, y2 - 16 * direction), (x2 + 10, y2 - 16 * direction)),
            fill=color,
        )


def panel(draw: ImageDraw.ImageDraw, bounds: tuple[int, int, int, int], title: str,
          lines: tuple[str, ...], *, accent: str, dark: bool = False,
          title_size: int = 21, body_size: int = 16) -> None:
    left, top, right, bottom = bounds
    fill = NAVY if dark else WHITE
    outline = accent if dark else LINE
    draw.rounded_rectangle(bounds, radius=7, fill=fill, outline=outline, width=2)
    draw.rectangle((left, top, left + 10, bottom), fill=accent)
    title_fill = "#f4fbfc" if dark else INK
    body_fill = "#d4e4e8" if dark else MUTED
    draw.text((left + 27, top + 17), title, font=font(title_size, bold=True), fill=title_fill)
    for index, line in enumerate(lines):
        draw.text(
            (left + 27, top + 57 + index * (body_size + 12)),
            line,
            font=font(body_size),
            fill=body_fill,
        )


def badge(draw: ImageDraw.ImageDraw, bounds: tuple[int, int, int, int], text: str, *,
          fill: str, foreground: str = WHITE) -> None:
    draw.rounded_rectangle(bounds, radius=6, fill=fill)
    centered(draw, bounds, text, size=14, fill=foreground, bold=True)


def raster_strip(text: str, *, cells: int = 10, scale: int = 2) -> Image.Image:
    patch = 32
    image = Image.new("L", (cells * patch, patch), 255)
    draw = ImageDraw.Draw(image)
    draw.text((6, -1), text, font=cjk_font(27), fill=0)
    for index in range(1, cells):
        draw.line((index * patch, 0, index * patch, patch), fill=225, width=1)
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST).convert("RGB")


def section_label(draw: ImageDraw.ImageDraw, y: int, number: str, title: str,
                  detail: str) -> None:
    badge(draw, (58, y, 108, y + 42), number, fill=TEAL)
    title_face = font(25, bold=True)
    title_x = 126
    draw.text((title_x, y + 2), title, font=title_face, fill=INK)
    title_bounds = draw.textbbox((title_x, y + 2), title, font=title_face)
    detail_x = max(490, title_bounds[2] + 28)
    draw.text((detail_x, y + 7), detail, font=font(17), fill=MUTED)


def training_figure(path: Path) -> None:
    canvas = Image.new("RGB", (2400, 1500), PAPER)
    draw = ImageDraw.Draw(canvas)
    draw.text((58, 36), "Causal Glyph Flow V35: implemented training route", font=font(37, bold=True), fill=INK)
    draw.text(
        (60, 91),
        "129.09M parameters | 32 x 32 binary writing patches | continuous 768-D latent | one RTX 4090 evidence run",
        font=font(19),
        fill=MUTED,
    )
    badge(draw, (1980, 44, 2338, 94), "IMPLEMENTED ARCHITECTURE", fill=TEAL)

    section_label(draw, 145, "1", "Audited foundations", "Useful external work is retained with explicit provenance")
    panel(
        draw,
        (58, 205, 720, 390),
        "Rendered visual corpus",
        ("public Chinese continuation", "short Chinese instructions + copy", "text/metadata stay outside student tensors"),
        accent=BLUE,
    )
    panel(
        draw,
        (760, 205, 1458, 390),
        "V34 continuous glyph codec",
        ("7.42M parameters, frozen", "pixels <-> normalized 768-D field", "codebook-free; held-font + historic qualified"),
        accent=GREEN,
    )
    panel(
        draw,
        (1498, 205, 2340, 390),
        "Public PIXAR initialization",
        ("12-layer, width-768 causal core", "resized pixel projection is alignment teacher", "teacher discarded before causal training/runtime"),
        accent=VIOLET,
    )

    section_label(draw, 430, "2", "Stage A: visual-coordinate alignment", "2,000 updates; only residual adapter learns")
    boxes = (
        ((58, 500, 430, 660), "held-out raster patches", ("Noto Sans train", "Noto Serif development"), BLUE),
        ((490, 500, 840, 660), "frozen V34 encoder", ("32 x 32 -> 768-D", "continuous glyph latent"), GREEN),
        ((900, 500, 1250, 660), "residual adapter", ("Linear-SiLU-Linear", "trainable in Stage A"), TEAL),
        ((1310, 500, 1715, 660), "PIXAR projection target", ("offline teacher only", "MSE + cosine"), VIOLET),
        ((1775, 500, 2340, 660), "frozen gate passed", ("2,461 held-out patches", "cosine 0.9793 | MSE 0.00851", "codec/core hashes unchanged"), AMBER),
    )
    for bounds, title, lines, accent in boxes:
        panel(draw, bounds, title, lines, accent=accent, title_size=18, body_size=14)
    for left, right in ((430, 490), (840, 900), (1250, 1310), (1715, 1775)):
        arrow(draw, (left, 580), (right, 580))

    section_label(draw, 705, "3", "Stages B/C: causal raster learning", "20,000 updates after alignment; codec and adapter frozen")
    pipeline = (
        ((58, 780, 375, 995), "visual strip", ("pixels + patch mask", "up to 96 patches", "no strings/IDs"), BLUE),
        ((425, 780, 735, 995), "V34 encode", ("frozen retina", "visible patches", "-> 768-D"), GREEN),
        ((785, 780, 1165, 995), "causal field", ("PIXAR-initialized", "12 x 768 QKV", "RoPE + SwiGLU"), VIOLET),
        ((1215, 780, 1580, 995), "two writers", ("deterministic anchor", "3-block rectified flow", "+ stop head"), TEAL),
        ((1630, 780, 1940, 995), "V34 decode", ("continuous latent", "-> raster logits", "visual losses"), GREEN),
        ((1990, 780, 2340, 995), "target raster", ("next visible patch", "teacher-forced only", "never a class ID"), AMBER),
    )
    for bounds, title, lines, accent in pipeline:
        panel(draw, bounds, title, lines, accent=accent, dark=title == "causal field", title_size=18, body_size=14)
    for left, right in ((375, 425), (735, 785), (1165, 1215), (1580, 1630), (1940, 1990)):
        arrow(draw, (left, 885), (right, 885))

    demo = raster_strip("\u7167\u5199\uff1a\u5929\u5730 \u7b54\uff1a", cells=10)
    canvas.paste(demo.resize((278, 56)), (78, 925))

    draw.rounded_rectangle((58, 1040, 2340, 1300), radius=7, fill=WHITE, outline=LINE, width=2)
    draw.text((82, 1063), "Fixed objective and curriculum", font=font(23, bold=True), fill=INK)
    objective_boxes = (
        ((82, 1110, 490, 1248), "Anchor", "cosine + 0.25 MSE", BLUE),
        ((518, 1110, 926, 1248), "Flow", "velocity MSE in latent field", VIOLET),
        ((954, 1110, 1362, 1248), "Visual", "BCE + edge + ink Dice", GREEN),
        ((1390, 1110, 1798, 1248), "Stop", "masked binary cross-entropy", AMBER),
        ((1826, 1110, 2316, 1248), "Stage C mix", "75% instruction | 12.5% copy | 12.5% replay", TEAL),
    )
    for bounds, title, line, accent in objective_boxes:
        left, top, right, bottom = bounds
        draw.rounded_rectangle(bounds, radius=6, fill="#f7fafb", outline=accent, width=2)
        draw.text((left + 18, top + 16), title, font=font(18, bold=True), fill=accent)
        wrapped(draw, (left + 18, top + 53), line, size=15, width=max(18, (right - left) // 12), fill=MUTED)

    draw.rounded_rectangle((58, 1340, 2340, 1448), radius=7, fill=NAVY)
    draw.text((84, 1363), "Student boundary", font=font(20, bold=True), fill="#e8f7f8")
    draw.text(
        (310, 1365),
        "pixels -> continuous states -> pixels   |   no token/Unicode IDs, embedding table, codebook, OCR, retrieval, or runtime teacher",
        font=font(17, bold=True),
        fill="#cbe3e7",
    )
    draw.text((84, 1407), "PIXAR and the rasterizer are preparation tools; the exported checkpoint contains the complete independent student.", font=font(15), fill="#9fc2c9")

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def inference_figure(path: Path) -> None:
    canvas = Image.new("RGB", (2400, 1500), PAPER)
    draw = ImageDraw.Draw(canvas)
    draw.text((58, 36), "Causal Glyph Flow V35: raster-only inference", font=font(37, bold=True), fill=INK)
    draw.text(
        (60, 91),
        "The public model boundary begins after optional UI rasterization and ends at generated writing pixels.",
        font=font(19),
        fill=MUTED,
    )
    badge(draw, (2007, 44, 2338, 94), "IMPLEMENTED RUNTIME", fill=GREEN)

    panel(
        draw,
        (58, 180, 485, 470),
        "A. Prompt source",
        ("typed Chinese/English", "or uploaded writing image", "single-line V35 scope"),
        accent=BLUE,
    )
    panel(
        draw,
        (535, 180, 965, 470),
        "B. UI boundary",
        ("typed text -> deterministic raster", "image -> crop + binary strip", "32 x 32 visual patches"),
        accent=AMBER,
    )
    panel(
        draw,
        (1015, 180, 1445, 470),
        "C. Student input",
        ("pixels [B,1,32,32L]", "patch_mask [B,L]", "floating tensors only"),
        accent=TEAL,
    )
    panel(
        draw,
        (1495, 180, 2340, 470),
        "No hidden symbolic channel",
        ("no tokenizer or Unicode lookup", "no OCR transcript or external LM", "no candidate bank, database, or network"),
        accent=RED,
    )
    for left, right in ((485, 535), (965, 1015)):
        arrow(draw, (left, 325), (right, 325))
    arrow(draw, (1445, 325), (1495, 325), color=RED, dashed=True)

    prompt = raster_strip("\u7167\u5199\uff1a\u5929\u5730 \u7b54\uff1a", cells=11)
    canvas.paste(prompt.resize((385, 70)), (78, 376))

    draw.text((58, 540), "Autoregressive visual step", font=font(27, bold=True), fill=INK)
    draw.text((470, 546), "Repeated until the learned stop head fires or 31 output patches are reached", font=font(17), fill=MUTED)

    pipeline = (
        ((58, 620, 400, 850), "frozen V34 E", ("encode visible raster", "normalized 768-D", "no quantization"), GREEN),
        ((450, 620, 790, 850), "frozen adapter", ("aligned coordinates", "residual map", "Stage A hash locked"), TEAL),
        ((840, 620, 1250, 850), "causal field", ("12 layers x width 768", "visual history only", "predict next state"), VIOLET),
        ((1300, 620, 1690, 850), "writer + stop", ("anchor or 8-step flow", "continuous 768-D", "stop probability"), BLUE),
        ((1740, 620, 2070, 850), "frozen V34 D", ("decode latent", "binary threshold 0.5", "visible patch"), GREEN),
        ((2120, 620, 2340, 850), "append", ("output PNG", "+ context", "repeat"), AMBER),
    )
    for bounds, title, lines, accent in pipeline:
        panel(draw, bounds, title, lines, accent=accent, dark=title == "causal field", title_size=18, body_size=14)
    for left, right in ((400, 450), (790, 840), (1250, 1300), (1690, 1740), (2070, 2120)):
        arrow(draw, (left, 735), (right, 735))

    draw.line((2230, 850, 2230, 940, 225, 940), fill=GREEN, width=5)
    draw.line((225, 940, 225, 850), fill=GREEN, width=5)
    draw.polygon(((225, 850), (214, 869), (236, 869)), fill=GREEN)
    centered(draw, (755, 895, 1710, 955), "visible feedback only: decode -> threshold -> re-encode", size=18, fill=GREEN, bold=True)

    draw.rounded_rectangle((58, 1005, 1605, 1325), radius=7, fill=WHITE, outline=LINE, width=2)
    draw.text((84, 1030), "Primary result", font=font(23, bold=True), fill=INK)
    draw.text((84, 1074), "Generated raster strip / PNG", font=font(19, bold=True), fill=TEAL)
    output = raster_strip("\u5929\u5730\u7384\u9ec4\u5b87\u5b99\u6d2a\u8352", cells=12, scale=3)
    canvas.paste(output.resize((1450, 145)), (84, 1120))
    draw.text((84, 1280), "The pixels above illustrate the output type; measured autonomous V35 samples are shown only in the evidence figure.", font=font(14), fill=MUTED)

    draw.rounded_rectangle((1655, 1005, 2340, 1325), radius=7, fill="#fbf7ee", outline=AMBER, width=2)
    draw.text((1681, 1030), "Optional post-processing", font=font(23, bold=True), fill=INK)
    draw.text((1681, 1080), "Tesseract OCR sidecar", font=font(19, bold=True), fill=AMBER)
    wrapped(
        draw,
        (1681, 1125),
        "Runs after the PNG exists. It can expose searchable text for encodable regions but never changes the generated raster or the model state.",
        size=16,
        width=46,
        fill=MUTED,
    )
    arrow(draw, (1605, 1165), (1655, 1165), color=AMBER, dashed=True)

    draw.rounded_rectangle((58, 1370, 2340, 1450), radius=7, fill=NAVY)
    centered(
        draw,
        (80, 1380, 2318, 1440),
        "Independent means self-contained at inference, not trained without public knowledge: PIXAR/V34 provenance remains part of the checkpoint receipt.",
        size=17,
        fill="#d6ebee",
        bold=True,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    args = parse_args()
    training_figure(args.training_out)
    inference_figure(args.inference_out)
    print(args.training_out)
    print(args.inference_out)


if __name__ == "__main__":
    main()
