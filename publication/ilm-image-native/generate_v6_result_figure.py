#!/usr/bin/env python3
"""Compose the measured V5/V6 autonomous-result figure from inference artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V5 = ROOT / (
    "artifacts/retinal_flow_chinese_mvp_v6_closed_loop/"
    "autonomous_32_v5_control/complete_page.png"
)
DEFAULT_V6 = ROOT / (
    "artifacts/retinal_flow_chinese_mvp_v6_closed_loop/"
    "autonomous_32_step_5200/complete_page.png"
)
DEFAULT_OUT = Path(__file__).resolve().parent / "figures/closed_loop_v6_result.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the RFLM V5/V6 autonomous comparison figure."
    )
    parser.add_argument("--v5", type=Path, default=DEFAULT_V5)
    parser.add_argument("--v6", type=Path, default=DEFAULT_V6)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/truetype/noto")
        / ("NotoSans-Bold.ttf" if bold else "NotoSans-Regular.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def fit_page(image: Image.Image, width: int) -> Image.Image:
    image = image.convert("RGB")
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def draw_metric(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    label: str,
    value: str,
    fraction: float,
    color: str,
) -> None:
    label_font = load_font(22)
    value_font = load_font(25, bold=True)
    draw.text((x, y), label, fill="#2a3037", font=label_font)
    value_box = draw.textbbox((0, 0), value, font=value_font)
    draw.text(
        (x + width - (value_box[2] - value_box[0]), y + 31),
        value,
        fill=color,
        font=value_font,
    )
    bar_y = y + 67
    draw.rounded_rectangle((x, bar_y, x + width, bar_y + 12), radius=6, fill="#e5e9ed")
    draw.rounded_rectangle(
        (x, bar_y, x + max(12, round(width * max(0.0, min(1.0, fraction)))), bar_y + 12),
        radius=6,
        fill=color,
    )


def draw_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    page: Image.Image,
    title: str,
    subtitle: str,
    accent: str,
    ink_ratio: float,
    sparse: float,
    top1: float,
    context_gain: float,
) -> None:
    panel_width = 790
    title_font = load_font(32, bold=True)
    subtitle_font = load_font(23)
    draw.rectangle((x, y, x + 9, y + 72), fill=accent)
    draw.text((x + 28, y - 4), title, fill="#171b20", font=title_font)
    draw.text((x + 28, y + 39), subtitle, fill="#59636f", font=subtitle_font)

    page_y = y + 102
    page = fit_page(page, panel_width)
    draw.rectangle(
        (x - 2, page_y - 2, x + panel_width + 2, page_y + page.height + 2),
        outline="#aeb7c0",
        width=2,
    )
    canvas.paste(page, (x, page_y))
    draw.text(
        (x, page_y + page.height + 12),
        "Rendered prompt followed by 32 model-generated image cells",
        fill="#69737d",
        font=load_font(19),
    )

    metric_y = page_y + page.height + 56
    metric_width = 360
    draw_metric(
        draw,
        x=x,
        y=metric_y,
        width=metric_width,
        label="Late / early ink",
        value=f"{ink_ratio:.3f}",
        fraction=min(ink_ratio / 1.25, 1.0),
        color=accent,
    )
    draw_metric(
        draw,
        x=x + 430,
        y=metric_y,
        width=metric_width,
        label="Sparse cells",
        value=f"{sparse * 100:.1f}%",
        fraction=sparse,
        color="#b4473f",
    )
    draw_metric(
        draw,
        x=x,
        y=metric_y + 113,
        width=metric_width,
        label="Frozen-bank top-1",
        value=f"{top1 * 100:.3f}%",
        fraction=min(top1 / 0.02, 1.0),
        color="#326e9b",
    )
    draw_metric(
        draw,
        x=x + 430,
        y=metric_y + 113,
        width=metric_width,
        label="Generated context signal",
        value=f"+{context_gain:.4f}",
        fraction=min(context_gain / 0.03, 1.0),
        color="#8065a8",
    )


def main() -> None:
    args = parse_args()
    for path in (args.v5, args.v6):
        if not path.is_file():
            raise FileNotFoundError(f"missing autonomous inference image: {path}")

    width, height = 1800, 930
    canvas = Image.new("RGB", (width, height), "#f8fafb")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (70, 42),
        "Closed-loop visual trajectory training: what changed",
        fill="#11161b",
        font=load_font(47, bold=True),
    )
    draw.text(
        (72, 103),
        "Matched 32-cell autonomous generation; same prompt, seed, sampler, and 11.69M-parameter model",
        fill="#58636e",
        font=load_font(25),
    )

    draw_panel(
        canvas,
        draw,
        x=70,
        y=170,
        page=Image.open(args.v5),
        title="V5  Clean-prefix training",
        subtitle="Ink thins under its own visual feedback",
        accent="#b4473f",
        ink_ratio=0.4828550111,
        sparse=0.375,
        top1=0.0090796533,
        context_gain=0.0210505883,
    )
    draw_panel(
        canvas,
        draw,
        x=940,
        y=170,
        page=Image.open(args.v6),
        title="V6  Model-induced rollout training",
        subtitle="Ink survives, but the writing remains unreadable",
        accent="#16835b",
        ink_ratio=1.1676816709,
        sparse=0.1875,
        top1=0.0119686339,
        context_gain=0.0077425047,
    )

    verdict_y = 812
    draw.rectangle((70, verdict_y, 1730, verdict_y + 78), fill="#202830")
    draw.text(
        (95, verdict_y + 13),
        "Measured verdict",
        fill="#ffffff",
        font=load_font(26, bold=True),
    )
    draw.text(
        (365, verdict_y + 14),
        "closed-loop stability improved; contextual language and readable generation did not",
        fill="#f1f4f6",
        font=load_font(25),
    )
    draw.text(
        (95, verdict_y + 48),
        "Frozen glyph bank SHA-256 543987ddd754135cff34dbee0f6f91ce1971741d0807754c65650031121181c4",
        fill="#aeb9c2",
        font=load_font(17),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, optimize=True)
    print(args.out)


if __name__ == "__main__":
    main()
