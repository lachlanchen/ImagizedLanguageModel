from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from torch.utils.data import Dataset

from .instruction_data import VisualInstructionRecord


INK_FONT_PATHS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Light.ttc",
    "/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf",
    "/usr/share/fonts/truetype/arphic-bkai00mp/bkai00mp.ttf",
)


@dataclass(frozen=True)
class InkRibbonConfig:
    height: int = 48
    strip_width: int = 8
    maximum_strips: int = 256
    font_size: int = 30
    minimum_font_size: int = 17
    horizontal_padding: int = 12
    prefix_loss_weight: float = 0.20
    answer_loss_weight: float = 1.0
    augment: bool = True

    def __post_init__(self) -> None:
        if self.height < 24 or self.strip_width < 2:
            raise ValueError("ribbon resolution is too small")
        if self.maximum_strips < 32:
            raise ValueError("maximum_strips must be at least 32")


def _font(variant: int, size: int) -> ImageFont.FreeTypeFont:
    available = [path for path in INK_FONT_PATHS if Path(path).exists()]
    if not available:
        return ImageFont.load_default()
    return ImageFont.truetype(available[variant % len(available)], size=size)


def _single_line(text: str) -> str:
    return "  ".join(part.strip() for part in text.replace("\r", "").split("\n") if part.strip())


def _text_width(text: str, font: ImageFont.FreeTypeFont) -> int:
    canvas = Image.new("L", (8, 8), 255)
    draw = ImageDraw.Draw(canvas)
    box = draw.textbbox((0, 0), text or " ", font=font)
    return max(1, box[2] - box[0])


def choose_shared_font_size(
    prompt: str,
    answer: str,
    *,
    config: InkRibbonConfig,
    variant: int,
    available_width: int,
) -> int:
    fixed_control_width = config.strip_width * 8 + config.horizontal_padding * 4
    for size in range(config.font_size, config.minimum_font_size - 1, -1):
        font = _font(variant, size)
        total = _text_width(prompt, font) + _text_width(answer, font) + fixed_control_width
        if total <= available_width:
            return size
    return config.minimum_font_size


def render_text_ink(
    text: str,
    *,
    height: int,
    font: ImageFont.FreeTypeFont,
    horizontal_padding: int,
) -> torch.Tensor:
    text = _single_line(text)
    width = _text_width(text, font) + horizontal_padding * 2
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    box = draw.textbbox((0, 0), text, font=font)
    text_height = box[3] - box[1]
    y = (height - text_height) // 2 - box[1]
    draw.text((horizontal_padding, y), text, font=font, fill=0)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(1.0 - array)


def visual_separator(config: InkRibbonConfig) -> torch.Tensor:
    width = config.strip_width * 4
    image = torch.zeros((config.height, width), dtype=torch.float32)
    centre = config.height // 2
    image[7 : config.height - 7, config.strip_width - 1 : config.strip_width + 1] = 1.0
    image[7 : config.height - 7, config.strip_width * 2 - 1 : config.strip_width * 2 + 1] = 1.0
    image[centre - 1 : centre + 1, config.strip_width * 2 : config.strip_width * 3] = 1.0
    return image


def visual_end_mark(config: InkRibbonConfig) -> torch.Tensor:
    width = config.strip_width * 3
    image = torch.zeros((config.height, width), dtype=torch.float32)
    size = min(config.height // 3, config.strip_width * 2)
    top = (config.height - size) // 2
    left = (width - size) // 2
    image[top : top + size, left : left + size] = 1.0
    image[top + 3 : top + size - 3, left + 3 : left + size - 3] = 0.0
    return image


def _blank(config: InkRibbonConfig, strips: int = 1) -> torch.Tensor:
    return torch.zeros((config.height, config.strip_width * strips), dtype=torch.float32)


def augment_ribbon(ink: torch.Tensor, *, seed: int) -> torch.Tensor:
    rng = random.Random(seed)
    image = Image.fromarray(((1.0 - ink.numpy()).clip(0, 1) * 255).astype(np.uint8), "L")
    if rng.random() < 0.35:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.78, 1.20))
    if rng.random() < 0.22:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.08, 0.48)))
    array = np.asarray(image, dtype=np.float32) / 255.0
    if rng.random() < 0.30:
        noise = np.random.default_rng(seed).normal(0.0, rng.uniform(0.002, 0.018), array.shape)
        array = np.clip(array + noise, 0.0, 1.0)
    return torch.from_numpy(1.0 - array.astype(np.float32))


def trim_ink(ink: torch.Tensor, maximum_width: int, *, keep: str) -> torch.Tensor:
    if ink.shape[1] <= maximum_width:
        return ink
    if keep == "start":
        return ink[:, :maximum_width]
    if keep == "end":
        return ink[:, -maximum_width:]
    raise ValueError("keep must be start or end")


def split_ink_strips(ink: torch.Tensor, strip_width: int) -> torch.Tensor:
    if ink.ndim != 2:
        raise ValueError("ink ribbon must be [height, width]")
    remainder = ink.shape[1] % strip_width
    if remainder:
        ink = torch.nn.functional.pad(ink, (0, strip_width - remainder))
    strips = ink.unfold(1, strip_width, strip_width).permute(1, 0, 2).contiguous()
    return strips[:, None]


def strips_to_image(strips: torch.Tensor) -> Image.Image:
    if strips.ndim == 5:
        if strips.shape[0] != 1:
            raise ValueError("only one batch can be visualized at a time")
        strips = strips[0]
    if strips.ndim != 4 or strips.shape[1] != 1:
        raise ValueError("expected [time, 1, height, width] visual strips")
    ink = strips[:, 0].permute(1, 0, 2).reshape(strips.shape[2], -1)
    array = ((1.0 - ink.detach().float().cpu().clamp(0, 1).numpy()) * 255).round().astype(np.uint8)
    return Image.fromarray(array, "L")


def render_qa_stream(
    prompt: str,
    answer: str,
    *,
    config: InkRibbonConfig,
    variant: int,
) -> tuple[torch.Tensor, int, dict[str, Any]]:
    maximum_width = (config.maximum_strips + 1) * config.strip_width
    size = choose_shared_font_size(
        _single_line(prompt),
        _single_line(answer),
        config=config,
        variant=variant,
        available_width=maximum_width,
    )
    font = _font(variant, size)
    prompt_ink = render_text_ink(
        prompt,
        height=config.height,
        font=font,
        horizontal_padding=config.horizontal_padding,
    )
    answer_ink = render_text_ink(
        answer,
        height=config.height,
        font=font,
        horizontal_padding=config.horizontal_padding,
    )
    controls = visual_separator(config)
    ending = visual_end_mark(config)
    fixed_width = config.strip_width * 5 + controls.shape[1] + ending.shape[1]
    available_content = max(config.strip_width * 8, maximum_width - fixed_width)
    prompt_budget = int(available_content * 0.45) // config.strip_width * config.strip_width
    answer_budget = available_content - prompt_budget
    prompt_ink = trim_ink(prompt_ink, prompt_budget, keep="start")
    answer_ink = trim_ink(answer_ink, answer_budget, keep="start")
    prefix = torch.cat((_blank(config, 1), prompt_ink, _blank(config, 1), controls, _blank(config, 1)), dim=1)
    answer_start = math_ceil_div(prefix.shape[1], config.strip_width)
    ribbon = torch.cat((prefix, answer_ink, _blank(config, 1), ending, _blank(config, 2)), dim=1)
    if config.augment:
        ribbon = augment_ribbon(ribbon, seed=variant)
    strips = split_ink_strips(ribbon, config.strip_width)
    strips = strips[: config.maximum_strips + 1]
    return strips, min(answer_start, strips.shape[0] - 1), {
        "font_size": size,
        "strips": int(strips.shape[0]),
        "answer_start": int(answer_start),
    }


def render_prompt_stream(
    prompt: str,
    *,
    config: InkRibbonConfig,
    variant: int,
) -> torch.Tensor:
    maximum_width = (config.maximum_strips - 8) * config.strip_width
    font = _font(variant, config.font_size)
    prompt_ink = render_text_ink(
        prompt,
        height=config.height,
        font=font,
        horizontal_padding=config.horizontal_padding,
    )
    if prompt_ink.shape[1] > maximum_width:
        for size in range(config.font_size - 1, config.minimum_font_size - 1, -1):
            font = _font(variant, size)
            prompt_ink = render_text_ink(
                prompt,
                height=config.height,
                font=font,
                horizontal_padding=config.horizontal_padding,
            )
            if prompt_ink.shape[1] <= maximum_width:
                break
    prompt_ink = trim_ink(prompt_ink, maximum_width, keep="start")
    prefix = torch.cat(
        (_blank(config, 1), prompt_ink, _blank(config, 1), visual_separator(config), _blank(config, 1)),
        dim=1,
    )
    return split_ink_strips(prefix, config.strip_width)[: config.maximum_strips]


def math_ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


class InkStreamDataset(Dataset):
    """Rasterize records offline and expose only continuous ink strips."""

    def __init__(
        self,
        records: Sequence[VisualInstructionRecord],
        *,
        config: InkRibbonConfig,
        seed: int = 0,
        length: int | None = None,
    ):
        if not records:
            raise ValueError("InkStreamDataset requires records")
        self.records = list(records)
        self.config = config
        self.seed = int(seed)
        self.length = int(length) if length is not None else len(records)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.seed + self.epoch * 10_000_019 + index * 104_729)
        record = self.records[index % len(self.records)] if self.length <= len(self.records) else rng.choice(self.records)
        variant = rng.randrange(2**31)
        strips, answer_start, render_meta = render_qa_stream(
            record.prompt,
            record.response,
            config=self.config,
            variant=variant,
        )
        if strips.shape[0] < 2:
            raise RuntimeError("rendered stream has no prediction target")
        inputs = strips[:-1]
        targets = strips[1:]
        weights = torch.full((inputs.shape[0],), self.config.prefix_loss_weight, dtype=torch.float32)
        weights[max(0, answer_start - 1) :] = self.config.answer_loss_weight
        return {
            "input": inputs,
            "target": targets,
            "weight": weights,
            "metadata": {
                "identifier": record.identifier,
                "source": record.source,
                "language": record.language,
                **render_meta,
            },
        }


def ink_stream_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty stream batch")
    maximum = max(item["input"].shape[0] for item in batch)
    _, channels, height, width = batch[0]["input"].shape
    inputs = torch.zeros((len(batch), maximum, channels, height, width), dtype=torch.float32)
    targets = torch.zeros_like(inputs)
    weights = torch.zeros((len(batch), maximum), dtype=torch.float32)
    for index, item in enumerate(batch):
        length = item["input"].shape[0]
        inputs[index, :length] = item["input"]
        targets[index, :length] = item["target"]
        weights[index, :length] = item["weight"]
    return {
        "input": inputs,
        "target": targets,
        "weight": weights,
        "metadata": [item["metadata"] for item in batch],
    }
