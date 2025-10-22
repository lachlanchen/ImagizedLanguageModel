"""Sequential English glyph tiling with multi-scale layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def find_font_path() -> str:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""


@dataclass(frozen=True)
class GridSpec:
    """Tiling configuration for sequential glyph placement."""

    grid_size: int
    tile_size: int
    canvas_size: int | None = None

    @property
    def max_length(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def image_size(self) -> int:
        return self.canvas_size or (self.grid_size * self.tile_size)

    @property
    def grid_extent(self) -> int:
        return self.grid_size * self.tile_size


# Canonical specs
SPEC_2X2_64 = GridSpec(grid_size=2, tile_size=64, canvas_size=128)
SPEC_3X3_40 = GridSpec(grid_size=3, tile_size=40, canvas_size=128)
SPEC_4X4_32 = GridSpec(grid_size=4, tile_size=32, canvas_size=128)
SPEC_5X5_24 = GridSpec(grid_size=5, tile_size=24, canvas_size=128)
SPEC_6X6_20 = GridSpec(grid_size=6, tile_size=20, canvas_size=128)
SPEC_8X8_16 = GridSpec(grid_size=8, tile_size=16, canvas_size=128)

DEFAULT_SPEC = SPEC_8X8_16

# Default dynamic rules (max length inclusive)
DEFAULT_RULES: Sequence[tuple[int, GridSpec]] = (
    (3, SPEC_2X2_64),      # <=3 letters: 2x2, 64px cells
    (8, SPEC_3X3_40),      # <=8 letters: 3x3, 40px cells, symmetric padding
    (16, SPEC_4X4_32),     # <=16 letters: 4x4, 32px cells
    (25, SPEC_5X5_24),     # <=25 letters: 5x5, 24px cells
    (36, SPEC_6X6_20),     # <=36 letters: 6x6, 20px cells
    (64, SPEC_8X8_16),     # <=64 letters: 8x8, 16px cells
)


def select_spec(
    word: str,
    *,
    rules: Sequence[tuple[int, GridSpec]] = DEFAULT_RULES,
    fallback: GridSpec = DEFAULT_SPEC,
    dynamic: bool = True,
) -> GridSpec:
    """Choose an appropriate tiling spec for the supplied word length."""

    if not dynamic or not word:
        return fallback
    length = len(word)
    for max_len, spec in rules:
        if length <= max_len:
            return spec
    return fallback


def _load_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = find_font_path()
    if font_path:
        return ImageFont.truetype(font_path, size=font_size)
    return ImageFont.load_default()


def render_char_patch(character: str, *, tile_size: int, intensity: int = 255) -> np.ndarray:
    """Render a single character centred in a tile-sized grayscale patch."""

    img = Image.new("L", (tile_size, tile_size), color=0)
    draw = ImageDraw.Draw(img)

    if not character or character.isspace():
        return np.array(img, dtype=np.uint8)

    font_size = max(6, int(tile_size * 0.85))
    for _ in range(8):
        font = _load_font(font_size)
        bbox = font.getbbox(character)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= tile_size - 2 and height <= tile_size - 2:
            break
        font_size = max(6, int(font_size * 0.9))
    else:
        font = _load_font(font_size)
        bbox = font.getbbox(character)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

    try:
        draw.text((tile_size // 2, tile_size // 2), character, font=font, fill=intensity, anchor="mm")
    except Exception:
        x = (tile_size - width) // 2
        y = (tile_size - height) // 2
        draw.text((x, y), character, font=font, fill=intensity)

    return np.array(img, dtype=np.uint8)


def render_word_tile_tensor(
    word: str,
    *,
    rules: Sequence[tuple[int, GridSpec]] = DEFAULT_RULES,
    fallback: GridSpec = DEFAULT_SPEC,
    dynamic: bool = True,
) -> np.ndarray:
    """Return a tensor of shape (grid, grid, tile, tile) for the word."""

    spec = select_spec(word, rules=rules, fallback=fallback, dynamic=dynamic)
    grid = spec.grid_size
    tile = spec.tile_size
    tensor = np.zeros((grid, grid, tile, tile), dtype=np.uint8)
    if not word:
        return tensor

    capped = word[: spec.max_length]
    for idx, ch in enumerate(capped):
        row = idx // grid
        col = idx % grid
        tensor[row, col] = render_char_patch(ch, tile_size=tile)
    return tensor


def render_word_tile_image(
    word: str,
    *,
    rules: Sequence[tuple[int, GridSpec]] = DEFAULT_RULES,
    fallback: GridSpec = DEFAULT_SPEC,
    dynamic: bool = True,
) -> np.ndarray:
    """Render the word as a grayscale image using sequential tiling."""

    spec = select_spec(word, rules=rules, fallback=fallback, dynamic=dynamic)
    tensor = render_word_tile_tensor(
        word,
        rules=rules,
        fallback=fallback,
        dynamic=dynamic,
    )
    grid = spec.grid_size
    tile = spec.tile_size
    image_size = spec.image_size
    canvas = np.zeros((image_size, image_size), dtype=np.uint8)

    margin = max(0, (image_size - spec.grid_extent) // 2)
    for row in range(grid):
        for col in range(grid):
            patch = tensor[row, col]
            y0 = margin + row * tile
            x0 = margin + col * tile
            canvas[y0 : y0 + tile, x0 : x0 + tile] = patch
    return canvas


def render_words(
    words: Iterable[str],
    *,
    rules: Sequence[tuple[int, GridSpec]] = DEFAULT_RULES,
    fallback: GridSpec = DEFAULT_SPEC,
    dynamic: bool = True,
) -> dict[str, np.ndarray]:
    """Render multiple words, optionally with dynamic scaling."""

    return {
        word: render_word_tile_image(
            word,
            rules=rules,
            fallback=fallback,
            dynamic=dynamic,
        )
        for word in words
    }


__all__ = [
    "GridSpec",
    "SPEC_2X2_64",
    "SPEC_3X3_40",
    "SPEC_4X4_32",
    "SPEC_8X8_16",
    "DEFAULT_SPEC",
    "DEFAULT_RULES",
    "select_spec",
    "render_char_patch",
    "render_word_tile_tensor",
    "render_word_tile_image",
    "render_words",
]
