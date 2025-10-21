"""Sequential English glyph tiling with dynamic layouts.

Each character is rendered into an individual tile and placed row-wise across a
square grid. For long strings the default layout uses 16×16 tiles over an 8×8
grid (64 positions). For shorter words (≤16 characters) we up-scale the tiles to
32×32 and place them on a 4×4 grid so the glyph makes fuller use of the canvas
while keeping the final image size at 128×128.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ilm.utils.glyphs import find_font_path


@dataclass(frozen=True)
class GridSpec:
    """Tiling configuration for sequential glyph placement."""

    grid_size: int = 8
    tile_size: int = 16

    @property
    def max_length(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def image_size(self) -> int:
        return self.grid_size * self.tile_size


DEFAULT_SPEC = GridSpec(grid_size=8, tile_size=16)
SHORT_WORD_SPEC = GridSpec(grid_size=4, tile_size=32)


def select_spec(
    word: str,
    *,
    default: GridSpec = DEFAULT_SPEC,
    short_word: GridSpec | None = SHORT_WORD_SPEC,
    dynamic: bool = True,
) -> GridSpec:
    """Choose an appropriate tiling spec for the supplied word."""

    if not dynamic or not word:
        return default
    if short_word and len(word) <= short_word.max_length:
        return short_word
    return default


def _load_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = find_font_path()
    if font_path:
        return ImageFont.truetype(font_path, size=font_size)
    return ImageFont.load_default()


def render_char_patch(character: str, *, tile_size: int = 16, intensity: int = 255) -> np.ndarray:
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
    spec: GridSpec = DEFAULT_SPEC,
    short_spec: GridSpec | None = SHORT_WORD_SPEC,
    dynamic: bool = True,
) -> np.ndarray:
    """Return a tensor of shape (grid, grid, tile, tile) for the word."""

    spec_used = select_spec(word, default=spec, short_word=short_spec, dynamic=dynamic)
    grid = spec_used.grid_size
    tile = spec_used.tile_size
    tensor = np.zeros((grid, grid, tile, tile), dtype=np.uint8)
    if not word:
        return tensor

    capped = word[: spec_used.max_length]
    for idx, ch in enumerate(capped):
        row = idx // grid
        col = idx % grid
        tensor[row, col] = render_char_patch(ch, tile_size=tile)
    return tensor


def render_word_tile_image(
    word: str,
    *,
    spec: GridSpec = DEFAULT_SPEC,
    short_spec: GridSpec | None = SHORT_WORD_SPEC,
    dynamic: bool = True,
) -> np.ndarray:
    """Render the word as a grayscale image using sequential tiling."""

    spec_used = select_spec(word, default=spec, short_word=short_spec, dynamic=dynamic)
    tensor = render_word_tile_tensor(
        word,
        spec=spec_used,
        short_spec=short_spec,
        dynamic=False,
    )
    grid = spec_used.grid_size
    tile = spec_used.tile_size
    image = np.zeros((spec_used.image_size, spec_used.image_size), dtype=np.uint8)
    for row in range(grid):
        for col in range(grid):
            patch = tensor[row, col]
            y0 = row * tile
            x0 = col * tile
            image[y0 : y0 + tile, x0 : x0 + tile] = patch
    return image


def render_words(
    words: Iterable[str],
    *,
    spec: GridSpec = DEFAULT_SPEC,
    short_spec: GridSpec | None = SHORT_WORD_SPEC,
    dynamic: bool = True,
) -> dict[str, np.ndarray]:
    """Render multiple words, optionally with dynamic short-word handling."""

    return {
        word: render_word_tile_image(
            word,
            spec=spec,
            short_spec=short_spec,
            dynamic=dynamic,
        )
        for word in words
    }


__all__ = [
    "GridSpec",
    "DEFAULT_SPEC",
    "SHORT_WORD_SPEC",
    "select_spec",
    "render_char_patch",
    "render_word_tile_tensor",
    "render_word_tile_image",
    "render_words",
]
