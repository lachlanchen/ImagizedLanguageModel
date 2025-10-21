"""Sequential English glyph tiling.

This module renders each character of a word into a 16×16 patch and places
patches sequentially across an 8×8 grid, yielding a 128×128 square glyph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ilm.utils.glyphs import find_font_path


@dataclass(frozen=True)
class GridSpec:
    grid_size: int = 8
    tile_size: int = 16

    @property
    def max_length(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def image_size(self) -> int:
        return self.grid_size * self.tile_size


DEFAULT_SPEC = GridSpec()


def _load_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = find_font_path()
    if font_path:
        return ImageFont.truetype(font_path, size=font_size)
    return ImageFont.load_default()


def render_char_patch(character: str, *, tile_size: int = 16, intensity: int = 255) -> np.ndarray:
    """Render a single character centred in a tile-sized image."""

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


def render_word_tile_tensor(word: str, *, spec: GridSpec = DEFAULT_SPEC) -> np.ndarray:
    """Return a tensor of shape (grid, grid, tile, tile) for the word."""

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


def render_word_tile_image(word: str, *, spec: GridSpec = DEFAULT_SPEC) -> np.ndarray:
    """Render the word as a 128×128 grayscale image."""

    tensor = render_word_tile_tensor(word, spec=spec)
    grid = spec.grid_size
    tile = spec.tile_size
    image = np.zeros((spec.image_size, spec.image_size), dtype=np.uint8)
    for row in range(grid):
        for col in range(grid):
            patch = tensor[row, col]
            y0 = row * tile
            x0 = col * tile
            image[y0 : y0 + tile, x0 : x0 + tile] = patch
    return image


def render_words(words: Iterable[str], *, spec: GridSpec = DEFAULT_SPEC) -> dict[str, np.ndarray]:
    """Convenience helper that renders many words using the same spec."""

    return {word: render_word_tile_image(word, spec=spec) for word in words}


__all__ = [
    "GridSpec",
    "DEFAULT_SPEC",
    "render_char_patch",
    "render_word_tile_tensor",
    "render_word_tile_image",
    "render_words",
]
