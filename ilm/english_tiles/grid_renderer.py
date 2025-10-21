"""English glyph tiling utilities.

This module builds a 16x16x64 tensor representation for English words where each
channel corresponds to a canonical character bucket. The 64 tiles are arranged
into an 8x8 grid so they can be rasterized as a 128x128 image reminiscent of a
CJK square glyph.
"""

from __future__ import annotations

import string
from functools import lru_cache
from typing import Dict, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ilm.utils.glyphs import find_font_path

# 64 canonical slots: 26 uppercase letters, 10 digits, 27 punctuation/space, 1 UNK
DEFAULT_TILE_SET = [
    *list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    *list("0123456789"),
    " ",
    ".",
    ",",
    ";",
    ":",
    "?",
    "!",
    "'",
    '"',
    "-",
    "_",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "/",
    "\\",
    "@",
    "#",
    "$",
    "%",
    "&",
    "+",
    "*",
    "=",
    "UNK",
]

ENGLISH_TILE_TO_INDEX: Dict[str, int] = {token: idx for idx, token in enumerate(DEFAULT_TILE_SET)}

_WHITESPACE = set(string.whitespace)


@lru_cache(maxsize=None)
def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = find_font_path()
    if font_path:
        return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def _tile_key_for_char(ch: str) -> str:
    if not ch:
        return "UNK"
    if ch in _WHITESPACE:
        return " "
    if ch in ENGLISH_TILE_TO_INDEX:
        return ch
    upper = ch.upper()
    if upper in ENGLISH_TILE_TO_INDEX and upper.isalpha():
        return upper
    if ch in {";", ":", "?", "!", "'", '"', "-", "_", "/", "\\", "@", "#", "$", "%", "&", "+", "*", "="}:
        return ch
    return "UNK"


def render_char_patch(char: str, size: int = 16, intensity: int = 255) -> np.ndarray:
    """Render a single character into a centered square glyph patch."""

    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)

    if not char:
        char = " "

    if char in _WHITESPACE:
        radius = max(1, size // 8)
        cx = cy = size // 2
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=intensity)
        return np.array(img, dtype=np.uint8)

    if char == "UNK":
        draw.line((2, 2, size - 3, size - 3), fill=intensity, width=1)
        draw.line((2, size - 3, size - 3, 2), fill=intensity, width=1)
        return np.array(img, dtype=np.uint8)

    font_size = max(10, int(size * 0.9))
    for _ in range(6):
        font = _load_font(font_size)
        bbox = font.getbbox(char)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= size - 2 and height <= size - 2:
            break
        font_size = max(6, int(font_size * 0.9))
    else:
        font = _load_font(font_size)
        bbox = font.getbbox(char)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

    try:
        draw.text((size // 2, size // 2), char, font=font, fill=intensity, anchor="mm")
    except Exception:
        x_pos = (size - width) // 2
        y_pos = (size - height) // 2
        draw.text((x_pos, y_pos), char, font=font, fill=intensity)

    return np.array(img, dtype=np.uint8)


def render_word_tile_tensor(
    word: str,
    *,
    tile_size: int = 16,
    tile_set: Sequence[str] = DEFAULT_TILE_SET,
) -> np.ndarray:
    """Return a (len(tile_set) x tile_size x tile_size) tensor for the word."""

    num_tiles = len(tile_set)
    tensor = np.zeros((num_tiles, tile_size, tile_size), dtype=np.float32)
    counts = np.zeros(num_tiles, dtype=np.int32)
    if not word:
        return tensor.astype(np.uint8)

    total = max(1, len(word) - 1)
    for pos, raw_ch in enumerate(word):
        key = _tile_key_for_char(raw_ch)
        idx = ENGLISH_TILE_TO_INDEX.get(key, ENGLISH_TILE_TO_INDEX["UNK"])
        patch_char = raw_ch if key != "UNK" else "UNK"
        patch = render_char_patch(patch_char, size=tile_size, intensity=200)
        weight = 0.6 + 0.4 / (counts[idx] + 1)
        tensor[idx] = np.clip(tensor[idx] + patch.astype(np.float32) * weight, 0.0, 255.0)

        row = int(round((pos / total) * (tile_size - 1))) if total else tile_size // 2
        col = min(tile_size - 1, counts[idx])
        tensor[idx, row, col] = 255.0

        counts[idx] += 1

    return tensor.astype(np.uint8)


def render_word_tile_image(
    word: str,
    *,
    tile_size: int = 16,
    grid_cols: int = 8,
    tile_set: Sequence[str] = DEFAULT_TILE_SET,
) -> np.ndarray:
    """Render the tiled tensor as a square grayscale image."""

    tensor = render_word_tile_tensor(word, tile_size=tile_size, tile_set=tile_set)
    num_tiles = len(tile_set)
    grid_rows = (num_tiles + grid_cols - 1) // grid_cols
    height = grid_rows * tile_size
    width = grid_cols * tile_size
    grid = np.zeros((height, width), dtype=np.uint8)
    for idx in range(num_tiles):
        row = idx // grid_cols
        col = idx % grid_cols
        y0 = row * tile_size
        x0 = col * tile_size
        grid[y0 : y0 + tile_size, x0 : x0 + tile_size] = tensor[idx]
    return grid


__all__ = [
    "DEFAULT_TILE_SET",
    "ENGLISH_TILE_TO_INDEX",
    "render_char_patch",
    "render_word_tile_tensor",
    "render_word_tile_image",
]
