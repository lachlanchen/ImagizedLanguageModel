from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ilm.english_tiles import render_word_tile_image as render_en_word_tiled


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


def render_word_glyph_center(word: str, size: int = 128) -> np.ndarray:
    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)
    font_path = find_font_path()
    fs = int(size * 0.7)
    for _ in range(12):
        font = ImageFont.truetype(font_path, size=max(8, fs)) if font_path else ImageFont.load_default()
        bbox = draw.textbbox((0, 0), word, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w <= size * 0.9 and h <= size * 0.9:
            break
        fs = max(8, int(fs * 0.9))
    try:
        draw.text((size // 2, size // 2), word, fill=255, font=font, anchor="mm")
    except Exception:
        x = (size - w) // 2
        y = (size - h) // 2
        draw.text((x, y), word, fill=255, font=font)
    return np.array(img, dtype=np.uint8)


def ascii_matrix_centered(word: str, size: int = 128, pos_width: int = 126,
                          punct_col: int = 0, stop_col: int | None = None,
                          stop_row: int | None = None) -> np.ndarray:
    if stop_col is None:
        stop_col = size - 1
    if stop_row is None:
        stop_row = size - 1
    img = np.zeros((size, size), dtype=np.uint8)
    L = min(len(word), pos_width)
    start = 1 + (pos_width - L) // 2
    placed = 0
    for j in range(len(word)):
        if placed >= L:
            break
        ch = word[j]
        oc = ord(ch)
        if 0 <= oc < size:
            img[oc, start + placed] = 255
        placed += 1
    punct_set = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    pchar = next((c for c in word if c in punct_set), None)
    if pchar is not None and 0 <= ord(pchar) < size:
        img[ord(pchar), punct_col] = 255
    if 0 <= stop_row < size and 0 <= stop_col < size:
        img[stop_row, stop_col] = 255
    return img


def render_char_image(ch: str, size: int = 128) -> np.ndarray:
    font_path = find_font_path()
    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)
    fs = int(size * 0.8)
    for _ in range(8):
        font = ImageFont.truetype(font_path, size=max(8, fs)) if font_path else ImageFont.load_default()
        bbox = draw.textbbox((0, 0), ch, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w <= size * 0.9 and h <= size * 0.9:
            break
        fs = max(8, int(fs * 0.9))
    try:
        draw.text((size // 2, size // 2), ch, fill=255, font=font, anchor="mm")
    except Exception:
        x = (size - w) // 2
        y = (size - h) // 2
        draw.text((x, y), ch, fill=255, font=font)
    return np.array(img, dtype=np.uint8)


def make_rgb_token_image(lang: str, token: str, size: int = 128) -> np.ndarray:
    """Build a 3-channel token image.

    - English ("en"): use the new dynamic tiled renderer to produce a centered
      128x128 grayscale word glyph; replicate into RGB (G channel primary).
    - Chinese ("zh"): center-render a single character into a 128x128 tile; replicate.
    - Others: fallback to centered text render replicated to RGB.
    """
    if lang == "en":
        g = render_en_word_tiled(token)
        r = g.copy()
        b = np.zeros_like(g)
        return np.stack([r, g, b], axis=-1)
    elif lang == "zh":
        g = render_char_image(token, size=size)
        r = g.copy()
        b = np.zeros_like(r)
        return np.stack([r, g, b], axis=-1)
    else:
        g = render_word_glyph_center(token, size=size)
        return np.stack([g, g, g], axis=-1)


def save_rgb_image(path: str | Path, rgb: np.ndarray) -> str:
    path = Path(path)
    if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(str(path), format="PNG")
    return str(path)
