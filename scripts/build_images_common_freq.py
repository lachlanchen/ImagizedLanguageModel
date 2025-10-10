#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np


def ensure_packages():
    import importlib, subprocess, sys  # noqa

    def pip_install(pkg):
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

    try:
        import PIL  # noqa: F401
    except Exception:
        pip_install("pillow")
    try:
        import wordfreq  # noqa: F401
    except Exception:
        pip_install("wordfreq")


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
    from PIL import Image, ImageDraw, ImageFont

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
    # Use anchor 'mm' if available for robust centering
    try:
        draw.text((size // 2, size // 2), word, fill=255, font=font, anchor="mm")
    except Exception:
        x = (size - w) // 2
        y = (size - h) // 2
        draw.text((x, y), word, fill=255, font=font)
    return np.array(img, dtype=np.uint8)


def ascii_matrix_centered(word: str, size: int = 128, pos_width: int = 126,
                          punct_col: int = 0, stop_col: int = 127,
                          stop_row: int = 127) -> np.ndarray:
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
    if 0 <= stop_row < size:
        img[stop_row, stop_col] = 255
    return img


def render_char_image(ch: str, size: int = 128) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

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


def is_cjk(ch: str) -> bool:
    oc = ord(ch)
    return (
        (0x3400 <= oc <= 0x4DBF) or
        (0x4E00 <= oc <= 0x9FFF) or
        (0xF900 <= oc <= 0xFAFF)
    )


def top_en_words(n: int) -> List[str]:
    from wordfreq import top_n_list
    words = top_n_list("en", n)
    # keep reasonably short words (<=126)
    return [w for w in words if 0 < len(w) <= 126]


def top_zh_chars(n_chars: int, sample_words: int = 20000) -> List[str]:
    from wordfreq import top_n_list
    words = top_n_list("zh", sample_words)
    cnt = Counter()
    for w in words:
        for ch in w:
            if is_cjk(ch):
                cnt[ch] += 1
    return [ch for ch, _ in cnt.most_common(n_chars)]


def build_common_freq(out_dir: str, size: int, n_en: int, n_zh_chars: int) -> Tuple[int, int]:
    ensure_packages()
    from PIL import Image

    out_root = Path(out_dir)
    (out_root / "en").mkdir(parents=True, exist_ok=True)
    (out_root / "zh").mkdir(parents=True, exist_ok=True)
    idx = open(out_root / "index.tsv", "w", encoding="utf-8")
    idx.write("lang\ttoken\tpath\n")

    en_words = top_en_words(n_en)
    zh_chars = top_zh_chars(n_zh_chars)

    # English
    wrote_en = 0
    for w in en_words:
        r = ascii_matrix_centered(w, size=size)
        g = render_word_glyph_center(w, size=size)
        b = np.zeros_like(r)
        rgb = np.stack([r, g, b], axis=-1)
        fname = f"en_{wrote_en}_{''.join(c if c.isalnum() else '_' for c in w)[:40]}.png"
        path = out_root / "en" / fname
        Image.fromarray(rgb).save(path)
        idx.write(f"en\t{w}\t{path.as_posix()}\n")
        wrote_en += 1

    # Chinese
    wrote_zh = 0
    for ch in zh_chars:
        g = render_char_image(ch, size=size)
        r = g.copy()
        r[-1, -1] = 255
        b = np.zeros_like(r)
        rgb = np.stack([r, g, b], axis=-1)
        fname = f"zh_{wrote_zh}_{ord(ch)}.png"
        path = out_root / "zh" / fname
        Image.fromarray(rgb).save(path)
        idx.write(f"zh\t{ch}\t{path.as_posix()}\n")
        wrote_zh += 1

    idx.close()
    return wrote_en, wrote_zh


def main():
    ap = argparse.ArgumentParser(description="Build common EN/ZN images from frequency lists (several thousand)")
    ap.add_argument("--out", default="data/processed/images_common_freq", help="output directory")
    ap.add_argument("--size", type=int, default=128, help="image size")
    ap.add_argument("--en", type=int, default=5000, help="number of English words")
    ap.add_argument("--zh", type=int, default=5000, help="number of Chinese characters")
    args = ap.parse_args()

    wrote_en, wrote_zh = build_common_freq(args.out, args.size, args.en, args.zh)
    print(json.dumps({"wrote_en": wrote_en, "wrote_zh": wrote_zh, "out": args.out}))


if __name__ == "__main__":
    main()

