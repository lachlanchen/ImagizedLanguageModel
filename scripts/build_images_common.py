#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np


def ensure_pillow():
    try:
        import PIL  # noqa: F401
    except Exception:
        import subprocess, sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])  # noqa


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


def render_word_glyph(word: str, size: int = 128) -> np.ndarray:
    ensure_pillow()
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("L", (size, size), color=0)  # black background
    draw = ImageDraw.Draw(img)

    font_path = find_font_path()
    # Find a font size that fits the width with a small margin
    if not word:
        return np.array(img, dtype=np.uint8)
    # Start from 3/4 of image size and downscale if needed
    fs = int(size * 0.7)
    for trial in range(12):
        font = ImageFont.truetype(font_path, size=max(8, fs)) if font_path else ImageFont.load_default()
        try:
            bbox = draw.textbbox((0, 0), word, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        except Exception:
            w = draw.textlength(word, font=font)
            h = int(size * 0.6)
        if w <= size * 0.9 and h <= size * 0.9:
            break
        fs = max(8, int(fs * 0.9))

    x = (size - w) // 2
    y = (size - h) // 2
    draw.text((x, y), word, fill=255, font=font)
    return np.array(img, dtype=np.uint8)


def ascii_matrix_centered(word: str, size: int = 128, pos_width: int = 126,
                          punct_col: int = 0, stop_col: int = 127,
                          stop_row: int = 127) -> np.ndarray:
    """Centered ASCII one-hot rows across columns with reserved punctuation and stop columns.

    - size: image height/width (128)
    - pos_width: number of columns reserved for characters (126), leaving 0 for punctuation and 127 for STOP
    - For a word of length L<=pos_width, positions are centered: start = 1 + floor((pos_width-L)/2)
    - Punctuation: first ASCII punctuation in the word is marked in column 0 at its ASCII row.
    - STOP: a single pixel at (stop_row, stop_col) is set to 255.
    """
    img = np.zeros((size, size), dtype=np.uint8)
    L = min(len(word), pos_width)
    start = 1 + (pos_width - L) // 2  # centered in [1..126]

    # place characters
    placed = 0
    for j in range(len(word)):
        if placed >= L:
            break
        ch = word[j]
        oc = ord(ch)
        if 0 <= oc < size:
            img[oc, start + placed] = 255
        placed += 1

    # punctuation column
    punct_set = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    pchar = next((c for c in word if c in punct_set), None)
    if pchar is not None:
        img[ord(pchar), punct_col] = 255

    # stop bit
    if 0 <= stop_row < size:
        img[stop_row, stop_col] = 255

    return img


def render_char_image(ch: str, size: int = 128) -> np.ndarray:
    ensure_pillow()
    from PIL import Image, ImageDraw, ImageFont
    font_path = find_font_path()
    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)
    fs = int(size * 0.8)
    for _ in range(8):
        font = ImageFont.truetype(font_path, size=max(8, fs)) if font_path else ImageFont.load_default()
        try:
            bbox = draw.textbbox((0, 0), ch, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        except Exception:
            w = draw.textlength(ch, font=font)
            h = int(size * 0.6)
        if w <= size * 0.9 and h <= size * 0.9:
            break
        fs = max(8, int(fs * 0.9))
    x = (size - w) // 2
    y = (size - h) // 2
    draw.text((x, y), ch, fill=255, font=font)
    return np.array(img, dtype=np.uint8)


COMMON_EN: List[str] = [
    # 100 common English words (lowercase) incl. some punctuation examples
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    "hello!", "okay.", "done?"
]


COMMON_ZH: List[str] = [
    # 150 common Chinese characters
    "的","一","是","了","我","不","在","人","有","他","这","个","上","们","来","到","时","大","地","为",
    "子","中","你","说","生","国","年","着","就","那","和","要","她","出","也","得","里","后","自","以",
    "会","家","可","下","而","过","天","去","能","对","小","多","然","于","心","学","么","之","都","好",
    "看","起","发","当","没","成","只","如","事","把","还","用","第","样","道","想","作","种","开","美",
    "总","从","无","情","己","面","最","女","但","现","前","些","所","同","日","手","又","行","意","动",
    "方","期","它","头","经","长","儿","回","位","分","爱","老","因","很","给","名","法","间","斯","知",
    "世","把","被","向","进","此","话","更","比","次","光","谁","高","已","呢","别","打","找","真","怎"
]


def build_common(out_dir: str, size: int = 128) -> Tuple[int, int]:
    ensure_pillow()
    from PIL import Image

    out_root = Path(out_dir)
    (out_root / "en").mkdir(parents=True, exist_ok=True)
    (out_root / "zh").mkdir(parents=True, exist_ok=True)
    idx = open(out_root / "index.tsv", "w", encoding="utf-8")
    idx.write("lang\ttoken\tpath\n")

    wrote_en = 0
    wrote_zh = 0

    # English: R channel = centered ASCII matrix (with punctuation & stop); G channel = glyph; B=0
    for w in COMMON_EN:
        r = ascii_matrix_centered(w, size=size)
        g = render_word_glyph(w, size=size)
        b = np.zeros_like(r)
        rgb = np.stack([r, g, b], axis=-1)
        fname = f"en_{wrote_en}_{''.join(c if c.isalnum() else '_' for c in w)[:40]}.png"
        path = out_root / "en" / fname
        Image.fromarray(rgb).save(path)
        idx.write(f"en\t{w}\t{path.as_posix()}\n")
        wrote_en += 1

    # Chinese: use glyph in G channel, replicate to R for visibility; add stop pixel at (127,127) in R
    for ch in COMMON_ZH:
        g = render_char_image(ch, size=size)
        r = g.copy()
        r[-1, -1] = 255  # stop marker for consistency
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
    ap = argparse.ArgumentParser(description="Build common EN/ZN word/char images (128x128) with centered ASCII+glyph")
    ap.add_argument("--out", default="data/processed/images_common", help="output directory")
    ap.add_argument("--size", type=int, default=128, help="image size (default 128)")
    args = ap.parse_args()

    wrote_en, wrote_zh = build_common(args.out, size=args.size)
    print(json.dumps({"wrote_en": wrote_en, "wrote_zh": wrote_zh, "out": args.out}))


if __name__ == "__main__":
    main()

