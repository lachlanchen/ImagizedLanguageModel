#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple, List, Iterable

import numpy as np


def ensure_pillow():
    try:
        import PIL  # noqa: F401
    except Exception:
        import subprocess, sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])  # noqa


def ascii_matrix_image(word: str, size: int = 128) -> np.ndarray:
    """Return a (size,size) uint8 image where rows=ASCII(0..127), cols=positions.
    Pixel is 255 if that ASCII char appears at that position; else 0.
    Non-ASCII chars are ignored. Columns beyond size are truncated.
    """
    img = np.zeros((size, size), dtype=np.uint8)
    cols = min(len(word), size)
    for j in range(cols):
        ch = word[j]
        oc = ord(ch)
        if 0 <= oc < size:
            img[oc, j] = 255
    return img


def find_font_path() -> str:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""


def render_char_image(ch: str, size: int = 128) -> np.ndarray:
    """Render a single character to a centered grayscale image (size×size)."""
    ensure_pillow()
    from PIL import Image, ImageDraw, ImageFont

    font_path = find_font_path()
    if font_path:
        font = ImageFont.truetype(font_path, size=int(size * 0.75))
    else:
        font = ImageFont.load_default()

    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)
    # Compute text bounding box and center
    try:
        bbox = draw.textbbox((0, 0), ch, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
    except Exception:
        w, h = draw.textlength(ch, font=font), int(size * 0.6)
    x = (size - w) // 2
    y = (size - h) // 2
    draw.text((x, y), ch, fill=0, font=font)
    return np.array(img, dtype=np.uint8)


def is_cjk(ch: str) -> bool:
    oc = ord(ch)
    return (
        (0x3400 <= oc <= 0x4DBF) or
        (0x4E00 <= oc <= 0x9FFF) or
        (0xF900 <= oc <= 0xFAFF)
    )


def build_images(in_jsonl: str, out_dir: str, max_tokens_en: int = 3000, max_tokens_zh: int = 3000, size: int = 128) -> Tuple[int, int]:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    en_dir = out_root / "en"
    zh_dir = out_root / "zh"
    en_dir.mkdir(parents=True, exist_ok=True)
    zh_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_root / "index.tsv"
    wrote_en = 0
    wrote_zh = 0
    seen_en: set = set()
    seen_zh: set = set()

    with open(in_jsonl, "r", encoding="utf-8") as f_in, open(index_path, "w", encoding="utf-8") as f_idx:
        f_idx.write("lang\ttoken\tpath\n")
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except Exception:
                continue
            txt = (ex.get("text") or "").strip()
            lang = (ex.get("lang") or "").strip().lower()
            if not txt or lang not in {"en", "zh"}:
                continue

            if lang == "en":
                # simple whitespace tokenization
                for tok in txt.split():
                    token = tok.strip()
                    if not token or token in seen_en:
                        continue
                    img = ascii_matrix_image(token, size=size)
                    fname = f"{len(seen_en)}_{''.join(c if c.isalnum() else '_' for c in token)[:40]}.png"
                    path = en_dir / fname
                    # save
                    from PIL import Image  # ensure pillow
                    Image.fromarray(img).save(path)
                    f_idx.write(f"en\t{token}\t{path.as_posix()}\n")
                    seen_en.add(token)
                    wrote_en += 1
                    if wrote_en >= max_tokens_en:
                        break
                if wrote_en >= max_tokens_en:
                    continue
            else:  # zh
                for ch in txt:
                    if not is_cjk(ch):
                        continue
                    if ch in seen_zh:
                        continue
                    img = render_char_image(ch, size=size)
                    fname = f"{len(seen_zh)}_{ord(ch)}.png"
                    path = zh_dir / fname
                    from PIL import Image
                    Image.fromarray(img).save(path)
                    # store raw char as token
                    f_idx.write(f"zh\t{ch}\t{path.as_posix()}\n")
                    seen_zh.add(ch)
                    wrote_zh += 1
                    if wrote_zh >= max_tokens_zh:
                        break
                if wrote_zh >= max_tokens_zh:
                    continue
            if wrote_en >= max_tokens_en and wrote_zh >= max_tokens_zh:
                break
    return wrote_en, wrote_zh


def main():
    ap = argparse.ArgumentParser(description="Build 128x128 images for EN words and ZH characters from a test JSONL")
    ap.add_argument("--in", dest="inp", default="data/processed/test_100.jsonl", help="input JSONL with {text,lang}")
    ap.add_argument("--out", default="data/processed/images", help="output directory")
    ap.add_argument("--max_tokens_en", type=int, default=1000, help="max English tokens to render")
    ap.add_argument("--max_tokens_zh", type=int, default=1000, help="max Chinese chars to render")
    ap.add_argument("--size", type=int, default=128, help="image size")
    args = ap.parse_args()

    # ensure pillow
    ensure_pillow()

    wrote_en, wrote_zh = build_images(args.inp, args.out, args.max_tokens_en, args.max_tokens_zh, args.size)
    print(json.dumps({"wrote_en": wrote_en, "wrote_zh": wrote_zh, "out": args.out}))


if __name__ == "__main__":
    main()

