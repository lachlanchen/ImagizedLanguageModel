#!/usr/bin/env python3
"""Generate 128x128 tiled glyph images for English words."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from ilm.english_tiles import render_word_tile_image, render_word_tile_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render tiled English glyph images")
    parser.add_argument("text", help="Word or short string to render")
    parser.add_argument("out", help="Output image path (PNG)")
    parser.add_argument(
        "--tile-size",
        type=int,
        default=16,
        help="Tile dimension in pixels (default: 16)",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=8,
        help="Number of tiles per row in the final grid (default: 8)",
    )
    parser.add_argument(
        "--save-tensor",
        action="store_true",
        help="Also save the raw 16x16x64 tensor alongside the image",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = render_word_tile_image(
        args.text,
        tile_size=args.tile_size,
        grid_cols=args.grid_cols,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out_path.with_suffix(".png"))

    if args.save_tensor:
        tensor = render_word_tile_tensor(args.text, tile_size=args.tile_size)
        np.save(out_path.with_suffix(".npy"), tensor)


if __name__ == "__main__":
    main()
