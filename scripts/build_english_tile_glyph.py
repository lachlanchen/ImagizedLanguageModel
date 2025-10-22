#!/usr/bin/env python3
"""Generate sequential English glyph grids for a single word."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from PIL import Image

from ilm.english_tiles import GridSpec, render_word_tile_image, render_word_tile_tensor

CanvasRules = Sequence[Tuple[int, GridSpec]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render tiled English glyph images")
    parser.add_argument("text", help="Word or short string to render")
    parser.add_argument("out", help="Output path (without extension)")
    parser.add_argument("--len-2x2", type=int, default=3, help="Max length using the 2x2 layout (default: 3)")
    parser.add_argument("--len-3x3", type=int, default=8, help="Max length using the 3x3 layout (default: 8)")
    parser.add_argument("--len-4x4", type=int, default=16, help="Max length using the 4x4 layout (default: 16)")
    parser.add_argument("--len-8x8", type=int, default=64, help="Max length using the 8x8 layout (default: 64)")
    parser.add_argument("--tile-2x2", type=int, default=64, help="Tile size for the 2x2 layout (default: 64)")
    parser.add_argument("--tile-3x3", type=int, default=40, help="Tile size for the 3x3 layout (default: 40)")
    parser.add_argument("--tile-4x4", type=int, default=32, help="Tile size for the 4x4 layout (default: 32)")
    parser.add_argument("--tile-8x8", type=int, default=16, help="Tile size for the 8x8 layout (default: 16)")
    parser.add_argument("--canvas", type=int, default=128, help="Overall canvas size (default: 128)")
    parser.add_argument("--no-dynamic", action="store_true", help="Disable dynamic layout selection")
    parser.add_argument("--save-tensor", action="store_true", help="Also save the raw tensor of per-tile glyph patches")
    return parser.parse_args()


def build_rules(args: argparse.Namespace) -> tuple[CanvasRules, GridSpec]:
    canvas = args.canvas
    rules: list[tuple[int, GridSpec]] = []

    if args.len_2x2 > 0:
        rules.append((args.len_2x2, GridSpec(grid_size=2, tile_size=args.tile_2x2, canvas_size=canvas)))
    if args.len_3x3 > 0:
        rules.append((args.len_3x3, GridSpec(grid_size=3, tile_size=args.tile_3x3, canvas_size=canvas)))
    if args.len_4x4 > 0:
        rules.append((args.len_4x4, GridSpec(grid_size=4, tile_size=args.tile_4x4, canvas_size=canvas)))

    fallback = GridSpec(grid_size=8, tile_size=args.tile_8x8, canvas_size=canvas)
    rules.append((args.len_8x8, fallback))
    return tuple(rules), fallback


def main() -> None:
    args = parse_args()
    rules, fallback = build_rules(args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image = render_word_tile_image(
        args.text,
        rules=rules,
        fallback=fallback,
        dynamic=not args.no_dynamic,
    )
    Image.fromarray(image).save(out_path.with_suffix(".png"))

    if args.save_tensor:
        tensor = render_word_tile_tensor(
            args.text,
            rules=rules,
            fallback=fallback,
            dynamic=not args.no_dynamic,
        )
        np.save(out_path.with_suffix(".npy"), tensor)


if __name__ == "__main__":
    main()
