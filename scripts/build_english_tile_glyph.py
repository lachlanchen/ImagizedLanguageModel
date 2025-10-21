#!/usr/bin/env python3
"""Generate sequential English glyph grids for a single word."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from ilm.english_tiles import (
    DEFAULT_SPEC,
    SHORT_WORD_SPEC,
    GridSpec,
    render_word_tile_image,
    render_word_tile_tensor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render tiled English glyph images")
    parser.add_argument("text", help="Word or short string to render")
    parser.add_argument("out", help="Output path (without extension)")
    parser.add_argument(
        "--tile-size",
        type=int,
        default=DEFAULT_SPEC.tile_size,
        help="Tile size for the main layout (default: 16)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_SPEC.grid_size,
        help="Grid size for the main layout (default: 8)",
    )
    parser.add_argument(
        "--short-grid-size",
        type=int,
        default=SHORT_WORD_SPEC.grid_size,
        help="Grid size for short words (default: 4)",
    )
    parser.add_argument(
        "--short-tile-size",
        type=int,
        default=SHORT_WORD_SPEC.tile_size,
        help="Tile size for short words (default: 32)",
    )
    parser.add_argument(
        "--no-dynamic",
        action="store_true",
        help="Disable dynamic short-word scaling",
    )
    parser.add_argument(
        "--save-tensor",
        action="store_true",
        help="Also save the raw tensor of per-tile glyph patches",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = GridSpec(grid_size=args.grid_size, tile_size=args.tile_size)
    short_spec = GridSpec(grid_size=args.short_grid_size, tile_size=args.short_tile_size)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image = render_word_tile_image(
        args.text,
        spec=spec,
        short_spec=short_spec,
        dynamic=not args.no_dynamic,
    )
    Image.fromarray(image).save(out_path.with_suffix(".png"))

    if args.save_tensor:
        tensor = render_word_tile_tensor(
            args.text,
            spec=spec,
            short_spec=short_spec,
            dynamic=not args.no_dynamic,
        )
        np.save(out_path.with_suffix(".npy"), tensor)


if __name__ == "__main__":
    main()
