"""Translate an aligned printed image, optionally normalizing its typeface from pixels."""
import argparse
from pathlib import Path

import numpy as np
import torch

from ilm.visual_lm.field_to_glyph import ROOT
from ilm.visual_lm.glyph_typeface_reader import load_typeface_reader
from ilm.visual_lm.target_glyph_teaching import load_writer
from scripts.write_neural_glyphs import grid_bits, save_ink


def normalize_print(grids, device):
    palette = np.load(ROOT / "models/ordered_translation/palette.npy")[2:]
    adapter = load_typeface_reader(ROOT / "models/book_breadth/typeface", palette, device)
    sizes = [len(x) for x in grids]
    packed = np.packbits(np.concatenate(grids).reshape(-1, 1024), axis=1)
    changed = adapter.transport(packed, palette)
    pixels = np.unpackbits(changed, axis=1).reshape(-1, 32, 32).astype(bool)
    return np.split(pixels, np.cumsum(sizes)[:-1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--cells", type=int, required=True)
    p.add_argument("--to", choices=("en", "zh"), required=True)
    p.add_argument("--normalize-print", action="store_true")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    x = grid_bits(args.input, args.cells)
    if args.normalize_print:
        x, = normalize_print([x], args.device)
    writer = load_writer("guided", args.device)
    ink, n, end = writer.write(torch.from_numpy(x)[None].to(args.device),
                               torch.tensor([len(x)], device=args.device), ("en", "zh").index(args.to))
    save_ink(ink[0, :int(n[0])].cpu().numpy(), args.output)
    print(f"{int(n[0])} emitted cells, stopped={bool(end[0])}")


if __name__ == "__main__":
    main()
