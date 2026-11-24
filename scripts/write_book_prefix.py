"""Continue an aligned glyph sheet as a PNG. Typed text is a convenience renderer."""
import argparse
from pathlib import Path
import json
import numpy as np
from ilm.visual_lm.book_predict import Predictor, checked
from ilm.visual_lm.portable_writer import read_image, read_text, DEFAULT_FONT, save_grid

DEFAULT = Path(__file__).resolve().parents[1]/"models/book_prefix"


def write(bundle, raw, steps, path, device):
    if not len(raw) or steps < 1:
        raise ValueError("Supply a nonempty prefix and a positive output length")
    manifest = json.loads((bundle/"bundle.json").read_text())
    train = np.load(checked(bundle/"palette.npy", manifest["palette"]))
    # The dummy ID zero marks missing history; a real empty sheet cell gets its own ID.
    cells = [row.tobytes() for row in train]
    lookup = {cell:i for i,cell in enumerate(cells) if i}
    ids = []
    for row in raw[-64:]:
        key = row.tobytes()
        if key not in lookup:
            lookup[key] = len(cells)
            cells.append(key)
        ids.append(lookup[key])
    tiles = np.frombuffer(b"".join(cells), np.uint8).reshape(-1,128)
    model = Predictor(bundle, "ink-hybrid", {"tiles":tiles}, device)
    output = model.generate(np.array([ids], np.uint16), steps)[0]
    save_grid(tiles[output].tobytes(), path)
    return tiles[output]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--input",type=Path)
    sources.add_argument("--text")
    parser.add_argument("--cells",type=int,help="Occupied cells in a padded input sheet")
    parser.add_argument("--steps",type=int,default=16)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--bundle",type=Path,default=DEFAULT)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args = parser.parse_args()
    raw = read_image(args.input,args.cells) if args.input else read_text(args.text,DEFAULT_FONT)
    write(args.bundle,raw,args.steps,args.output,args.device)
