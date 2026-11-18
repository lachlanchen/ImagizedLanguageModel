"""Write an answer on CPU from a pictured question and a two-entry evidence card."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch

from ilm.visual_lm.entry_span_writer import EntryWriter, alternatives, parse_card
from scripts.infer_continuous_image_language import image_cells

DEFAULT = Path(__file__).resolve().parents[1] / "models/pictured_entry_writer"


def page(stream, width=32):
    width = min(width, max(1, len(stream) // 128))
    cells = np.unpackbits(np.frombuffer(stream, np.uint8)).reshape(-1, 32, 32)
    grid = np.zeros(
        (max(1, (len(cells) + width - 1) // width) * width, 32, 32), dtype=np.uint8
    )
    grid[: len(cells)] = cells
    return Image.fromarray(
        255
        * (
            1
            - grid.reshape(-1, width, 32, 32)
            .transpose(0, 2, 1, 3)
            .reshape(-1, 32 * width)
        )
    )


def bitmap(path, count):
    return np.packbits(image_cells(path, count).reshape(-1, 1024), axis=-1).tobytes()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--question", type=Path, required=True)
    p.add_argument("--card", type=Path, required=True)
    p.add_argument("--question-cells", type=int)
    p.add_argument("--card-cells", type=int)
    p.add_argument("--model-dir", type=Path, default=DEFAULT)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    question = bitmap(args.question, args.question_cells)
    card = bitmap(args.card, args.card_cells)
    with np.load(args.model_dir / "appearance.npz") as z:
        marks, literals = tuple(map(bytes, z["marks"])), list(map(bytes, z["literals"]))
    torch.set_num_threads(2)
    writer = EntryWriter().eval()
    weights = args.model_dir / "matched.safetensors"
    writer.load_state_dict(load_file(str(weights)))
    [answer], [stop] = writer.write(
        [alternatives(parse_card(card, marks), question, literals)]
    )
    args.output.mkdir(parents=True)
    page(answer).save(args.output / "answer.png")
    (args.output / "answer.bin").write_bytes(answer)
    digest = lambda x: hashlib.sha256(x).hexdigest()
    receipt = dict(
        stopped=stop,
        cells=len(answer) // 128,
        output_sha=digest(answer),
        card_sha=digest(card),
        question_sha=digest(question),
        weights_sha=digest(weights.read_bytes()),
    )
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
