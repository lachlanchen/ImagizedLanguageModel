"""Read archived test PNGs with the released model and compare physical answers."""

import hashlib
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.portable_writer import read_image
from scripts.write_visual_discourse import Writer, encode_pages
from scripts.write_visual_cursor import generate, load_cursor
from scripts.prepare_visual_cursor import MODEL, PUBLIC, dump, read


def main():
    torch.set_num_threads(3)
    writer = Writer()
    net = load_cursor("persistent", "cpu")
    receipt = {}
    for p in sorted((MODEL / "examples/persistent").glob("*/*")):
        sample = read(p / "readout.json")
        image = read_image(p / "input.png", sample["prompt_cells"])
        tiles, seq = encode_pages(writer.palette, [image])
        features = torch.from_numpy(observations({"tiles": tiles}, "ink"))
        out, end = generate(
            writer.model, net, features, seq, len(writer.palette), 96, True
        )
        raw = tiles[np.asarray(out[0], dtype=int)].tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        receipt[str(p.relative_to(MODEL))] = dict(
            sha256=digest,
            match=digest == sample["sha256"] and end[0] == sample["stopped"],
        )
    dump(PUBLIC / "png-replay.json", receipt)
    print(receipt)


if __name__ == "__main__":
    main()
