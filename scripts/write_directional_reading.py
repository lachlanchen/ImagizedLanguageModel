"""Generate an answer PNG from a pictured question and its source regions."""

import argparse
from pathlib import Path
from hashlib import sha256

import numpy as np
import torch

from scripts.train_directional_reading import MODES, restore
from scripts.write_fact_ranking import RankedAnswer
from scripts.train_fact_channel import pack, join_regions, split_span
from scripts.prepare_fact_channel import Ink, regions, read, dump
from scripts.write_region_writer import bitmap_ids
from scripts.write_open_exam import read_image
from scripts.evaluate_visual_discourse import draw
from scripts.verse_gpu_runtime import configure_cuda_cli


class DirectionalAnswer(RankedAnswer):
    def __init__(self, mode, device):
        super().__init__("whole", device)
        self.reader = restore(mode, device)

    def select(self, features, qs, pages):
        query = pack(qs, self.device)
        page = pack([join_regions(rs) for rs in pages], self.device)
        starts, lengths = self.reader.select(features, query, page, self.ranker)
        return [
            split_span(rs, s, n)
            for rs, s, n in zip(pages, starts.tolist(), lengths.tolist(), strict=True)
        ]


def write(args, make_model=DirectionalAnswer):
    net = make_model(args.mode, args.device)
    ink = Ink(net.palette)
    if args.input:
        spec = read(args.input)

        def image(r):
            return bitmap_ids(ink, read_image(args.input.parent / r["png"], r["cells"]))

        q = image(spec["question"])
        pages = [image(r) for r in spec["regions"]]
    else:
        if not args.question or not args.passage:
            raise ValueError("Supply the question and passage or the pictured manifest")
        q = ink.encode(args.question)
        pages = [ink.encode(b) for p in args.passage for b in regions(p)]
        if ink.bad:
            raise ValueError("Typed input needs missing glyphs; supply their images")
    tiles = ink.array()
    outputs, spans = net.answer(tiles, [q], [pages])
    ids, stop = outputs[0]
    region, offset, fact = spans[0]
    picture = tiles[np.asarray(ids, int)]
    selected = tiles[fact]
    args.output.mkdir(parents=True, exist_ok=False)
    draw(args.output, "answer", picture)
    draw(args.output, "fact", selected)
    result = dict(
        cells=len(ids),
        stopped=stop,
        sha256=sha256(picture.tobytes()).hexdigest(),
        region=region,
        offset=offset,
        fact_sha256=sha256(selected.tobytes()).hexdigest(),
    )
    dump(args.output / "receipt.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=MODES, default="signed")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--input", type=Path)
    p.add_argument("--question")
    p.add_argument("--passage", action="append")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.device == "cuda":
        configure_cuda_cli()
    torch.set_num_threads(3)
    print(write(args))
