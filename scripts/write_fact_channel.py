"""Write an answer PNG from a pictured question and its source regions."""

import argparse
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_fact_channel import MODEL, PUB
from scripts.prepare_region_writer import Ink, regions
from scripts.prepare_open_exam import read, dump, sha
from scripts.evaluate_visual_discourse import draw
from scripts.train_region_writer import join_regions, pack
from scripts.train_fact_channel import load, predict
from scripts.write_region_writer import bitmap_ids
from scripts.write_open_exam import read_image


class VisualAnswer:
    def __init__(self, arm, device="cpu", root=MODEL):
        self.device = device
        self.arm = arm
        self.writer = load(arm, device, root)
        self.reader = load("reader", device, root) if arm != "direct" else None
        spec = read(PUB / "corpus.json")
        for name in ("palette.npy", "readback.json"):
            if sha(root / name) != spec["files"][name]:
                raise ValueError("Changed " + name)
        self.palette = np.load(root / "palette.npy")
        self.n = len(self.palette)
        self.chars = read(root / "readback.json")

    @torch.inference_mode()
    def answer(self, tiles, qs, pages, supplied=None, limit=256, evidence="both"):
        if (
            len(qs) != len(pages)
            or any(not len(q) for q in qs)
            or any(not rs or any(not len(r) for r in rs) for rs in pages)
        ):
            raise ValueError("The question and all source regions must contain glyphs")
        features = torch.from_numpy(observations({"tiles": tiles}, "ink")).to(
            self.device
        )
        if supplied is not None:
            spans = supplied
        elif self.reader is not None:
            spans = predict(self.reader, features, qs, pages, self.device)
        else:
            spans = [None] * len(qs)
        if self.arm == "direct":
            source = (
                [join_regions(rs) for rs in pages]
                if supplied is None
                else [s[2] for s in supplied]
            )
        elif evidence == "region":
            source = [rs[s[0]] for rs, s in zip(pages, spans, strict=True)]
        elif self.arm == "human" or evidence == "fact":
            source = [s[2] for s in spans]
        else:
            source = [
                join_regions([s[2], rs[s[0]]])
                for s, rs in zip(spans, pages, strict=True)
            ]
        context, h = self.writer.encode(
            features, self.n, pack(qs, self.device), pack(source, self.device)
        )
        return self.writer.write(context, h, limit), spans


def main(args):
    if args.device == "cpu":
        torch.set_num_threads(3)
    model = VisualAnswer(args.arm, args.device, args.model_dir)
    ink = Ink(model.palette)
    if args.input:
        doc = read(args.input)

        def picture(r):
            return bitmap_ids(
                ink, read_image(args.input.parent / r["png"], int(r["cells"]))
            )

        q = picture(doc["question"])
        page = [picture(r) for r in doc["regions"]]
    else:
        if not args.question or not args.passage:
            raise ValueError("Supply a pictured manifest or question and passage")
        q = ink.encode(args.question)
        page = [ink.encode(r) for p in args.passage for r in regions(p)]
        if ink.bad:
            raise ValueError(
                "Typed input has missing font coverage; supply the original pixels"
            )
    tiles = ink.array()
    answers, spans = model.answer(tiles, [q], [page])
    ids, stop = answers[0]
    raw = tiles[np.asarray(ids, int)]
    args.output.mkdir(exist_ok=False, parents=True)
    draw(args.output, "answer", raw)
    rc = dict(stopped=stop, cells=len(ids), sha256=sha256(raw.tobytes()).hexdigest())
    if spans[0] is not None:
        b, i, s = spans[0]
        fact = tiles[np.asarray(s, int)]
        draw(args.output, "fact", fact)
        rc.update(
            region=b,
            offset=i,
            fact_cells=len(s),
            fact_sha256=sha256(fact.tobytes()).hexdigest(),
        )
    dump(args.output / "receipt.json", rc)
    print(args.output / "answer.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path)
    p.add_argument("--question")
    p.add_argument("--passage", action="append")
    p.add_argument(
        "--arm", choices=["direct", "human", "predicted"], default="predicted"
    )
    p.add_argument("--model-dir", type=Path, default=MODEL)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
