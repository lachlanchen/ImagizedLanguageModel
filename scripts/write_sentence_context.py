"""Generate an answer image from a pictured question and a page of source strips."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.sentence_context import ContextWriter
from scripts.prepare_sentence_context import MODEL, PUB, read, dump, sha, Ink, regions
from scripts.train_sentence_context import ARMS, extended_page
from scripts.train_fact_channel import load, pack
from scripts.train_fact_ranking import restore
from scripts.write_fact_ranking import RankedAnswer
from scripts.write_region_writer import bitmap_ids
from scripts.write_open_exam import read_image
from scripts.evaluate_visual_discourse import draw
from scripts.verse_gpu_runtime import configure_cuda_cli


class SentenceAnswer(RankedAnswer):
    def __init__(self, mode, device):
        self.device = device
        self.capture = False
        rc = read(MODEL / mode / "fit.json")
        wp = MODEL / mode / "weights.safetensors"
        corpus = read(PUB / "optical.json")
        if (
            sha(wp) != rc["weights"]
            or sha(MODEL / "palette.npy") != corpus["palette"]
            or sha(MODEL / "readback.json") != corpus["files"]["readback.json"]
        ):
            raise ValueError("Changed model or optical inventory")
        self.palette = np.load(MODEL / "palette.npy")
        self.n = len(self.palette)
        self.chars = read(MODEL / "readback.json")
        self.reader = load("reader", device)
        self.ranker = restore("whole", device)
        self.writer = ContextWriter().to(device).eval()
        self.writer.load_state_dict(load_file(str(wp)))

    @torch.inference_mode()
    def answer(self, tiles, qs, pages, supplied=None, limit=1024):
        if (
            not qs
            or len(qs) != len(pages)
            or any(not len(q) for q in qs)
            or any(not rs or any(not len(b) for b in rs) for rs in pages)
        ):
            raise ValueError("Supply a nonempty question and nonempty pictured regions")
        if limit < 1:
            raise ValueError("A positive output budget is required")
        features = torch.from_numpy(observations({"tiles": tiles}, "ink")).to(
            self.device
        )
        spans = self.select(features, qs, pages) if supplied is None else supplied
        sources = [
            extended_page(rs, f) for rs, (_, _, f) in zip(pages, spans, strict=True)
        ]
        ctx, h = self.writer.encode(
            features, self.n, pack(qs, self.device), pack(sources, self.device)
        )
        return self.writer.write(ctx, h, limit), spans


def write(args):
    net = SentenceAnswer(args.arm, args.device)
    ink = Ink(net.palette)
    if args.input:
        r = read(args.input)

        def picture(k):
            return bitmap_ids(
                ink, read_image(args.input.parent / k["png"], int(k["cells"]))
            )

        q = picture(r["question"])
        pages = [picture(p) for p in r["regions"]]
    else:
        if not args.question:
            raise ValueError("Provide a question or an input manifest")
        q = ink.encode(args.question)
        pages = [
            ink.encode(s) for p in (args.passage or [args.question]) for s in regions(p)
        ]
        if ink.bad:
            raise ValueError("Missing font coverage for typed input")
    tiles = ink.array()
    out, spans = net.answer(tiles, [q], [pages], limit=args.limit)
    ids, stop = out[0]
    b, s, f = spans[0]
    image = tiles[np.asarray(ids, int)]
    fact = tiles[np.asarray(f, int)]
    args.output.mkdir(parents=True, exist_ok=False)
    draw(args.output, "answer", image)
    draw(args.output, "fact", fact)
    rc = dict(
        stopped=stop,
        cells=len(ids),
        sha256=sha256(image.tobytes()).hexdigest(),
        region=b,
        offset=s,
        fact_sha256=sha256(fact.tobytes()).hexdigest(),
    )
    dump(args.output / "receipt.json", rc)
    return rc


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--input", type=Path)
    p.add_argument("--question")
    p.add_argument("--passage", action="append")
    p.add_argument("--limit", type=int, default=1024)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.device == "cuda":
        configure_cuda_cli()
    torch.set_num_threads(3)
    print(write(args))
