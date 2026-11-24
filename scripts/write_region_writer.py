"""Read pictured question and regions; write the complete answer as a PNG grid."""

import argparse
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.region_writer import RegionSelector, RegionWriter
from scripts.evaluate_visual_discourse import draw
from scripts.prepare_region_writer import MODEL, PUB, Ink, regions
from scripts.prepare_open_exam import dump, read, sha
from scripts.train_region_writer import selector_input, pack, join_regions
from scripts.write_open_exam import read_image


def model(arm, device, model_dir=MODEL):
    path = model_dir / arm / "weights.safetensors"
    if sha(path) != read(path.parent / "fit.json")["weights"]:
        raise ValueError("Invalid checkpoint")
    net = (RegionSelector() if arm == "selector" else RegionWriter()).to(device)
    net.load_state_dict(load_file(str(path)))
    return net.eval()


def phrase_model(directory, device, model_dir=MODEL):
    from ilm.visual_lm.phrase_region import PhraseRegionWriter

    receipt = read(directory / "fit.json")
    weights = directory / "weights.safetensors"
    if (
        sha(weights) != receipt["weights"]
        or sha(model_dir / "selector/weights.safetensors")
        != receipt["binding"]["selector"]
    ):
        raise ValueError("Phrase writer or region selector differs from its receipt")
    net = PhraseRegionWriter().to(device)
    net.load_state_dict(load_file(str(weights)))
    return net.eval()


@torch.inference_mode()
def answer(net, selector, arm, tiles, n, queries, pages, device, *, supplied=None):
    for q, p in zip(queries, pages, strict=True):
        if (
            not 0 < len(q) <= 192
            or not 1 <= len(p) <= 24
            or any(not 1 <= len(r) <= 256 for r in p)
            or sum(map(len, p)) > 1536
        ):
            raise ValueError("Input exceeds the registered region geometry")
    f = torch.from_numpy(observations({"tiles": tiles}, "ink")).to(device)
    if supplied is None:
        ix = (
            selector(f, *selector_input(queries, pages, device))
            .argmax(-1)
            .cpu()
            .tolist()
            if arm == "selected"
            else [None] * len(queries)
        )
    else:
        ix = supplied
    p = [
        join_regions(rows) if j is None else rows[j]
        for rows, j in zip(pages, ix, strict=True)
    ]
    ctx, h = net.encode(f, n, pack(queries, device), pack(p, device))
    return net.write(ctx, h), ix


def bitmap_ids(ink, values):
    ids = []
    for v in values:
        b = bytes(v)
        if b not in ink.byink:
            ink.byink[b] = len(ink.tiles)
            ink.tiles.append(b)
        ids.append(ink.byink[b])
    return ids


def main(args):
    device = args.device
    if device == "cpu":
        torch.set_num_threads(3)
    palette_path = args.model_dir / "palette.npy"
    if sha(palette_path) != read(PUB / "corpus.json")["files"]["palette.npy"]:
        raise ValueError("Changed output bitmap bank")
    palette = np.load(palette_path)
    ink = Ink(palette)
    if args.input:
        doc = read(args.input)

        def raster(row):
            return bitmap_ids(
                ink, read_image(args.input.parent / row["png"], int(row["cells"]))
            )

        q = raster(doc["question"])
        pages = [raster(row) for row in doc["regions"]]
    else:
        if not args.question or not args.passage:
            raise ValueError("Supply a pictured document or a question and passage")
        q = ink.encode(args.question)
        pages = [ink.encode(s) for r in args.passage for s in regions(r)]
        if ink.bad:
            raise ValueError("The supplied font cannot render every requested cell")
    tiles = ink.array()
    if args.phrase_checkpoint and args.arm != "selected":
        raise ValueError("Phrase checkpoints use the learned region selector")
    net = (
        phrase_model(args.phrase_checkpoint, device, args.model_dir)
        if args.phrase_checkpoint
        else model(args.arm, device, args.model_dir)
    )
    pred, selected = answer(
        net,
        model("selector", device, args.model_dir) if args.arm == "selected" else None,
        args.arm,
        tiles,
        len(palette),
        [q],
        [pages],
        device,
    )
    ids, end = pred[0]
    pixels = tiles[np.asarray(ids, int)]
    args.output.mkdir(parents=True, exist_ok=False)
    draw(args.output, "answer", pixels)
    if selected[0] is not None:
        draw(args.output, "selected-region", tiles[pages[selected[0]]])
    dump(
        args.output / "receipt.json",
        dict(
            cells=len(ids),
            stopped=end,
            selected_region=selected[0],
            sha256=sha256(pixels.tobytes()).hexdigest(),
        ),
    )
    print(args.output / "answer.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=MODEL)
    p.add_argument("--arm", choices=("direct", "selected"), default="selected")
    p.add_argument("--question")
    p.add_argument("--passage", action="append")
    p.add_argument("--phrase-checkpoint", type=Path)
    p.add_argument("--input", type=Path)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
