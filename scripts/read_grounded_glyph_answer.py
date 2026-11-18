"""Read two glyph strips and write the locally trained model's complete answer as a PNG."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from scripts.infer_continuous_image_language import image_cells, text_cells
from scripts.infer_pictured_entry import page
from scripts.prepare_pictured_entry_answers import sha, write_json
from scripts.train_grounded_glyph_writer import LIMIT, WORK, ShapeBank, load_writer


@torch.inference_mode()
def respond(net, tiles, passage, question, device):
    if not len(passage) or not len(question):
        raise ValueError("Both images require at least one occupied cell")
    bank = ShapeBank([bytes(128), *tiles])
    p, q = [torch.as_tensor(bank.merge(s), device=device)[None] for s in (passage, question)]
    memory = torch.arange(1, len(tiles) + 1, device=device)
    ink = net.ink(bank.features(device))
    output, stop = net.generate(ink, p, q, net.write_key(ink[memory]), memory, LIMIT)
    cells = [bank.tiles[i] for i in output[0].tolist() if i]
    return b"".join(cells), bool(stop[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("passage", "question"):
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--" + key + "-image", type=Path)
        group.add_argument("--" + key + "-text")
        parser.add_argument("--" + key + "-cells", type=int)
    parser.add_argument("--output", required=True, type=Path)
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--arm", choices=("paired", "deranged"), default="paired")
    weights.add_argument("--checkpoint-dir", type=Path,
                         help="Use a fitted grounded writer with the original bitmap memory")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory.")
    streams = []
    for key in ("passage", "question"):
        path, count = getattr(args, key + "_image"), getattr(args, key + "_cells")
        if count is not None and path is None:
            parser.error("An occupied cell count requires a corresponding image.")
        cells = image_cells(path, count) if path else text_cells(getattr(args, key + "_text"))
        streams.append(np.packbits(cells.reshape(-1, 1024), axis=1))
    torch.set_num_threads(2)
    directory = args.checkpoint_dir or WORK / args.arm
    net = load_writer(directory, args.device)
    memory = WORK / "memory.npy"
    raw, stopped = respond(net, np.load(memory), *streams, args.device)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        dest = Path(tmp) / "answer"
        dest.mkdir()
        (dest / "answer.bin").write_bytes(raw)
        page(raw).save(dest / "answer.png")
        write_json(dest / "receipt.json", dict(stopped=stopped, cells=len(raw)//128,
                   weights_sha256=sha(directory/"final.safetensors"), memory_sha256=sha(memory)))
        dest.rename(args.output)
    print(args.output / "answer.png")


if __name__ == "__main__":
    main()
