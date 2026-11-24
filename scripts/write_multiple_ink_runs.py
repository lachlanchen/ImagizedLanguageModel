"""Write a visual list from a pictured question and page, without an external service."""

import argparse
from pathlib import Path
import numpy as np
import torch

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.multiple_ink_runs import intervals
from ilm.visual_lm.portable_writer import read_image, read_text, DEFAULT_FONT
from scripts.evaluate_visual_discourse import draw
from scripts.prepare_multiple_ink_runs import dump
from scripts.train_multiple_ink_runs import load
from scripts.train_visual_cursor import pad
from scripts.write_visual_discourse import encode_pages


@torch.inference_mode()
def forward(model, bank, qs, ps):
    device = bank.device
    q, p = [
        torch.from_numpy(pad(v)).to(device=device, dtype=torch.long) for v in (qs, ps)
    ]
    roles, counts = model(bank, q, p)
    return roles.cpu().numpy(), counts.cpu().numpy()


def write(parts, dest):
    dest.mkdir(parents=True, exist_ok=True)
    joiner = read_text(" | ", DEFAULT_FONT)
    seq = []
    for i, part in enumerate(parts):
        if i:
            seq.append(joiner)
        seq.append(part)
    pixels = np.concatenate(seq) if seq else np.empty((0, 128), np.uint8)
    draw(dest, "answer", pixels)
    (dest / "answer.bin").write_bytes(pixels.tobytes())
    return len(pixels)


@torch.inference_mode()
def infer(model, q, p, mode="count"):
    if not 1 <= len(q) <= 160 or not 1 <= len(p) <= 1200:
        raise ValueError("Expected 1–160 question and 1–1,200 page cells")
    tiles, (qi, pi) = encode_pages(np.zeros((2, 128), np.uint8), [q, p])
    features = torch.from_numpy(observations({"tiles": tiles}, "ink")).to(
        next(model.parameters()).device
    )
    role, count = forward(model, model.ink(features), [qi], [pi])
    return intervals(role[0], count[0], mode)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("question", "page"):
        g = ap.add_mutually_exclusive_group(required=True)
        g.add_argument("--" + name)
        g.add_argument("--" + name + "-image", type=Path)
        ap.add_argument("--" + name + "-cells", type=int)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--mode", choices=("count", "single", "free"), default="count")
    args = ap.parse_args()
    if args.device == "cuda":
        from scripts.verse_gpu_runtime import configure_cuda_cli

        configure_cuda_cli()
    torch.set_num_threads(3)

    def pixels(name):
        image = getattr(args, name + "_image")
        return (
            read_image(image, getattr(args, name + "_cells"))
            if image
            else read_text(getattr(args, name), DEFAULT_FONT)
        )

    q, p = pixels("question"), pixels("page")
    spans = infer(load(args.device), q, p, args.mode)
    args.output.mkdir(parents=True, exist_ok=False)
    n = write([p[i:j] for i, j in spans], args.output)
    dump(
        args.output / "result.json",
        dict(spans=spans, cells=n, parts=len(spans), termination="end of source scan"),
    )
    print(args.output / "answer.png")


if __name__ == "__main__":
    main()
