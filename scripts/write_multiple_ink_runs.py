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
def forward(model, bank, qs, ps, survival=False):
    device = bank.device
    q, p = [
        torch.from_numpy(pad(v)).to(device=device, dtype=torch.long) for v in (qs, ps)
    ]
    roles, counts = model(bank, q, p)
    if survival:
        from ilm.visual_lm.run_survival import utilities

        roles = utilities(roles, p >= 2)
    return roles.cpu().numpy(), counts.cpu().numpy()


def write(parts, dest):
    dest.mkdir(parents=True, exist_ok=True)
    pixels = joined(parts, read_text(" | ", DEFAULT_FONT))
    draw(dest, "answer", pixels)
    (dest / "answer.bin").write_bytes(pixels.tobytes())
    return len(pixels)


@torch.inference_mode()
def infer(model, q, p, mode="count", survival=False):
    if not 1 <= len(q) <= 160 or not 1 <= len(p) <= 1200:
        raise ValueError("Expected 1–160 question and 1–1,200 page cells")
    tiles, (qi, pi) = encode_pages(np.zeros((2, 128), np.uint8), [q, p])
    features = torch.from_numpy(observations({"tiles": tiles}, "ink")).to(
        next(model.parameters()).device
    )
    role, count = forward(model, model.ink(features), [qi], [pi], survival)
    return intervals(role[0], count[0], mode)


def joined(parts, separator):
    cells = []
    for part in parts:
        if cells:
            cells.append(separator)
        cells.append(part)
    return np.concatenate(cells) if cells else np.empty((0, 128), np.uint8)


@torch.inference_mode()
def sentences(writer, head, questions, lists, conjunction):
    from scripts.write_visual_statements import realize

    pages = []
    for q, parts in zip(questions, lists, strict=True):
        pages.append(q)
        pages.extend(parts)
    palette, sequences = encode_pages(writer.palette, [conjunction, *pages])
    sep, *sequences = sequences
    cursor = iter(sequences)
    qs, facts = [], []
    for parts in lists:
        qs.append(next(cursor))
        fact = []
        for i, _ in enumerate(parts):
            if i:
                fact.extend(sep)
            fact.extend(next(cursor))
        facts.append(fact)
    features = torch.from_numpy(observations({"tiles": palette}, "ink")).to(
        writer.device
    )
    answers = realize(
        writer.model, head, features, len(writer.palette), qs, facts, sep[0]
    )
    return [(palette[np.asarray(ids, dtype=int)], stopped) for ids, stopped in answers]


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
    ap.add_argument("--survival", action="store_true")
    ap.add_argument(
        "--sentence",
        action="store_true",
        help="Also run the English statement writer on the selected ink",
    )
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
    spans = infer(load(args.device, args.survival), q, p, args.mode, args.survival)
    args.output.mkdir(parents=True, exist_ok=False)
    parts = [p[i:j] for i, j in spans]
    n = write(parts, args.output)
    report = dict(
        spans=spans, cells=n, parts=len(spans), termination="end of source scan"
    )
    if args.sentence:
        from scripts.write_visual_discourse import Writer
        from scripts.write_visual_statements import editor

        pixels, stopped = sentences(
            Writer(device=args.device),
            editor(args.device),
            [q],
            [parts],
            read_text(" and ", DEFAULT_FONT),
        )[0]
        draw(args.output, "sentence", pixels)
        (args.output / "sentence.bin").write_bytes(pixels.tobytes())
        report.update(sentence_cells=len(pixels), sentence_stopped=stopped)
    dump(args.output / "result.json", report)
    print(args.output / "answer.png")


if __name__ == "__main__":
    main()
