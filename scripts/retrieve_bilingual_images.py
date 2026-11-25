"""Rank pictured English or Chinese sentences by their proximity to a pictured query."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file

from ilm.visual_lm.bilingual_alignment import BilingualTraining, read_sentence
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.portable_writer import read_image
from scripts.prepare_bilingual_ink import MODEL, WORK, read, sha
from scripts.train_ordered_translation import pack
from scripts.write_ordered_translation import restore


def load_reader(device):
    writer = restore("bilingual-aligned", device)
    bundle = MODEL / "aligned"
    weights = bundle / "alignment-projection.safetensors"
    if not weights.is_file():
        weights = WORK / "alignment-projection.safetensors"
    if sha(weights) != read(bundle / "fit.json")["training_projection"]:
        raise ValueError("Alignment weights differ from the training receipt")
    reader = BilingualTraining(writer).to(device).eval()
    reader.project.load_state_dict(load_file(str(weights)))
    return reader


@torch.inference_mode()
def embed(reader, batches, device):
    # Handles are local pixel addresses. Blank ink has a real address distinct from padding.
    vectors = []
    for i in range(0, len(batches), 32):
        local = batches[i : i + 32]
        if any(not 1 <= len(cells) <= 96 for cells in local):
            raise ValueError("Expected 1–96 occupied cells per image")
        ink = [bytes(128)]
        address = {}
        seq = []
        for cells in local:
            ids = []
            for cell in cells:
                bitmap = bytes(cell)
                if bitmap not in address:
                    address[bitmap] = len(ink)
                    ink.append(bitmap)
                ids.append(address[bitmap])
            seq.append(ids)
        tiles = np.frombuffer(b"".join(ink), np.uint8).reshape(-1, 128)
        pixels = torch.tensor(observations({"tiles": tiles}, "ink"), device=device)
        bank = reader.writer.optical(pixels)
        _, _, hidden = read_sentence(reader.writer, bank, pack(seq, device))
        vectors.append(F.normalize(reader.project(hidden).float(), dim=1).cpu())
    return torch.cat(vectors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument(
        "--gallery",
        type=Path,
        required=True,
        help="JSON list with relative image path and occupied cells for each sentence",
    )
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    rows = read(args.gallery)
    if not rows or args.top < 1:
        parser.error("Supply a nonempty gallery and positive --top")
    torch.set_num_threads(2)
    model = load_reader(args.device)
    q = embed(model, [read_image(args.image, args.cells)], args.device)
    chunks = []
    for i in range(0, len(rows), 32):
        candidates = [
            read_image(args.gallery.parent / r["image"], r["cells"])
            for r in rows[i : i + 32]
        ]
        chunks.extend((embed(model, candidates, args.device) @ q[0]).tolist())
    order = sorted(range(len(rows)), key=chunks.__getitem__, reverse=True)[: args.top]
    print(
        json.dumps(
            [dict(image=rows[i]["image"], cosine=chunks[i]) for i in order],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
