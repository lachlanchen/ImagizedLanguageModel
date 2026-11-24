"""Write a list of source glyphs from two pictured streams with the released reader."""

import argparse
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.fact_binding import BindingHead, decode, fixed, frozen_features
from ilm.visual_lm.portable_writer import read_image
from scripts.prepare_complete_fact_sets import MODEL as BASE
from scripts.prepare_fact_binding import MODEL, dump, read, sha
from scripts.train_fact_binding import tensors
from scripts.write_multiple_ink_runs import load_reader, write
from scripts.write_visual_discourse import encode_pages


def load(arm, device):
    reader = load_reader(device, True, BASE / "contrast")
    if arm == "parent":
        return reader, None
    path = MODEL / arm / "head.safetensors"
    rec = read(path.with_name("fit.json"))
    if rec["weights"] != sha(path) or rec["bindings"]["parent"] != sha(
        BASE / "contrast/survival.safetensors"
    ):
        raise ValueError("Reader or head changed")
    head = BindingHead().to(device)
    head.load_state_dict(load_file(str(path), device=device))
    return reader, head.eval()


@torch.inference_mode()
def read_batch(reader, head, bank, rows):
    q, p = tensors(rows, bank.device)
    feat, role, count = frozen_features(reader, bank, q, p)
    h = head(feat) if head is not None else role.new_zeros(*role.shape[:2], 2)
    return tuple(fixed(x).cpu() for x in (role, count, h))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("question", "page"):
        parser.add_argument("--" + name + "-image", type=Path, required=True)
        parser.add_argument("--" + name + "-cells", type=int, required=True)
    parser.add_argument(
        "--arm", choices=("parent", "linear", "mixture"), default="mixture"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.question_cells <= 160 or not 1 <= args.page_cells <= 1200:
        parser.error("Expected 1–160 question and 1–1,200 page cells")
    q = read_image(args.question_image, args.question_cells)
    p = read_image(args.page_image, args.page_cells)
    tiles, (qi, pi) = encode_pages(np.zeros((2, 128), np.uint8), [q, p])
    torch.set_num_threads(3)
    with torch.inference_mode():
        reader, head = load(args.arm, "cpu")
        bank = reader.ink(torch.from_numpy(observations({"tiles": tiles}, "ink")))
        s, c, h = read_batch(reader, head, bank, [dict(qi=qi, pi=pi)])
        spans, value = decode(
            s[0].numpy(), c[0].numpy(), h[0].numpy(), args.arm == "mixture"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    n = write([p[a:b] for a, b in spans], args.output)
    dump(args.output / "result.json", dict(spans=spans, cells=n, path_utility=value))
    print(args.output / "answer.png")


if __name__ == "__main__":
    main()
