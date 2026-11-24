"""Write one missing word image from a pictured prefix and optional suffix."""

import argparse
from pathlib import Path
import numpy as np
from ilm.visual_lm.causal_glyph_writer import write_strip
from ilm.visual_lm.portable_writer import read_image
from ilm.visual_lm.context_cloze import ContextCloze
from ilm.visual_lm.visual_gap import fill, continue_words
from ilm.visual_lm.optical_passage_qa import word_runs
from scripts.answer_dialogue_surface import load
from scripts.prepare_context_cloze import MODEL

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL / "ink/passage")
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--words",
        type=int,
        default=1,
        help="Write a fixed length after a prefix of at least two words",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    net = load(args.model, args.device)
    if not isinstance(net, ContextCloze) or net.mode != "ink":
        parser.error("Choose an image context-cloze checkpoint")
    left = read_image(args.left, None)
    if args.words < 1 or (args.words > 1 and args.right):
        parser.error("Multiple words require a prefix without a suffix")
    if args.words == 1:
        raw = fill(net, left, read_image(args.right, None) if args.right else None)
    else:
        before = [left[a:b] for a, b in word_runs(left)][-2:]
        if len(before) != 2:
            parser.error("Supply at least two pictured prefix words")
        prefix = net.encode_words(net.reader, None, before)[None, :, :50]
        ids = continue_words(net, prefix, args.words)[0].tolist()
        raw = bytes(128).join(net.prototype_words[i] for i in ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_strip(np.frombuffer(raw, np.uint8).reshape(-1, 128), args.output)
