"""Read an English passage and question from two 32-pixel-cell images."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

from ilm.visual_lm.visual_action_meaning import OpticalWordMeaning
from ilm.visual_lm.optical_passage_qa import OpticalPassageQA, word_runs
from ilm.visual_lm.portable_writer import DEFAULT_FONT, read_image, read_text
from ilm.visual_lm.causal_glyph_writer import write_strip
from scripts.read_optical_role import digest as sha

ROOT = Path(__file__).resolve().parents[1]

RELEASE = ROOT / "models/optical_passage_qa"
PARENT = ROOT / "models/visual_action_meaning/aligned/lookup.safetensors"


def optical_words(device):
    if (
        sha(PARENT)
        != json.loads((PARENT.parent / "weights.json").read_text())["sha256"]
    ):
        raise ValueError("Changed optical encoder weights")
    values = load_file(str(PARENT))
    model = OpticalWordMeaning().to(device)
    model.load_state_dict(
        {k[6:]: v for k, v in values.items() if k.startswith("words.")}
    )
    return model.eval().requires_grad_(False)


def load(device="cpu", folder=RELEASE):
    net = OpticalPassageQA().to(device)
    meta = json.loads((folder / "weights.json").read_text())
    file = folder / "reader.safetensors"
    if sha(file) != meta["sha256"] or sha(PARENT) != meta["optical_parent"]:
        raise ValueError("Reader weight checksum mismatch")
    net.load_state_dict(load_file(str(file)))
    return net.eval(), optical_words(device)


@torch.no_grad()
def visual_words(net, encoder, runs):
    device = net.sketch.device
    lengths = torch.tensor([len(run) for run in runs], device=device)
    packed = np.zeros((len(runs), int(lengths.max()), 128), np.uint8)
    for j, run in enumerate(runs):
        packed[j, : len(run)] = run
    return net.surface(torch.from_numpy(packed).to(device), lengths, encoder)


@torch.inference_mode()
def answer(net, encoder, passage, question):
    p, q = word_runs(passage), word_runs(question)
    if not 1 <= len(p) <= 192 or not 1 <= len(q) <= 32:
        raise ValueError("Expected 1–192 passage words and 1–32 question words")
    ps = torch.tensor([len(p)], device=net.sketch.device)
    qs = torch.tensor([len(q)], device=net.sketch.device)

    def embed(raw, spans):
        return visual_words(net, encoder, [raw[s:e] for s, e in spans])[None]

    scores, trim = net(embed(passage, p), ps, embed(question, q), qs)
    widths = torch.from_numpy(p[:, 1] - p[:, 0]).to(ps.device)[None]
    l, r, dl, dr = net.select(scores, trim, ps, widths)[0].tolist()
    first, end = int(p[l, 0] + dl), int(p[r, 1] - dr)
    return passage[first:end].copy(), (first, end)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("passage", "question"):
        inputs = parser.add_mutually_exclusive_group(required=True)
        inputs.add_argument("--" + name, type=Path)
        inputs.add_argument("--" + name + "-text")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)

    def pixels(path, text):
        return (
            read_image(path, None)
            if path
            else read_text(" ".join(text.split()), DEFAULT_FONT)
        )

    result, bounds = answer(
        *load(),
        pixels(args.passage, args.passage_text),
        pixels(args.question, args.question_text),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_strip(result, args.output)
    print(json.dumps(dict(start_cell=bounds[0], end_exclusive=bounds[1])))
