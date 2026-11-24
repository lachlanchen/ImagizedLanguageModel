"""Answer a pictured English question by selecting word images and spelling new words."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

from ilm.visual_lm.dialogue_ink import DialogueInk, WordCodeDialogueInk
from ilm.visual_lm.optical_passage_qa import word_runs
from ilm.visual_lm.causal_glyph_writer import write_strip
from ilm.visual_lm.portable_writer import DEFAULT_FONT, read_image, read_text
from scripts.answer_optical_passage_qa import optical_words, PARENT, sha
from scripts.dialogue_ink_images import WordImages
from scripts.answer_optical_passage_qa import ROOT

RELEASE = ROOT / "models/dialogue_ink"


def load(folder, device):
    file = folder / "writer.safetensors"
    meta = json.loads((folder / "weights.json").read_text())
    if meta["sha256"] != sha(file) or meta["optical_encoder"] != sha(PARENT):
        raise ValueError("Released image weights have changed")
    state = load_file(str(file))
    classes = {
        "dialogue_ink_v1": DialogueInk,
        "word_code_dialogue_ink_v1": WordCodeDialogueInk,
    }
    net = classes[meta["architecture"]](state["palette"]).to(device).eval()
    net.load_state_dict(state)
    return net, optical_words(device)


@torch.inference_mode()
def generate(
    net, optical, bank, inputs, variants, *, separator=bytes(128), feedback=None
):
    feedback = net.reader.surface if feedback is None else feedback
    mem, _ = net.memory(*inputs)
    initial = mem[-1]
    n = len(initial)
    state = initial[None]
    prev = initial.new_zeros(n, 1, 50)
    done = torch.zeros(n, dtype=torch.bool, device=initial.device)
    complete = torch.ones_like(done)
    results = [[] for _ in range(n)]
    paths = [[] for _ in range(n)]
    raw_palette = net.palette.cpu().numpy()
    for t in range(net.max_words + 1):
        ch, ptr, z, state = net.step(mem, prev, state)
        route = ch[:, 0].argmax(1).masked_fill(done, 0)
        done = done | (route == 0)
        if t == net.max_words or bool(done.all()):
            break
        pointer = ptr[:, 0].masked_fill(variants == 0, -1e4).flatten(1).argmax(1)
        match = variants.flatten(1).gather(1, pointer[:, None]).long()
        prev = bank.features[match, :50].clone()
        copy = (route == 1).nonzero().flatten()
        for j, k in zip(copy.tolist(), match[copy, 0].tolist()):
            results[j].append(bank.raw[k])
            paths[j].append("copy")
        writer = (route == 2).nonzero().flatten()
        if len(writer):
            glyphs, ended = net.write_words(z[writer, 0])
            complete[writer] &= ended
            packed = net.palette[glyphs]
            length = (glyphs != 0).sum(1)
            prev[writer, 0] = feedback(packed, length, optical)[:, :50]
            for j, g in zip(writer.tolist(), glyphs.tolist()):
                results[j].append(raw_palette[[k for k in g if k]].tobytes())
                paths[j].append("spell")
    return [separator.join(x) for x in results], (done & complete).tolist(), paths


@torch.inference_mode()
def answer(net, optical, page, question, history):
    sets = [
        [raw[a:b].tobytes() for a, b in word_runs(raw)]
        for raw in (page, question, history)
    ]
    if not 1 <= len(sets[0]) <= 320 or not 1 <= len(sets[1]) <= 32 or len(sets[2]) > 48:
        raise ValueError(
            "Expected 1–320 passage words, 1–32 question words and at most 48 history words"
        )
    bank = WordImages(set(w for s in sets for w in s), net.reader, optical)
    inputs = []
    indices = []
    for words in sets:
        ix = torch.tensor([[bank.ids[w] for w in words] or [0]], device=bank.device)
        indices.append(ix)
        inputs.extend(
            [bank.features[ix], torch.tensor([len(words)], device=bank.device)]
        )
    images, stopped, mode = generate(
        net, optical, bank, inputs, bank.edits[torch.cat(indices, 1)]
    )
    return np.frombuffer(images[0], np.uint8).reshape(-1, 128), dict(
        stopped=stopped[0], actions=mode[0]
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("passage", "question", "history"):
        g = p.add_mutually_exclusive_group(required=name != "history")
        g.add_argument("--" + name, type=Path)
        g.add_argument("--" + name + "-text")
    p.add_argument("--model", type=Path, default=RELEASE)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)

    def pixels(name):
        f = getattr(a, name)
        text = getattr(a, name + "_text")
        if f:
            return read_image(f, None)
        return (
            read_text(" ".join(text.split()), DEFAULT_FONT)
            if text
            else np.empty((0, 128), np.uint8)
        )

    result, decision = answer(
        *load(a.model, "cpu"), *(pixels(k) for k in ("passage", "question", "history"))
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_strip(result, a.output)
    print(json.dumps(decision))
