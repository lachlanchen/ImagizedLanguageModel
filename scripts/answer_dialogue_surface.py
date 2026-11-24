"""Answer using the matched cold visual or ASCII byte conversational checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from ilm.visual_lm.causal_glyph_writer import write_strip
from ilm.visual_lm.dialogue_surface import SurfaceDialogue
from ilm.visual_lm.optical_passage_qa import word_runs
from ilm.visual_lm.portable_writer import read_image, read_text, DEFAULT_FONT
from scripts.answer_dialogue_ink import generate
from scripts.dialogue_ink_images import WordImages


def load(folder, device="cpu"):
    meta = json.loads((folder / "weights.json").read_text())
    file = folder / "writer.safetensors"
    if (
        hashlib.sha256(file.read_bytes()).hexdigest()
        != meta["checkpoints"]["8000"]["sha256"]
    ):
        raise ValueError("Wrong surface checkpoint")
    state = load_file(str(file))
    net = SurfaceDialogue(state["palette"], meta["mode"]).to(device)
    net.load_state_dict(state)
    return net.eval()


@torch.inference_mode()
def answer(net, fields):
    if net.mode == "ink":
        words = [
            [value[a:b].tobytes() for a, b in word_runs(value)] for value in fields
        ]
        unit, sep = 128, bytes(128)
    else:
        words = [[w.encode("ascii") for w in value.split()] for value in fields]
        unit, sep = 1, b" "
    if (
        not 1 <= len(words[0]) <= 320
        or not 1 <= len(words[1]) <= 32
        or len(words[2]) > 48
    ):
        raise ValueError(
            "Expected a passage of 1–320 words, a question of 1–32 and history of at most 48"
        )
    bank = WordImages(
        {v for field in words for v in field}, net.reader, None, unit, net.encode_words
    )
    inputs = []
    indices = []
    for field in words:
        ix = torch.tensor([[bank.ids[w] for w in field] or [0]], device=bank.device)
        inputs.extend(
            (bank.features[ix], torch.tensor([len(field)], device=bank.device))
        )
        indices.append(ix)
    output, stopped, actions = generate(
        net,
        None,
        bank,
        inputs,
        bank.edits[torch.cat(indices, 1)],
        separator=sep,
        feedback=net.surface,
    )
    return output[0], dict(stopped=stopped[0], actions=actions[0])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models/dialogue_surfaces/ink",
    )
    for name in ("passage", "question", "history"):
        g = p.add_mutually_exclusive_group(required=name != "history")
        g.add_argument("--" + name, type=Path)
        g.add_argument("--" + name + "-text")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    net = load(a.model)
    fields = []
    for name in ("passage", "question", "history"):
        path = getattr(a, name)
        text = getattr(a, name + "_text") or ""
        if net.mode == "ink":
            fields.append(
                read_image(path, None)
                if path
                else read_text(" ".join(text.split()), DEFAULT_FONT)
                if text
                else np.empty((0, 128), np.uint8)
            )
        elif path:
            p.error(
                "The byte baseline accepts --passage-text, --question-text and --history-text"
            )
        else:
            fields.append(text)
    raw, result = answer(net, fields)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if net.mode == "ink":
        write_strip(np.frombuffer(raw, np.uint8).reshape(-1, 128), a.output)
    else:
        a.output.write_bytes(raw)
    print(json.dumps(result))
