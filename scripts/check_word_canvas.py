"""Check that hidden target ink cannot affect logits and that PNG inference reproduces the report."""

import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from scripts.answer_dialogue_surface import load, answer
from scripts.prepare_word_canvas import MODEL, PUB, reserved
from scripts.train_dialogue_surface import representation, corpus
from scripts.evaluate_optical_passage_qa import save_page
from scripts.train_span_choice import gpu_owner
from scripts.prepare_dialogue_ink import dump
from ilm.visual_lm.portable_writer import read_image


@torch.inference_mode()
def check(device):
    data = reserved()
    sample = json.loads((PUB / "examples.json").read_text())["draft"]
    i = sample["index"]
    row = data["rows"][i]
    face, unit, _ = representation("ink")
    net = load(MODEL / "ink/masked", device)
    small = dict(rows=[row], pages={row[1]: data["pages"][row[1]]})
    ds = corpus(small, "fresh", net, face, unit)
    fields, variants = ds.inputs(np.array([0]))
    memory, _ = net.memory(*fields)
    length = torch.tensor([4], device=device)
    features = torch.arange(200, device=device).reshape(1, 4, 50).float() / 100
    visible = torch.tensor([[True, False, False, True]], device=device)
    scores = net.paint(memory, features, visible, length)
    hidden_change = net.paint(
        memory, features.masked_fill(~visible[:, :, None], 27.0), visible, length
    )
    hidden_drift = max(
        float((a - b).abs().max()) for a, b in zip(scores, hidden_change, strict=True)
    )
    altered = features.clone()
    altered[:, 3] = 0
    observed = net.paint(memory, altered, visible, length)
    right_effect = float((scores[1][:, 1] - observed[1][:, 1]).abs().max())
    if hidden_drift != 0 or right_effect <= 1e-5:
        raise ValueError("Mask or right context check failed")
    with tempfile.TemporaryDirectory() as tmp:
        fields = []
        for k, s in enumerate((data["pages"][row[1]], row[3], row[4])):
            if s:
                p = Path(tmp) / f"{k}.png"
                save_page(s, p, face)
                fields.append(read_image(p, None))
            else:
                fields.append(np.empty((0, 128), np.uint8))
        raw, meta = answer(net, fields)
        if raw != face(sample["generated"]) or meta != sample["state"]:
            raise ValueError("Single PNG answer changed")
    return dict(
        checked_answer=sample["generated"],
        commit_rounds=meta["commit_rounds"],
        hidden_target_max_difference=hidden_drift,
        right_context_pointer_max_difference=right_effect,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    with gpu_owner():
        report = {d: check(d) for d in ("cpu", "cuda")}
    if args.output:
        dump(args.output, report)
    print(report)
