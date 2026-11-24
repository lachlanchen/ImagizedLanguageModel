"""Verify hidden-centre independence and equivalence of individual and batched image paths."""

import numpy as np
import torch
from ilm.visual_lm.visual_gap import fill
from scripts.prepare_context_cloze import MODEL, PUB, reserved
from scripts.prepare_dialogue_ink import dump
from scripts.train_context_cloze import page_state
from scripts.train_dialogue_surface import corpus, representation
from scripts.train_span_choice import gpu_owner
from scripts.answer_dialogue_surface import load


@torch.inference_mode()
def check():
    torch.set_num_threads(2)
    data = reserved()
    face, unit, separator = representation("ink")
    net = load(MODEL / "ink/passage", "cuda")
    ds = corpus(data, "fresh", net, face, unit)
    pages = [data["pages"][k].split() for k in sorted(data["pages"])]

    def pack(words):
        return np.frombuffer(separator.join(face(w) for w in words), np.uint8).reshape(
            -1, unit
        )

    checks = 0
    for b, words in enumerate(pages[:8]):
        for t in (0, 1, 9, len(words) - 1):
            pos = torch.tensor([[b, t]], device=ds.device)
            expected = int(net.retrieve_logits(page_state(net, ds, pos)).argmax())
            raw = fill(net, pack(words[max(t - 2, 0) : t]), pack(words[t + 1 : t + 3]))
            if raw != net.prototype_words[expected]:
                raise AssertionError("Packed PNG strip and batched page disagree")
            checks += 1
    # Changing the physical input in a hidden QA slot cannot change any output logit.
    x, _ = ds.inputs(np.arange(4))
    mem, _ = net.memory(*x)
    f = ds.bank.features[ds.q[:4, :3].long(), :50].clone()
    visible = torch.ones(4, 3, dtype=torch.bool, device=ds.device)
    visible[:, 1] = False
    lengths = torch.full((4,), 3, device=ds.device, dtype=torch.long)

    def outputs(feats):
        route, ptr, z = net.paint(mem, feats, visible, lengths)
        return route, ptr, net.retrieve_logits(z), net.gate_logits(z)

    a = outputs(f)
    f[:, 1] += 100
    diff = max(float((p - q).abs().max()) for p, q in zip(a, outputs(f), strict=True))
    if diff != 0:
        raise AssertionError("Hidden image affects predictions")
    dump(
        PUB / "image_paths.json",
        dict(
            held_pages=8,
            word_image_decisions=checks,
            max_hidden_input_logit_change=diff,
        ),
    )
    print(checks, diff)


if __name__ == "__main__":
    with gpu_owner():
        check()
