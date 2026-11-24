"""Measure fixed-length greedy continuation and correct-history prediction on real pages."""

import numpy as np
import torch
from ilm.visual_lm.visual_gap import continue_words
from ilm.visual_lm.causal_glyph_writer import write_strip
from scripts.answer_dialogue_surface import load
from scripts.prepare_context_cloze import MODEL, PUB, reserved
from scripts.prepare_dialogue_ink import dump
from scripts.evaluate_word_canvas import write_bound
from scripts.train_dialogue_surface import representation, corpus
from scripts.train_span_choice import gpu_owner


@torch.inference_mode()
def measure():
    torch.set_num_threads(2)
    data = reserved()
    keys = sorted(data["pages"])
    result = {}
    for mode in ("ink", "byte"):
        net = load(MODEL / mode / "passage", "cuda")
        face, unit, sep = representation(mode)
        ds = corpus(data, "fresh", net, face, unit)
        target_ids = ds.page[:, 32:40].long()
        truth = net.lookup(ds.bank)[target_ids]
        f = ds.bank.features[ds.page[:, 30:32].long(), :50]
        own = continue_words(net, f, 8)
        # Fixed-prefix exchange across distinct held pages.
        swapped = continue_words(net, f.roll(1, 0), 8)
        supplied = torch.stack(
            [
                continue_words(
                    net, ds.bank.features[ds.page[:, i - 2 : i].long(), :50], 1
                )[:, 0]
                for i in range(32, 40)
            ],
            1,
        )
        results = {}
        out = []
        for label, drawn in (
            ("free", own),
            ("swapped", swapped),
            ("correct_history", supplied),
        ):
            right = drawn == truth
            results[label] = dict(
                correct_at_position=right.sum(0).tolist(),
                correct=int(right.sum()),
                exact_streams=int(right.all(1).sum()),
                unique_streams=len(set(map(tuple, drawn.tolist()))),
            )
        for b, k in enumerate(keys):
            out.append(
                [
                    k,
                    truth[b].tolist(),
                    own[b].tolist(),
                    swapped[b].tolist(),
                    supplied[b].tolist(),
                ]
            )
        result[mode] = dict(
            stories=len(keys),
            words=int(truth.numel()),
            stored=int((truth >= 0).sum()),
            scores=results,
            outputs=write_bound(PUB / "responses", mode + "-continuations", out),
        )
        if mode == "ink":
            raw = sep.join(net.prototype_words[i] for i in own[0].tolist())
            write_strip(
                np.frombuffer(raw, np.uint8).reshape(-1, unit), PUB / "continuation.png"
            )
            left = data["pages"][keys[0]].split()[30:32]
            write_strip(
                np.frombuffer(sep.join(face(w) for w in left), np.uint8).reshape(
                    -1, unit
                ),
                PUB / "prefix.png",
            )
            alpha = {face(chr(i)): chr(i) for i in range(32, 127)}
            result["example"] = dict(
                prefix=left,
                output="".join(
                    alpha.get(raw[i : i + unit], " ") for i in range(0, len(raw), unit)
                ),
                reference=data["pages"][keys[0]].split()[32:40],
            )
    dump(PUB / "continuations.json", result)
    print(result)


if __name__ == "__main__":
    with gpu_owner():
        measure()
