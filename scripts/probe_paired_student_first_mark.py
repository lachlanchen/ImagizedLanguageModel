"""Bound the probability of the first emitted image across overlapping routes."""

import numpy as np
import torch
from ilm.visual_lm.causal_glyph_translator import COPY, DRAW, END, EXEMPLAR
from ilm.visual_lm.idiom_ink import visual_input
from scripts.evaluate_paired_student import (
    EVAL,
    OUT,
    ROOT,
    load,
    read_streams,
    sha,
    write_new,
)
from scripts.train_paired_student import restore


def pooled_choice(copy, examples, identifiers, stop, draw):
    # The final bin is reserved for END. Open drawing can add mass to any ink.
    lower = np.bincount(identifiers, weights=np.r_[copy, examples])
    lower = np.r_[lower, stop]
    best = int(lower.argmax())
    upper = lower.copy()
    upper[:-1] += draw
    upper[best] = -1
    return best, bool(lower[best] > max(upper.max(), draw))


@torch.inference_mode()
def run():
    torch.set_num_threads(2)
    checkpoint = OUT / "teacher.safetensors"
    assert sha(checkpoint) == load(OUT / "fit.json")["arms"]["teacher"]["checkpoint"]
    model = restore(checkpoint).cpu().eval()
    library = np.packbits(model.exemplars.numpy(), axis=1)
    physical = [row.tobytes() for row in library]
    keys = model.image_keys()
    source, saved = read_streams("source"), read_streams("teacher")
    rc = load(OUT / "target_teacher/receipt.json")
    refs = {name: read_streams(name) for name in ("reference", "teacher_target")}
    details = []
    for i, row in enumerate(source):
        pics = torch.from_numpy(np.unpackbits(row, axis=1)).bool()[None]
        n = torch.tensor([len(row)])
        memory, sk, mask, hidden = model.read(visual_input(pics, model.landmarks), n)
        q, _ = model.history(model.start[None, None], hidden)
        state, attn, _ = model.plan_steps(q, memory, sk, mask, n, torch.tensor([-0.65]))
        state = state[:, 0]
        routes = model.route(state).double().softmax(-1)[0].numpy()
        loc = attn[0, 0].double().softmax(-1).numpy()
        exemplars = model.prototype_scores(state, keys).double().softmax(-1)[0].numpy()
        cells = [x.tobytes() for x in row] + physical
        unique = list(dict.fromkeys(cells))
        index = {cell: j for j, cell in enumerate(unique)}
        identifiers = np.fromiter((index[x] for x in cells), int)
        best, cert = pooled_choice(
            routes[COPY] * loc,
            routes[EXEMPLAR] * exemplars,
            identifiers,
            routes[END],
            routes[DRAW],
        )
        proposed = (unique + [None])[best]
        mode = routes.argmax()
        if mode == DRAW:
            bits = model.draw.choose(state)
            ink = model.motor(bits) > 0
            ink[:, model.landmarks] = bits.bool()
            original = np.packbits(ink.numpy()).tobytes()
        else:
            original = {
                COPY: cells[loc.argmax()],
                EXEMPLAR: physical[exemplars.argmax()],
                END: None,
            }[mode]
        actual = saved[i][0].tobytes() if len(saved[i]) else None
        emitted = proposed if cert else actual
        record = dict(
            row=i, certified=cert, replay=original == actual, changed=actual != emitted
        )
        for key, streams in refs.items():
            if key == "teacher_target" and not rc["complete_renderable"][i]:
                continue
            target = streams[i][0].tobytes()
            record[key] = [target == actual, target == emitted]
        details.append(record)
    comparisons = {}
    for key in refs:
        a = np.array([r[key] for r in details if key in r])
        comparisons[key] = dict(
            n=len(a),
            original=int(a[:, 0].sum()),
            pooled=int(a[:, 1].sum()),
            gains=int((~a[:, 0] & a[:, 1]).sum()),
            losses=int((a[:, 0] & ~a[:, 1]).sum()),
        )
    dest = ROOT / "references/evidence/paired_student_first_mark_20261119.json"
    write_new(
        dest,
        dict(
            checkpoint=sha(checkpoint),
            roster=sha(EVAL / "roster.json"),
            certified=sum(r["certified"] for r in details),
            certified_changes=sum(r["changed"] for r in details),
            replay_disagreements=sum(not r["replay"] for r in details),
            comparison=comparisons,
            rows=details,
        ),
    )
    print(comparisons)


if __name__ == "__main__":
    run()
