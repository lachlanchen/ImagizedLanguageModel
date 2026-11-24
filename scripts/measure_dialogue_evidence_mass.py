"""Check the evidence-enrichment theorem on a frozen reader's actual copy probabilities."""

from collections import Counter
import json
import numpy as np
import torch
from scripts.answer_dialogue_surface import load
from scripts.train_dialogue_surface import ROOT_MODEL, representation, corpus
from scripts.dialogue_ink_data import read_data, pack
from scripts.dialogue_ink_images import WordImages
from scripts.dialogue_teacher_stories import new_stories
from scripts.prepare_dialogue_ink import printable
from scripts.train_dialogue_evidence import evidence_mask
from scripts.qualify_dialogue_teacher import PUB, dump, sha


@torch.inference_mode()
def measure():
    data, _, _ = new_stories(read_data(), "long")
    net = load(ROOT_MODEL / "ink", "cpu")
    face, unit, _ = representation("ink")
    ds = corpus(data, "fresh", net, face, unit)
    gold = [
        [face(w) for w in r[5].split()[:16]] if printable(r[5]) else [] for r in ds.rows
    ]
    gb = WordImages(
        {w for ws in gold for w in ws}, net.reader, None, unit, net.encode_words
    )
    counts = Counter()
    prob = Counter()
    max_delta = 0.0
    for off in range(0, len(ds.rows), 32):
        idx = np.arange(off, min(off + 32, len(ds.rows)))
        ws = gold[off : off + len(idx)]
        previous, _ = pack([[0] + [gb.ids[w] for w in words] for words in ws], "cpu")
        target, _ = pack(
            [[ds.bank.ids.get(w, -1) for w in words] for words in ws], "cpu"
        )
        if not target.size(1):
            continue
        inputs, var = ds.inputs(idx)
        mem, _ = net.memory(*inputs)
        _, ptr, _, _ = net.step(mem, gb.features[previous.long(), :50], mem[-1][None])
        scores = ptr[:, : target.size(1)].flatten(2)
        candidates = var.flatten(1)
        scores = scores.masked_fill(candidates[:, None] == 0, -torch.inf).double()
        distribution = scores.softmax(-1)
        matches = (candidates[:, None] == target[:, :, None]) & (target[:, :, None] > 0)
        mask = evidence_mask(ds.ev[idx], scores.size(2))[:, None]
        keep = matches.any(-1) & (ds.ev[idx, 0] >= 0)[:, None]
        counts["within_word_limit"] += sum(map(len, ws))
        counts["reference_has_evidence_and_copy"] += int(keep.sum())
        a = (distribution * (matches & mask)).sum(-1)[keep]
        b = (distribution * (~matches & mask)).sum(-1)[keep]
        c = (distribution * (matches & ~mask)).sum(-1)[keep]
        d = (distribution * (~matches & ~mask)).sum(-1)[keep]
        base = a + c
        change = scores + mask.double() * float(np.log(4.0))
        boosted = (change.softmax(-1) * matches).sum(-1)[keep]
        # Integer boost exactly four; double precision tolerance is expressed on probabilities.
        algebra = (3 * (a * d - b * c)) / (4 * (a + b) + c + d)
        delta = boosted - base
        max_delta = max(
            max_delta, float((algebra - delta).abs().max()) if delta.numel() else 0
        )
        counts["sign_disagreement_outside_roundoff"] += int(
            ((algebra * delta < 0) & (delta.abs() > 1e-10)).sum()
        )
        for key, value in [
            ("higher", delta > 1e-10),
            ("lower", delta < -1e-10),
            ("level", delta.abs() <= 1e-10),
            ("no_matching_word_inside", ~(matches & mask).any(-1)[keep]),
        ]:
            counts[key] += int(value.sum())
        pos = scores.argmax(-1)
        zoom = change.argmax(-1)
        before = matches.gather(-1, pos[:, :, None]).squeeze(-1)[keep]
        after = matches.gather(-1, zoom[:, :, None]).squeeze(-1)[keep]
        for key, value in [
            ("location_correct", before),
            ("boosted_location_correct", after),
            ("fixed", after & ~before),
            ("lost", before & ~after),
        ]:
            counts[key] += int(value.sum())
        prob["original"] += float(base.sum())
        prob["quadrupled"] += float(boosted.sum())
    output = dict(
        mode="cpu_float64_after_pointer_logits",
        counts=counts,
        mean_probability={
            k: v / counts["reference_has_evidence_and_copy"] for k, v in prob.items()
        },
        max_probability_identity_error=max_delta,
        roundoff_cutoff=1e-10,
        weights=sha(ROOT_MODEL / "ink/writer.safetensors"),
    )
    if max_delta > 1e-9 or counts["sign_disagreement_outside_roundoff"]:
        raise ValueError("Evidence theorem failed numerical check")
    dump(PUB / "mass.json", output)
    print(json.dumps(output))


if __name__ == "__main__":
    torch.set_num_threads(2)
    measure()
