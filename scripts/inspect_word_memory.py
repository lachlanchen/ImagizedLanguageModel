"""Replay actual answer PNGs and locate the failure on stored but absent words."""

import gzip
import json
from collections import Counter
import numpy as np
import torch
from ilm.visual_lm.causal_glyph_writer import write_strip
from ilm.visual_lm.portable_writer import read_image
from scripts.answer_dialogue_surface import answer, load
from scripts.dialogue_ink_images import WordImages
from scripts.evaluate_optical_passage_qa import metrics
from scripts.prepare_compose_dialogue import kind
from scripts.prepare_dialogue_ink import sha, dump
from scripts.prepare_word_memory import MODEL, PUB, WORK, reserved
from scripts.train_dialogue_surface import representation, corpus
from scripts.train_span_choice import gpu_owner


@torch.inference_mode()
def inspect():
    torch.set_num_threads(2)
    data = reserved()
    rows = data["rows"]
    study = json.loads((PUB / "results.json").read_text())
    net = load(MODEL / "ink", "cuda")
    face, unit, sep = representation("ink")
    ds = corpus(data, "fresh", net, face, unit)
    rec = {}
    for arm in ("memory", "disabled", "parent"):
        f = PUB / "responses" / ("ink-" + arm + ".json.gz")
        if sha(f) != study["models"]["ink-" + arm]["outputs"]:
            raise ValueError("Unbound evaluation")
        rec[arm] = json.loads(gzip.decompress(f.read_bytes()))
    f = PUB / "responses/ink-memory-slots.json.gz"
    if sha(f) != study["models"]["ink-memory"]["slots"]:
        raise ValueError("Unbound conditional results")
    slots = json.loads(gzip.decompress(f.read_bytes()))
    wanted = [(i, t) for i, t, k, _ in slots if k == "absent_stored"]
    freq = Counter()
    stats = []
    for i, t in wanted:
        r = rows[i]
        raw = [face(w) for w in r[5].split()]
        ix = net.prototype_ids[raw[t]]
        visible = WordImages(raw, net.reader, None, unit, net.encode_words)
        ids = torch.tensor([[visible.ids[w] for w in raw]], device="cuda")
        seen = torch.ones_like(ids, dtype=torch.bool)
        seen[0, t] = False
        x, _ = ds.inputs(np.array([i]))
        m, _ = net.memory(*x)
        _, _, z = net.paint(
            m, visible.features[ids, :50], seen, ids.new_tensor([len(raw)])
        )
        logits = net.retrieve_logits(z[0, t])
        rank = 1 + int((logits > logits[ix]).sum())
        stats.append(
            dict(
                row=i,
                slot=t,
                rank=rank,
                gate=float(net.gate(z[0, t]).sigmoid()),
                target_probability_given_retrieval=float(logits.softmax(-1)[ix]),
            )
        )
        freq.update(top1=rank == 1, top8=rank <= 8, top32=rank <= 32)

    def correct(trial, r):
        return trial["stop"] and bool(metrics(trial["answer"], [r[5]])[0])

    examples = {}
    for i, r in enumerate(rows):
        own = rec["memory"][i]
        ans = own["original"]
        different = ans != rec["disabled"][i]["original"]
        eligible = [
            (
                "retrieval-change",
                different
                and correct(ans, r)
                and not correct(rec["disabled"][i]["original"], r),
            ),
            (
                "grounded",
                len(ans["answer"].split()) >= 2
                and correct(ans, r)
                and not any(correct(own[c], r) for c in ("question", "passage")),
            ),
            (
                "miss",
                kind(r[5], (data["pages"][r[1]], r[3], r[4])) != "ordinary"
                and not correct(ans, r),
            ),
        ]
        for name, ok in eligible:
            if not ok or name in examples:
                continue
            fields = []
            for j, s in enumerate((data["pages"][r[1]], r[3], r[4])):
                raw = sep.join(face(w) for w in s.split())
                cells = np.frombuffer(raw, np.uint8).reshape(-1, unit)
                if raw:
                    png = WORK / f"{name}-{j}.png"
                    write_strip(cells, png)
                    cells = read_image(png, None)
                fields.append(cells)
            output, info = answer(net, fields)
            expected = sep.join(face(w) for w in ans["answer"].split())
            if output != expected or info["stopped"] != ans["stop"]:
                raise ValueError("Standalone PNG replay mismatch")
            f = PUB / (name + ".png")
            write_strip(np.frombuffer(output, np.uint8).reshape(-1, unit), f)
            examples[name] = dict(
                row=i,
                question=r[3],
                reference=r[5],
                generated=ans["answer"],
                disabled=rec["disabled"][i]["original"]["answer"],
                after_question_exchange=own["question"]["answer"],
                after_passage_exchange=own["passage"]["answer"],
                png_sha256=sha(f),
            )
    dump(PUB / "examples.json", examples)
    dump(
        PUB / "stored_ranks.json",
        dict(
            counts=dict(freq),
            n=len(stats),
            median_rank=float(np.median([s["rank"] for s in stats])),
            median_gate=float(np.median([s["gate"] for s in stats])),
            slots=stats,
        ),
    )
    print(dict(freq), len(stats), examples, flush=True)


if __name__ == "__main__":
    with gpu_owner():
        inspect()
