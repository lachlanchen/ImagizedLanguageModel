"""Evaluate a frozen writer using its estimated passage-interval prior."""

import gzip
import json
import torch
from scripts.qualify_dialogue_teacher import PUB, WORK, dump, sha
from scripts.dialogue_ink_data import read_data
from scripts.dialogue_teacher_stories import new_stories, absent
from scripts.train_dialogue_evidence import RELEASE
from scripts.train_dialogue_surface import representation, corpus
from scripts.evaluate_dialogue_evidence import summarize, difference, predict
from scripts.answer_dialogue_surface import load
from scripts.train_span_choice import gpu_owner
from ilm.visual_lm.dialogue_surface import interval_occupancy


def check_intervals():
    rng = torch.Generator().manual_seed(147)
    max_error = 0.0
    for n in (1, 4, 19, 48):
        x = torch.randn(4, n, 2, dtype=torch.float64, generator=rng) * 9
        for j in range(4):
            x[j, max(1, n - j) :] = -10000.0
        score = x[:, :, 0, None] + x[:, None, :, 1]
        score = score.masked_fill(
            ~torch.ones((n, n), dtype=torch.bool).triu(), -torch.inf
        )
        weight = score.flatten(1).softmax(1).view_as(score)
        ref = torch.stack([weight[:, : i + 1, i:].sum((1, 2)) for i in range(n)], 1)
        max_error = max(max_error, float((ref - interval_occupancy(x)).abs().max()))
    if max_error > 1e-12:
        raise ValueError("Ordered-span marginal mismatch")
    return max_error


@torch.inference_mode()
def study():
    result = {"enumeration_error": check_intervals(), "groups": {}}
    registry = read_data()
    for subset in ("long", "longer"):
        data, markers, evaluator = new_stories(registry, subset)
        rows = data["rows"]
        hard = [
            i
            for i, r in enumerate(rows)
            if absent(
                evaluator.gold_data[markers[i]], (data["pages"][r[1]], r[3], r[4])
            )
        ]
        group = result["groups"][subset] = dict(
            turns=len(rows),
            stories=len({r[1] for r in rows}),
            absent=len(hard),
            modes={},
        )
        for mode in ("ink", "byte"):
            folder = RELEASE / mode / "human"
            net = load(folder, "cuda")
            face, unit, sep = representation(mode)
            decode = {face(chr(i)): chr(i) for i in range(32, 127)}
            ds = corpus(data, "fresh", net, face, unit)
            outs = []
            for prior in (False, True):
                net.use_predicted_interval = prior
                rec = predict(folder, ds, net, sep, unit, decode, rows)
                if subset == "long" and not prior:
                    old = json.loads(
                        gzip.decompress(
                            (WORK / f"fresh-{mode}-human.json.gz").read_bytes()
                        )
                    )
                    if rec != old:
                        raise ValueError("Control no longer reproduces")
                file = WORK / f"interval-{subset}-{mode}-{int(prior)}.json.gz"
                file.write_bytes(
                    gzip.compress(
                        json.dumps(rec, separators=(",", ":")).encode(), mtime=0
                    )
                )
                outs.append(rec)
            entry = group["modes"][mode] = dict(
                paired=difference(*outs, rows, markers, evaluator, set(hard)),
                weights=sha(folder / "writer.safetensors"),
            )
            for name, out in zip(("ordinary", "interval"), outs):
                entry[name] = summarize(out, markers, evaluator, hard)
        print(subset, group["modes"]["ink"]["paired"], flush=True)
    dump(PUB / "predicted_interval.json", result)


if __name__ == "__main__":
    with gpu_owner():
        study()
