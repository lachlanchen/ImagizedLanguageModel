"""Measure free image drafts and both exchanged inputs on the reserved stories."""

import argparse
import gzip
import json
from pathlib import Path
from time import monotonic
import numpy as np
import torch
from ilm.visual_lm.image_copy_search import copy_proposals
from ilm.visual_lm.word_canvas import draw
from scripts.answer_dialogue_surface import load
from scripts.evaluate_compose_dialogue import assignment, confidence, score, summarize
from scripts.evaluate_dialogue_evidence import predict
from scripts.evaluate_optical_passage_qa import normalized
from scripts.measure_writer_energy import snapshot
from scripts.prepare_compose_dialogue import MODEL as PARENTS, kind
from scripts.prepare_dialogue_ink import dump, sha
from scripts.prepare_word_canvas import MODEL, reserved, PUB
from scripts.train_dialogue_surface import representation, corpus
from scripts.train_dialogue_subword import timed
from scripts.train_span_choice import gpu_owner


def write_bound(out, name, rows):
    path = out / (name + ".json.gz")
    path.write_bytes(
        gzip.compress(
            json.dumps(rows, separators=(",", ":"), allow_nan=False).encode(), mtime=0
        )
    )
    return sha(path)


def passes(data, indices, swaps, size):
    q, p = swaps
    for arm in ("original", "question", "passage"):
        for start in range(0, len(indices), size):
            ix = indices[start : start + size]
            x, v = data.inputs(
                ix,
                q[ix] if arm == "question" else None,
                p[ix] if arm == "passage" else None,
            )
            yield arm, ix, x, v


@torch.inference_mode()
def evaluate(root, out):
    out.mkdir(parents=True, exist_ok=False)
    data = reserved()
    rows = data["rows"]
    refs = [[r[5]] for r in rows]
    classes = np.array([kind(r[5], (data["pages"][r[1]], r[3], r[4])) for r in rows])
    subsets = {
        "all": np.arange(len(rows)),
        "difficult": np.flatnonzero(classes != "ordinary"),
        "assembly": np.flatnonzero(classes == "assembly"),
        "spell": np.flatnonzero(classes == "spell"),
    }
    swaps = assignment(rows)
    study = dict(
        reservation=sha(PUB / "reservation.json"), models={}, differences={}, search={}
    )
    for mode in ("ink", "byte"):
        face, unit, sep = representation(mode)
        decode = {face(chr(i)): chr(i) for i in range(32, 127)}

        def read(w):
            return "".join(decode[w[i : i + unit]] for i in range(0, len(w), unit))

        values = {}
        grounded = {}
        for arm in ("parent", "causal", "masked"):
            folder = PARENTS / mode / "lists" if arm == "parent" else root / mode / arm
            clock = monotonic()
            meter = snapshot()
            torch.cuda.reset_peak_memory_stats()
            net = load(folder, "cuda")
            ds = corpus(data, "fresh", net, face, unit)
            if arm == "masked":
                records = [{} for _ in rows]
                for condition, ix, x, v in passes(ds, subsets["all"], swaps, 48):
                    timeline = []
                    batch = draw(net, ds.bank, x, v, commit_rounds=timeline)
                    for i, paths, steps in zip(ix, batch, timeline, strict=True):
                        words, stop, _ = paths[0]
                        records[i][condition] = dict(
                            answer=" ".join(map(read, words)),
                            stop=stop,
                            commit_rounds=steps,
                        )
            else:
                records = predict(folder, ds, net, sep, unit, decode, rows, swaps)
            cost = timed(clock, meter)
            vals = score(records, refs, None, None)
            scores, grounded[arm] = summarize(vals, subsets)
            values[arm] = vals
            tag = f"{mode}-{arm}"
            for label, indices in subsets.items():
                scores[label]["length_correct"] = sum(
                    len(records[i]["original"]["answer"].split())
                    == len(rows[i][5].split())
                    for i in map(int, indices)
                )
            study["models"][tag] = dict(
                weights=sha(folder / "writer.safetensors"),
                scores=scores,
                outputs=write_bound(out, tag, records),
                cost=cost,
            )
            print(
                tag,
                "all",
                scores["all"]["exact"],
                "difficult",
                scores["difficult"],
                flush=True,
            )
            # At equal candidate budgets compare parent and masked inference on identical copyable cases.
            # The masked search also includes all turns requiring new words.
            if arm == "causal":
                continue
            indices = subsets["difficult" if arm == "masked" else "assembly"]
            candidates = {int(i): {} for i in indices}
            batch_disagreements = []
            clock = monotonic()
            meter = snapshot()
            torch.cuda.reset_peak_memory_stats()
            for condition, ix, x, v in passes(
                ds, indices, swaps, 4 if arm == "masked" else 8
            ):
                if arm == "masked":
                    batch = draw(net, ds.bank, x, v, explore=True)
                else:
                    batch = [
                        [(tuple(ds.bank.raw[j] for j in seq), True, s) for seq, s in rr]
                        for rr in copy_proposals(net, ds.bank, x, v)
                    ]
                for i, paths in zip(ix, batch, strict=True):
                    if arm == "masked" and (
                        (" ".join(map(read, paths[0][0])), paths[0][1])
                        != (
                            records[i][condition]["answer"],
                            records[i][condition]["stop"],
                        )
                    ):
                        batch_disagreements.append([int(i), condition])
                    best = {}
                    for words, stopped, logscore in paths:
                        if stopped:
                            text = " ".join(map(read, words))
                            best[text] = max(best.get(text, -float("inf")), logscore)
                    candidates[int(i)][condition] = sorted(
                        best.items(), key=lambda x: -x[1]
                    )
            cost = timed(clock, meter)
            summaries = {}
            for label in ("assembly", "spell", "difficult"):
                ix = [int(i) for i in subsets[label] if int(i) in candidates]
                if not ix:
                    continue
                summary = {}
                for condition in ("original", "question", "passage"):
                    ranks = []
                    for i in ix:
                        target = normalized(rows[i][5])
                        ranks.append(
                            next(
                                (
                                    k
                                    for k, (s, _) in enumerate(
                                        candidates[i][condition], 1
                                    )
                                    if normalized(s) == target
                                ),
                                0,
                            )
                        )
                    summary[condition] = dict(
                        n=len(ix),
                        top1=ranks.count(1),
                        top8=sum(0 < k <= 8 for k in ranks),
                        top32=sum(k > 0 for k in ranks),
                    )
                summaries[label] = summary
            study["search"][tag] = dict(
                default_batch_disagreements=batch_disagreements,
                scores=summaries,
                cost=cost,
                outputs=write_bound(out, tag + "-proposals", list(candidates.items())),
            )
            print(tag, "proposals", summaries["assembly"]["original"], flush=True)
        for control in ("parent", "causal"):
            for label, ix in subsets.items():
                diff = values["masked"][:, 0, 1] - values[control][:, 0, 1]
                study["differences"][mode + "-" + control + "-" + label] = dict(
                    f1=confidence(diff, rows, ix),
                    gained=int((grounded["masked"][ix] & ~grounded[control][ix]).sum()),
                    lost=int((~grounded["masked"][ix] & grounded[control][ix]).sum()),
                )
    dump(out / "results.json", study)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, default=MODEL)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    with gpu_owner():
        evaluate(a.models, a.output)
