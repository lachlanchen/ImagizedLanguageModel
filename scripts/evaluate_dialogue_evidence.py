"""Measure independent image and byte students on complete previously excluded stories."""

from collections import defaultdict
import argparse
import gzip
import json
from pathlib import Path
import hashlib
import numpy as np
import torch
from scripts.answer_dialogue_surface import load, answer
from scripts.answer_dialogue_ink import generate
from scripts.dialogue_ink_data import read_data
from scripts.train_dialogue_surface import ROOT_MODEL, corpus, representation
from scripts.dialogue_teacher_stories import new_stories, absent
from scripts.train_dialogue_evidence import RELEASE
from scripts.qualify_dialogue_teacher import PUB, dump, sha
from scripts.evaluate_dialogue_ink import exchanges
from scripts.evaluate_optical_passage_qa import normalized, save_page
from scripts.train_span_choice import gpu_owner
from ilm.visual_lm.causal_glyph_writer import write_strip
from ilm.visual_lm.portable_writer import read_image


def turn_metrics(evaluator, marker, r):
    if not r["stop"]:
        return dict(f1=0.0, em=0.0, any_exact=False)
    values = evaluator.compute_turn_score(*marker, r["answer"])
    return dict(
        f1=values["f1"],
        em=values["em"],
        any_exact=any(
            normalized(a) == normalized(r["answer"])
            for a in evaluator.gold_data[marker]
        ),
    )


@torch.inference_mode()
def predict(folder, ds, net, sep, unit, decode, rows, swaps=None):
    q, p = exchanges(rows) if swaps is None else swaps
    records = [{} for _ in rows]
    for key, qi, pi in [
        ("original", None, None),
        ("question", q, None),
        ("passage", None, p),
    ]:
        for start in range(0, len(rows), 48):
            ix = np.arange(start, min(len(rows), start + 48))
            x, variants = ds.inputs(
                ix, None if qi is None else qi[ix], None if pi is None else pi[ix]
            )
            values, stops, _ = generate(
                net, None, ds.bank, x, variants, separator=sep, feedback=net.surface
            )
            for i, raw, stop in zip(ix, values, stops, strict=True):
                text = "".join(
                    decode[raw[k : k + unit]] for k in range(0, len(raw), unit)
                )
                records[i][key] = dict(answer=text, stop=bool(stop))
    return records


def summarize(records, markers, evaluator, hard):
    scores = {}
    for key, idx in (("all", list(range(len(records)))), ("absent", hard)):
        scores[key] = {}
        for arm in ("original", "question", "passage"):
            vals = [turn_metrics(evaluator, markers[i], records[i][arm]) for i in idx]
            scores[key][arm] = dict(
                turns=len(vals),
                f1=np.mean([v["f1"] for v in vals]),
                em=np.mean([v["em"] for v in vals]),
                any_exact=sum(v["any_exact"] for v in vals),
                stopped=sum(records[i][arm]["stop"] for i in idx),
            )
    return scores


def difference(control, test, rows, markers, evaluator, hard):
    by_story = defaultdict(list)
    gained = lost = 0
    for i, (c, t) in enumerate(zip(control, test, strict=True)):
        score = lambda r: turn_metrics(evaluator, markers[i], r)
        by_story[rows[i][1]].append(
            score(t["original"])["f1"] - score(c["original"])["f1"]
        )
        if i in hard:

            def credit(r):
                return score(r["original"])["any_exact"] and all(
                    not score(r[k])["any_exact"] for k in ("question", "passage")
                )

            gained += credit(t) and not credit(c)
            lost += credit(c) and not credit(t)
    a = np.array([(sum(v), len(v)) for v in by_story.values()])
    draws = np.random.default_rng(420121).integers(len(a), size=(5000, len(a)))
    s = a[draws].sum(1)
    ci = np.quantile(s[:, 0] / s[:, 1], [0.025, 0.975])
    return dict(
        delta=float(a[:, 0].sum() / a[:, 1].sum()),
        interval95=ci.tolist(),
        stories=len(a),
        gained_absent=gained,
        lost_absent=lost,
    )


def evaluate(root, output):
    output.mkdir(parents=True, exist_ok=False)
    private = output / "decisions"
    private.mkdir()
    data, markers, evaluator = new_stories(read_data(), "long")
    rows = data["rows"]
    hard = [
        i
        for i, r in enumerate(rows)
        if absent(evaluator.gold_data[markers[i]], (data["pages"][r[1]], r[3], r[4]))
    ]
    if not hard:
        raise ValueError("No absent-phrase turns")
    passed = json.loads((PUB / "current_last/qualification.json").read_text())[
        "admitted"
    ]
    fits = ["human", "teacher"] if passed else ["plain", "human"]
    summary = {}
    paired = {}
    files = {}
    examples = []
    for mode in ("ink", "byte"):
        face, unit, sep = representation(mode)
        decode = {face(chr(i)): chr(i) for i in range(32, 127)}
        outputs = {}
        for arm in ["parent"] + fits:
            folder = ROOT_MODEL / mode if arm == "parent" else root / mode / arm
            net = load(folder, "cuda")
            if arm == "parent":
                ds = corpus(data, "fresh", net, face, unit)
            rec = predict(folder, ds, net, sep, unit, decode, rows)
            outputs[arm] = rec
            summary[mode + "/" + arm] = summarize(rec, markers, evaluator, hard)
            file = private / f"fresh-{mode}-{arm}.json.gz"
            file.write_bytes(
                gzip.compress(json.dumps(rec, separators=(",", ":")).encode(), mtime=0)
            )
            files[mode + "/" + arm] = dict(
                outputs=sha(file), weights=sha(folder / "writer.safetensors")
            )
        paired[mode] = difference(
            outputs[fits[0]], outputs[fits[1]], rows, markers, evaluator, set(hard)
        )
        if mode == "ink":
            # First changed success and first changed loss against the matched human control, in corpus order.
            for gain, name in ((True, "gain"), (False, "loss")):
                idx = next(
                    (
                        i
                        for i in range(len(rows))
                        if turn_metrics(
                            evaluator, markers[i], outputs[fits[1]][i]["original"]
                        )["any_exact"]
                        == gain
                        and turn_metrics(
                            evaluator, markers[i], outputs[fits[0]][i]["original"]
                        )["any_exact"]
                        != gain
                    ),
                    None,
                )
                if idx is None:
                    continue
                r = rows[idx]
                paths = []
                example_folder = private / "examples"
                example_folder.mkdir(exist_ok=True)
                for field, text in zip(
                    ("passage", "question", "history"),
                    (data["pages"][r[1]], r[3], r[4]),
                ):
                    f = example_folder / f"{name}-{field}.png"
                    if text:
                        save_page(text, f, face)
                        paths.append(read_image(f, None))
                    else:
                        paths.append(np.empty((0, 128), np.uint8))
                for arm in fits:
                    raw, state = answer(load(root / mode / arm), paths)
                    ref = outputs[arm][idx]["original"]
                    if raw != face(ref["answer"]) or state["stopped"] != ref["stop"]:
                        raise ValueError("Original PNG CPU replay differs")
                    write_strip(
                        np.frombuffer(raw, np.uint8).reshape(-1, 128),
                        output / f"{name}-{arm}.png",
                    )
                examples.append(
                    dict(
                        kind=name,
                        index=idx,
                        domain=r[7],
                        question=r[3],
                        references=evaluator.gold_data[markers[idx]],
                        answers={k: outputs[k][idx]["original"] for k in fits},
                    )
                )
    result = dict(
        stories=len(set(r[1] for r in rows)),
        turns=len(rows),
        absent=len(hard),
        comparison=fits,
        scores=summary,
        paired=paired,
        bindings=files,
        examples=examples,
        held_hash=hashlib.sha256(json.dumps(markers).encode()).hexdigest(),
    )
    dump(output / "fresh.json", result)
    print("fresh comparison", paired, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, default=RELEASE)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    with gpu_owner():
        evaluate(a.model_root, a.output)
