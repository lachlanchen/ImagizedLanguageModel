"""Verify counts and replay the illustrative predictions from their actual PNG fields."""

import argparse
import json
import gzip
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from scripts.prepare_word_canvas import reserved, PUB, MODEL
from scripts.prepare_compose_dialogue import kind
from scripts.prepare_dialogue_ink import sha, dump
from scripts.answer_dialogue_surface import load, answer
from scripts.evaluate_optical_passage_qa import normalized, metrics, save_page
from scripts.train_dialogue_surface import representation
from ilm.visual_lm.portable_writer import read_image
from ilm.visual_lm.causal_glyph_writer import write_strip


def restore(path, expected):
    if sha(path) != expected:
        raise ValueError(path)
    return json.loads(gzip.decompress(path.read_bytes()))


def audit(out, images):
    torch.set_num_threads(2)
    data = reserved()
    rows = data["rows"]
    r = json.loads((PUB / "results.json").read_text())
    records = {}
    hard = [kind(x[5], (data["pages"][x[1]], x[3], x[4])) != "ordinary" for x in rows]
    for tag, model in r["models"].items():
        rec = records[tag] = restore(
            PUB / "responses" / (tag + ".json.gz"), model["outputs"]
        )
        for subset, indices in [
            ("all", range(len(rows))),
            ("difficult", [i for i, x in enumerate(hard) if x]),
        ]:
            hits = [0, 0, 0]
            grounded = 0
            for i in indices:
                decisions = [
                    a["stop"] and normalized(a["answer"]) == normalized(rows[i][5])
                    for a in (rec[i]["original"], rec[i]["question"], rec[i]["passage"])
                ]
                for j in range(3):
                    hits[j] += decisions[j]
                grounded += decisions == [True, False, False]
            expected = model["scores"][subset]
            if expected["exact"] != hits or expected["grounded"] != grounded:
                raise ValueError(tag + "/" + subset)
    for tag, s in r["search"].items():
        cases = restore(PUB / "responses" / (tag + "-proposals.json.gz"), s["outputs"])
        for label, groups in s["scores"].items():
            for arm, expected in groups.items():
                ranks = []
                for i, cs in cases:
                    row = rows[i]
                    cls = kind(row[5], (data["pages"][row[1]], row[3], row[4]))
                    if label != "difficult" and label != cls:
                        continue
                    vals = cs[arm]
                    if (
                        len(vals) > 32
                        or len({a for a, _ in vals}) != len(vals)
                        or any(vals[j][1] > vals[j - 1][1] for j in range(1, len(vals)))
                    ):
                        raise ValueError("Invalid proposals")
                    ranks.append(
                        next(
                            (
                                j
                                for j, (v, _) in enumerate(vals, 1)
                                if normalized(v) == normalized(row[5])
                            ),
                            0,
                        )
                    )
                if expected != dict(
                    n=len(ranks),
                    top1=ranks.count(1),
                    top8=sum(0 < x <= 8 for x in ranks),
                    top32=sum(x > 0 for x in ranks),
                ):
                    raise ValueError(tag + "/" + label + "/" + arm)
    diagnosis = json.loads((PUB / "diagnosis.json").read_text())
    for mode, v in diagnosis.items():
        cases = restore(PUB / "responses" / (mode + "-diagnosis.json.gz"), v["outputs"])
        ct = Counter()
        for case in cases:
            i = case["row"]
            gold = rows[i][5]
            pred = case["oracle_length_answer"]
            steps = case["commit_rounds"]
            if len(steps) != len(gold.split()):
                raise ValueError("Wrong supplied length")
            bad = min(
                (
                    k
                    for k, a, b in zip(steps, pred.split(), gold.split(), strict=True)
                    if a != b
                ),
                default=0,
            )
            if bad != case["first_literal_error_pass"]:
                raise ValueError("Wrong first error")
            ct["first_error_pass_" + str(bad)] += 1
            for word in case["single_mask"]:
                key = "copyable" if word["copyable"] else "absent"
                ct[key + "_words"] += 1
                ct[key + "_correct"] += word["correct"]
        if any(v["counts"][k] != n for k, n in ct.items()):
            raise ValueError("Diagnosis recount failed")

    def sensitive(tag, i):
        return [
            r["stop"] and normalized(r["answer"]) == normalized(rows[i][5])
            for r in records[tag][i].values()
        ] == [True, False, False]

    def choose(predicate):
        return next((i for i in range(len(rows)) if predicate(i)), None)

    success = next(
        i
        for i in [
            choose(
                lambda j: hard[j]
                and sensitive("ink-masked", j)
                and not sensitive("ink-causal", j)
            ),
            choose(lambda j: hard[j] and sensitive("ink-masked", j)),
            choose(
                lambda j: sensitive("ink-masked", j)
                and len(normalized(rows[j][5]).split()) >= 2
            ),
        ]
        if i is not None
    )
    miss = choose(
        lambda j: hard[j]
        and not metrics(records["ink-masked"][j]["original"]["answer"], [rows[j][5]])[0]
    )
    control = choose(lambda j: hard[j] and sensitive("ink-causal", j))
    face, _, _ = representation("ink")
    report = {}
    images.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    for name, i, arm in [
        ("draft", success, "masked"),
        ("miss", miss, "masked"),
        ("causal", control, "causal"),
    ]:
        if i is None:
            continue
        row = rows[i]
        fields = []
        for key, s in zip(
            ("passage", "question", "history"), (data["pages"][row[1]], row[3], row[4])
        ):
            if not s:
                fields.append(np.empty((0, 128), np.uint8))
                continue
            f = images / f"{name}-{key}.png"
            save_page(s, f, face)
            fields.append(read_image(f, None))
        raw, state = answer(load(MODEL / "ink" / arm), fields)
        record = records["ink-" + arm][i]["original"]
        if face(record["answer"]) != raw or state["stopped"] != record["stop"]:
            raise ValueError("Different standalone image")
        write_strip(
            np.frombuffer(raw, np.uint8).reshape(-1, 128), out / (name + ".png")
        )
        if name == "draft":
            if state["commit_rounds"] != record["commit_rounds"]:
                raise ValueError("Different unmasking order")
            words = record["answer"].split()
            for k in (1, 2, 3, 4):
                original = [face(w) for w in words]
                strip = bytes(128).join(
                    w if step <= k else bytes(len(w))
                    for w, step in zip(original, state["commit_rounds"], strict=True)
                )
                write_strip(
                    np.frombuffer(strip, np.uint8).reshape(-1, 128),
                    images / f"pass-{k}.png",
                )
            (out / "first-pass.png").write_bytes((images / "pass-1.png").read_bytes())
        report[name] = dict(
            index=i,
            story=row[1],
            question=row[3],
            reference=row[5],
            generated=record["answer"],
            state=state,
        )
    dump(out / "examples.json", report)
    print(report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--inputs", required=True, type=Path)
    a = p.parse_args()
    audit(a.output, a.inputs)
