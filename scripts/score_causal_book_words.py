"""Recount free word and prefix accuracy from archived output pixels."""

from zipfile import ZipFile
import numpy as np
from scripts.evaluate_causal_book_words import ARCHIVE, FRESH, OLD, load_table
from scripts.evaluate_passage_selector import read_ink, read_array
from scripts.prepare_causal_book_words import PUB, WORK, MODEL, old
from scripts.score_passage_selector import exact, bounds

ARMS = ("dense", "endpoint", "released", "counts")
PRIMARY = ("absent", "prefix2", "prefix3")


def measure(z, split):
    tab = load_table(split)
    groups = tab["group"]
    tasks = tab["task"]
    gold = [
        read_ink(z, split + f"/gold{j+1}") for j in range(3 if split in FRESH else 1)
    ]
    truth = old.base.Ink()
    forms = old.base.read(WORK / "forms.json")
    for j, ref in enumerate(gold):
        for k, pixels in zip(tab["target"][:, j], ref, strict=True):
            idx = truth.encode(forms[k])
            assert np.array_equal(truth.array()[idx], pixels)
    fresh = split in FRESH
    within = (tab["source"] == tab["target"][:, 0, None]).any(1)
    masks = {
        "all": np.ones(len(tasks), bool),
        "absent": ~within,
        "present": within,
        "ordinary": tasks == 0,
        "final": tasks == 1,
    }
    if fresh:
        for j in range(3):
            masks[f"prefix{j+1}"] = (tasks == 0) & (~tab["ends"][:, :j].any(1))
            masks[f"word{j+1}"] = masks[f"prefix{j+1}"]
    correct = {}
    stops = {}
    for arm in ARMS + (("donor",) if fresh else ()):
        prefix = split + "/" + arm
        assert np.array_equal(read_array(z, prefix + "/indices"), np.arange(len(tasks)))
        end = read_array(z, prefix + "/stops")
        hits = []
        for j in range(3 if fresh and arm != "donor" else 1):
            emitted = read_ink(z, prefix + f"/word{j+1}")
            active = np.array([len(w) > 0 for w in emitted])
            assert np.array_equal(
                active, ((end == 0) | (end >= j + 1)) & ((j == 0) | (tasks == 0))
            )
            hits.append(exact(emitted, gold[j]))
        report = {}
        for k, m in masks.items():
            if k.startswith(("prefix", "word")):
                j = int(k[-1])
                if j > len(hits):
                    continue
                h = (
                    np.logical_and.reduce(hits[:j])
                    if k.startswith("prefix")
                    else hits[j - 1]
                )
            else:
                h = hits[0]
            report[k] = h
        correct[arm] = report
        if fresh and arm != "donor":
            ref = np.where(tab["ends"].any(1), tab["ends"].argmax(1) + 1, 0)
            ordinary = tasks == 0
            stops[arm] = dict(
                predicted=int(((end > 0) & ordinary).sum()),
                matched=int(((end > 0) & (end == ref) & ordinary).sum()),
                reference=int(((ref > 0) & ordinary).sum()),
            )
    return masks, correct, groups, stops


def summarize(masks, correct, groups):
    res = {}
    for k, m in masks.items():
        arms = [a for a in correct if k in correct[a]]
        res[k] = dict(
            n=int(m.sum()),
            hits={a: int(correct[a][k][m].sum()) for a in arms},
            intervals={
                a: bounds(correct["dense"][k][m], correct[a][k][m], groups[m])
                for a in arms
                if a != "dense"
            }
            if m.any()
            else {},
        )
    return res


def main():
    with ZipFile(ARCHIVE) as z:
        raw = {s: measure(z, s) for s in (*FRESH, *OLD)}
    outcome = {s: dict(scores=summarize(*r[:3]), stopping=r[3]) for s, r in raw.items()}
    m0, c0, g0, _ = raw["fiction"]
    m1, c1, g1, _ = raw["gutenberg"]
    combined = summarize(
        {k: np.r_[m0[k], m1[k]] for k in m0},
        {
            a: {k: np.r_[v, c1[a][k]] for k, v in values.items()}
            for a, values in c0.items()
        },
        np.r_[g0, g1 + g0.max() + 1],
    )
    passed = all(
        combined[k]["intervals"]["endpoint"][0] > 0
        and combined[k]["hits"]["dense"] > combined[k]["hits"]["counts"]
        for k in PRIMARY
    )
    old.base.dump(
        PUB / "scores.json",
        dict(
            archive=old.base.sha(ARCHIVE),
            weights={
                a: old.base.sha(MODEL / a / "weights.safetensors")
                for a in ("dense", "endpoint")
            },
            splits=outcome,
            combined=combined,
            composition_gate=bool(passed),
        ),
    )
    for k in ("all", *PRIMARY):
        print(k, combined[k], flush=True)
    print("gate", passed, flush=True)


if __name__ == "__main__":
    main()
