"""Fit and evaluate a question-side relation channel without changing the visual reader."""

from collections import Counter, defaultdict
import json
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from ilm.visual_lm.pictured_relation import (
    BUCKETS,
    THRESHOLD,
    PicturedRelation,
    features,
)
from scripts.prepare_visual_facts import (
    ROOT,
    WORK,
    SEED,
    Face,
    FONTS,
    LANG,
    RELATIONS,
    statement,
    write_json,
)
from scripts.train_visual_facts import rows

OUT = ROOT / "artifacts/pictured_relation_v1"
EVIDENCE = ROOT / "references/evidence/pictured_relation_20261123"


def fit(data, face):
    vectors = []
    targets = []

    def add(q, s, r):
        vectors.append(features(face(q), face(s)))
        targets.append(RELATIONS.index(r))

    for x in data:
        add(x["q"], x["s"], x["r"])
    for r in RELATIONS:
        distinct = list(dict.fromkeys(x["s"] for x in data if x["r"] == r))
        for i, s in enumerate(distinct[:200]):
            noun = LANG[r][0]
            q = f"请问{s}的{noun}是哪位？" if i % 2 == 0 else f"谁是{s}的{noun}？"
            add(q, s, r)
    offsets = np.cumsum([0] + [len(v) for v in vectors])
    cols = np.concatenate(vectors)
    mat = csr_matrix(
        (np.ones(len(cols), dtype=np.float32), cols, offsets),
        shape=(len(vectors), BUCKETS),
    )
    for mode in ("genuine", "permuted"):
        y = np.asarray(targets)
        if mode == "permuted":
            y = np.random.default_rng(SEED + 31).permutation(y)
        clf = LogisticRegression(solver="lbfgs", C=1, max_iter=160, tol=1e-4)
        with threadpool_limits(2):
            clf.fit(mat, y)
        assert np.array_equal(clf.classes_, np.arange(len(RELATIONS)))
        np.savez_compressed(
            OUT / (mode + ".npz"),
            weights=clf.coef_.astype(np.float32),
            bias=clf.intercept_.astype(np.float32),
        )
    return len(data), len(vectors) - len(data)


def evaluate():
    OUT.mkdir(exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    face = Face((FONTS[0], 2))
    n, extra = fit(rows("training"), face)
    held = rows("held")
    outputs = json.loads((WORK / "genuine_readings.json").read_text())
    result = {}
    for mode in ("genuine", "permuted"):
        model = PicturedRelation(OUT / (mode + ".npz"))
        totals = defaultdict(Counter)
        ledger = []
        notes = []
        for x, z in zip(held, outputs, strict=True):
            assert (x["key"], x["group"]) == (z["key"], z["group"])
            base = RELATIONS.index(z["relation"])
            r, conf = model.choose(face(x["q"]), face(z["subject"]), base)
            rel = RELATIONS[r]
            ans = statement(z["subject"], rel, z["object"])
            valid = face(ans) == face(statement(x["s"], x["r"], x["o"]))
            old = z["metrics"]["exact_pixels"]
            c = totals[x["group"]]
            c.update(
                n=1,
                exact=valid,
                base=old,
                correct_role=rel == x["r"],
                changed=base != r,
                repaired=valid and not old,
                lost=old and not valid,
                used=conf >= THRESHOLD,
            )
            ledger.append(
                dict(
                    key=x["key"],
                    group=x["group"],
                    chosen=rel,
                    confidence=conf,
                    output=ans,
                    exact=valid,
                )
            )
            if x["group"] == "notes":
                notes.append(valid)
        result[mode] = dict(
            groups=totals,
            both_roles_both_assignments=sum(
                all(notes[i : i + 4]) for i in range(0, len(notes), 4)
            ),
        )
        write_json(OUT / (mode + "_readings.json"), ledger)
    write_json(
        EVIDENCE / "result.json",
        dict(primary=n, additional=extra, threshold=THRESHOLD, arms=result),
    )
    print(result)


if __name__ == "__main__":
    evaluate()
