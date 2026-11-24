"""Recount paired, source-absent and previously unwritten replies from the saved outputs."""

from collections import defaultdict
from pathlib import Path
import argparse
import numpy as np
from scripts.prepare_evidence_views import PUB, WORK, read
from scripts.prepare_open_exam import dump
from scripts.train_open_exam import corpus
from scripts.score_dialogue_ink_references import official_index
from scripts.evaluate_visual_discourse import unpack


def main(work, report):
    judge, _, _, _, _ = official_index()
    norm = judge.normalize_answer
    from scripts.prepare_evidence_views import TRAIN

    chinese, _, _ = corpus("train")
    # Novelty below uses a deliberately stronger exclusion: every original CoQA and C³
    # training answer, including the turns omitted by geometry or split screening.
    seen = {norm(a["input_text"]) for d in read(TRAIN)["data"] for a in d["answers"]}
    seen.update(norm(r["answer"]) for r in chinese)
    all_outputs = {}
    per_arm = {}
    for arm in ("parent", "whole", "highlight"):
        rows = unpack(work / (arm + "-responses.json.gz"))
        pub = unpack(report / (arm + "-outcomes.json.gz"))
        assert len(rows) == len(pub) and all(
            all(r[k] == v for k, v in b.items()) for r, b in zip(rows, pub, strict=True)
        )
        data = defaultdict(dict)
        for r in rows:
            key = (r["doc"], r["turn"])
            group = data[r["group"]]
            if key in group:
                raise ValueError("Repeated evaluation turn")
            group[key] = r
        all_outputs[arm] = data
        r = data["composition"]
        hits = [norm(z["text"]) for z in r.values() if z["exact"]]
        per_arm[arm] = dict(
            turns=len(r),
            exact=len(hits),
            distinct=len(set(hits)),
            outside_training=sum(h not in seen for h in hits),
            distinct_outside_training=len(set(hits) - seen),
        )
    primary = {}
    for group in ("coqa", "drop", "composition"):
        a, b = [all_outputs[arm][group] for arm in ("whole", "highlight")]
        if a.keys() != b.keys():
            raise ValueError("Unmatched questions")
        bydoc = defaultdict(list)
        for k, r in a.items():
            bydoc[k[0]].append((r["exact"], b[k]["exact"]))
        pairs = group != "composition"
        if pairs:
            assert all(len(x) == 2 for x in bydoc.values())
        # Bootstrap the document as a cluster while retaining its actual question count.
        doc = np.asarray(
            [
                [
                    sum(int(y) - int(x) for x, y in g),
                    len(g),
                    int(all(y for x, y in g)) - int(all(x for x, y in g)),
                ]
                for g in bydoc.values()
            ]
        )
        rng = np.random.default_rng(71125)
        samples = doc[rng.integers(0, len(doc), (5000, len(doc)))].sum(1)
        primary[group] = dict(
            docs=len(doc),
            turns=len(a),
            gained=sum(not r["exact"] and b[k]["exact"] for k, r in a.items()),
            lost=sum(r["exact"] and not b[k]["exact"] for k, r in a.items()),
            exact_diff_pp_ci95=np.percentile(
                100 * samples[:, 0] / samples[:, 1], [2.5, 97.5]
            ).tolist(),
            pair_diff_pp_ci95=np.percentile(
                100 * samples[:, 2] / len(doc), [2.5, 97.5]
            ).tolist()
            if pairs
            else None,
        )
    dump(report / "paired-audit.json", dict(composition=per_arm, paired=primary))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--report", type=Path, default=PUB)
    a = p.parse_args()
    main(a.work, a.report)
