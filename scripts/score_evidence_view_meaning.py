"""Calibrate an anonymous local model judge on alternate human answers before use."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.benchmark_open_exam import JUDGE, ask, server
from scripts.benchmark_competing_qwen import STORE, MODELS
from scripts.evaluate_evidence_views import groups, official_index
from scripts.evaluate_visual_discourse import unpack
from scripts.prepare_open_exam import dump, sha
from scripts.prepare_evidence_views import WORK, PUB
from scripts.train_paired_evidence import gpu_owner


def main(work, report):
    judge, _, _, dev, _ = official_index()
    candidates = groups(judge, dev)["composition"]

    def distinct(r):
        vals = list(dict.fromkeys(judge.normalize_answer(t) for t in r["refs"]))
        return len(vals) >= 2

    rows = []
    used = set()
    for r in sorted(
        filter(distinct, candidates),
        key=lambda r: sha256((r["doc"] + str(r["turn"]) + "judge").encode()).digest(),
    ):
        if r["doc"] in used:
            continue
        rows.append(r)
        used.add(r["doc"])
        if len(rows) == 24:
            break
    assert len(rows) == 24
    outputs = {
        a: {
            (z["doc"], z["turn"]): z
            for z in unpack(work / (a + "-responses.json.gz"))
            if z["group"] == "composition"
        }
        for a in ("whole", "highlight")
    }
    cats = ["whole", "highlight", "human", "other_document"]
    letters = "ABCD"
    schema = dict(
        type="object",
        properties={k: dict(type="boolean") for k in letters},
        required=list(letters),
        additionalProperties=False,
    )
    items = []
    for i, r in enumerate(rows):
        human = r["refs"][-1]
        refs = [
            t
            for t in r["refs"]
            if judge.normalize_answer(t) != judge.normalize_answer(human)
        ]
        assert refs
        foil = next(
            z["refs"][0]
            for z in rows[i + 1 :] + rows[:i]
            if judge.normalize_answer(z["refs"][0])
            not in set(map(judge.normalize_answer, r["refs"]))
        )
        answers = {a: outputs[a][r["doc"], r["turn"]]["text"] for a in cats[:2]}
        answers.update(human=human, other_document=foil)
        order = list(np.random.default_rng(929 + i).permutation(cats))
        for names in (order, order[::-1]):
            content = (
                "Judge each anonymous answer independently. Mark true only if it gives a complete correct answer to the current question in context, accepting equivalent wording. The pictured dialogue has been transcribed for you. Previous dialogue, if present, precedes a vertical bar. Return one boolean per answer.\n"
                + json.dumps(
                    dict(
                        passage=r["page"],
                        question=r["question"],
                        human_references=refs,
                        answers={k: answers[a] for k, a in zip(letters, names)},
                    )
                )
            )
            items.append((r["doc"], r["turn"], names, content))
    if sha(STORE / "blobs" / ("sha256-" + MODELS[JUDGE])) != MODELS[JUDGE]:
        raise ValueError("Changed local judge")
    with gpu_owner(), server() as url:

        def grade(item):
            doc, turn, names, content = item
            text, stop = ask(url, JUDGE, dict(role="user", content=content), schema)
            votes = json.loads(text)
            if not stop or any(type(votes.get(k)) is not bool for k in letters):
                raise ValueError("Invalid anonymous review")
            return dict(
                doc=doc,
                turn=turn,
                ratings={a: votes[k] for k, a in zip(letters, names)},
            )

        with ThreadPoolExecutor(max_workers=4) as pool:
            out = list(pool.map(grade, items))
    both = {
        a: sum(u["ratings"][a] and v["ratings"][a] for u, v in zip(out[::2], out[1::2]))
        for a in cats
    }
    rejected = sum(
        not u["ratings"]["other_document"] and not v["ratings"]["other_document"]
        for u, v in zip(out[::2], out[1::2])
    )
    dump(
        report / "meaning-review.json",
        dict(
            n=len(rows),
            judge=MODELS[JUDGE],
            accepted_in_both_orders=both,
            foils_rejected_in_both_orders=rejected,
            qualification=both["human"] >= 23 and rejected >= 23,
            criterion="At least 23/24 distinct human alternatives accepted and 23/24 cross-document foils rejected in both orders.",
            votes=out,
        ),
    )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=WORK)
    p.add_argument("--report", type=Path, default=PUB)
    a = p.parse_args()
    main(a.work, a.report)
