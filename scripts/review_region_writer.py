"""Judge full natural responses, calibrated with human replies and minimally changed facts."""

from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
from scripts.prepare_region_writer import WORK, PUB, digest
from scripts.prepare_open_exam import read, dump, sha
from scripts.evaluate_region_writer import changed
from scripts.evaluate_visual_discourse import unpack
from scripts.benchmark_open_exam import JUDGE, ask, server
from scripts.benchmark_competing_qwen import STORE, MODELS
from scripts.train_paired_evidence import gpu_owner


def main():
    rows = read(WORK / "dev-source.json")
    edits = changed(rows)
    outputs = {
        a: unpack(WORK / (a + "-predictions.json.gz"))["original"]
        for a in ("direct", "selected")
    }
    examples = []
    for t in ("catalogue", "statements"):
        pool = sorted(
            (e for e in edits if e["task"] == t and e["answers"]),
            key=lambda e: digest("review-natural-region" + e["family"]),
        )
        assert len(pool) >= 16
        examples.extend(pool[:16])
    names = ["direct", "selected", "human", "wrong_fact"]
    keys = list("ABCD")
    rng = np.random.default_rng(4261125)
    schema = dict(
        type="object", properties={k: dict(type="boolean") for k in keys}, required=keys
    )
    items = []
    for e in examples:
        i = e["original"]
        r = rows[i]
        gold = next(s for s in r["answers"] if e["old"] in s)
        variants = {a: z[i]["text"] for a, z in outputs.items()}
        variants.update(human=gold, wrong_fact=gold.replace(e["old"], e["new"]))
        assert variants["human"] != variants["wrong_fact"]
        shuffled = list(rng.permutation(names))
        for order in (shuffled, shuffled[::-1]):
            payload = dict(
                question=r["q"],
                regions=r["blocks"],
                answers={k: variants[n] for k, n in zip(keys, order)},
            )
            prompt = (
                "Decide independently for each answer whether it answers the pictured question completely, in intelligible language, based on the source facts. Accept faithful rephrasing. Reject a wrong value, an incomplete statement or an unsupported claim. The source and replies are transcribed below. Reply in the requested JSON schema.\n"
                + json.dumps(payload, ensure_ascii=False)
            )
            items.append((r["task"], r["family"], order, prompt))
    assert sha(STORE / "blobs" / ("sha256-" + MODELS[JUDGE])) == MODELS[JUDGE]
    with gpu_owner(), server() as url:

        def grade(item):
            task, fam, order, prompt = item
            text, stop = ask(url, JUDGE, dict(role="user", content=prompt), schema)
            vals = json.loads(text)
            if not stop or any(type(vals.get(k)) != bool for k in keys):
                raise ValueError("Invalid review")
            return dict(
                task=task, family=fam, votes={n: vals[k] for k, n in zip(keys, order)}
            )

        with ThreadPoolExecutor(max_workers=4) as workers:
            votes = list(workers.map(grade, items))
    counts = {}
    for task in ("catalogue", "statements"):
        pairs = [
            (a["votes"], b["votes"])
            for a, b in zip(votes[::2], votes[1::2])
            if a["task"] == task
        ]
        c = {n: sum(a[n] and b[n] for a, b in pairs) for n in names}
        c["foil_rejected"] = sum(
            not a["wrong_fact"] and not b["wrong_fact"] for a, b in pairs
        )
        c["qualified"] = c["human"] >= 15 and c["foil_rejected"] >= 15
        counts[task] = c
    dump(
        PUB / "semantic-review.json",
        dict(
            criterion="At least 15 of 16 original human answers accepted and 15 of 16 minimally changed foils rejected in both orders in each domain.",
            counts=counts,
            votes=votes,
            judge=MODELS[JUDGE],
        ),
    )


if __name__ == "__main__":
    main()
