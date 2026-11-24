"""Test complete-answer supervision on new and prior narrative, bilingual and historical tasks."""

import argparse
from collections import defaultdict
from time import monotonic
import numpy as np
import torch
from scripts.prepare_story_answer_fit import (
    WORK,
    PUB,
    MODEL,
    read,
    dump,
    sha,
)
from scripts.prepare_optical_competition import WORK as LAST, PUB as LAST_PUB
from scripts.write_story_answer import StoryAnswer, DirectionalAnswer
from scripts.evaluate_directional_reading import measure, summary
from scripts.evaluate_fact_channel import observed
from scripts.evaluate_visual_discourse import packed
from scripts.evaluate_sentence_context import long_scores
from scripts.prepare_sentence_context import WORK as EARLY, PUB as EARLY_PUB
from scripts.evaluate_region_writer import offpage
from scripts.prepare_fact_channel import MODEL as BASE, Ink
from scripts.measure_writer_energy import snapshot, consumed
from scripts.review_directional_reading import (
    compare,
    review,
    MODELS,
    JUDGE,
    VLM,
    STORE,
)
from scripts.transfer_fact_channel import main as transfer
from scripts.train_fact_channel import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli

ARMS = ("signed", "single", "marginal")


def actors():
    return {
        k: (
            lambda m=k: DirectionalAnswer("signed", "cuda")
            if m == "signed"
            else StoryAnswer(m, "cuda")
        )
        for k in ARMS
    }


def material():
    rows = []
    for folder, pub in ((WORK, PUB), (LAST, LAST_PUB)):
        f = folder / "confirmation.json"
        if sha(f) != read(pub / "reservation.json")["files"][f.name]:
            raise ValueError("Changed test images")
        values = read(f)
        if folder == LAST:
            values = [
                dict(r, task="previous-stories") if r["task"] == "stories" else r
                for r in values
            ]
        rows.extend(values)
    return rows


def evaluate():
    rows = material()
    for f in ("confirmation.json", "long.json"):
        if sha(EARLY / f) != read(EARLY_PUB / "corpus.json")["files"][f]:
            raise ValueError(f)
    prior = read(EARLY / "confirmation.json")
    long = read(EARLY / "long.json")
    dialogue, _, norm = offpage()
    scores = {}
    individual = {}
    for m, make in actors().items():
        ink = Ink(np.load(BASE / "palette.npy"))
        with gpu_owner(), torch.inference_mode():
            net = make()
            start = monotonic()
            energy = snapshot()
            a = observed(net, rows, ink)
            torch.cuda.synchronize()
            cost = dict(
                seconds=monotonic() - start,
                joules=consumed(energy, snapshot())["joules"],
            )
            old = observed(net, prior, ink)
            prose = observed(net, long, ink)
            chat = observed(net, dialogue, ink)
        packed(
            WORK / (m + ".json.gz"),
            dict(original=a, development=old, long=prose, dialogue=chat),
        )
        individual[m] = measure(rows, a, ink)
        scores[m] = dict(
            free=summary(rows, individual[m]),
            development=summary(prior, measure(prior, old, ink)),
            cost=cost,
            long=long_scores(long, prose, ink),
            dialogue=dict(
                n=len(chat),
                exact=sum(
                    v["stop"] and norm(v["text"]) in {norm(w) for w in r["answers"]}
                    for r, v in zip(dialogue, chat, strict=True)
                ),
            ),
        )
        dump(PUB / "scores.json", scores)
        print(m, scores[m]["free"]["stories"], flush=True)
    intervals = defaultdict(dict)
    for comp in ARMS[:2]:
        for task in sorted({r["task"] for r in rows}):
            ids = [i for i, r in enumerate(rows) if r["task"] == task]
            fields = {}
            for k in ("fact", "exact", "mentions"):
                d = np.array(
                    [individual["marginal"][i][k] - individual[comp][i][k] for i in ids]
                )
                pairs = d.reshape(-1, 2).mean(1)
                samples = (
                    np.random.default_rng(7421)
                    .choice(pairs, (5000, len(pairs)))
                    .mean(1)
                )
                fields[k] = dict(
                    gained=int((d > 0).sum()),
                    lost=int((d < 0).sum()),
                    interval_pp=(100 * np.quantile(samples, [0.025, 0.975])).tolist(),
                )
            intervals[comp][task] = fields
    dump(PUB / "intervals.json", intervals)
    transfer(actors(), PUB, MODEL)


def judgment():
    rows = material()
    for m in (JUDGE, VLM):
        if sha(STORE / "blobs" / ("sha256-" + MODELS[m])) != MODELS[m]:
            raise ValueError("Teacher mismatch")
    path = PUB / "comparison.json"
    # The upstream helper hashes confirmation.json. Bind the additional old test as well.
    if path.exists():
        rc = read(path)
        if rc.get("previous_sha256") != sha(LAST / "confirmation.json"):
            raise ValueError("Previous source mismatch")
    else:
        rc = compare(
            rows, actors(), ("stories", "exam", "mlqa-en", "mlqa-zh"), WORK, PUB
        )
        rc["previous_sha256"] = sha(LAST / "confirmation.json")
        dump(path, rc)
    if (
        rc["cohort"] != sha(WORK / "confirmation.json")
        or rc["vision_weights"] != MODELS[VLM]
    ):
        raise ValueError("Comparator mismatch")
    review(rows, rc, ARMS, PUB)


if __name__ == "__main__":
    configure_cuda_cli()
    p = argparse.ArgumentParser()
    p.add_argument("actions", nargs="+", choices=("evaluate", "judge"))
    for action in p.parse_args().actions:
        (evaluate if action == "evaluate" else judgment)()
