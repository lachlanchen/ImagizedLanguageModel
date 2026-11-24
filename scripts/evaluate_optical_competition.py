"""Measure actual image answers, set dependence and human-reference semantic credit."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from time import monotonic
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.optical_competition import shortlist
from scripts.evaluate_directional_reading import bitmap, measure, summary
from scripts.evaluate_fact_channel import observed
from scripts.evaluate_visual_discourse import packed, unpack
from scripts.evaluate_sentence_context import long_scores
from scripts.evaluate_region_writer import offpage, changed
from scripts.prepare_optical_competition import (
    WORK,
    PUB,
    read,
    dump,
    sha,
    MODEL,
    FIFTH,
    FIFTH_PUB,
)
from scripts.prepare_fact_channel import MODEL as BASE, Ink
from scripts.write_optical_competition import CompetitionAnswer, DirectionalAnswer
from scripts.train_region_writer import gpu_owner, pack, join_regions
from scripts.measure_writer_energy import snapshot, consumed
from scripts.review_directional_reading import (
    compare,
    review,
    MODELS,
    STORE,
    VLM,
    JUDGE,
)
from scripts.transfer_fact_channel import main as transfer
from scripts.verse_gpu_runtime import configure_cuda_cli

ARMS = ("signed", "independent", "competing")
TASKS = ("exam", "stories", "mlqa-en", "mlqa-zh", "drop")


def actors():
    return {
        arm: (
            lambda m=arm: DirectionalAnswer("signed", "cuda")
            if m == "signed"
            else CompetitionAnswer(m, "cuda")
        )
        for arm in ARMS
    }


def fixed():
    f = WORK / "confirmation.json"
    if sha(f) != read(PUB / "reservation.json")["files"][f.name]:
        raise ValueError("Unsealed test")
    return read(f)


def evaluate():
    rows = fixed()
    long_file = FIFTH / "long.json"
    if sha(long_file) != read(FIFTH_PUB / "corpus.json")["files"][long_file.name]:
        raise ValueError("Altered long-form test")
    long = read(long_file)
    dialogue, _, normalize = offpage()
    development = read(FIFTH / "confirmation.json")
    edits = changed(development)
    stats = {}
    outcomes = {}
    for mode, make in actors().items():
        ink = Ink(np.load(BASE / "palette.npy"))
        with gpu_owner(), torch.inference_mode():
            net = make()
            meter = snapshot()
            start = monotonic()
            raw = observed(net, rows, ink)
            torch.cuda.synchronize()
            seconds = monotonic() - start
            energy = consumed(meter, snapshot())["joules"]
            books = observed(net, long, ink)
            spoken = observed(net, dialogue, ink)
            prior = observed(net, development, ink)
            altered = observed(net, edits, ink)
        packed(
            WORK / (mode + ".json.gz"),
            dict(
                original=raw, long=books, dialogue=spoken, prior=prior, edited=altered
            ),
        )
        scores = measure(rows, raw, ink)
        outcomes[mode] = scores
        other = measure(rows, [raw[i ^ 1] for i in range(len(raw))], ink)
        orig = measure(development, prior, ink)
        new = measure(edits, altered, ink)
        paired = defaultdict(Counter)
        for e, r in zip(edits, new, strict=True):
            k = orig[e["original"]]
            paired[e["task"]].update(
                n=1, both_exact=r["exact"] * k["exact"], both_fact=r["fact"] * k["fact"]
            )
        stats[mode] = dict(
            free=summary(rows, scores),
            question_exchange=summary(rows, other),
            long=long_scores(long, books, ink),
            development=summary(development, orig),
            edits=paired,
            dialogue=sum(
                o["stop"]
                and normalize(o["text"]) in {normalize(v) for v in r["answers"]}
                for r, o in zip(dialogue, spoken, strict=True)
            ),
            dialogue_total=len(spoken),
            joules=energy,
            seconds=seconds,
        )
        print(mode, stats[mode]["free"], flush=True)
        dump(PUB / "scores.json", stats)
    intervals = {}
    for comp in ARMS[:2]:
        groups = defaultdict(dict)
        for task in sorted({r["task"] for r in rows}):
            ix = [i for i, r in enumerate(rows) if r["task"] == task]
            for k in ("fact", "exact", "mentions"):
                delta = np.array(
                    [outcomes["competing"][i][k] - outcomes[comp][i][k] for i in ix]
                )
                u = delta.reshape(-1, 2).mean(1)
                bound = (
                    np.quantile(
                        np.random.default_rng(48213).choice(u, (5000, len(u))).mean(1),
                        [0.025, 0.975],
                    )
                    * 100
                )
                groups[task][k] = dict(
                    gained=int((delta > 0).sum()),
                    lost=int((delta < 0).sum()),
                    interval_pp=bound.tolist(),
                )
        intervals[comp] = groups
    dump(PUB / "intervals.json", intervals)
    transfer(actors(), PUB, MODEL)


@torch.inference_mode()
def intervene():
    rows = fixed()
    ink = Ink(np.load(BASE / "palette.npy"))
    qs = [ink.encode(r["q"]) for r in rows]
    ps = [join_regions([ink.encode(b) for b in r["blocks"]]) for r in rows]
    targets = [
        {
            bitmap(ink, v)
            for v in (
                [r["facts"]["answer"]] if r["task"] == "statements" else r["answers"]
            )
        }
        for r in rows
    ]
    tiles = ink.array()
    result = {}
    for mode in ARMS[1:]:
        with gpu_owner():
            net = actors()[mode]()
            features = torch.from_numpy(observations({"tiles": tiles}, "ink")).cuda()
            saved = unpack(WORK / (mode + ".json.gz"))["original"]
            counts = defaultdict(Counter)
            max_drift = 0.0
            for i in range(0, len(rows), 12):
                im, base, s, l = shortlist(
                    net.reader,
                    features,
                    pack(qs[i : i + 12], "cuda"),
                    pack(ps[i : i + 12], "cuda"),
                )
                head = net.ranker.residual
                b = net.ranker.parent(im, base, l)
                full = head(im, b, l)
                alone = head(im, b, l, within=True)
                rev = head(im.flip(1), b.flip(1), l.flip(1)).flip(1)
                mask = base.isfinite()
                max_drift = max(max_drift, float((rev - full)[mask].abs().max()))
                for j, k in enumerate(range(i, i + len(s))):
                    a, prior = int(full[j].argmax()), int(alone[j].argmax())
                    items = [
                        tiles[np.asarray(ps[k][x : x + y], int)].tobytes()
                        for x, y in zip(s[j].tolist(), l[j].tolist())
                    ]
                    if sha256(items[a]).hexdigest() != saved[k]["fact_sha256"]:
                        raise ValueError("Cannot reproduce original selected image")
                    g = counts[rows[k]["task"]]
                    hit = items[a] in targets[k]
                    old = items[prior] in targets[k]
                    valid = mask[j]
                    gap = float(torch.topk(alone[j], 2).values.diff().abs().item())
                    # Use exact float64 differences of the observed FP32 logits.
                    f = full[j].double()
                    u = alone[j].double()
                    d = f - u
                    certified = gap > 2 * float(d[valid].abs().max())
                    if a != prior:
                        assert d[a] - d[prior] >= u[prior] - u[a]
                    if certified:
                        assert a == prior
                    g.update(
                        n=1,
                        full_correct=hit,
                        own_correct=old,
                        changed=items[a] != items[prior],
                        gained=hit and not old,
                        lost=old and not hit,
                        certified=certified,
                        available=any(
                            t in targets[k] for t, v in zip(items, valid.tolist()) if v
                        ),
                    )
        if mode == "independent" and any(g["changed"] for g in counts.values()):
            raise ValueError("Independent model depends on rivals")
        if max_drift > 3e-4:
            raise ValueError(f"Candidate order dependence: {max_drift}")
        result[mode] = dict(counts=counts, permutation_max_abs=max_drift)
    dump(PUB / "intervention.json", result)


def judgment():
    rows = fixed()
    for m in (JUDGE, VLM):
        if sha(STORE / "blobs" / ("sha256-" + MODELS[m])) != MODELS[m]:
            raise ValueError("External model mismatch")
    path = PUB / "comparison.json"
    c = read(path) if path.exists() else compare(rows, actors(), TASKS, WORK, PUB)
    if (
        c["cohort"] != sha(WORK / "confirmation.json")
        or c["vision_weights"] != MODELS[VLM]
    ):
        raise ValueError("Comparison mismatch")
    review(rows, c, ARMS, PUB)


if __name__ == "__main__":
    configure_cuda_cli()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "actions", nargs="+", choices=("evaluate", "intervene", "judge")
    )
    for action in parser.parse_args().actions:
        {"evaluate": evaluate, "intervene": intervene, "judge": judgment}[action]()
