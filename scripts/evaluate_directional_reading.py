"""Score fixed models on authentic held questions, natural shifts and arbitrary ink."""

from collections import Counter, defaultdict
from hashlib import sha256
from time import monotonic

import numpy as np
import torch

from scripts.prepare_directional_reading import WORK, PUB, MODEL, read, dump, sha
from scripts.prepare_sentence_context import WORK as FIFTH, PUB as FIFTH_PUB
from scripts.evaluate_sentence_context import long_scores
from scripts.evaluate_fact_channel import observed
from scripts.evaluate_region_writer import changed, offpage
from scripts.evaluate_visual_discourse import packed
from scripts.prepare_fact_channel import MODEL as BASE, Ink
from scripts.train_region_writer import gpu_owner
from scripts.write_directional_reading import DirectionalAnswer
from scripts.write_fact_ranking import RankedAnswer
from scripts.measure_writer_energy import snapshot, consumed
from scripts.verse_gpu_runtime import configure_cuda_cli
from scripts.transfer_fact_channel import main as transfer


def models():
    return {
        "original": lambda: RankedAnswer("whole", "cuda"),
        **{
            m: lambda mode=m: DirectionalAnswer(mode, "cuda")
            for m in ("unsigned", "signed")
        },
    }


def bitmap(ink, text):
    return b"".join(ink.tiles[k] for k in ink.encode(text))


def measure(rows, outputs, ink):
    people = []
    for r, o in zip(rows, outputs, strict=True):
        fact = bytes.fromhex(o["fact_hex"])
        image = bytes.fromhex(o["ink_hex"])
        if (
            sha256(image).hexdigest() != o["sha256"]
            or sha256(fact).hexdigest() != o["fact_sha256"]
        ):
            raise ValueError("Physical output hash mismatch")
        refs = {bitmap(ink, v) for v in r["answers"]}
        annotated = (
            {bitmap(ink, r["facts"]["answer"])} if r["task"] == "statements" else refs
        )
        source = [bitmap(ink, s) for s in r["blocks"]]

        def includes(raw, part):
            return any(
                raw[i : i + len(part)] == part
                for i in range(0, len(raw) - len(part) + 1, 128)
            )

        people.append(
            dict(
                fact=int(fact in annotated),
                exact=int(o["stop"] and image in refs),
                mentions=int(o["stop"] and any(includes(image, v) for v in annotated)),
                visible=int(any(includes(b, v) for b in source for v in annotated)),
                stopped=int(o["stop"]),
            )
        )
    return people


def summary(rows, vals, paired=True):
    totals = defaultdict(Counter)
    for r, v in zip(rows, vals, strict=True):
        totals[r["task"]].update(n=1, **v)
    if paired:
        for i in range(0, len(rows), 2):
            r, s = rows[i : i + 2]
            assert r["family"] == s["family"] and r["blocks"] == s["blocks"]
            a, b = vals[i : i + 2]
            totals[r["task"]].update(
                pages=1,
                both_fact=a["fact"] * b["fact"],
                both_exact=a["exact"] * b["exact"],
                both_mentions=a["mentions"] * b["mentions"],
            )
    return totals


def evaluate():
    for root, receipt, files in (
        (WORK, PUB / "reservation.json", ["confirmation.json"]),
        (FIFTH, FIFTH_PUB / "corpus.json", ["confirmation.json", "long.json"]),
    ):
        for f in files:
            if sha(root / f) != read(receipt)["files"][f]:
                raise ValueError(f)
    fresh = read(WORK / "confirmation.json")
    earlier = read(FIFTH / "confirmation.json")
    long = read(FIFTH / "long.json")
    edits = changed(earlier)
    coqa, excluded, norm = offpage()
    metrics = {}
    paired = {}
    costs = {}
    for name, make in models().items():
        ink = Ink(np.load(BASE / "palette.npy"))
        with gpu_owner(), torch.inference_mode():
            net = make()
            torch.cuda.synchronize()
            t = monotonic()
            meter = snapshot()
            original = observed(net, fresh, ink)
            torch.cuda.synchronize()
            costs[name] = dict(
                seconds=monotonic() - t, joules=consumed(meter, snapshot())["joules"]
            )
            dev = observed(net, earlier, ink)
            intervened = observed(net, edits, ink)
            prose = observed(net, long, ink)
            dialogue = observed(net, coqa, ink)
            del net
        packed(
            WORK / (name + ".json.gz"),
            dict(
                original=original,
                development=dev,
                edited=intervened,
                long=prose,
                dialogue=dialogue,
            ),
        )

        def score(rs, ans):
            return measure(rs, ans, ink)

        natural = score(fresh, original)
        prior = score(earlier, dev)
        modified = score(edits, intervened)
        paired[name] = natural
        both = defaultdict(Counter)
        for e, s in zip(edits, modified, strict=True):
            before = prior[e["original"]]
            both[e["task"]].update(
                n=1,
                both_exact=s["exact"] * before["exact"],
                both_fact=s["fact"] * before["fact"],
            )
        metrics[name] = dict(
            confirmation=summary(fresh, natural),
            prior=summary(earlier, prior),
            interventions=both,
            long=long_scores(long, prose, ink),
            dialogue=dict(
                n=len(coqa),
                outside=excluded,
                exact=sum(
                    o["stop"] and norm(o["text"]) in {norm(v) for v in r["answers"]}
                    for r, o in zip(coqa, dialogue, strict=True)
                ),
            ),
        )
        print(name, metrics[name]["confirmation"], flush=True)
        dump(PUB / "scores.json", metrics)
    intervals = {}
    for comp in ("original", "unsigned"):
        stats = defaultdict(dict)
        for task in sorted({r["task"] for r in fresh}):
            indices = [i for i, r in enumerate(fresh) if r["task"] == task]
            for key in ("fact", "exact", "mentions"):
                delta = np.array(
                    [paired["signed"][i][key] - paired[comp][i][key] for i in indices]
                )
                groups = delta.reshape(-1, 2).mean(1)
                draws = (
                    np.random.default_rng(7813)
                    .choice(groups, (5000, len(groups)), replace=True)
                    .mean(1)
                )
                stats[task][key] = dict(
                    gained=int((delta > 0).sum()),
                    lost=int((delta < 0).sum()),
                    interval_pp=(100 * np.quantile(draws, [0.025, 0.975])).tolist(),
                )
        intervals[comp] = stats
    dump(PUB / "intervals.json", intervals)
    dump(PUB / "costs.json", costs)
    transfer(models(), PUB, MODEL)


if __name__ == "__main__":
    configure_cuda_cli()
    evaluate()
