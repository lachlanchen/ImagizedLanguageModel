"""Evaluate complete generated streams from each writer on the reserved evidence."""

from collections import defaultdict
from hashlib import sha256
from time import monotonic
import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
from scripts.prepare_sentence_context import (
    WORK,
    PUB,
    MODEL,
    Ink,
    read,
    dump,
    sha,
)
from scripts.write_sentence_context import SentenceAnswer
from scripts.write_fact_ranking import RankedAnswer
from scripts.train_sentence_context import ARMS
from scripts.train_region_writer import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli
from scripts.evaluate_fact_channel import observed
from scripts.evaluate_phrase_region import measurements, group_counts, intervals
from scripts.evaluate_region_writer import changed, e2e_scorer, offpage
from scripts.audit_region_writer import wording
from scripts.evaluate_visual_discourse import packed
from scripts.measure_writer_energy import snapshot, consumed


class Original(RankedAnswer):
    def __init__(self):
        super().__init__("whole", "cuda")

    def answer(self, tiles, qs, pages, supplied=None, limit=1024):
        return super().answer(tiles, qs, pages, supplied, limit)


def models():
    return {
        "original": Original,
        **{mode: lambda mode=mode: SentenceAnswer(mode, "cuda") for mode in ARMS},
    }


def long_scores(rows, answers, ink):
    groups = defaultdict(list)
    for r, a in zip(rows, answers, strict=True):
        raw = bytes.fromhex(a["ink_hex"])
        if sha256(raw).hexdigest() != a["sha256"]:
            raise ValueError("Reply hash mismatch")
        cells = [raw[k : k + 128] for k in range(0, len(raw), 128)]
        refs = [[ink.tiles[k] for k in ink.encode(s)] for s in r["answers"]]
        novel = all(
            not any(wording(s) in wording(t) for t in [r["q"], " ".join(r["blocks"])])
            for s in r["answers"]
        )
        row = dict(
            exact=bool(a["stop"] and cells in refs),
            stop=a["stop"],
            cells=len(cells),
            edit=max(Levenshtein.normalized_similarity(cells, ref) for ref in refs),
            over_budget=all(len(s) >= 1024 for s in refs),
            novel=novel,
        )
        groups[r["task"] + ("_long" if len(refs[0]) >= 256 else "_short")].append(row)
    return {
        g: dict(
            n=len(v),
            bitmap_exact=sum(x["exact"] for x in v),
            stops=sum(x["stop"] for x in v),
            mean_emitted_cells=float(np.mean([x["cells"] for x in v])),
            mean_physical_edit_similarity=float(np.mean([x["edit"] for x in v])),
            outside_budget=sum(x["over_budget"] for x in v),
            source_absent=sum(x["novel"] for x in v),
        )
        for g, v in groups.items()
    }


def main():
    spec = read(PUB / "corpus.json")
    for f in ("confirmation.json", "long.json"):
        if sha(WORK / f) != spec["files"][f]:
            raise ValueError(f)
    rows = read(WORK / "confirmation.json")
    edited = changed(rows)
    longs = read(WORK / "long.json")
    scorer = e2e_scorer()
    coqa, excluded, norm = offpage()
    order = models()
    common_bank = np.load(MODEL / "palette.npy")
    stats = {}
    metrics = {}
    costs = {}
    orig_spans = None
    with gpu_owner(), torch.inference_mode():
        for name, make in order.items():
            net = make()
            assert np.array_equal(net.palette, common_bank[: len(net.palette)])
            ink = Ink(common_bank)
            # The original cannot spell from the expansion. It can copy new visible ink.
            if name == "original":
                net.chars = read(MODEL / "readback.json")
            start = monotonic()
            meter = snapshot()
            outputs = observed(net, rows, ink)
            choices = [(o["selected"], o["offset"], o["fact_sha256"]) for o in outputs]
            if orig_spans is None:
                orig_spans = choices
            elif choices != orig_spans:
                raise ValueError("Evidence reader changed between arms")
            altered = observed(net, edited, ink)
            gold = observed(net, rows, ink, gold=True)
            long = observed(net, longs, ink)
            dialogue = observed(net, coqa, ink)
            metrics[name] = measurements(rows, outputs, ink, scorer)
            result = group_counts(
                rows, metrics[name], edited, measurements(edited, altered, ink, scorer)
            )
            result["supplied"] = group_counts(
                rows, measurements(rows, gold, ink, scorer), [], []
            )
            result["instructions"] = long_scores(longs, long, ink)
            result["dialogue"] = dict(
                tested=len(coqa),
                outside_geometry=excluded,
                exact=sum(
                    o["stop"] and norm(o["text"]) in {norm(a) for a in r["answers"]}
                    for r, o in zip(coqa, dialogue, strict=True)
                ),
            )
            packed(
                WORK / (name + ".json.gz"),
                dict(original=outputs, edited=altered, gold=gold, long=long),
            )
            torch.cuda.synchronize()
            costs[name] = dict(
                seconds=monotonic() - start,
                joules=consumed(meter, snapshot())["joules"],
            )
            stats[name] = result
            print(
                name,
                {
                    k: (
                        result[k]["bitmap_exact"],
                        result["supplied"][k]["bitmap_exact"],
                    )
                    for k in ("exam", "statements")
                },
                flush=True,
            )
        dump(PUB / "scores.json", stats)
        dump(PUB / "costs.json", costs)
        dump(
            PUB / "intervals.json",
            {
                n: intervals(rows, dict(single=metrics[n], phrase=metrics["surround"]))
                for n in ("original", "direct")
            },
        )
    # Keep whole families together in the sensitivity analysis.
    affected = set(read(PUB / "overlap.json")["affected_ids"])
    bad_families = {r["family"] for j, r in enumerate(rows) if j in affected}
    keep = [i for i, r in enumerate(rows) if r["family"] not in bad_families]
    sensitivity = {
        n: group_counts([rows[i] for i in keep], [metrics[n][i] for i in keep], [], [])
        for n in order
    }
    from scripts.evaluate_visual_discourse import unpack

    for n in order:
        long = unpack(WORK / (n + ".json.gz"))["long"]
        keep_long = [i for i in range(len(longs)) if len(rows) + i not in affected]
        sensitivity[n]["instructions"] = long_scores(
            [longs[i] for i in keep_long],
            [long[i] for i in keep_long],
            Ink(common_bank),
        )
    dump(PUB / "overlap-sensitivity.json", sensitivity)
    from scripts.transfer_fact_channel import main as transfer

    transfer(order, PUB, MODEL, palette_dir=MODEL)
    from scripts.benchmark_fact_channel import main as benchmark

    benchmark(WORK, PUB, order, palette_dir=MODEL)
    from scripts.review_phrase_region import review

    review(WORK, PUB, "", tuple(order))


if __name__ == "__main__":
    configure_cuda_cli()
    main()
