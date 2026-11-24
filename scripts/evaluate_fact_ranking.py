"""Score whole answers and all candidate selections in the reserved fourth cohort."""

from collections import Counter, defaultdict
from time import monotonic
import numpy as np
import torch

from scripts.evaluate_fact_channel import observed
from scripts.evaluate_phrase_region import measurements, group_counts, intervals
from scripts.evaluate_region_writer import changed, e2e_scorer, offpage
from scripts.evaluate_visual_discourse import packed
from scripts.measure_writer_energy import consumed, snapshot
from scripts.prepare_fact_channel import Ink, MODEL as BASE_MODEL, read, dump, sha
from scripts.prepare_fact_ranking_confirmation import PUB, WORK
from scripts.prepare_region_writer import WORK as OLD
from scripts.train_fact_channel import join_regions, gpu_owner
from scripts.train_fact_ranking import MODES, MODEL
from scripts.verse_gpu_runtime import configure_cuda_cli
from scripts.write_fact_channel import VisualAnswer
from scripts.write_fact_ranking import RankedAnswer


def audit_choices(rows, lists, outputs, ink):
    groups = defaultdict(Counter)
    for row, options, saved in zip(rows, lists, outputs, strict=True):
        g = groups[row["task"]]
        source = join_regions([ink.encode(s) for s in row["blocks"]])
        targets = row["answers"] if row["task"] == "exam" else [row["facts"]["answer"]]
        targets = {tuple(ink.encode(a)) for a in targets}
        strips = [tuple(source[s : s + n]) for s, n, _, _ in options]
        positive = np.array([v in targets for v in strips])
        scores = np.array([s for _, _, _, s in options])
        chosen = int(scores.argmax())
        mass = np.rint(np.exp(scores - scores.max()) * 10**9).astype(np.int64)
        accepted = int(mass[positive].sum())
        rival = int(mass[~positive].sum())
        certified = accepted > int(positive.sum()) * rival
        if certified and not positive[chosen]:
            raise ValueError("Violated integral mass bound")
        raw = b"".join(ink.tiles[k] for k in strips[chosen])
        if raw.hex() != saved["fact_hex"]:
            raise ValueError("Selected candidate differs from emitted fact")
        g.update(
            n=1,
            positions=len(options),
            duplicate_positions=len(options) - len(set(strips)),
            supported=int(positive.any()),
            selected=int(positive[chosen]),
            certified=int(certified),
            multiple_correct=int(positive.sum() > 1),
            recovered=int(positive[chosen] and not positive[0]),
            lost=int(positive[0] and not positive[chosen]),
        )
    return groups


def main():
    if sha(WORK / "confirmation.json") != read(PUB / "reservation.json")["sha256"]:
        raise ValueError("Changed confirmation")
    rows = read(WORK / "confirmation.json")
    edits = changed(rows)
    prior = [r for r in read(OLD / "dev-source.json") if r["task"] == "catalogue"]
    coqa, excluded, norm = offpage()
    scorer = e2e_scorer()
    scores = {}
    metrics = {}
    ranking = {}
    costs = {}
    candidate_base = None
    with gpu_owner(), torch.inference_mode():
        torch.set_num_threads(3)
        for mode in ("frozen", *MODES):
            net = RankedAnswer(mode, "cuda", capture=True)
            ink = Ink(net.palette)
            meter = snapshot()
            start = monotonic()
            orig = observed(net, rows, ink)
            torch.cuda.synchronize()
            costs[mode] = dict(
                seconds=monotonic() - start,
                joules=consumed(meter, snapshot())["joules"],
            )
            lists = net.lists
            ranking[mode] = audit_choices(rows, lists, orig, ink)
            options = [[(i, n, f) for i, n, f, _ in o] for o in lists]
            if candidate_base is None:
                candidate_base = options
            elif candidate_base != options:
                raise ValueError("Candidate lists changed across arms")
            net.capture = False
            altered = observed(net, edits, ink)
            m = measurements(rows, orig, ink, scorer)
            metrics[mode] = m
            scores[mode] = group_counts(
                rows, m, edits, measurements(edits, altered, ink, scorer)
            )
            catalogue = observed(net, prior, ink)
            conversation = observed(net, coqa, ink)
            scores[mode]["catalogue"] = group_counts(
                prior, measurements(prior, catalogue, ink, scorer), [], []
            )["catalogue"]
            scores[mode]["coqa"] = dict(
                n=len(coqa),
                outside_geometry=excluded,
                exact=sum(
                    o["stop"] and norm(o["text"]) in {norm(a) for a in r["answers"]}
                    for r, o in zip(coqa, conversation, strict=True)
                ),
            )
            packed(
                WORK / (mode + ".json.gz"),
                dict(original=orig, edited=altered, lists=lists),
            )
            print(mode, scores[mode], flush=True)
            if mode == "frozen":
                base = VisualAnswer("human", "cuda")
                control = observed(base, rows, ink)
                for a, b in zip(control, orig, strict=True):
                    if any(
                        a[k] != b[k]
                        for k in ("stop", "sha256", "selected", "offset", "fact_cells")
                    ):
                        raise ValueError("Original selector changed")
                gold = observed(net, rows, ink, gold=True)
                oracle = group_counts(
                    rows, measurements(rows, gold, ink, scorer), [], []
                )
                dump(PUB / "formulation.json", oracle)
        dump(PUB / "scores.json", scores)
        dump(PUB / "reader.json", ranking)
        dump(PUB / "cost.json", costs)
        dump(
            PUB / "intervals.json",
            {
                k: intervals(rows, dict(single=metrics[k], phrase=metrics["whole"]))
                for k in ("frozen", "endpoints")
            },
        )
    from scripts.transfer_fact_channel import main as transfer

    transfer(
        {k: lambda mode=k: RankedAnswer(mode, "cuda") for k in ("frozen", *MODES)},
        PUB,
        MODEL,
    )


if __name__ == "__main__":
    configure_cuda_cli()
    main()
