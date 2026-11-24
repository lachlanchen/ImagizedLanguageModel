"""Recount original answers, and independently replay selected raw question and answer images."""

from collections import Counter
from hashlib import sha256
from math import comb
from statistics import median
import torch

from scripts.evaluate_page_competition import setup, ARMS, release
from scripts.evaluate_multiple_ink_runs import grade
from scripts.prepare_multiple_ink_runs import multi
from scripts.evaluate_visual_discourse import unpack, draw
from scripts.prepare_page_competition import PUB, MODEL, read, dump
from scripts.write_multiple_ink_runs import infer, load_reader, write
from ilm.visual_lm.portable_writer import read_image


def main():
    torch.set_num_threads(3)
    rows, tiles = setup()
    index = {r["id"]: r for r in rows}
    all_scores = {}
    samples = {}
    for arm in ARMS:
        detail = unpack(PUB / (arm + "-answers.json.gz"))
        decoded = {
            r["id"]: r
            for r in detail
            if r["condition"] == "original" and r["mode"] == "count"
        }
        swaps = {
            r["id"]: r
            for r in detail
            if r["condition"] == "exchanged" and r["mode"] == "count"
        }
        samples[arm] = decoded
        groups = {}
        for l, plural in (("en", True), ("zh", True), ("zh", False)):
            ds = [
                v
                for k, v in decoded.items()
                if index[k]["lang"] == l and multi(index[k]) == plural
            ]
            fixed = [grade(v["parts"], index[v["id"]]["refs"][:1]) for v in ds]
            gaps = [
                v["gap"]
                for v in read(PUB / (arm + "-gaps.json"))
                if v["group"] == l + ("-multi" if plural else "-single")
            ]
            groups[l + ("-multi" if plural else "-single")] = dict(
                n=len(ds),
                exact=sum(v["exact"] for v in ds),
                exchanged=sum(swaps[v["id"]]["exact"] for v in ds),
                both=sum(v["exact"] and swaps[v["id"]]["exact"] for v in ds),
                matched=sum(v["matched"] for v in fixed),
                written=sum(v["emitted"] for v in fixed),
                reference=sum(v["expected"] for v in fixed),
                output_counts=dict(sorted(Counter(v["emitted"] for v in ds).items())),
                aligned=len(gaps),
                median_gap=median(gaps),
            )
        all_scores[arm] = groups
    changes = {}
    for l, plural in (("en", True), ("zh", True), ("zh", False)):
        ids = [r["id"] for r in rows if r["lang"] == l and multi(r) == plural]
        up = sum(
            samples["competition"][i]["exact"] and not samples["control"][i]["exact"]
            for i in ids
        )
        down = sum(
            samples["control"][i]["exact"] and not samples["competition"][i]["exact"]
            for i in ids
        )
        p = min(
            1,
            2
            * sum(comb(up + down, k) for k in range(1 + min(up, down)))
            / 2 ** (up + down),
        )
        changes[l + ("-multi" if plural else "-single")] = dict(
            gained=up, lost=down, paired_two_sided=p
        )
    dump(PUB / "summary.json", dict(scores=all_scores, changes=changes))
    # A fixed linguistic example, plus the first held successes/regressions (if present).
    ids = ["186530"]
    for group in ("en", "zh"):
        for selected in ("gained", "lost"):
            for r in rows:
                i = r["id"]
                if r["lang"] != group or not multi(r):
                    continue
                wins = [samples[a][i]["exact"] for a in ("control", "competition")]
                if wins == ([False, True] if selected == "gained" else [True, False]):
                    ids.append(i)
                    break
    for arm in ARMS:
        net = load_reader("cpu", True, release(arm))
        for i in dict.fromkeys(ids):
            r = index[i]
            dest = MODEL / "examples" / i
            dest.mkdir(parents=True, exist_ok=True)
            if arm == ARMS[0]:
                draw(dest, "page", tiles[r["pi"]])
                draw(dest, "question", tiles[r["qi"]])
            q = read_image(dest / "question.png", len(r["qi"]))
            p = read_image(dest / "page.png", len(r["pi"]))
            bounds = infer(net, q, p, survival=True)
            parts = [p[a:b] for a, b in bounds]
            expected = samples[arm][i]
            if (
                bounds != expected["spans"]
                or [sha256(z.tobytes()).hexdigest() for z in parts] != expected["ink"]
            ):
                raise ValueError("Standalone PNG replay differs from GPU evaluation")
            n = write(parts, dest / arm)
            dump(
                dest / arm / "receipt.json",
                dict(
                    id=i,
                    question=r["q"],
                    question_cells=len(q),
                    page_cells=len(p),
                    parts=expected["parts"],
                    cells=n,
                    spans=bounds,
                    exact=expected["exact"],
                    kind=r["lang"],
                    reference=r["refs"][0],
                ),
            )
    dump(
        PUB / "replayed.json",
        dict(
            ids=list(dict.fromkeys(ids)),
            arms=ARMS,
            selection="One fixed Chinese language question and first gained/lost plural cases per language in reserved order.",
        ),
    )


if __name__ == "__main__":
    main()
