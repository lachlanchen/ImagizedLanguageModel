"""Enumerate physical fact ranks and restrict the frozen reader to annotated locations."""

from collections import defaultdict
from hashlib import sha256
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_fact_channel import WORK, PUB, MODEL, Ink, read, dump
from scripts.train_fact_channel import load, pack, join_regions, split_span
from scripts.evaluate_visual_discourse import unpack


def locations(page, needles):
    # One Unicode scalar per local raster index gives aligned substring positions.
    s = "".join(map(chr, page))
    out = set()
    for ref in needles:
        v = "".join(map(chr, ref))
        start = s.find(v)
        while start >= 0:
            out.add((start, len(v)))
            start = s.find(v, start + 1)
    return sorted(out)


@torch.inference_mode()
def main():
    torch.set_num_threads(3)
    rows = read(WORK / "confirmation.json")
    old = unpack(WORK / "fresh-human.json.gz")["original"]
    ink = Ink(np.load(MODEL / "palette.npy"))
    reader = load("reader", "cpu")
    qs = [ink.encode(r["q"]) for r in rows]
    pages = [[ink.encode(b) for b in r["blocks"]] for r in rows]
    references = [
        [
            ink.encode(a)
            for a in (r["answers"] if r["task"] == "exam" else [r["facts"]["answer"]])
        ]
        for r in rows
    ]
    tiles = ink.array()
    features = torch.from_numpy(observations({"tiles": tiles}, "ink"))
    totals = defaultdict(lambda: defaultdict(int))
    for i, (r, q, regions, refs) in enumerate(
        zip(rows, qs, pages, references, strict=True)
    ):
        out = totals[r["task"]]
        out["n"] += 1
        pg = join_regions(regions)

        def scores(p):
            return reader(features, pack([q], "cpu"), pack([p], "cpu"))[0].numpy()

        raw = scores(pg)
        chosen = np.unravel_index(raw.argmax(), raw.shape)
        b, start, v = split_span(regions, chosen[0], chosen[1] + 1)
        out["cpu_gpu_span_agreement"] += (b, start, len(v)) == (
            old[i]["selected"],
            old[i]["offset"],
            old[i]["fact_cells"],
        )
        alias = locations(pg, refs)
        if not alias:
            continue
        out["available"] += 1
        good = np.zeros(raw.shape, bool)
        for s, n in alias:
            good[s, n - 1] = True
        rank = 1 + int((raw > raw[good].max()).sum())
        for k in (1, 8, 32):
            out["rank_at_" + str(k)] += rank <= k
        regionmask = np.zeros(len(pg), bool)
        pos = 0
        for j, reg in enumerate(regions):
            if j in r["valid"]:
                regionmask[pos : pos + len(reg)] = True
            pos += len(reg) + 1
        out["full_choice"] += bool(good[chosen])
        out["restricted_region"] += bool(
            good.flat[np.where(regionmask[:, None], raw, -np.inf).argmax()]
        )
        out["restricted_start"] += bool(
            good.flat[np.where(good.any(1)[:, None], raw, -np.inf).argmax()]
        )
        ends = np.arange(len(pg))[:, None] + np.arange(raw.shape[1])
        allowed = {s + n - 1 for s, n in alias}
        mask = np.isin(ends, list(allowed))
        out["restricted_end"] += bool(good.flat[np.where(mask, raw, -np.inf).argmax()])
        crop = regions[r["valid"][0]]
        cropped = scores(crop)
        s, k = np.unravel_index(cropped.argmax(), cropped.shape)
        out["reencoded_first_region"] += crop[s : s + k + 1] in refs
    result = {t: {k: int(v) for k, v in d.items()} for t, d in totals.items()}
    dump(PUB / "span-localization.json", result)
    print(result)


if __name__ == "__main__":
    main()
