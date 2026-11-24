"""Recount held outcomes using full stopped bitmaps and preserved English word boundaries."""

import re
import unicodedata
from hashlib import sha256
import numpy as np
from scripts.evaluate_region_writer import changed, contains_fact, e2e_scorer
from scripts.evaluate_visual_discourse import unpack
from scripts.prepare_region_writer import MODEL, WORK, PUB, Ink
from scripts.prepare_open_exam import read, dump


def wording(s):
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", s).casefold()))


def recount(row, r, ink, score):
    raw = bytes.fromhex(r["ink_hex"])
    assert len(raw) == 128 * r["cells"] and sha256(raw).hexdigest() == r["sha256"]

    def bits(s):
        return [ink.tiles[i] for i in ink.encode(s)]

    refs = [b"".join(bits(s)) for s in row["answers"]]
    source = [bits(s) for s in [row["q"], *row["blocks"]]]
    # Compare complete 128-byte cells at aligned offsets.
    glyphs = [raw[j : j + 128] for j in range(0, len(raw), 128)]

    def substring(needle, hay):
        key = lambda x: "".join(chr(ink.byink[b]) for b in x)
        return bool(needle) and key(needle) in key(hay)

    copied = any(substring(glyphs, s) for s in source)
    assert copied == r["visible_reply"]
    exact = r["stop"] and any(wording(r["text"]) == wording(s) for s in row["answers"])
    eligible = all(
        not any(substring(bits(a), s) for s in source)
        and not any(wording(a) in wording(v) for v in [row["q"], *row["blocks"]])
        for a in row["answers"]
    )
    fact = r["stop"] and (
        not any(score(r["text"], row["facts"]))
        if row["task"] == "restaurant"
        else contains_fact(r["text"], next(iter(row["facts"].values())), row["task"])
    )
    return dict(
        exact=exact,
        bitmap_exact=r["stop"] and raw in refs,
        novel=eligible,
        novel_exact=eligible and exact,
        stopped=r["stop"],
        fact=fact,
        cells=r["cells"],
        copied=copied,
    )


def main():
    rows = read(WORK / "dev-source.json")
    edits = changed(rows)
    ink = Ink(np.load(MODEL / "palette.npy"))
    score = e2e_scorer()
    summ = {}
    details = {}
    for a in ("direct", "selected", "oracle"):
        values = unpack(WORK / (a + "-predictions.json.gz"))
        base = [
            recount(r, v, ink, score)
            for r, v in zip(rows, values["original"], strict=True)
        ]
        eff = [
            recount(r, v, ink, score)
            for r, v in zip(edits, values["edited"], strict=True)
        ]
        details[a] = base
        summ[a] = {}
        for t in ("restaurant", "catalogue", "statements"):
            g = [v for r, v in zip(rows, base) if r["task"] == t]
            e = [
                (r, v, p)
                for r, v, p in zip(edits, eff, values["edited"])
                if r["task"] == t
            ]
            correct = lambda r, p: contains_fact(
                p["text"], r["new"], t
            ) and not contains_fact(p["text"], r["old"], t)
            both = sum(base[r["original"]]["fact"] and v["fact"] for r, v, _ in e)
            summ[a][t] = dict(
                n=len(g),
                **{
                    k: sum(z[k] for z in g)
                    for k in (
                        "exact",
                        "bitmap_exact",
                        "novel",
                        "novel_exact",
                        "stopped",
                        "fact",
                        "copied",
                    )
                },
                edits=len(e),
                both_fact=both,
                edit_new_without_old=sum(p["stop"] and correct(r, p) for r, v, p in e),
            )
    intervals = {}
    for t in ("restaurant", "catalogue", "statements"):
        ix = [i for i, r in enumerate(rows) if r["task"] == t]
        assert len({rows[i]["family"] for i in ix}) == len(ix)
        draws = np.random.default_rng(4261125).integers(
            0, len(ix), size=(5000, len(ix))
        )
        intervals[t] = {}
        for k in ("fact", "exact", "novel_exact"):
            diff = np.array(
                [
                    int(details["selected"][i][k]) - int(details["direct"][i][k])
                    for i in ix
                ]
            )
            intervals[t][k] = dict(
                gained=int((diff > 0).sum()),
                lost=int((diff < 0).sum()),
                diff_pp_ci95=np.percentile(
                    100 * diff[draws].mean(1), [2.5, 97.5]
                ).tolist(),
            )
    dump(PUB / "pixel-audit.json", dict(measurements=summ, paired=intervals))


if __name__ == "__main__":
    main()
