"""Score the saved physical images, contrasted meanings and training-only baselines."""

from collections import Counter, defaultdict
import json
import numpy as np
from PIL import Image
from scripts.evaluate_visual_word_senses import load_replies
from scripts.prepare_visual_word_senses import (
    PUBLIC,
    WORK,
    BOOKS,
    RADIUS,
    SEED,
    save,
    sha,
)
from scripts.train_visual_word_senses import corpus


def read(path):
    return json.loads(path.read_text())


def mark_window(tiles):
    pixels = np.unpackbits(tiles, axis=-1).reshape(-1, 32, 32)
    img = np.full((37, len(pixels) * 32), 255, np.uint8)
    img[:32] = 255 * (1 - pixels.transpose(1, 0, 2).reshape(32, -1))
    img[34:36, RADIUS * 32 : (RADIUS + 1) * 32] = 0
    return Image.fromarray(img)


def strip(raw):
    if not raw:
        return Image.new("L", (1, 32), 255)
    bits = np.unpackbits(np.frombuffer(raw, np.uint8)).reshape(-1, 32, 32)
    return Image.fromarray(255 * (1 - bits.transpose(1, 0, 2).reshape(32, -1)))


def backoff_key(row):
    r = RADIUS
    return (
        tuple(row[r - 2 : r + 3]),
        tuple(row[r - 1 : r + 2]),
        tuple(row[r - 1 : r + 1]),
        tuple(row[r : r + 2]),
        (row[r],),
    )


def audit():
    train_x, _ = corpus("train")
    training = read(WORK / "train.json")
    pixels = np.load(WORK / "pixels.npy")[0]
    codes = read(WORK / "cells.json")
    reverse = {p.tobytes(): c for p, c in zip(pixels, codes) if c.isascii()}
    form = np.unique(pixels, axis=0, return_inverse=True)[1]
    stored = defaultdict(Counter)
    seen = set()
    for (_, _, _, _, _, _, g), x in zip(training, train_x, strict=True):
        seen.add(g)
        for level, k in enumerate(backoff_key(form[x])):
            stored[level, k][g] += 1

    def vote(counter):
        return min(counter, key=lambda g: (-counter[g], g)) if counter else ""

    rank = {k: vote(v) for k, v in stored.items()}
    pairs = read(PUBLIC / "pairs.json")
    aliases = read(PUBLIC / "aliases.json")
    report = {}
    parsed = {}
    examples = {}
    for book in BOOKS:
        X, Y = corpus(book)
        R = read(WORK / (book + ".json"))
        truth = np.array([r[-1] for r in R])
        pair = np.array(pairs[book])
        groups = np.array([R[i][4] for i in pair[:, 0]])
        assert len(set(pair.flatten())) == pair.size and all(
            R[i][4] == R[j][4] and R[i][-1] != R[j][-1] for i, j in pair
        )
        held = np.array([g not in seen for g in truth])
        alt = {
            arm: {} for arm in ("aligned", "shuffled", "head_only", "modal", "backoff")
        }
        oks = {}
        meanings = {}
        allraw = {}
        for arm in alt:
            malformed = 0
            length = Counter()
            out = []
            if arm in ("aligned", "shuffled", "head_only"):
                path = PUBLIC / (arm + "_" + book + ".npz")
                assert (
                    sha(path)
                    == read(PUBLIC / (arm + "_outputs.json"))["books"][book]["outputs"]
                )
                streams = load_replies(path)
                allraw[arm] = streams
                for i, (raw, end) in enumerate(streams):
                    letters = [
                        reverse.get(raw[j : j + 128], "�")
                        for j in range(0, len(raw), 128)
                    ]
                    text = "".join(letters)
                    malformed += "�" in text
                    length[len(letters)] += 1
                    out.append(text if end else "<unfinished>")
                    assert (text == truth[i] and end) == (
                        raw == pixels[Y[i, Y[i] != 0]].tobytes() and end
                    )
                stopped = sum(e for _, e in streams)
            else:
                for x in X:
                    keys = backoff_key(form[x])
                    candidates = [4] if arm == "modal" else range(5)
                    out.append(
                        next(
                            (
                                rank[level, keys[level]]
                                for level in candidates
                                if (level, keys[level]) in rank
                            ),
                            "",
                        )
                    )
                stopped = None
            a = np.array(out) == truth
            s = np.array(
                [word == g or word in aliases.get(g, []) for word, g in zip(out, truth)]
            )
            complete = a[pair].all(1)
            oks[arm] = complete
            meanings[arm] = out
            alt[arm] = dict(
                exact=int(a.sum()),
                accepted=int(s.sum()),
                cases=len(a),
                paired_exact=int(complete.sum()),
                paired_accepted=int(s[pair].all(1).sum()),
                paired_individual=int(a[pair].sum()),
                pairs=len(pair),
                stopped=stopped,
                malformed=malformed if stopped is not None else None,
                unseen_glosses=int(held.sum()),
                unseen_exact=int((a & held).sum()),
            )
        headtypes, ids = np.unique(groups, return_inverse=True)
        delta = np.bincount(ids, weights=oks["aligned"].astype(float) - oks["shuffled"])
        sizes = np.bincount(ids)
        draw = np.random.default_rng(SEED).integers(
            len(sizes), size=(10000, len(sizes))
        )
        values = delta[draw].sum(1) / sizes[draw].sum(1)
        ci = np.quantile(values, [0.025, 0.975]).tolist()
        yan = groups == "言"
        report[book] = dict(
            scores=alt,
            paired_difference=float(delta.sum() / sizes.sum()),
            headword_interval=ci,
            resampled_headwords=len(sizes),
            yan_pairs=int(yan.sum()),
            yan_complete=int((oks["aligned"] & yan).sum()),
        )
        parsed[book] = meanings
        for label, mask in (
            ("first_complete", oks["aligned"]),
            ("first_incomplete", ~oks["aligned"]),
            ("yan", oks["aligned"] & yan),
        ):
            if not mask.any():
                continue
            n = int(np.flatnonzero(mask)[0])
            example = []
            for side, i in enumerate(pair[n]):
                r = R[i]
                raw, end = allraw["aligned"][i]
                name = book + "_" + label + "_" + str(side)
                folder = PUBLIC / "examples"
                folder.mkdir(exist_ok=True)
                mark_window(pixels[X[i]]).save(folder / (name + "_page.png"))
                strip(raw).save(folder / (name + "_answer.png"))
                strip(pixels[Y[i, Y[i] != 0]].tobytes()).save(
                    folder / (name + "_reference.png")
                )
                example.append(
                    dict(
                        index=int(i),
                        source=r[1],
                        left=r[3],
                        head=r[4],
                        right=r[5],
                        target=r[-1],
                        reply=meanings["aligned"][i],
                        stopped=end,
                        shuffled=meanings["shuffled"][i],
                        file=name,
                    )
                )
            examples[book + "/" + label] = example
    ok = all(r["headword_interval"][0] > 0 for r in report.values()) and any(
        r["yan_complete"] > 0 for r in report.values()
    )
    save(PUBLIC / "result.json", dict(books=report, contextual_criterion=ok))
    save(PUBLIC / "examples.json", examples)
    save(WORK / "interpreted_outputs.json", parsed)
    print(
        json.dumps(
            dict(
                eligible=ok,
                books={
                    b: (
                        r["scores"]["aligned"]["exact"],
                        r["scores"]["aligned"]["paired_exact"],
                        r["scores"]["shuffled"]["paired_exact"],
                        r["yan_complete"],
                    )
                    for b, r in report.items()
                },
            )
        )
    )


def choices():
    rows = read(PUBLIC / "comparison_roster.json")
    reports = {}
    for arm in ("visual", "text"):
        r = read(PUBLIC / ("choices_" + arm + ".json"))
        assert r["roster"] == sha(PUBLIC / "comparison_roster.json")
        assert len(rows) == len(r["rows"]) == 128
        reports[arm] = {}
        for book in BOOKS:
            res = []
            for t, p in zip(rows, r["rows"]):
                assert (t["book"], t["index"]) == (p["book"], p["index"])
                if t["book"] == book:
                    res.append(p["choice"] == t["gold"])
            z = np.array(res)
            reports[arm][book] = dict(
                exact=int(z.sum()),
                n=len(z),
                complete=int(z.reshape(-1, 2).all(1).sum()),
                pairs=len(z) // 2,
            )
    save(PUBLIC / "comparison.json", reports)


if __name__ == "__main__":
    import sys

    choices() if sys.argv[1:] == ["choices"] else audit()
