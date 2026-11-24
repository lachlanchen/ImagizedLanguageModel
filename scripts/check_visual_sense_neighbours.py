"""Test whether identical local ink can require two different generated readings."""

import argparse
from collections import defaultdict
import json
import numpy as np
from scripts.prepare_visual_word_senses import PUBLIC, WORK, BOOKS, RADIUS, key, save
from scripts.train_visual_word_senses import corpus


def prepare():
    bank = np.load(WORK / "pixels.npy")[0]
    ids = np.unique(bank, axis=0, return_inverse=True)[1]
    tx, ty = corpus("train")
    counts = defaultdict(int)
    for p, v in zip(tx, ty):
        counts[ids[p[RADIUS]], v.tobytes()] += 1
    out = {}
    for book in BOOKS:
        xs, ys = corpus(book)
        meta = json.loads((WORK / (book + ".json")).read_text())
        for radius in (1, 2):
            groups = defaultdict(lambda: defaultdict(list))
            pairs = []
            for i, (x, y) in enumerate(zip(xs, ys)):
                if counts[ids[x[RADIUS]], y.tobytes()] < 3:
                    continue
                # Retain the exact central neighbourhood, including white page boundaries.
                groups[tuple(ids[x[RADIUS - radius : RADIUS + radius + 1]])][
                    y.tobytes()
                ].append(i)
            for local, answers in sorted(groups.items()):
                for seq in answers.values():
                    seq.sort(key=lambda i: key(repr(meta[i])))
                while len(pool := [k for k, v in answers.items() if v]) >= 2:
                    a, b = sorted(pool, key=lambda g: (-len(answers[g]), g))[:2]
                    i, j = answers[a].pop(), answers[b].pop()
                    assert np.array_equal(
                        bank[xs[i, RADIUS - radius : RADIUS + radius + 1]],
                        bank[xs[j, RADIUS - radius : RADIUS + radius + 1]],
                    )
                    pairs.append([i, j])
            out[book + "/" + str(radius)] = pairs
    save(PUBLIC / "local_pairs.json", out)
    print({k: len(v) for k, v in out.items()})


def score():
    models = json.loads((WORK / "interpreted_outputs.json").read_text())
    pairs = json.loads((PUBLIC / "local_pairs.json").read_text())
    out = {}
    for group, ps in pairs.items():
        book, r = group.split("/")
        meta = json.loads((WORK / (book + ".json")).read_text())
        result = {}
        for arm, pred in models[book].items():
            good = [p for p in ps if all(pred[i] == meta[i][-1] for i in p)]
            result[arm] = dict(
                complete=len(good),
                pairs=len(ps),
                first_complete=good[0] if good else None,
            )
        out[group] = result
    save(PUBLIC / "local_result.json", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "score"))
    prepare() if ap.parse_args().action == "prepare" else score()
