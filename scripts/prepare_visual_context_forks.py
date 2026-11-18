"""Find cross-chapter training pairs with the same answer history and two non-copyable next glyphs."""

from collections import Counter, defaultdict
from hashlib import sha256

import numpy as np
from scripts.prepare_visual_translation_coverage import read_rows
from scripts.train_visual_translation_coverage import (
    DATA,
    ROOT,
    Rasterizer,
    sha,
    write_new,
)


def key(value):
    return sha256(str(value).encode()).digest()


def prepare():
    corpus = read_rows("train")
    render = Rasterizer()
    groups = defaultdict(list)
    candidates = {}
    with np.load(DATA / "training.npz") as z:
        x, y, xo, yo, bank = (
            z[k] for k in ("source", "target", "x_offsets", "y_offsets", "cells")
        )
    ids = {bytes(v): i for i, v in enumerate(bank)}
    for i, r in enumerate(corpus):
        source = x[xo[i] : xo[i + 1]].tolist()
        target = y[yo[i] : yo[i + 1]].tolist()
        # Text is used offline to exclude punctuation and present the mined pairs for review.
        if target != [ids[render.tile(c)] for c in r["new"]] or source != [
            ids[render.tile(c)] for c in r["source"]
        ]:
            raise ValueError(f"Corpus transcription does not match physical row {i}")
        candidates[i] = (frozenset(source), r["group"])
        for n in range(2, min(len(target), 9)):
            if target[n] in source or not ("\u3400" <= r["new"][n] <= "\u9fff"):
                continue
            if not any("\u3400" <= c <= "\u9fff" for c in r["new"][:n]):
                continue
            groups[tuple(target[:n])].append((i, target[n]))
    used = set()
    pairs = []
    for prefix, items in sorted(groups.items(), key=lambda v: (-len(v[0]), key(v[0]))):
        pool = sorted((r for r in items if r[0] not in used), key=lambda r: key(r[0]))
        for pos, (i, a) in enumerate(pool):
            if i in used:
                continue
            xs, chapter = candidates[i]
            for j, b in pool[pos + 1 :]:
                ys, other = candidates[j]
                if j in used or a == b or chapter == other or b in xs or a in ys:
                    continue
                n = len(prefix)
                left, right = corpus[i], corpus[j]
                pairs.append(
                    dict(
                        rows=[i, j],
                        chapters=[chapter, other],
                        prefix_cells=n,
                        prefix_text=left["new"][:n],
                        candidate_text=[left["new"][n], right["new"][n]],
                        prefix_pixel_sha=sha256(
                            bank[list(prefix)].tobytes()
                        ).hexdigest(),
                        candidate_pixel_sha=[
                            sha256(bank[k].tobytes()).hexdigest() for k in (a, b)
                        ],
                    )
                )
                used.update((i, j))
                break
    counts = Counter(p["prefix_cells"] for p in pairs)
    dest = ROOT / "references/evidence/context_fork_candidates_20261119.json"
    write_new(
        dest,
        dict(
            training_sha=sha(DATA / "training.npz"),
            rows=len(corpus),
            pair_count=len(pairs),
            lengths=dict(counts),
            meaning_reviewed=False,
            pairs=pairs,
        ),
    )
    print(len(pairs), counts)


if __name__ == "__main__":
    prepare()
