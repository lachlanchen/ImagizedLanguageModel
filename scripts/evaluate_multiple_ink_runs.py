"""Compare unconstrained visual lists, single runs, question exchange and changed ink."""

from collections import Counter
from hashlib import sha256
import numpy as np
import torch

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.multiple_ink_runs import intervals
from scripts.prepare_multiple_ink_runs import (
    PUB,
    WORK,
    dump,
    read,
    sha,
    multi,
    normalize,
)
from scripts.evaluate_visual_discourse import packed
from scripts.train_multiple_ink_runs import load
from scripts.train_paired_evidence import gpu_owner
from scripts.write_multiple_ink_runs import forward


def grade(parts, refs):
    p = Counter(normalize(w) for w in parts)
    scored = []
    for ref in refs:
        r = Counter({normalize(w): 1 for w in ref})
        overlap = sum((p & r).values())
        n = sum(r.values())
        m = sum(p.values())
        scored.append((p == r, 2 * overlap / (n + m) if n + m else 1.0, overlap, m, n))
    exact, f1, tp, m, n = max(scored)
    return dict(exact=exact, f1=f1, matched=tp, emitted=m, expected=n)


def edit(r):
    ref = r["refs"][0]
    if len(ref) < 2 or not all(w in r["p"] for w in ref):
        return
    for target in ref:
        if not 2 <= len(target) <= 16 or target in r["q"] or r["p"].count(target) != 1:
            continue
        if any(target != w and (target in w or w in target) for w in ref):
            continue
        # Require that at least one occurrence of every other fact survives the edit.
        i = r["p"].index(target)
        j = i + len(target)
        mark = "".join(chr(0xE200 + n) for n in range(j - i))
        text = r["p"][:i] + mark + r["p"][j:]
        if any(w != target and w not in text for w in ref):
            continue
        return i, j, text, [[mark if w == target else w for w in ref]]


def unseen(tiles):
    seen = {bytes(x) for x in tiles}
    result = []
    rng = np.random.default_rng(317093)
    while len(result) < 16:
        # Asymmetric marks built from random strokes within a 24-pixel frame.
        from PIL import Image, ImageDraw

        im = Image.new("1", (32, 32))
        d = ImageDraw.Draw(im)
        for _ in range(4):
            d.line(tuple(map(int, rng.integers(4, 28, 4))), fill=1, width=2)
        a = np.packbits(np.asarray(im)).astype(np.uint8)
        if a.tobytes() not in seen:
            seen.add(a.tobytes())
            result.append(a)
    return np.stack(result)


def corpus():
    meta = read(PUB / "corpus.json")
    for f, h in meta["files"].items():
        if sha(WORK / f) != h:
            raise ValueError("Changed evaluation data")
    with np.load(WORK / "training.npz") as z:
        tiles = z["tiles"]
    rows = read(WORK / "held.json")
    return rows, tiles


def aggregate(items):
    result = {}
    for lang, plural in (("en", True), ("zh", True), ("zh", False)):
        group = [r for r in items if r["lang"] == lang and r["multi"] == plural]
        if not group:
            continue
        tp = sum(r["matched"] for r in group)
        m = sum(r["emitted"] for r in group)
        n = sum(r["expected"] for r in group)
        result[lang + ("-multi" if plural else "-single")] = dict(
            pages=len(group),
            exact=sum(r["exact"] for r in group),
            visible=sum(r["visible"] for r in group),
            exact_and_visible=sum(r["exact"] and r["visible"] for r in group),
            extra=m - tp,
            missing=n - tp,
            matched=tp,
            part_precision=tp / max(m, 1),
            part_recall=tp / max(n, 1),
        )
    return result


@torch.inference_mode()
def main(survival=False, confirm=False):
    rows, tiles = corpus()
    if confirm:
        spec = read(PUB / "confirmation-split.json")
        assert (
            sha(WORK / "confirm.json") == spec["sha256"]
            and sha(WORK / "confirm-tiles.npy") == spec["tiles"]
        )
        rows = read(WORK / "confirm.json")
        tiles = np.concatenate((tiles, np.load(WORK / "confirm-tiles.npy")))
    prefix = ("survival-" if survival else "") + ("confirm-" if confirm else "")
    offset = len(tiles)
    tiles = np.concatenate((tiles, unseen(tiles)))
    variants = []
    swaps = {}
    for lang, plural in (("en", True), ("zh", True), ("zh", False)):
        g = sorted(
            [r for r in rows if r["lang"] == lang and multi(r) == plural],
            key=lambda r: (len(r["qi"]), r["id"]),
        )
        for i, r in enumerate(g):
            swaps[r["id"]] = g[(i + 1) % len(g)]
    for r in rows:
        variants.append((r, "original", r["qi"], r["pi"], r["p"], r["refs"]))
        variants.append(
            (r, "exchanged", swaps[r["id"]]["qi"], r["pi"], r["p"], r["refs"])
        )
        e = edit(r)
        if e:
            i, j, text, refs = e
            variants.append(
                (
                    r,
                    "novel",
                    r["qi"],
                    r["pi"][:i] + list(range(offset, offset + j - i)) + r["pi"][j:],
                    text,
                    refs,
                )
            )
    detail = []
    with gpu_owner():
        model = load("cuda", survival)
        bank = model.ink(torch.from_numpy(observations({"tiles": tiles}, "ink")).cuda())
        for i in range(0, len(variants), 12):
            group = variants[i : i + 12]
            roles, counts = forward(
                model, bank, [x[2] for x in group], [x[3] for x in group], survival
            )
            for j, (r, condition, q, p, text, refs) in enumerate(group):
                for mode in ("count", "single", "free"):
                    spans = intervals(roles[j, : len(p)], counts[j], mode)
                    parts = [text[a:b] for a, b in spans]
                    detail.append(
                        dict(
                            id=r["id"],
                            lang=r["lang"],
                            multi=multi(r),
                            condition=condition,
                            mode=mode,
                            spans=spans,
                            parts=parts,
                            ink=[
                                sha256(tiles[p[a:b]].tobytes()).hexdigest()
                                for a, b in spans
                            ],
                            visible=any(all(w in text for w in alt) for alt in refs),
                            primary_exact=grade(parts, refs[:1])["exact"],
                            **grade(parts, refs),
                        )
                    )
    scores = {}
    recount = {}
    for mode in ("count", "single", "free"):
        scores[mode] = {
            condition: aggregate(
                [d for d in detail if d["mode"] == mode and d["condition"] == condition]
            )
            for condition in ("original", "exchanged", "novel")
        }
        orig = {
            r["id"]: r
            for r in detail
            if r["condition"] == "original" and r["mode"] == mode
        }
        for lang in ("en", "zh"):
            edited = [
                r
                for r in detail
                if r["lang"] == lang and r["mode"] == mode and r["condition"] == "novel"
            ]
            recount[mode + "/" + lang] = dict(
                n=len(edited),
                both=sum(r["exact"] and orig[r["id"]]["primary_exact"] for r in edited),
            )
    packed(PUB / (prefix + "answers.json.gz"), detail)
    dump(
        PUB / (prefix + "scores.json"),
        dict(
            weights=read(PUB / ("survival.json" if survival else "fit.json"))[
                "weights"
            ],
            outputs=sha(PUB / (prefix + "answers.json.gz")),
            scores=scores,
            edit_pairs=recount,
        ),
    )
    print(scores["count"], recount, flush=True)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--survival", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    main(a.survival, a.confirm)
