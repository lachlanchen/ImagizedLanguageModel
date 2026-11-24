"""Score complete copied answers and switched pronoun questions on reserved works."""

from collections import Counter, defaultdict
import gzip
import hashlib
import json
from itertools import combinations
import torch

from ilm.visual_lm.book_prefix import observations
from scripts.prepare_story_mentions import PUBLIC, LANGS, SPLITS, dump
from scripts.train_story_mentions import data, load
from scripts.train_semantic_page import gpu_owner
from scripts.write_story_mentions import location, pooled


def picture(source, span):
    return None if span is None else source[slice(*span)].tobytes()


def grade(row, source, span):
    truth = [(a, b) for a, b, e in row["names"] if e == row["entity"]]
    gold_ink = {picture(source, t) for t in truth}
    raw = picture(source, span)
    return dict(
        span=span,
        present=bool(truth),
        stopped=span is None,
        exact=span in truth if truth else span is None,
        ink=raw in gold_ink if truth else raw is None,
        bitmap_sha256=None if raw is None else hashlib.sha256(raw).hexdigest(),
    )


def metrics(results):
    out = Counter()
    for r in results:
        out["cases"] += 1
        out["correct"] += r["ink"]
        out["exact"] += r["exact"]
        out["present"] += r["present"]
        out["present_correct"] += r["ink"] and r["present"]
        out["false_null"] += r["stopped"] and r["present"]
        out["false_ink"] += not r["stopped"] and not r["present"]
    return dict(out)


def baselines(row):
    names = row["names"]
    nearest = max(names, key=lambda n: (n[1], n[0]))
    count = Counter(e for a, b, e in names)
    frequent = max(names, key=lambda n: (count[n[2]], n[1], n[0]))
    return {"near": tuple(nearest[:2]), "frequent": tuple(frequent[:2])}


def switches(rows, idxs):
    by = defaultdict(list)
    for i in idxs:
        by[rows[i]["doc"], rows[i]["offset"], rows[i]["text"]].append(i)
    found = []
    for group in by.values():
        for i, j in combinations(group, 2):
            a, b = rows[i], rows[j]
            if a["entity"] == b["entity"]:
                continue

            def alternatives(x):
                return {
                    x["text"][lo:hi] for lo, hi, e in x["names"] if e == x["entity"]
                }

            first, second = alternatives(a), alternatives(b)
            if not first or not second or first & second:
                continue
            found.append((i, j, baselines(a)["near"] == baselines(b)["near"]))
    return found


@torch.inference_mode()
def evaluate(split):
    rows, bank, (src, bounds, lengths, known, truth) = data(split, "cuda")
    output = defaultdict(list)
    for arm in ("ink", "byte"):
        reader = load(arm, "cuda")
        features = torch.as_tensor(observations(bank, arm), device="cuda")
        for first in range(0, len(rows), 32):
            ix = slice(first, first + 32)
            x = features[src[ix].long()]
            scores = reader(x, bounds[ix], lengths[ix])[0]
            scores_np = scores.cpu().numpy()
            if arm == "ink":
                erased = (
                    reader(torch.zeros_like(x), bounds[ix], lengths[ix])[0]
                    .argmax(1)
                    .tolist()
                )
            masks = torch.cat(
                (
                    known[ix].flatten(1),
                    torch.ones((len(scores), 1), device="cuda", dtype=torch.bool),
                ),
                dim=1,
            )
            restricted = scores.masked_fill(~masks, -float("inf")).argmax(1).tolist()
            cells_np = src[ix].cpu().numpy()
            for t, r in enumerate(rows[ix]):
                source = bank["tiles"][cells_np[t, : len(r["text"])]]
                size = x.shape[1]
                v = scores_np[t]
                q = r["query"][0]
                results = {
                    arm: location(v.argmax(), size),
                    arm + "-given-names": location(restricted[t], size),
                    arm + "-pooled": pooled(v, source, q),
                }
                if arm == "ink":
                    results.update(baselines(r))
                    results["erase-ink"] = location(erased[t], size)
                for name, span in results.items():
                    output[name].append(grade(r, source, span))
    summary = {}
    for lang in LANGS:
        idxs = [i for i, r in enumerate(rows) if r["lang"] == lang]
        pairs = switches(rows, idxs)
        stats = {}
        for name, dec in output.items():
            scores = metrics([dec[i] for i in idxs])
            for key, pop in (
                ("switched", pairs),
                ("same_nearest", [p for p in pairs if p[2]]),
            ):
                scores[key] = [
                    sum(dec[a]["ink"] and dec[b]["ink"] for a, b, _ in pop),
                    len(pop),
                ]
            stats[name] = scores
        works = {
            doc: {
                k: metrics([v[i] for i in idxs if rows[i]["doc"] == doc])
                for k, v in output.items()
            }
            for doc in sorted({rows[i]["doc"] for i in idxs})
        }
        summary[lang] = dict(arms=stats, works=works)
    payload = json.dumps(output, ensure_ascii=False, separators=(",", ":")).encode()
    (PUBLIC / (split + "-decisions.json.gz")).write_bytes(
        gzip.compress(payload, mtime=0)
    )
    dump(PUBLIC / (split + "-scores.json"), summary)
    print(
        split,
        {
            l: {k: (v["correct"], v["cases"]) for k, v in s["arms"].items()}
            for l, s in summary.items()
        },
        flush=True,
    )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    with gpu_owner():
        for split in SPLITS[1:]:
            evaluate(split)
