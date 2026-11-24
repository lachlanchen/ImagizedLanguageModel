"""Construct name exchanges from original, independently authored venue descriptions."""

from collections import defaultdict
import numpy as np
from scripts.prepare_prose_fields import LIMIT


def eligible(rows):
    return [
        r
        for r in rows
        if 1 <= len(r["name"])
        and r["text"].count(r["name"]) == 1
        and len(r["text"]) < LIMIT - 2
    ]


def pairs(rows, seed, each=3):
    rows = eligible(rows)
    by = defaultdict(list)
    for i, r in enumerate(rows):
        for k in r["spans"]:
            by[r["domain"], k].append(i)
    rng = np.random.default_rng(seed)
    out = []
    seen = set()
    for i, a in enumerate(rows):
        for k, ab in a["spans"].items():
            pool = by[a["domain"], k]
            count = 0
            for j in rng.choice(pool, size=min(32, len(pool)), replace=False):
                b = rows[j]
                sig = (k, *sorted((a["text"], b["text"])))
                if (
                    sig in seen
                    or a["dialog"] == b["dialog"]
                    or a["name"] in b["text"]
                    or b["name"] in a["text"]
                ):
                    continue
                if len(a["text"]) + len(b["text"]) + 1 > LIMIT:
                    continue
                av = a["text"][slice(*ab)]
                bv = b["text"][slice(*b["spans"][k])]
                if (
                    av in b["text"]
                    or bv in a["text"]
                    or a["text"].count(av) != 1
                    or b["text"].count(bv) != 1
                ):
                    continue
                out.append((a, b, k))
                seen.add(sig)
                count += 1
                if count == each:
                    break
    return out


def page(a, b, target, k):
    sep = "" if a["text"][-1] in "。！？?!" else "。"
    shift = 0 if target is a else len(a["text"]) + len(sep)
    return dict(
        text=a["text"] + sep + b["text"],
        domain=a["domain"],
        name=target["name"],
        spans={
            f: [left + shift, h + shift] for f, (left, h) in target["spans"].items()
        },
        keys=[k],
        q=f"请问这家的{k}是多少？",
        answer="",
        card={f: target["text"][left:h] for f, (left, h) in target["spans"].items()},
    )
