"""Test whether restricting a visual answer to one retrieved Chinese excerpt repairs it."""

from collections import Counter
from zipfile import ZipFile
import json
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.multiple_ink_runs import best_path
from scripts.evaluate_complete_fact_sets import setup, run_dir
from scripts.evaluate_multiple_ink_runs import grade
from scripts.evaluate_visual_discourse import packed
from scripts.prepare_complete_fact_sets import PUB
from scripts.prepare_multiple_ink_runs import ZH, string, dump, normalize
from scripts.write_multiple_ink_runs import load_reader, forward


def documents(rows):
    ids = {r["id"] for r in rows}
    result = {}
    with ZipFile(ZH) as z:
        for src in ("search", "zhidao"):
            with z.open(f"raw/devset/{src}.dev.json") as raw:
                for line in raw:
                    rec = json.loads(line)
                    i = str(rec["question_id"])
                    if i in ids:
                        result[i] = [
                            string(" ".join(d["paragraphs"]))[:300]
                            for d in rec["documents"][:4]
                        ]
    return result


def decode(util, terminal, text, offset=0):
    tags, score = best_path(util, terminal)
    spans = []
    for i, t in enumerate(tags, offset):
        if t == 1:
            spans.append([i, i + 1])
        elif t == 2:
            spans[-1][1] = i + 1
    return score, [text[a:b] for a, b in spans]


@torch.inference_mode()
def main():
    torch.set_num_threads(3)
    rows, tiles = setup()
    rows = [
        r
        for r in rows
        if r["lang"] == "zh"
        and r["visible_parts"]
        and r["visible_parts"][0]
        and 2 <= len(r["refs"][0]) <= 16
    ]
    docs = documents(rows)
    records = []
    for arm in ("control", "contrast"):
        model = load_reader("cpu", True, run_dir(arm))
        bank = model.ink(torch.from_numpy(observations({"tiles": tiles}, "ink")))
        for i in range(0, len(rows), 12):
            group = rows[i : i + 12]
            roles, counts = forward(
                model, bank, [r["qi"] for r in group], [r["pi"] for r in group], True
            )
            for role, prior, r in zip(roles, counts, group, strict=True):
                text = r["p"]
                gold = r["refs"][0]
                k = len(gold)
                assert " ".join(docs[r["id"]])[:1200] == text
                prior = np.rint(prior.astype(np.float64) * 1000).astype(np.int64)
                util = np.rint(role[: len(text)].astype(np.float64) * 1000).astype(
                    np.int64
                )
                util = util - util[:, :1]
                fixed = np.full(k + 1, -(1 << 45), np.int64)
                fixed[k] = 0
                original = decode(util, prior, text)[1]
                known = decode(util, fixed, text)[1]

                def correct(parts):
                    return grade(parts, [gold])["exact"]

                slices = []
                at = 0
                for doc in docs[r["id"]]:
                    end = min(at + len(doc), len(text))
                    if end > at:
                        block = util[at:end]
                        score, parts = decode(block, prior, text, at)
                        given = (
                            decode(block, fixed, text, at)[1] if len(block) >= k else []
                        )
                        slices.append(
                            dict(
                                score=int(score),
                                parts=parts,
                                exact=correct(parts),
                                known=correct(given),
                                visible=all(s in text[at:end] for s in gold),
                            )
                        )
                    at += len(doc) + 1
                best = max(slices, key=lambda d: d["score"])
                records.append(
                    dict(
                        arm=arm,
                        id=r["id"],
                        k=k,
                        single_document=any(d["visible"] for d in slices),
                        original=correct(original),
                        exact_count=correct(known),
                        best_document=best["exact"],
                        oracle_document=any(d["exact"] for d in slices),
                        oracle_document_and_count=any(d["known"] for d in slices),
                    )
                )
    result = {}
    for arm in ("control", "contrast"):
        g = [r for r in records if r["arm"] == arm]
        result[arm] = dict(
            n=len(g),
            **{
                field: sum(r[field] for r in g)
                for field in (
                    "single_document",
                    "original",
                    "exact_count",
                    "best_document",
                    "oracle_document",
                    "oracle_document_and_count",
                )
            },
        )
    packed(PUB / "locality-answers.json.gz", records)
    dump(PUB / "locality.json", result)
    print(result)


if __name__ == "__main__":
    main()
