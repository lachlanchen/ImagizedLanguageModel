"""Recount every retained answer from original SQuAD annotations and output bounds."""

import gzip
import json
import runpy
from collections import Counter
from pathlib import Path
import argparse

from scripts.prepare_optical_passage_qa import PUB, SOURCES, dump, sha, signature


def audit(official):
    if (
        sha(official)
        != "f5a673dbbd173e29e9ea38f1b2091d883583b77b3a4c17144b223fb0f2f9bd09"
    ):
        raise ValueError("Unexpected official SQuAD 1.1 scorer")
    score = runpy.run_path(str(official))
    predictions = json.loads(gzip.decompress((PUB / "decisions.json.gz").read_bytes()))
    by_id = {r["id"]: r for r in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("Duplicate predictions")
    counts = Counter()
    paired = Counter()
    paired_wins = Counter()
    exchanged = Counter()
    f1_sums = Counter()
    all_training_questions = set()
    eligible_questions = []
    for index, (path, h) in enumerate(SOURCES):
        assert sha(path) == h
        for article in json.loads(path.read_text())["data"]:
            held_title = int(signature(article["title"]), 16) % 17 == 0
            group = "dev" if index else ("reserved" if held_title else "train")
            for paragraph in article["paragraphs"]:
                context = " ".join(paragraph["context"].split())
                answers = []
                for qa in paragraph["qas"]:
                    question = " ".join(qa["question"].split())
                    if group == "train":
                        all_training_questions.add(question)
                    if qa["id"] not in by_id:
                        continue
                    r = by_id.pop(qa["id"])
                    a, b = r["bounds"]
                    assert 0 <= a < b <= len(context)
                    assert (group, context[a:b], signature(context), question) == (
                        r["group"],
                        r["prediction"],
                        r["page"],
                        r["question"],
                    )
                    refs = [ref["text"] for ref in qa["answers"]]
                    values = {score["normalize_answer"](x) for x in refs}
                    pred = score["normalize_answer"](r["prediction"])
                    em = pred in values
                    f1 = max(score["f1_score"](r["prediction"], x) for x in refs)
                    assert (
                        refs == r["answers"]
                        and em == r["exact"]
                        and abs(r["f1"] - f1) < 1e-9
                    )
                    counts[group] += em
                    f1_sums[group] += f1
                    answers.append((question, pred, values))
                    eligible_questions.append((group, question))
                if answers:
                    q1, p1, a1 = answers[0]
                    second = next(
                        (
                            (p, a)
                            for q, p, a in answers[1:]
                            if q != q1 and a.isdisjoint(a1)
                        ),
                        None,
                    )
                    if second:
                        p2, a2 = second
                        paired[group] += 1
                        paired_wins[group] += p1 in a1 and p2 in a2
                        exchanged[group] += p2 in a1 and p1 in a2
    assert not by_id
    results = json.loads((PUB / "result.json").read_text())["groups"]
    for g, r in results.items():
        assert (
            r["exact"],
            r["both_exact"],
            r["both_exchanged"],
            r["paired_total"],
        ) == (counts[g], paired_wins[g], exchanged[g], paired[g])
        assert abs(f1_sums[g] / r["queries"] - r["mean_f1"]) < 1e-9
    return dict(
        scorer_sha256=sha(official),
        all_decisions_verified=len(predictions),
        evidence_sha256=sha(PUB / "decisions.json.gz"),
        exact=counts,
        both_exact=paired_wins,
        both_exchanged=exchanged,
        exact_questions_shared_with_any_original_training_article=Counter(
            g for g, q in eligible_questions if q in all_training_questions
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("official_scorer", type=Path)
    result = audit(ap.parse_args().official_scorer)
    dump(PUB / "audit.json", result)
    print(result)
