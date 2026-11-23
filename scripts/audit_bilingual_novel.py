"""Locate reading errors using original answer cells, positions and stop flags."""

import json
from collections import Counter

from scripts.evaluate_bilingual_novel_writer import corpus
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.train_bilingual_novel_writer import PUBLIC, WORK, match_page
from scripts.train_causal_glyph_writer import in_cells, record, sha


def substring(page, reply):
    """Only cell-aligned starts are visible source positions."""
    if not reply:
        return False
    i = page.find(reply)
    while i >= 0:
        if i % 128 == 0:
            return True
        i = page.find(reply, i + 1)
    return False


def first_divergence(case, reply, stopped):
    """Follow any still-compatible reference until the first incompatible action."""
    candidates = [
        list(zip(in_cells(s), in_cells(c), strict=True))
        for s, c in zip(case.source, case.canonical, strict=True)
    ]
    output = in_cells(reply)
    for i, cell in enumerate(output):
        live = [ref for ref in candidates if i < len(ref) and cell in ref[i]]
        if not live:
            if all(len(ref) == i for ref in candidates):
                return "continued_past_answer"
            return "wrong_first_glyph" if i == 0 else "wrong_later_glyph"
        candidates = live
    if not stopped:
        return "no_stop"
    if any(len(ref) == len(output) for ref in candidates):
        return "exact"
    return "early_stop"


def characterize(cases, replies, stops):
    counts = Counter()
    for i, (c, y, stop) in enumerate(zip(cases, replies, stops, strict=True)):
        if c.grade(y, stop)[2]:
            continue
        kind = first_divergence(c, y, stop)
        assert kind != "exact"
        counts["first_failure_" + kind] += 1
        counts["miss"] += 1
        if not stop:
            counts["unstopped"] += 1
        if not y:
            counts["empty"] += 1
            continue
        if substring(c.p, y):
            counts["quoted_span"] += 1
        if any(
            len(y) != len(ref) and (substring(y, ref) or substring(ref, y))
            for ref in c.source
        ):
            counts["contains_reference_or_fragment"] += 1
        if cases[i ^ 1].grade(y, stop)[2]:
            counts["other_questions_exact_answer"] += 1
    return dict(counts)


def unseen_book_cases(cases):
    """Screen exact question strings and annotated answer strings seen in book fitting."""
    queries = {"en": set(), "zh": set()}
    answers = set()
    titles = set()

    def title(q):
        return q["id"].split("_", 1)[1].rsplit("_", 3)[0]

    for line in (WORK / "train.jsonl").read_text().splitlines():
        for q, _ in match_page(json.loads(line)["questions"]):
            titles.add(title(q))
            answers.add(q["refs"][0])
            for lang in queries:
                queries[lang].add(q[lang].strip())
    valid = {
        q["id"]: q
        for line in (WORK / "valid.jsonl").read_text().splitlines()
        for q in json.loads(line)["questions"]
    }
    selected = [valid[c.question_id] for c in cases]
    return dict(
        new_title=[title(q) not in titles for q in selected],
        both_questions_unseen=[
            all(q[lang].strip() not in known for lang, known in queries.items())
            for q in selected
        ],
        all_answers_unseen=[answers.isdisjoint(q["refs"]) for q in selected],
    )


def audit():
    all_cases = corpus()
    result = json.loads((PUBLIC / "result.json").read_text())
    unseen = unseen_book_cases(all_cases["book_zh"])
    stats = {}
    for arm, scores in result["scores"].items():
        group = {}
        stats[arm] = group
        for name in ("book", "xquad_regular", "xquad_hana"):
            by_lang = {}
            for lang in ("en", "zh"):
                k = name + "_" + lang
                path = WORK / "evaluation" / arm / (k + ".npz")
                assert sha(path) == scores[k]["sha256"]
                ys, ends = load_streams(path)
                rows = all_cases[k]
                measures = characterize(rows, ys, ends)
                assert measures.get("miss", 0) + scores[k]["exact"] == len(rows)
                physical = list(zip(ys, ends, strict=True))
                by_lang[lang] = physical
                measures["same_reply_to_distinct_questions"] = sum(
                    a == b for a, b in zip(physical[::2], physical[1::2], strict=True)
                )
                group[k] = measures
                if name == "book":
                    for subset, mask in unseen.items():
                        counts = [
                            c.grade(y, stop)[2]
                            for c, (y, stop), use in zip(
                                rows, physical, mask, strict=True
                            )
                            if use
                        ]
                        measures[subset] = dict(n=len(counts), exact=sum(counts))
            group[name + "_same_between_languages"] = sum(
                a == b for a, b in zip(by_lang["en"], by_lang["zh"], strict=True)
            )
    record(PUBLIC / "error_audit.json", stats)


if __name__ == "__main__":
    audit()
