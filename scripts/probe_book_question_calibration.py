"""Test a pictured-question calibration on archived mistakes from four earlier works."""

import json
from collections import Counter

import torch
from ilm.visual_lm.pointing_causal_writer import PointingCausalWriter
from scripts.audit_book_sequence_probability import score_batch
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.mine_stopped_book_replies import PUBLIC, PARENT
from scripts.pointing_glyph_corpus import Row
from scripts.reserve_first_glyph_books import reserved
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_first_glyph_book import WORK as PRIOR, PUBLIC as PRIOR_PUBLIC
from scripts.train_pointing_causal_writer import inputs
from scripts.train_span_choice import gpu_owner


@torch.inference_mode()
def probe():
    scores = json.loads((PRIOR_PUBLIC / "result.json").read_text())["scores"][
        "ordinary"
    ]
    original = json.loads((PRIOR_PUBLIC / "sequence_probability.json").read_text())
    assert original["arm"] == "ordinary"
    weights = json.loads((PARENT.parent / "page_fit.json").read_text())["weights"]
    assert all(sha(PARENT / n) == h for n, h in weights.items())
    writer = PointingCausalWriter.restore(PARENT, "cuda")
    rows, requests = [], []
    for name, cases in reserved().items():
        if name == "fresh_erased":
            continue
        path = PRIOR / "evaluation/ordinary" / (name + ".npz")
        assert sha(path) == scores[name]["sha256"]
        ys, ends = load_streams(path)
        for i, (case, y, stop) in enumerate(zip(cases, ys, ends, strict=True)):
            if not stop:
                continue
            refs = list(dict.fromkeys(case.source))
            sibling = list(dict.fromkeys(cases[i ^ 1].source))
            answers = list(dict.fromkeys([*refs, *sibling, y]))
            start = len(rows)
            for answer in answers:
                for question in (case.q, bytes(128)):
                    rows.append(
                        Row("score", case.question_id, case.p, question, answer, b"")
                    )
            requests.append(
                (
                    case.question_id,
                    name,
                    start,
                    answers,
                    refs,
                    sibling,
                    y,
                    bool(case.grade(y, stop)[2]),
                )
            )
    read, alphabet, pairs, targets = inputs(writer.writer, rows)
    values = []
    for i in range(0, len(rows), 8):
        v, _ = score_batch(writer, read, alphabet, pairs[i : i + 8], targets[i : i + 8])
        values.extend(v)
    counts = Counter()
    details = []
    for id_, name, start, answers, refs, sibling, y, exact in requests:
        score = {
            a: values[start + 2 * j : start + 2 * j + 2] for j, a in enumerate(answers)
        }

        def compare(weight):
            value = {
                a: actual + weight * (actual - blank)
                for a, (actual, blank) in score.items()
            }
            correct = max(value[a] for a in refs)
            return correct - value[y], correct - max(value[a] for a in sibling)

        original_gap, orig_sibling = compare(0)
        changed_gap, changed_sibling = compare(1)
        if exact:
            counts["exact"] += 1
            counts["exact_over_sibling"] += orig_sibling > 1e-5
            counts["adjusted_exact_over_sibling"] += changed_sibling > 1e-5
        elif original_gap < -1e-5:
            counts["original_strict_loss"] += 1
            counts["reversed"] += changed_gap > 1e-5
        details.append(
            dict(
                id=id_,
                group=name,
                exact=exact,
                original=original_gap,
                adjusted=changed_gap,
                sibling=orig_sibling,
                adjusted_sibling=changed_sibling,
            )
        )
    assert counts["original_strict_loss"] == original["counts"]["favors_emitted"] == 94
    assert counts["exact"] == 30
    record(
        PUBLIC / "question_calibration.json",
        dict(
            counts=dict(counts),
            requests=details,
            eligible=counts["reversed"] >= 47
            and counts["adjusted_exact_over_sibling"] >= 29,
        ),
    )


if __name__ == "__main__":
    with gpu_owner():
        probe()
