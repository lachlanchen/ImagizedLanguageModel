"""Lower bounds on changes to divergent action log probabilities in literary misses."""

import json
from statistics import median
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.reserve_first_glyph_books import reserved
from scripts.train_causal_glyph_writer import in_cells, record, sha
from scripts.train_first_glyph_book import PUBLIC, WORK


def suffix_actions(a, b):
    # The first unequal action can be the stop of the shorter reply.
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    return len(a) + len(b) + 2 - 2 * common


def run():
    audit = json.loads((PUBLIC / "sequence_probability.json").read_text())
    arm = audit["arm"]
    stats = json.loads((PUBLIC / "result.json").read_text())["scores"][arm]
    cases = reserved()
    lookup = {}
    for group in ("fresh_en", "fresh_zh"):
        path = WORK / "evaluation" / arm / (group + ".npz")
        assert sha(path) == stats[group]["sha256"]
        replies, stopped = load_streams(path)
        for c, y, end in zip(cases[group], replies, stopped, strict=True):
            lookup[group, c.question_id] = c, in_cells(y), end
    records = []
    for r in audit["samples"]:
        c, y, end = lookup[r["group"], r["id"]]
        if not end or r["exact"]:
            continue
        gap = r["emitted"] - r["best_reference"]
        if gap <= 1e-5:
            continue
        longest = max(suffix_actions(in_cells(ref), y) for ref in c.source)
        records.append(
            dict(
                group=r["group"],
                id=r["id"],
                gap_nats=gap,
                max_suffix_actions=longest,
                lower_bound_nats=gap / longest,
            )
        )
    assert len(records) == audit["counts"]["favors_emitted"]
    summary = dict(
        n=len(records),
        median_gap_nats=median(r["gap_nats"] for r in records),
        median_lower_bound_nats=median(r["lower_bound_nats"] for r in records),
        largest_lower_bound_nats=max(r["lower_bound_nats"] for r in records),
    )
    record(PUBLIC / "stopped_margin.json", dict(summary=summary, rows=records))
    print(summary)


if __name__ == "__main__":
    run()
