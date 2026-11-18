"""Apply frozen whole-answer credits and recount the physical comparisons."""

import csv
from collections import defaultdict
from scripts.evaluate_anchor_gap_writer import (
    CONTROLS,
    EVIDENCE,
    ROSTER,
    load,
    sha,
    write_new,
)


def run():
    control = load(CONTROLS)
    rows = load(ROSTER)["rows"]
    mapping = load(EVIDENCE / "mapping.json")
    packet = load(EVIDENCE / "review.json")
    path = EVIDENCE / "readings.tsv"
    with path.open() as handle:
        grades = list(csv.DictReader(handle, delimiter="\t"))
    assert len(grades) == len(mapping) == len(packet) == 24
    meaning = defaultdict(list)
    for j, (g, b, masked) in enumerate(zip(grades, mapping, packet)):
        assert int(g["packet"]) == j
        credit = (
            set()
            if g["credited_zero_based_choices"] == "-"
            else set(map(int, g["credited_zero_based_choices"].split(",")))
        )
        assert credit <= set(range(len(masked["choices"])))
        for k, idx in b["options"].items():
            if idx in credit:
                meaning[k].append(b["row"])
            else:
                meaning[k]
    gen = load(EVIDENCE / "generation.json")
    scores = {}
    for stream in gen["streams"]:
        cases = stream["per_case"]
        scores[stream["policy"]] = dict(
            totals={
                k: sum(r[k] for r in cases)
                for k in (
                    "edits",
                    "cells",
                    "stop",
                    "selected",
                    "true_source_positions",
                    "selected_true",
                )
            },
            bands=[sum(r["edits"] for r in cases if r["band"] == j) for j in range(4)],
            exact=sum(r["edits"] == 0 for r in cases),
        )
    for k, costs in control["per_case_edit_counts"].items():
        scores[k] = dict(
            totals=dict(edits=sum(costs)),
            exact=costs.count(0),
            bands=[
                sum(c for r, c in zip(rows, costs) if r["length_band"] == j)
                for j in range(4)
            ],
        )
    write_new(
        EVIDENCE / "scores.json",
        dict(
            readings_sha256=sha(path),
            mapping_sha256=sha(EVIDENCE / "mapping.json"),
            scores=scores,
            faithful_rows=dict(meaning),
            faithful_counts={k: len(v) for k, v in meaning.items()},
            reader="Single model reader, method labels hidden; one reading, no independent human adjudication.",
        ),
    )


if __name__ == "__main__":
    run()
