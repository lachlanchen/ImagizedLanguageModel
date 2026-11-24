"""Measure unavoidable interval errors when two queries have the same physical input."""

import gzip
import hashlib
import json
from collections import Counter, defaultdict

from scripts.prepare_chinese_card_reply import Rasterizer
from scripts.prepare_prose_fields import PUBLIC, dump
from scripts.write_prose_fields import render


def errors(counts):
    return sum(
        sum(values.values()) - max(values.values()) for values in counts.values()
    )


def main():
    with gzip.open(PUBLIC / "entity-decisions.json.gz", "rt") as handle:
        decisions = json.load(handle)
    cases = decisions["rows"]["test"]
    raster = Rasterizer()
    original = defaultdict(Counter)
    erased = defaultdict(Counter)
    scores = Counter()
    for i, case in enumerate(cases):
        (field,) = case["keys"]
        wanted = tuple(case["spans"][field])
        cells = render(case["q"], case["domain"], case["name"], case["text"], raster)
        original[hashlib.sha256(cells.tobytes()).digest()][wanted] += 1
        cells[2] = 0
        erased[hashlib.sha256(cells.tobytes()).digest()][wanted] += 1
        for arm, output in decisions["results"].items():
            guess = output["test"][i]["intervals"].get(field)
            scores[arm] += guess is not None and tuple(guess) == wanted
    if not all(sorted(c.values()) == [1, 1] for c in erased.values()):
        raise ValueError("Expected balanced, conflicting name-erased observations")
    report = dict(
        queries=len(cases),
        original_unique_pages=len(original),
        original_minimum_errors=errors(original),
        erased_unique_pages=len(erased),
        erased_minimum_errors=errors(erased),
        exact_requested_intervals=dict(scores),
    )
    dump(PUBLIC / "interval-observability.json", report)
    print(report)


if __name__ == "__main__":
    main()
