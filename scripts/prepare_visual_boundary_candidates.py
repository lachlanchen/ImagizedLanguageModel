"""Score alternative source boundaries for observed modern phrases using training ink only."""

from collections import Counter
from rapidfuzz.distance import Levenshtein

from scripts.prepare_contextual_phrase_realization import (
    OUT as TRAIN,
    ROOT,
    Rasterizer,
    clauses,
    load,
    sha,
    write_new,
)
from scripts.train_causal_glyph_translator import read_rows

OTHER = {"谢": ("道歉", "感谢"), "举": ("举起", "推荐")}


def additional(rows):
    for row in rows:
        s, t = clauses(row["source"]), clauses(row["new"])
        if len(s) != len(t):
            continue
        for x, y in zip(s, t):
            for p, choices in OTHER.items():
                if x.count(p) != 1 or any(g in x for g in choices):
                    continue
                matches = [g for g in choices if g in y]
                if len(matches) == 1 and y.count(matches[0]) == 1:
                    yield dict(
                        predicate=p,
                        source=x,
                        target=y,
                        expression=matches[0],
                        gloss=matches[0],
                        chapter=row["group"],
                    )


def lattice(source, target, i, a, b):
    candidates = []
    for left in range(min(1, i) + 1):
        for right in range(min(1, len(source) - i - 1) + 1):
            l, r = i - left, i + 1 + right
            cost = Levenshtein.distance(source[:l], target[:a]) + Levenshtein.distance(
                source[r:], target[b:]
            )
            candidates.append(dict(left=left, right=right, outer_edit_cost=cost))
    return candidates


def main():
    rows = load(TRAIN / "train.json")
    assert sha(TRAIN / "train.json") == load(TRAIN / "admission.json")["train_sha"]
    render = Rasterizer()
    seen = set()
    records = []
    types = Counter()
    ambiguous = 0
    duplicates = 0
    for raw in rows + list(additional(read_rows("train"))):
        x, y, p, w = [raw[k] for k in ("source", "target", "predicate", "expression")]
        if y.count(w) != 1 or x.count(p) != 1:
            continue
        s = tuple(map(render.tile, x))
        t = tuple(map(render.tile, y))
        if s in seen:
            duplicates += 1
            continue
        seen.add(s)
        i = x.index(p)
        a = y.index(w)
        b = a + len(w)
        options = lattice(s, t, i, a, b)
        best = min(v["outer_edit_cost"] for v in options)
        winners = [v for v in options if v["outer_edit_cost"] == best]
        ambiguous += len(winners) > 1
        types[p] += 1
        records.append(
            dict(
                source=x,
                target=y,
                predicate=p,
                sense=raw["gloss"],
                expression=w,
                chapter=raw["chapter"],
                source_index=i,
                target_slice=[a, b],
                candidates=options,
                minimum_tied=len(winners) > 1,
            )
        )
    overlap = [r for r in records if "相去" in r["source"] and r["sense"] == "距离"]
    report = dict(
        training_only=True,
        heldout_access=False,
        updates=0,
        counts=dict(types),
        duplicates=duplicates,
        ambiguous_minimum=ambiguous,
        overlap_expressions=dict(Counter(r["expression"] for r in overlap)),
        overlap_bad_composition=sum("相距离" in r["target"] for r in overlap),
        records=records,
    )
    write_new(
        ROOT / "references/evidence/visual_boundary_candidates_20261119.json", report
    )
    print(
        len(records),
        types,
        "ambiguous",
        ambiguous,
        "overlap",
        report["overlap_expressions"],
    )


if __name__ == "__main__":
    main()
