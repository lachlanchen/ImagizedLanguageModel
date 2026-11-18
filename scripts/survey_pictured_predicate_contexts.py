"""Find paired training clauses for two readings of the same visible Chinese predicate."""

import json
import re
from collections import Counter

from scripts.align_visual_translation import ROOT, write_new
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.train_causal_glyph_translator import read_rows


PREDICATES = {"去": ("离开", "距离"), "伐": ("砍伐", "攻打")}
BOUNDARY = re.compile(r"[，、。；！？：,;!?:]+")


def clauses(text):
    return [
        clause.strip(" “”（）「」")
        for clause in BOUNDARY.split(text)
        if clause.strip(" “”（）「」")
    ]


def survey():
    rows = read_rows("train")
    raster = Rasterizer()
    records = []
    for predicate, alternatives in PREDICATES.items():
        for gloss in alternatives:
            (other,) = set(alternatives) - {gloss}
            seen = set()
            for row in rows:
                x, y = row["source"], row["new"]
                if (
                    x.count(predicate) != 1
                    or y.count(gloss) != 1
                    or other in y
                    or any(term in x for term in alternatives)
                ):
                    continue
                src, tgt = clauses(x), clauses(y)
                if len(src) != len(tgt):
                    continue
                i = next(j for j, fragment in enumerate(src) if predicate in fragment)
                if gloss not in tgt[i]:
                    continue
                key = b"".join(raster.tile(c) for c in src[i])
                if key in seen:
                    continue
                seen.add(key)
                records.append(
                    dict(
                        predicate=predicate,
                        gloss=gloss,
                        source=src[i],
                        target=tgt[i],
                        chapter=row["group"],
                        ordinal=row["ordinal"],
                    )
                )
    assert len({r["source"] for r in records}) == len(
        records
    ), "Inspect conflicting clause labels before export"
    counts = Counter((r["predicate"], r["gloss"]) for r in records)
    return dict(
        training_pairs_scanned=len(rows),
        heldout_access=False,
        training_updates=0,
        inclusion="Same ordinal clause after punctuation split, one pictured predicate and one contrasting modern expression. Candidate alignments require reading.",
        counts={
            p: {g: counts[p, g] for g in options} for p, options in PREDICATES.items()
        },
        candidates=records,
    )


if __name__ == "__main__":
    receipt = survey()
    write_new(
        ROOT
        / "references/evidence/pictured_predicate_training_candidates_20261119.json",
        receipt,
    )
    print(json.dumps(receipt["counts"], ensure_ascii=False))
