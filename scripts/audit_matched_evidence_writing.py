"""Locate failures after the reader supplies an already correct physical excerpt."""

from collections import Counter
import json
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.evaluate_question_evidence_head import mlqa
from scripts.reserve_mlqa_pictures import WORK, PUBLIC
from scripts.train_causal_glyph_writer import in_cells, record, sha


def audit():
    rows, _ = mlqa()
    source = json.loads((PUBLIC / "result.json").read_text())["scores"]["matched"][
        "mlqa"
    ]
    writer = json.loads((PUBLIC / "writer.json").read_text())["matched"]
    stats = {}
    for lang, items in rows.items():
        paths = [
            WORK / f"mlqa_{lang}_{arm}.npz" for arm in ("matched", "writer_matched")
        ]
        assert [sha(p) for p in paths] == [
            source[lang]["sha256"],
            writer[lang]["sha256"],
        ]
        selected, _ = load_streams(paths[0])
        emitted, stops = load_streams(paths[1])
        counts = Counter()
        for c, x, y, end in zip(items, selected, emitted, stops, strict=True):
            if not c.grade(x, True)[2] or c.grade(y, end)[2]:
                continue
            if not end:
                kind = "generation_limit"
            elif not y:
                kind = "empty_stop"
            elif x.startswith(y):
                kind = "stopped_on_proper_prefix"
            elif y.startswith(x):
                kind = "continued_past_excerpt"
            elif set(in_cells(y)).issubset(in_cells(x)):
                kind = "other_recombination"
            else:
                kind = "contains_ink_outside_excerpt"
            counts[kind] += 1
        assert sum(counts.values()) == writer[lang]["lost"]
        stats[lang] = counts
    record(PUBLIC / "lost_excerpts.json", stats)


if __name__ == "__main__":
    audit()
