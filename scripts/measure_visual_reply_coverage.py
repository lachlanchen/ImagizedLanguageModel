"""Measure which errors can be repaired by choosing among four existing replies."""

from collections import Counter
import json
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.evaluate_question_evidence_head import mlqa
from scripts.reserve_mlqa_pictures import WORK, PUBLIC
from scripts.train_causal_glyph_writer import record, sha


def measure():
    sets, _ = mlqa()
    reader = json.loads((PUBLIC / "result.json").read_text())["scores"]["matched"][
        "mlqa"
    ]
    written = json.loads((PUBLIC / "writer.json").read_text())
    out = {}
    for lang, items in sets.items():
        records = (
            ("matched", reader[lang]),
            ("writer_matched", written["matched"][lang]),
            ("writer_full", written["full"][lang]),
        )
        candidates = []
        for name, receipt in records:
            path = WORK / f"mlqa_{lang}_{name}.npz"
            assert sha(path) == receipt["sha256"]
            candidates.append(list(zip(*load_streams(path), strict=True)))
        counts = Counter()
        for c, selected, generated, full in zip(items, *candidates, strict=True):
            x = selected[0]
            stopped = (x, True) if generated[0].startswith(x) else generated
            success = [
                bool(c.grade(y, end)[2])
                for y, end in (selected, generated, full, stopped)
            ]
            counts.update(
                n=1,
                any_correct=int(any(success)),
                all_wrong=int(not any(success)),
                single_accepted_spelling=int(len(set(c.source)) == 1),
            )
        out[lang] = counts
    record(PUBLIC / "reply_coverage.json", out)


if __name__ == "__main__":
    measure()
