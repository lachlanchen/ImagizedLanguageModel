"""Counterfactual for explicit quotation requests on the already generated MLQA streams."""

from collections import Counter
import json
from scripts.evaluate_question_evidence_head import mlqa
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.reserve_mlqa_pictures import WORK, PUBLIC
from scripts.train_causal_glyph_writer import sha, record


def measure():
    sets, _ = mlqa()
    reader = json.loads((PUBLIC / "result.json").read_text())["scores"]["matched"][
        "mlqa"
    ]
    writer = json.loads((PUBLIC / "writer.json").read_text())["matched"]
    out = {}
    for lang, rows in sets.items():
        paths = [
            WORK / f"mlqa_{lang}_{name}.npz" for name in ("matched", "writer_matched")
        ]
        assert [sha(f) for f in paths] == [
            reader[lang]["sha256"],
            writer[lang]["sha256"],
        ]
        quotes, _ = load_streams(paths[0])
        ys, ends = load_streams(paths[1])
        counts = Counter()
        for c, x, y, end in zip(rows, quotes, ys, ends, strict=True):
            finish = bool(x and y.startswith(x))
            z, stop = (x, True) if finish else (y, end)
            was = c.grade(y, end)[2]
            now = c.grade(z, stop)[2]
            exact = c.grade(x, True)[2]
            counts.update(
                n=1,
                intercepted=int(finish and (z, stop) != (y, end)),
                correct=int(now),
                writer_correct=int(was),
                reader_correct=int(exact),
                gained=int(now and not was),
                lost=int(was and not now),
                lost_vs_excerpt=int(exact and not now),
                repaired_vs_excerpt=int(now and not exact),
            )
        out[lang] = counts
    record(PUBLIC / "quote_prefix_replay.json", out)


if __name__ == "__main__":
    measure()
