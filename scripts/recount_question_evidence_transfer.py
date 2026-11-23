"""Pair each exact result with its controls and resample whole MLQA pages."""

from collections import defaultdict
import json
import numpy as np
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.evaluate_question_evidence_head import mlqa
from scripts.reserve_mlqa_pictures import PUBLIC, WORK
from scripts.train_causal_glyph_writer import record, sha


def recount():
    sets, _ = mlqa()
    source = json.loads((PUBLIC / "result.json").read_text())["scores"]
    rows = {}
    for lang, items in sets.items():
        flags = {}
        for arm in ("parent", "matched", "exchanged"):
            path = WORK / f"mlqa_{lang}_{arm}.npz"
            assert sha(path) == source[arm]["mlqa"][lang]["sha256"]
            y, stop = load_streams(path)
            flags[arm] = np.array(
                [
                    bool(c.grade(s, e)[2])
                    for c, s, e in zip(items, y, stop, strict=True)
                ],
                int,
            )
            assert int(flags[arm].sum()) == source[arm]["mlqa"][lang]["exact"]
        comparisons = {}
        for arm in ("exchanged", "parent"):
            diffs = flags["matched"] - flags[arm]
            blocks = defaultdict(list)
            for c, d in zip(items, diffs, strict=True):
                blocks[c.board].append(d)
            sums = np.array([sum(v) for v in blocks.values()])
            sizes = np.array([len(v) for v in blocks.values()])
            rng = np.random.default_rng(241301)
            samples = rng.integers(len(blocks), size=(10000, len(blocks)))
            freq = 100 * sums[samples].sum(1) / sizes[samples].sum(1)
            comparisons[arm] = dict(
                gained=int((diffs > 0).sum()),
                lost=int((diffs < 0).sum()),
                pages=len(blocks),
                delta_percentage_points=float(diffs.mean() * 100),
                percentile95=np.quantile(freq, [0.025, 0.975]).tolist(),
            )
        rows[lang] = comparisons
    record(PUBLIC / "paired.json", rows)


if __name__ == "__main__":
    recount()
