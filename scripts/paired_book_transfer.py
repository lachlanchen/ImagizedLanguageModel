"""Estimate paired changes in complete replies by resampling held-out passages."""

from collections import defaultdict
import json

import numpy as np
from scripts.mine_stopped_book_replies import PUBLIC
from scripts.train_causal_glyph_writer import record, sha


def evaluate():
    evidence = PUBLIC / "case_scores.json"
    scores = json.loads(evidence.read_text())
    boards = {}
    for arm, groups in scores.items():
        by_page = defaultdict(list)
        for lang in ("zh", "en"):
            for v in groups["fresh_" + lang]:
                by_page[v["board"]].append((lang, v["id"], int(v["exact"])))
        assert all(len(qs) == 4 for qs in by_page.values())
        boards[arm] = dict(by_page)
    titles = json.loads((PUBLIC / "reserved_books.json").read_text())
    keys = sorted(boards["preference"])
    strata = defaultdict(list)
    for i, k in enumerate(keys):
        strata[titles[k]["title"]].append(i)
    rng = np.random.default_rng(20261124)
    samples = np.concatenate(
        [
            rng.choice(indices, size=(10000, len(indices)), replace=True)
            for indices in strata.values()
        ],
        axis=1,
    )
    result = {}
    for arm in ("ordinary", "replay"):
        changes = []
        for k in keys:
            ref, cand = boards[arm][k], boards["preference"][k]
            assert [q[:2] for q in ref] == [q[:2] for q in cand]
            changes.append(sum(q[2] for q in cand) - sum(q[2] for q in ref))
        diffs = np.asarray(changes)
        draws = diffs[samples].mean(axis=1) * 25
        result[arm] = dict(
            percentage_points=float(diffs.mean() * 25),
            interval95=np.quantile(draws, [0.025, 0.975]).tolist(),
            gained_pages=int((diffs > 0).sum()),
            lost_pages=int((diffs < 0).sum()),
            equal_pages=int((diffs == 0).sum()),
        )
    record(
        PUBLIC / "paired_transfer.json",
        dict(
            source_sha256=sha(evidence),
            resamples=10000,
            seed=20261124,
            strata="passages resampled inside each of six fixed works",
            preference_minus=result,
        ),
    )


if __name__ == "__main__":
    evaluate()
