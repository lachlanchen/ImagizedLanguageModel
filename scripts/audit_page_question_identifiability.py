"""Certify training answers attainable by a table indexed only by exact pictured page."""

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import numpy as np

from ilm.visual_lm.page_competition import best_reference, spans
from scripts.prepare_page_competition import WORK, PUB, read, sha, dump
from scripts.evaluate_multiple_ink_runs import grade


def main():
    path = WORK / "training.json"
    if sha(path) != read(PUB / "corpus.json")["files"][path.name]:
        raise ValueError("Changed training questions")
    rows = read(path)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        # One injective ID per distinct packed bitmap in this training bank.
        key = sha256(np.asarray(r["pi"], dtype="<u4").tobytes()).digest()
        groups[key].append(i)
    total = 0
    multi = 0
    conflict = 0
    records = []
    for ids in groups.values():
        if len(ids) == 1:
            # Data preparation already exhibited a legal placement on this page.
            total += 1
            continue
        multi += len(ids)
        g = [rows[i] for i in ids]
        p = g[0]["pi"]
        assert all(r["pi"] == p for r in g)
        candidates = set()
        u = np.zeros((len(p), 3), np.int64)
        c = np.zeros(17, np.int64)
        for r in g:
            for ref in r["options"]:
                y, _ = best_reference(u, c, [ref])
                candidates.add(tuple(map(tuple, spans(y))))
        hits = max(
            sum(grade([r["p"][a:b] for a, b in cand], r["refs"])["exact"] for r in g)
            for cand in candidates
        )
        total += hits
        conflict += hits < len(g)
        records.append(
            dict(
                ids=[r["id"] for r in g],
                questions=len({tuple(r["qi"]) for r in g}),
                witness_hits=hits,
            )
        )
    dump(
        PUB / "page-only-capacity.json",
        dict(
            corpus=sha(path),
            questions=len(rows),
            distinct_physical_pages=len(groups),
            questions_on_shared_pages=multi,
            groups_without_common_witness=conflict,
            constructive_page_only_hits=total,
            shared_groups=records,
        ),
    )


if __name__ == "__main__":
    main()
