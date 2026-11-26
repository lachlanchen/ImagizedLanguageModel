"""Paired document bootstrap of the mean change in edit cost after the same ink."""

from collections import defaultdict
import numpy as np
from scripts.prepare_visual_rollins import WORK, PUB, read, dump, sha


def main():
    path = WORK / "conditional.json"
    empirical = read(PUB / "results.json")
    if sha(path) != empirical["conditional"]:
        raise ValueError("Conditional outputs changed")
    rows = read(WORK / "generated.json")["rows"]
    domains = defaultdict(lambda: defaultdict(list))
    for i, case in read(path)["cases"].items():
        row = rows[int(i)]
        domains[row["group"]][row["doc"]].append(
            case["clean-draft"]["excess"] - case["recover-draft"]["excess"]
        )
    trials = 20000
    rng = np.random.default_rng(162421)
    num, den = np.zeros(trials), np.zeros(trials)
    difference, count = 0, 0
    for docs in domains.values():
        entries = list(docs.values())
        sizes = np.array(list(map(len, entries)))
        sums = np.array(list(map(sum, entries)))
        idx = rng.integers(len(docs), size=(trials, len(docs)))
        num += sums[idx].sum(1)
        den += sizes[idx].sum(1)
        difference += int(sums.sum())
        count += int(sizes.sum())
    dump(
        PUB / "edit-effect.json",
        dict(
            requests=count,
            edit_reduction=difference,
            mean=difference / count,
            interval=np.quantile(num / den, [0.025, 0.975]).tolist(),
            replicates=trials,
            seed=162421,
            document_counts={k: len(v) for k, v in domains.items()},
            conditional=sha(path),
        ),
    )


if __name__ == "__main__":
    main()
