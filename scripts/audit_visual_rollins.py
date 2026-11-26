"""Independently verify every frozen eight-glyph alignment and its attainable edit cost."""

from rapidfuzz.distance import Levenshtein
import numpy as np
from scripts.prepare_visual_rollins import training_rows, WORK, PUB, dump, sha


def audit():
    cells = training_rows()
    with np.load(WORK / "drafts.npz") as z:
        active, cuts, ends, costs, drafts = (
            z[n] for n in ("active", "cuts", "ends", "costs", "drafts")
        )
    count = 0
    bound = 0
    for i, t in np.argwhere(active):
        p = drafts[i, t, : np.count_nonzero(drafts[i, t])].tolist()
        y = cells[i, t, cuts[i, t] : np.count_nonzero(cells[i, t])].tolist()
        # Unit edit distance changes by at least the absolute length difference.
        options = range(min(len(y), 2 * len(p)) + 1)
        d, _, j = min(
            (Levenshtein.distance(p, y[:k]), abs(k - len(p)), k) for k in options
        )
        if (d, j) != (costs[i, t], ends[i, t]) or Levenshtein.distance(
            p + y[j:], y
        ) != d:
            raise ValueError((i, t))
        count += 1
        bound += d
    dump(
        PUB / "alignment-audit.json",
        dict(
            cases=count,
            total_inevitable_edits=bound,
            drafts=sha(WORK / "drafts.npz"),
            all_endpoints_match=True,
            all_completions_attain_bound=True,
        ),
    )


if __name__ == "__main__":
    audit()
