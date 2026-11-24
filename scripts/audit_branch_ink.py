"""Recount paired gains and uncertainty directly from the emitted optical streams."""

from collections import defaultdict
import numpy as np
from scripts.prepare_branch_ink import PUB, WORK, EXAM, MODEL
from scripts.prepare_open_exam import read, dump, sha
from scripts.prepare_recurrent_evidence import MODEL as INIT
from scripts.write_visual_discourse import DEFAULT
from scripts.evaluate_visual_discourse import unpack
from scripts.train_open_exam import corpus


def compare(rows, cohort, tiles):
    docs = defaultdict(list)
    for r in rows:
        docs[r["doc"]].append(r)
    arms = {}
    for arm in ("matched", "branch"):
        rs = [
            r
            for r in unpack(WORK / cohort / arm / (arm + "-responses.json.gz"))
            if r["condition"] == "intact"
        ]
        lookup = {r["id"]: r for r in rs}
        exact = {}
        for r in rows:
            out = lookup[r["id"]]
            ink = tiles[np.asarray(out["ids"], int)]
            answer = tiles[np.asarray(r["y"][:-1], int)]
            exact[r["id"]] = out["stop"] and np.array_equal(ink, answer)
            if exact[r["id"]] != out["exact"]:
                raise ValueError("Disagreement with the physical answer")
        arms[arm] = exact
    groups = defaultdict(list)
    for g in docs.values():
        groups[g[0]["genre"], g[0]["cohort"]].append(g)
    rng = np.random.default_rng(15731)
    estimates = np.zeros((20000, 2), int)
    both = {}
    gains = losses = 0
    for key, gs in groups.items():
        diffs = []
        for g in gs:
            a, b = [[arms[arm][r["id"]] for r in g] for arm in ("matched", "branch")]
            gains += sum(v and not u for u, v in zip(a, b, strict=True))
            losses += sum(u and not v for u, v in zip(a, b, strict=True))
            change = int(all(b)) - int(all(a)) if len(g) == 2 else 0
            if len(g) == 2:
                both[g[0]["doc"]] = change
            diffs.append([sum(b) - sum(a), change])
        deltas = np.array(diffs)
        estimates += deltas[rng.integers(len(gs), size=(len(estimates), len(gs)))].sum(
            1
        )
    return dict(
        questions=len(rows),
        pairs=len(both),
        gains=gains,
        losses=losses,
        pair_gains=sum(v > 0 for v in both.values()),
        pair_losses=sum(v < 0 for v in both.values()),
        difference_percentage_points_ci95=(
            np.quantile(estimates, [0.025, 0.975], axis=0)
            * 100
            / np.asarray([len(rows), len(both)])
        ).T.tolist(),
    )


if __name__ == "__main__":
    control = read(EXAM / "matched/fit.json")
    fit = read(MODEL / "branch/fit.json")
    old, tiles, _ = corpus("held")
    assert sha(INIT / "dynamic.safetensors") == control["bindings"]["init"]
    assert sha(DEFAULT / "ink-pair.safetensors") == control["bindings"]["prior"]
    assert (
        control["bindings"]["schedule"] == fit["bindings"]["schedule"]
        and fit["presentations"] == control["presentations"]
    )
    assert sha(MODEL / "branch/writer.safetensors") == fit["weights"]
    dump(
        PUB / "paired-audit.json",
        dict(
            fresh=compare(read(WORK / "held.json"), "fresh", tiles),
            previous=compare(old, "previous", tiles),
        ),
    )
