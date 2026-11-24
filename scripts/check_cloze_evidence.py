"""Independently recount held language scores and the suffix limit of two-word states."""

import json
from collections import Counter, defaultdict
import numpy as np
from scripts.audit_context_cloze import read_outputs
from scripts.evaluate_compose_dialogue import score, summarize
from scripts.prepare_compose_dialogue import kind
from scripts.prepare_context_cloze import PUB, reserved, MODEL
from scripts.prepare_dialogue_ink import dump, sha
from scripts.prepare_word_memory import PUB as MEMORY, MODEL as PARENT


def check():
    data = reserved()
    rows = data["rows"]
    results = json.loads((PUB / "results.json").read_text())
    assert results["reservation"] == sha(PUB / "reservation.json")
    cats = np.array([kind(r[5], (data["pages"][r[1]], r[3], r[4])) for r in rows])
    subsets = dict(
        all=np.arange(len(rows)),
        difficult=np.flatnonzero(cats != "ordinary"),
        assembly=np.flatnonzero(cats == "assembly"),
        spell=np.flatnonzero(cats == "spell"),
    )
    for model, res in results["models"].items():
        raw = read_outputs(model, res["outputs"])
        values = score(raw, [[r[5]] for r in rows], None, None)
        assert summarize(values, subsets)[0] == res["scores"]
        ranks = read_outputs(model + "-ranks", res["rank_outputs"])
        slots = read_outputs(model + "-slots", res["slots"])
        keep = {(i, j) for i, j, g, _ in slots if g == "absent_stored"}
        assert len(ranks) == len(keep) and keep == {
            (r["row"], r["slot"]) for r in ranks
        }
        for c, v in res["ranks"].items():
            rr = np.array([r[c] for r in ranks])
            assert v["top1"] == int((rr == 1).sum()) and np.isclose(
                (1 / rr).mean(), v["mrr"]
            )
        if "pages" in res:
            pages = res["pages"]
            d = np.array(read_outputs(model + "-page", pages["outputs"]))
            for k, col in (("local", 3), ("exchanged", 4), ("left", 5)):
                assert int((d[:, 2] == d[:, col]).sum()) == pages[k]
    streams = json.loads((PUB / "continuations.json").read_text())
    audit = {}
    previous = json.loads((MEMORY / "cost.json").read_text())
    for mode in ("ink", "byte"):
        r = read_outputs(mode + "-continuations", streams[mode]["outputs"])
        groups = defaultdict(Counter)
        outcomes = defaultdict(set)
        actual = 0
        for page, truth, generated, swapped, history in r:
            state = tuple(generated[:2])
            actual_suffix = tuple(data["pages"][page].split()[34:40])
            groups[state][actual_suffix] += 1
            outcomes[state].add(tuple(generated[2:]))
            actual += generated[2:] == truth[2:]
        assert all(len(v) == 1 for v in outcomes.values())
        upper = sum(max(c.values()) for c in groups.values())
        assert actual <= upper
        costs = {}
        for arm in ("qa", "passage"):
            m = json.loads((MODEL / mode / arm / "weights.json").read_text())
            assert m["parent"] == sha(PARENT / mode / "writer.safetensors") and m[
                "sha256"
            ] == sha(MODEL / mode / arm / "writer.safetensors")
            costs[arm] = (
                previous[mode]["measured_lineage_gpu_joules"]
                + m["fit"]["energy"]["joules"]
                + m["preparation"]["energy"]["joules"]
            )
        audit[mode] = dict(
            stories=len(r),
            distinct_two_output_states=len(groups),
            max_six_reference_suffixes=upper,
            actual_six_suffixes=actual,
            measured_ancestral_and_current_training_preparation_joules=costs,
        )
    dump(PUB / "state_risk.json", audit)
    print(audit)


if __name__ == "__main__":
    check()
