"""Locate the first wrong output cell, retaining the distinction between choice and writing."""

from collections import Counter
import numpy as np
from scripts.prepare_open_exam import WORK, PUB, read, dump
from scripts.evaluate_visual_discourse import unpack


def counts(rows, records):
    dist, absent, surfaces = Counter(), [], Counter()
    for r in records:
        row = rows[r["id"]]
        expected = row["y"]
        actual = r["ids"] + ([1] if r["stop"] else [])
        equal = actual == expected
        if equal != r["exact"]:
            raise ValueError("Recorded stopped credit differs")
        k = next(
            (k for k, (a, b) in enumerate(zip(actual, expected)) if a != b),
            min(len(actual), len(expected)),
        )
        if equal:
            dist["exact"] += 1
        elif k == 0:
            dist["wrong_first"] += 1
        elif k < len(actual) and actual[k] == 1:
            dist["early_stop_after_correct_prefix"] += 1
        elif k < len(expected) and expected[k] == 1:
            dist["overrun_after_complete_spelling"] += 1
        else:
            dist["wrong_later_or_cap"] += 1
        if r["ranked"] and not equal:
            dist["correct_option_wrong_free"] += 1
        if r["exact"] and r["absent_multi"]:
            absent.append(r["output_sha256"])
        surfaces[tuple(r["ids"])] += 1
    return dict(
        counts=dict(dist),
        distinct_exact_absent_multi=len(set(absent)),
        exact_absent_multi=len(absent),
        distinct_outputs=len(surfaces),
        most_frequent_count=max(surfaces.values()),
    )


if __name__ == "__main__":
    rows = {r["id"]: r for r in read(WORK / "held.json")}
    result = {
        a: counts(
            rows,
            [
                r
                for r in unpack(WORK / (a + "-responses.json.gz"))
                if r["condition"] == "intact"
            ],
        )
        for a in ("parent", "exchanged", "matched")
    }
    runs = {
        a: {
            r["id"]: int(r["exact"])
            for r in unpack(WORK / (a + "-responses.json.gz"))
            if r["condition"] == "intact"
        }
        for a in ("matched", "exchanged", "parent")
    }
    docs = {}
    for r in rows.values():
        docs.setdefault(r["doc"], []).append(r["id"])
    rng = np.random.default_rng(93211)
    # Resample whole held documents within genre and cohort, preserving sibling dependence.
    strata = {}
    for keys in docs.values():
        r = rows[keys[0]]
        strata.setdefault((r["genre"], r["cohort"]), []).append(keys)
    intervals = {}
    for other in ("exchanged", "parent"):
        boot = np.zeros(20000, int)
        for members in strata.values():
            diff = np.asarray(
                [
                    sum(runs["matched"][i] - runs[other][i] for i in ids)
                    for ids in members
                ]
            )
            boot += diff[rng.integers(len(diff), size=(len(boot), len(diff)))].sum(1)
        intervals[other] = dict(
            gains=sum(runs["matched"][i] > runs[other][i] for i in rows),
            losses=sum(runs["matched"][i] < runs[other][i] for i in rows),
            percentage_point_interval=(
                100 * np.quantile(boot, [0.025, 0.975]) / len(rows)
            ).tolist(),
        )
    result["exploratory_paired_bootstrap"] = dict(
        samples=20000, seed=93211, matched_minus=intervals
    )
    dump(PUB / "answer-audit.json", result)
