"""Compare the same 64 pages before and after exchange of the pictured question."""

from statistics import median
from math import comb

from scripts.prepare_fact_binding import PUB, dump
from scripts.evaluate_visual_discourse import unpack


def main():
    out = {}
    for arm in ("parent", "linear", "mixture"):
        detail = unpack(PUB / (arm + "-witness.json.gz"))
        rows = [
            {r["id"]: r for r in detail if r["exchanged"] == flag}
            for flag in (False, True)
        ]
        assert set(rows[0]) == set(rows[1])
        pairs = [
            [all(m > 0 for m in side[i]["margins"]) for side in rows] for i in rows[0]
        ]
        a = sum(x and not y for x, y in pairs)
        b = sum(y and not x for x, y in pairs)
        p = min(
            1.0, 2 * sum(comb(a + b, k) for k in range(min(a, b) + 1)) / 2 ** (a + b)
        )
        gaps = [r["free_over_best_supplied"] / 1000 for r in rows[0].values()]
        out[arm] = dict(
            pages=len(gaps),
            original_only=a,
            unrelated_only=b,
            both_questions=sum(x and y for x, y in pairs),
            exact_p_two_sided=p,
            strict_free_gap=sum(v > 0 for v in gaps),
            median_free_gap=median(gaps),
            free_gap_range=[min(gaps), max(gaps)],
        )
    dump(PUB / "paired-binding.json", out)


if __name__ == "__main__":
    main()
