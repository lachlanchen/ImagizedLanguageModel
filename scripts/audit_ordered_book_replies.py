"""Measure paired gains and localize failures using the original emitted bitmaps."""

from collections import Counter, defaultdict
from scripts.train_typeface_writer import Face, FACES
import json
import numpy as np

from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.prepare_ordered_book_replies import PUBLIC, WORK
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_ordered_book_replies import grades, items, positions


def audit():
    result = json.loads((PUBLIC / "result.json").read_text())["scores"]
    output = {}
    face = Face(FACES[0])
    for group in ("book", "new"):
        rows = list(items(group, "matched"))
        systems = {}
        counts = Counter()
        for arm, stats in result.items():
            dest = WORK / f"{group}_{arm}.npz"
            assert sha(dest) == stats[group]["streams_sha"]
            ys, ends = load_streams(dest)
            pages = defaultdict(list)
            for (row, case, lang, rev, facts, covered), y, end in zip(
                rows, ys, ends, strict=True
            ):
                good = grades(y, end, facts, rev)
                pages[case["key"]].append(int(good))
                if arm == "matched_matched":
                    a, b = (positions(y, facts[i]) for i in ((1, 0) if rev else (0, 1)))
                    counts["rows"] += 1
                    counts["first_present"] += bool(a)
                    counts["second_present"] += bool(b)
                    counts["both_exactly_once"] += len(a) == len(b) == 1
                    counts["both_in_banks"] += covered
                    counts["ordered_with_both_in_banks"] += good and covered
                    counts["output_is_source_substring"] += bool(
                        y and positions(face(case["page"]), y)
                    )
            systems[arm] = np.array(list(pages.values()))
            assert int(systems[arm].sum()) == sum(
                s["ordered"] for s in stats[group]["by_language"].values()
            )
        diffs = {}
        rng = np.random.default_rng(241209)
        baseline = systems["matched_matched"]
        indexes = rng.integers(len(baseline), size=(10000, len(baseline)))
        for arm, matrix in systems.items():
            if arm == "matched_matched":
                continue
            paired = (baseline - matrix).mean(axis=1)
            diffs[arm] = dict(
                gain_pp=100 * float(paired.mean()),
                page_bootstrap_95_pp=(
                    100 * np.quantile(paired[indexes].mean(axis=1), [0.025, 0.975])
                ).tolist(),
            )
        output[group] = dict(diagnosis=counts, paired_difference=diffs)
    record(PUBLIC / "audit.json", output)


if __name__ == "__main__":
    audit()
