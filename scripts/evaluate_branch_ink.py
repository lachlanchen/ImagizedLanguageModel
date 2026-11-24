"""Compare complete optical answers and the prefixes actually reached on two held cohorts."""

import argparse
from collections import defaultdict
from pathlib import Path
from scripts.prepare_branch_ink import EXAM, MODEL, WORK, PUB
from scripts.prepare_open_exam import dump, read, sha
from scripts.train_open_exam import corpus
from scripts.evaluate_open_exam import main as evaluate
from scripts.evaluate_visual_discourse import unpack
from scripts.check_open_exam_stage_order import main as stages


def details(rows, records):
    given = {r["id"]: r for r in records if r["condition"] == "intact"}
    pairs = defaultdict(list)
    absent = set()
    for r in rows:
        if r["cohort"] == "paired":
            pairs[r["doc"]].append(r)
        g = given[r["id"]]
        if g["exact"] and g["absent_multi"]:
            absent.add(g["output_sha256"])
    both = 0
    for a, b in pairs.values():
        k = next(i for i, (g, h) in enumerate(zip(a["y"], b["y"])) if g != h)

        def reaches(r):
            out = given[r["id"]]
            cells = out["ids"] + ([1] if out["stop"] else [])
            return cells[: k + 1] == r["y"][: k + 1]

        both += reaches(a) and reaches(b)
    return dict(both_reached=both, pages=len(pairs), different_exact_absent=len(absent))


def run(model_dir, work, report):
    old, tiles, n = corpus("held")
    assert sha(WORK / "held.json") == read(PUB / "corpus.json")["files"]["held.json"]
    for cohort, rows in (("fresh", read(WORK / "held.json")), ("previous", old)):
        out = {}
        for arm, root in (("matched", EXAM), ("branch", model_dir)):
            d = work / cohort / arm
            evaluate([arm], root, d, d, samples=(rows, tiles, n))
            records = unpack(d / (arm + "-responses.json.gz"))
            scores = read(d / "results.json")[arm]
            scores.update(details(rows, records))
            out[arm] = scores
        dump(report / (cohort + ".json"), out)
    stages("branch", model_dir, report, None)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL)
    parser.add_argument(
        "--output", type=Path, help="Separate directory for all replication outputs"
    )
    args = parser.parse_args()
    configure_cuda_cli()
    report = args.output or PUB
    report.mkdir(parents=True, exist_ok=True)
    run(args.model_dir, args.output or WORK, report)
