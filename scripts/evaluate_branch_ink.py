"""Compare complete optical answers and the prefixes actually reached on two held cohorts."""

from collections import defaultdict
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


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    old, tiles, n = corpus("held")
    assert sha(WORK / "held.json") == read(PUB / "corpus.json")["files"]["held.json"]
    for cohort, rows in (("fresh", read(WORK / "held.json")), ("previous", old)):
        out = {}
        for arm, root in (("matched", EXAM), ("branch", MODEL)):
            d = WORK / cohort / arm
            evaluate([arm], root, d, d, samples=(rows, tiles, n))
            records = unpack(d / (arm + "-responses.json.gz"))
            scores = read(d / "results.json")[arm]
            scores.update(details(rows, records))
            out[arm] = scores
        dump(PUB / (cohort + ".json"), out)
    stages("branch", MODEL, PUB, None)
