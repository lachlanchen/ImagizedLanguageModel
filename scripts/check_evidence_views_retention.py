"""Test held Chinese questions, factual edits, book prefixes and crossed period requests."""

import argparse
from pathlib import Path
from scripts.check_branch_ink_transfer import main as transfers
from scripts.check_open_exam_stage_order import main as stages
from scripts.evaluate_open_exam import main as exam
from scripts.evaluate_branch_ink import details
from scripts.evaluate_visual_discourse import unpack
from scripts.prepare_branch_ink import WORK as BRANCH, PUB as BRANCH_PUB
from scripts.prepare_evidence_views import MODEL, WORK, PUB
from scripts.prepare_open_exam import read, dump, sha
from scripts.train_open_exam import corpus
from scripts.replay_branch_history import group_cards


def main(root, work, report):
    _, tiles, n = corpus("held")
    path = BRANCH / "held.json"
    if sha(path) != read(BRANCH_PUB / "corpus.json")["files"]["held.json"]:
        raise ValueError("Changed Chinese cohort")
    rows = read(path)
    arms = ("whole", "highlight")
    result = {}
    for arm in arms:
        d = work / "exam" / arm
        exam([arm], root, d, d, samples=(rows, tiles, n))
        recs = unpack(d / (arm + "-responses.json.gz"))
        scores = read(d / "results.json")[arm]
        scores.update(details(rows, recs))
        hist = report / arm
        hist.mkdir(parents=True, exist_ok=True)
        stages(arm, root, hist, None)
        raw = read(hist / "stage-order.json")
        cards = group_cards(raw["rows"])
        summary = raw["summary"]
        for lang in summary:
            g = [c for (l, _), c in cards.items() if l == lang]
            summary[lang].update(
                both_original=sum(
                    c[0, 0, 0]["correct"] and c[0, 0, 1]["correct"] for c in g
                ),
                both_all_layouts=sum(all(v["correct"] for v in c.values()) for c in g),
            )
        result[arm] = dict(exam=scores, history=summary)
    dump(report / "retention.json", result)
    transfers([(a, root) for a in arms], report)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=MODEL)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    main(a.model_dir, a.output or WORK, a.output or PUB)
