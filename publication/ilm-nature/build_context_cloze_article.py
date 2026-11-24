"""Build the missing-word prediction study and verify its measured figure."""

import argparse
import fcntl
import json
import build_word_memory_article as preceding
from build_pictured_relation_article import build
from inspect_grounded_proposition import inspect

paper, base, previous, prior = (
    preceding.paper,
    preceding.base,
    preceding.previous,
    preceding.prior,
)
study = "context_cloze"
work = paper.ROOT / "artifacts/context_cloze_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "f714ce0a"
prior.SPECS[study] = (
    "Figure_134_context_cloze",
    "Context_cloze",
    "fig:context-cloze",
)
prior.configure(study, "v1")
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
literal = base.literal_figures
base.literal_figures = lambda: literal().replace(
    "build_word_memory_article.py", "build_context_cloze_article.py"
)


def figure_data():
    result = json.loads((paper.SOURCE / "results.json").read_text())
    bids = result["models"]
    p = bids["ink-passage"]["pages"]
    counts = json.loads((paper.SOURCE / "backoff.json").read_text())["groups"]["all"]
    series = [
        [round(100 * n / p["n"], 2) for n in row]
        for row in (
            (p["local"], p["left"]),
            (counts["backoff"], counts["left_backoff"]),
            (p["unigram"], p["unigram"]),
        )
    ]
    table = [["Visual writer", "Ranked first", "Word emitted", "Complete difficult"]]
    for arm in ("parent", "qa", "passage"):
        r = bids["ink-" + arm]
        h = r["hidden"]["absent_stored"]
        table.append(
            [
                {"parent": "Parent", "qa": "QA only", "passage": "QA + passages"}[arm],
                str(r["ranks"]["original"]["top1"]) + " / " + str(h["total"]),
                str(h["correct"]) + " / " + str(h["total"]),
                str(r["scores"]["difficult"]["exact"][0])
                + " / "
                + str(r["scores"]["difficult"]["n"]),
            ]
        )
    return dict(series=series, table=table)


def verify():
    if json.loads((paper.SOURCE / "figure.json").read_text()) != figure_data():
        raise ValueError("Different measured values")
    return prior.verify(study, "v1")


def promote():
    expected = verify()

    def check(table):
        if table != expected:
            raise ValueError("Different native table")

    paper.promote(check, "context_cloze_article_build.json")


if __name__ == "__main__":
    actions = dict(
        baseline=previous.baseline,
        figure=lambda: prior.export(study, "v1"),
        build=build,
        inspect=lambda: inspect(base, verify_existing_pages=True),
        promote=promote,
        verify=verify,
        captions=lambda: (paper.HERE / "complete_continuation_figures.tex").write_text(
            base.literal_figures()
        ),
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=actions)
    work.mkdir(exist_ok=True)
    with (paper.ROOT / "artifacts/publication_cpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        actions[parser.parse_args().action]()
