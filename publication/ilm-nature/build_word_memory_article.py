"""Build the visual word-image retrieval study and verify its measured figure."""

import argparse
import fcntl
import json
import build_word_canvas_article as preceding
from build_pictured_relation_article import build
from inspect_grounded_proposition import inspect

paper, base, previous, prior = (
    preceding.paper,
    preceding.base,
    preceding.previous,
    preceding.prior,
)
study = "word_memory"
work = paper.ROOT / "artifacts/word_memory_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "196289af"
prior.SPECS[study] = (
    "Figure_133_word_memory",
    "Word_memory",
    "fig:word-memory",
)
prior.configure(study, "v1")
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
literal = base.literal_figures
base.literal_figures = lambda: literal().replace(
    "build_word_canvas_article.py", "build_word_memory_article.py"
)


def figure_data():
    result = json.loads((paper.SOURCE / "results.json").read_text())
    arms = ("parent", "disabled", "memory")
    series = [
        [
            round(
                100
                * result["models"][mode + "-" + arm]["scores"]["difficult"]["f1"][0],
                2,
            )
            for arm in arms
        ]
        for mode in ("ink", "byte")
    ]
    table = [
        ["Observation and writer", "Exact / all", "Missing, stored", "Missing, other"]
    ]
    for mode in ("ink", "byte"):
        for arm in ("parent", "memory"):
            r = result["models"][mode + "-" + arm]
            h = r["hidden"]

            def count(k):
                return f"{h[k]['correct']} / {h[k]['total']}"

            table.append(
                [
                    ("Image" if mode == "ink" else "Byte") + " / " + arm,
                    str(r["scores"]["all"]["exact"][0])
                    + " / "
                    + str(r["scores"]["all"]["n"]),
                    count("absent_stored"),
                    count("absent_unstored"),
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

    paper.promote(check, "word_memory_article_build.json")


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
