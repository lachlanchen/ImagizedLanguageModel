"""Build the masked visual answer study and verify its measured figure."""

import argparse
import fcntl
import json
import build_compose_dialogue_article as preceding
from build_pictured_relation_article import build
from inspect_grounded_proposition import inspect

paper, base, previous, prior = (
    preceding.paper,
    preceding.base,
    preceding.previous,
    preceding.prior,
)
study = "word_canvas"
work = paper.ROOT / "artifacts/word_canvas_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "d7929b6d"
prior.SPECS[study] = (
    "Figure_132_word_canvas",
    "Word_canvas",
    "fig:word-canvas",
)
prior.configure(study, "v1")
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
literal = base.literal_figures
base.literal_figures = lambda: literal().replace(
    "build_compose_dialogue_article.py", "build_word_canvas_article.py"
)


def figure_data():
    result = json.loads((paper.SOURCE / "results.json").read_text())
    arms = ["parent", "causal", "masked"]
    series = [
        [
            round(100 * result["models"]["ink-" + arm]["scores"][subset]["f1"][0], 2)
            for arm in arms
        ]
        for subset in ("all", "difficult")
    ]
    table = [["Writer", "Top 1", "Top 8", "Top 32", "Questions"]]
    for mode in ("ink", "byte"):
        for arm in ("parent", "masked"):
            d = result["search"][mode + "-" + arm]["scores"]["assembly"]["original"]
            table.append(
                [
                    ("Image" if mode == "ink" else "Byte")
                    + " / "
                    + ("copy search" if arm == "parent" else "masked"),
                    str(d["top1"]),
                    str(d["top8"]),
                    str(d["top32"]),
                    str(d["n"]),
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

    paper.promote(check, "word_canvas_article_build.json")


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
