"""Build the pictured dialogue writing experiments in all three manuscript editions."""

import argparse
import fcntl
import json

import build_optical_passage_qa_article as preceding
from build_pictured_relation_article import build
from inspect_grounded_proposition import inspect

paper, base, previous, prior = (
    preceding.paper,
    preceding.base,
    preceding.previous,
    preceding.prior,
)
study = "dialogue_ink"
work = paper.ROOT / "artifacts/dialogue_ink_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "2c01ba5b"
prior.SPECS[study] = (
    "Figure_128_dialogue_ink",
    "Dialogue_ink",
    "fig:dialogue-ink",
)
prior.configure(study, "v1")
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
literal = base.literal_figures
base.literal_figures = lambda: literal().replace(
    "build_optical_passage_qa_article.py", "build_dialogue_ink_article.py"
)


def figure_data():
    root = paper.SOURCE
    outcomes = [
        json.loads((root / arm / "scores.json").read_text())
        for arm in ("plain", "word_code")
    ]
    series = []
    for result, arm in (
        (outcomes[0], "span"),
        (outcomes[0], "writer"),
        (outcomes[1], "writer"),
        (outcomes[1], "exchanged_question"),
        (outcomes[1], "exchanged_passage"),
    ):
        series.append(
            [
                round(
                    100
                    * result[g]["all"]["arms"][arm]["exact"]
                    / result[g]["all"]["n"],
                    2,
                )
                for g in ("reserved", "dev")
            ]
        )
    table = [["Evaluation split", "Turns", "Direct writer", "Word code", "Span reader"]]
    for g, name in (("reserved", "Reserved articles"), ("dev", "Development")):
        n = outcomes[0][g]["strict"]["n"]
        table.append(
            [
                name,
                str(n),
                str(outcomes[0][g]["strict"]["arms"]["writer"]["exact"]),
                str(outcomes[1][g]["strict"]["arms"]["writer"]["exact"]),
                str(outcomes[0][g]["strict"]["arms"]["span"]["exact"]),
            ]
        )
    return dict(series=series, table=table)


def verify_ink():
    if json.loads((paper.SOURCE / "figure.json").read_text()) != figure_data():
        raise ValueError("Figure differs from measured scores")
    return prior.verify(study, "v1")


def promote():
    expected = verify_ink()

    def verify(table):
        if table != expected:
            raise ValueError("Native table differs")

    paper.promote(verify, "dialogue_ink_article_build.json")


if __name__ == "__main__":
    actions = dict(
        baseline=previous.baseline,
        figure=lambda: prior.export(study, "v1"),
        build=build,
        inspect=lambda: inspect(base, verify_existing_pages=True),
        promote=promote,
        verify=verify_ink,
        captions=lambda: (paper.HERE / "complete_continuation_figures.tex").write_text(
            base.literal_figures()
        ),
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=actions)
    with (paper.ROOT / "artifacts/publication_cpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        actions[parser.parse_args().action]()
