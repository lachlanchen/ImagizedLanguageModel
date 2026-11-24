"""Build and verify the continuous prose and referent study."""

import argparse
import fcntl
import json
import build_two_fact_figure_article as preceding
from build_pictured_relation_article import build
from inspect_grounded_proposition import inspect

paper, base, previous, prior = (
    preceding.paper,
    preceding.base,
    preceding.previous,
    preceding.prior,
)
study = "prose_fields"
work = paper.ROOT / "artifacts/prose_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "1a9ccf90"
prior.SPECS[study] = (
    "Figure_142_prose_fields",
    "Prose_fields",
    "fig:prose-fields",
)
prior.configure(study, "v1")
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
literal = base.literal_figures
base.literal_figures = lambda: literal().replace(
    "build_two_fact_figure_article.py", "build_prose_fields_article.py"
)


def figure_data():
    root = paper.SOURCE
    scores = json.loads((root / "entity-scores.json").read_text())["results"]
    s = json.loads((root / "scores.json").read_text())["groups"]["ink"]
    arms = ("base", "entity-erased", "entity-ink", "entity-byte")
    series = [
        [
            round(100 * scores[a][g]["complete"] / scores[a][g]["groups"], 1)
            for a in arms
        ]
        for g in ("test", "authored-multi")
    ]
    table = [
        ["Continuous human evidence", "Correct clauses or refusals", "Question erased"]
    ]
    for key, n, title, measure in (
        ("test-natural", "1", "One requested property", "attached"),
        ("test-natural", "2", "Two requested properties", "attached"),
        ("test-assembled", "2", "Two assembled replies", "attached"),
        ("test-missing", "1", "Missing single property", "right_refusal"),
    ):
        row = s[key]
        item = row["actual"][n]
        table.append(
            [
                title,
                f'{item[measure]} / {item["n"]}',
                f'{row["erased"][n][measure]} / {item["n"]}',
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

    paper.promote(check, "prose_fields_article_build.json")


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
