"""Build the multi-source composition study and verify its measured figure."""

import argparse
import fcntl
import json
import build_dialogue_evidence_article as preceding
from build_pictured_relation_article import build
from inspect_grounded_proposition import inspect

paper, base, previous, prior = (
    preceding.paper,
    preceding.base,
    preceding.previous,
    preceding.prior,
)
study = "compose_dialogue"
work = paper.ROOT / "artifacts/compose_dialogue_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "beed6e06"
prior.SPECS[study] = (
    "Figure_131_compose_dialogue",
    "Compose_dialogue",
    "fig:compose-dialogue",
)
prior.configure(study, "v1")
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
literal = base.literal_figures
base.literal_figures = lambda: literal().replace(
    "build_dialogue_evidence_article.py", "build_compose_dialogue_article.py"
)


def figure_data():
    result = json.loads((paper.SOURCE / "results.json").read_text())
    arms = ["plain", "assembly", "lists"]
    series = [
        [
            result["lists"]["models"]["ink/" + arm]["scores"][subset][metric][0]
            if metric == "exact"
            else result["lists"]["models"]["ink/" + arm]["scores"][subset][metric]
            for arm in arms
        ]
        for subset, metric in [("all", "exact"), ("difficult", "grounded")]
    ]
    evidence = json.loads((paper.SOURCE / "exchange.json").read_text())
    table = [
        ["Visual training", "Original page", "Exchanged page", "Difference (points)"]
    ]
    for title, arm in zip(["Uniform", "Assembly", "Assembly + lists"], arms):
        v = evidence["models"]["ink/" + arm]["all"]
        table.append(
            [
                title,
                f"{v['original']} / {v['n']}",
                f"{v['exchanged']} / {v['n']}",
                f"{100*v['difference']['delta']:+.1f}",
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

    paper.promote(check, "compose_dialogue_article_build.json")


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
