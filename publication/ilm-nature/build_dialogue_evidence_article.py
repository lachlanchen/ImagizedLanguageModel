"""Build the human evidence and conditional pointer results."""

import argparse
import fcntl
import json
import build_dialogue_efficiency_article as preceding
from build_pictured_relation_article import build
from inspect_grounded_proposition import inspect

paper, base, previous, prior = (
    preceding.paper,
    preceding.base,
    preceding.previous,
    preceding.prior,
)
study = "dialogue_evidence"
work = paper.ROOT / "artifacts/dialogue_evidence_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "e90a565e"
prior.SPECS[study] = (
    "Figure_130_dialogue_evidence",
    "Dialogue_evidence",
    "fig:dialogue-evidence",
)
prior.configure(study, "v1")
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
literal = base.literal_figures
base.literal_figures = lambda: literal().replace(
    "build_dialogue_efficiency_article.py", "build_dialogue_evidence_article.py"
)


def figure_data():
    source = paper.ROOT / "references/evidence/dialogue_teacher_v1"
    replication = json.loads((source / "replication.json").read_text())
    series = [
        [
            round(
                100
                * replication["by_seed"][str(n)]["groups"][g]["comparison"]["delta"],
                3,
            )
            for n in replication["seeds"]
        ]
        for g in ("long", "longer")
    ]
    table = [["Teacher input order", "Difficult F1", "Any exact", "Valid quote"]]
    for label, file in [
        ("Earlier turn last", "qualification.json"),
        ("Current question last", "current_last/qualification.json"),
    ]:
        r = json.loads((source / file).read_text())["scores"]["absent"]["original"]
        table.append(
            [
                label,
                f"{100*r['f1']:.1f}%",
                f"{r['any_exact']} / {r['n']}",
                f"{r['quoted']} / {r['n']}",
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

    paper.promote(check, "dialogue_evidence_article_build.json")


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
