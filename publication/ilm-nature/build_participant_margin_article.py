"""Build the participant-span comparison in all three manuscript editions."""

import argparse
import fcntl

import build_pictured_relation_article as preceding
from inspect_grounded_proposition import inspect

paper, base, previous, prior = (
    preceding.paper,
    preceding.base,
    preceding.previous,
    preceding.prior,
)
study = "participant_margin"
work = paper.ROOT / "artifacts/participant_margin_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "14f2010b"
prior.SPECS[study] = (
    "Figure_95_participant_margin",
    "Participant_margin",
    "fig:participant-margin",
)
prior.configure(study, "20261123")
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
literal = base.literal_figures
base.literal_figures = lambda: literal().replace(
    "build_pictured_relation_article.py", "build_participant_margin_article.py"
)


def promote():
    expected = prior.verify(study, "20261123")

    def verify(table):
        if table != expected:
            raise ValueError("Native table differs")

    paper.promote(verify, "participant_margin_article_build.json")


if __name__ == "__main__":
    actions = dict(
        baseline=previous.baseline,
        figure=lambda: prior.export(study, "20261123"),
        build=preceding.build,
        inspect=lambda: inspect(base, verify_existing_pages=True),
        promote=promote,
        captions=lambda: (paper.HERE / "complete_continuation_figures.tex").write_text(
            base.literal_figures()
        ),
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=actions)
    with (paper.ROOT / "artifacts/publication_cpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        actions[parser.parse_args().action]()
