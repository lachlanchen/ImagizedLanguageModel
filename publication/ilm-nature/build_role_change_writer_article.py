"""Build the pictured role-change comparison in all three manuscript editions."""

import argparse
import fcntl

import build_ordered_book_replies_article as preceding
from build_pictured_relation_article import build
from inspect_grounded_proposition import inspect

paper, base, previous, prior = (
    preceding.paper,
    preceding.base,
    preceding.previous,
    preceding.prior,
)
study = "role_change_writer"
work = paper.ROOT / "artifacts/role_change_writer_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "8f518e4a"
prior.SPECS[study] = (
    "Figure_110_role_change_writer",
    "Role_change_writer",
    "fig:role-change-writer",
)
prior.configure(study, "20261124")
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
literal = base.literal_figures
base.literal_figures = lambda: literal().replace(
    "build_ordered_book_replies_article.py", "build_role_change_writer_article.py"
)


def promote():
    expected = prior.verify(study, "20261124")

    def verify(table):
        if table != expected:
            raise ValueError("Native table differs")

    paper.promote(verify, "role_change_writer_article_build.json")


if __name__ == "__main__":
    actions = dict(
        baseline=previous.baseline,
        figure=lambda: prior.export(study, "20261124"),
        build=build,
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
