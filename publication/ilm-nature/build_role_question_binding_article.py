"""Build the classical role-question comparison in all three manuscript editions."""

import argparse
import fcntl

import build_book_span_distillation_article as preceding
from build_pictured_relation_article import build
from inspect_grounded_proposition import inspect

paper, base, previous, prior = (
    preceding.paper,
    preceding.base,
    preceding.previous,
    preceding.prior,
)
study = "role_question_binding"
work = paper.ROOT / "artifacts/role_question_binding_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "5b284111"
prior.SPECS[study] = (
    "Figure_114_role_question_binding",
    "Role_question_binding",
    "fig:role-question-binding",
)
prior.configure(study, "20261124")
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
second_study = "classical_argument"
prior.SPECS[second_study] = (
    "Figure_115_classical_argument",
    "Classical_argument",
    "fig:classical-argument",
)
base.FIGURES.append(
    (
        prior.SPECS[second_study][0],
        second_study + "_caption.tex",
        prior.SPECS[second_study][2],
    )
)
third = "optical_role_adapter"
prior.SPECS[third] = (
    "Figure_116_optical_role_adapter",
    "Optical_role_adapter",
    "fig:optical-role-adapter",
)
base.FIGURES.append(
    (prior.SPECS[third][0], third + "_caption.tex", prior.SPECS[third][2])
)


def export_third():
    paper.WORK = work / "optical_export"
    paper.WORK.mkdir(exist_ok=True)
    try:
        prior.export(third, "20261124")
    finally:
        paper.WORK = work


literal = base.literal_figures
base.literal_figures = lambda: literal().replace(
    "build_book_span_distillation_article.py", "build_role_question_binding_article.py"
)


def promote():
    prior.verify(study, "20261124")
    prior.verify(second_study, "20261124")
    expected = prior.verify(third, "20261124")

    def verify(table):
        if table != expected:
            raise ValueError("Native table differs")

    paper.promote(verify, "role_question_binding_article_build.json")


if __name__ == "__main__":
    actions = dict(
        baseline=previous.baseline,
        figure=lambda: prior.export(study, "20261124"),
        second_figure=lambda: prior.export(second_study, "20261124"),
        third_figure=export_third,
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
