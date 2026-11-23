"""Build the span distillation comparison in all three manuscript editions."""

import argparse
import fcntl

import build_role_question_binding_article as preceding
from build_pictured_relation_article import build
from inspect_grounded_proposition import inspect

paper, base, previous, prior = (
    preceding.paper,
    preceding.base,
    preceding.previous,
    preceding.prior,
)
study = "fixed_action"
work = paper.ROOT / "artifacts/fixed_action_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "0582b117"
prior.SPECS[study] = (
    "Figure_117_fixed_action",
    "Fixed_action",
    "fig:fixed-action",
)
prior.configure(study, "20261124")
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
second = "grounded_ink_editor"
prior.SPECS[second] = (
    "Figure_118_grounded_ink_editor",
    "Grounded_ink_editor",
    "fig:grounded-ink-editor",
)
base.FIGURES.append(
    (prior.SPECS[second][0], second + "_caption.tex", prior.SPECS[second][2])
)
literal = base.literal_figures
base.literal_figures = lambda: literal().replace(
    "build_role_question_binding_article.py", "build_fixed_action_article.py"
)


def promote():
    prior.verify(study, "20261124")
    expected = prior.verify(second, "20261124")

    def verify(table):
        if table != expected:
            raise ValueError("Native table differs")

    paper.promote(verify, "fixed_action_article_build.json")


if __name__ == "__main__":
    actions = dict(
        baseline=previous.baseline,
        figure=lambda: prior.export(study, "20261124"),
        second_figure=lambda: prior.export(second, "20261124"),
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
