"""Build the source-pointing glyph study and verify the native figure and published editions."""

import argparse
import fcntl

import build_grounded_proposition_article as preceding
from inspect_grounded_proposition import inspect

prior = preceding.prior

paper, base, previous = prior.paper, prior.base, prior.previous
work = paper.ROOT / "artifacts/role_binding_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "0e5cb0a0"
study = "role_binding"
prior.SPECS[study] = (
    "Figure_93_role_binding",
    "Role_binding",
    "fig:role-binding",
)
prior.configure(study, "20261123")
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
literal = base.literal_figures
base.literal_figures = lambda: literal().replace(
    "build_grounded_proposition_article.py", "build_role_binding_article.py"
)


def promote():
    expected = prior.verify(study, "20261123")

    def verify(table):
        if table != expected:
            raise ValueError("Revision table changed")

    paper.promote(verify, "role_binding_article_build.json")


if __name__ == "__main__":
    actions = dict(
        baseline=previous.baseline,
        figure=lambda: prior.export(study, "20261123"),
        build=previous.build,
        inspect=lambda: inspect(base),
        promote=promote,
        captions=lambda: (paper.HERE / "complete_continuation_figures.tex").write_text(
            base.literal_figures()
        ),
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=actions)
    with (paper.ROOT / "artifacts/publication_cpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        actions[p.parse_args().action]()
