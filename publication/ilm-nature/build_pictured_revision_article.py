"""Build the pictured revision study and verify the native figure and published editions."""

import argparse
import fcntl

import build_clause_support_article as prior

paper, base, previous = prior.paper, prior.base, prior.previous
work = paper.ROOT / "artifacts/pictured_revision_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "9ff107a1"
study = "pictured_revision"
prior.SPECS[study] = (
    "Figure_60_pictured_revision",
    "Pictured_revision",
    "fig:pictured-revision",
)
prior.configure(study)
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
literal = base.literal_figures
base.literal_figures = lambda: literal().replace(
    "build_clause_support_article.py", "build_pictured_revision_article.py"
)


def promote():
    expected = prior.verify(study)

    def verify(table):
        if table != expected:
            raise ValueError("Revision table changed")

    paper.promote(verify, "pictured_revision_article_build.json")


if __name__ == "__main__":
    actions = dict(
        baseline=previous.baseline,
        figure=lambda: prior.export(study),
        build=previous.build,
        inspect=base.inspect,
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
