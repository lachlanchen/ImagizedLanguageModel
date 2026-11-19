"""Build the crossed participant study and verify the native figure and published editions."""

import argparse
import fcntl

import build_visual_inference_writer_article as preceding

prior = preceding.prior

paper, base, previous = prior.paper, prior.base, prior.previous
work = paper.ROOT / "artifacts/visual_role_writer_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "bb529aac"
study = "visual_role_writer"
prior.SPECS[study] = (
    "Figure_84_visual_role_writer",
    "Visual_role_writer",
    "fig:visual-role-writer",
)
prior.configure(study, "20261119")
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
literal = base.literal_figures
base.literal_figures = lambda: literal().replace(
    "build_visual_inference_writer_article.py", "build_visual_role_writer_article.py"
)


def promote():
    expected = prior.verify(study, "20261119")

    def verify(table):
        if table != expected:
            raise ValueError("Revision table changed")

    paper.promote(verify, "visual_role_writer_article_build.json")


if __name__ == "__main__":
    actions = dict(
        baseline=previous.baseline,
        figure=lambda: prior.export(study, "20261119"),
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
