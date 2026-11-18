"""Build the natural answer writing study and verify the native figure and published editions."""

import argparse
import fcntl

import build_traditional_passage_article as preceding

prior = preceding.prior

paper, base, previous = prior.paper, prior.base, prior.previous
work = paper.ROOT / "artifacts/composed_writer_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "7d202f02"
study = "composed_writer"
prior.SPECS[study] = (
    "Figure_75_composed_writer",
    "Composed_writer",
    "fig:composed-writer",
)
prior.configure(study, "20261119")
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
literal = base.literal_figures
def captions():
    source = literal().replace("build_traditional_passage_article.py", "build_composed_writer_article.py")
    # Flush the preceding full-page plates before queuing the new figure.
    head, label, tail = source.partition(r"\label{fig:composed-writer}")
    before, opening, last = head.rpartition(r"\begin{figure*}")
    if not opening or not label:
        raise ValueError("Missing final composition figure")
    return before + "\\clearpage\n" + opening + last + label + tail


base.literal_figures = captions


def promote():
    expected = prior.verify(study, "20261119")

    def verify(table):
        if table != expected:
            raise ValueError("Revision table changed")

    paper.promote(verify, "composed_writer_article_build.json")


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
