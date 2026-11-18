"""Build the English question and scope studies and verify the native figure and published editions."""

import argparse
import fcntl

import build_same_passage_writer_article as preceding

prior = preceding.prior

paper, base, previous = prior.paper, prior.base, prior.previous
work = paper.ROOT / "artifacts/english_request_writer_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "c4e01d39"
study = "english_request_writer"
prior.SPECS[study] = (
    "Figure_80_english_request_writer",
    "English_request_writer",
    "fig:english-request-writer",
)
prior.configure(study, "20261119")
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
literal = base.literal_figures
base.literal_figures = lambda: literal().replace(
    "build_same_passage_writer_article.py", "build_english_request_writer_article.py"
)


extra = "scoped_request_writer"
prior.SPECS[extra] = ("Figure_81_scoped_request_writer", "Scoped_request_writer", "fig:scoped-request-writer")
base.FIGURES.append((prior.SPECS[extra][0], extra + "_caption.tex", prior.SPECS[extra][2]))


def promote():
    extra_table = prior.verify(extra, "20261119")
    extra_sha = base.sha(paper.HERE / "figures" / paper.DECK)
    extra_files = {str(p.relative_to(paper.SOURCE)): base.sha(p) for p in paper.SOURCE.iterdir() if p.is_file()}
    expected = prior.verify(study, "20261119")

    def verify(table):
        if table != expected:
            raise ValueError("Revision table changed")

    receipt = paper.HERE / "evidence/english_request_writer_article_build.json"
    paper.promote(verify, receipt.name)
    import json
    record = base.load(receipt)
    record[extra] = dict(deck_sha256=extra_sha, table=extra_table, source_hashes=extra_files)
    receipt.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    actions = dict(
        baseline=previous.baseline,
        figure=lambda: prior.export(study, "20261119"),
        entry_figure=lambda: prior.export(extra, "20261119"),
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
