"""Build the original visual prefix study and verify its measured figure."""

import argparse
import fcntl
import json
import build_context_cloze_article as preceding
from build_pictured_relation_article import build
from inspect_grounded_proposition import inspect

paper, base, previous, prior = (
    preceding.paper,
    preceding.base,
    preceding.previous,
    preceding.prior,
)
study = "book_prefix"
work = paper.ROOT / "artifacts/book_prefix_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "a2f73c50"
prior.SPECS[study] = (
    "Figure_135_book_prefix",
    "Book_prefix",
    "fig:book-prefix",
)
prior.configure(study, "v1")
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
literal = base.literal_figures
base.literal_figures = lambda: literal().replace(
    "build_context_cloze_article.py", "build_book_prefix_article.py"
)


def figure_data():
    result = json.loads((paper.SOURCE / "results.json").read_text())
    models = result["models"]
    arms = ("ink-hybrid", "ink-neural", "counts", "byte-hybrid")
    labels = ("Ink + counts", "Ink", "Counts", "Bytes + counts")
    series = []
    for condition in ("teacher", "free"):
        groups = [models[arm]["groups"]["all"][condition] for arm in arms]
        series.append([round(100 * g["correct"] / g["total"], 2) for g in groups])
    table = [["Source prefix", "Analects", "Chu Ci", "Both works"]]
    for mode, name in (("free", "Original"), ("swapped", "30 cells exchanged"), ("two", "Two cells only")):
        groups = models["ink-hybrid"]["groups"]
        table.append([name] + [str(groups[w][mode]["correct"]) for w in ("KR1h0004","KR4a0001","all")])
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

    paper.promote(check, "book_prefix_article_build.json")


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
