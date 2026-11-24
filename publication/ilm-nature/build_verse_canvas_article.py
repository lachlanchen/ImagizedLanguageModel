"""Build the paired verse study and verify its measured figure."""

import argparse
import fcntl
import json
import build_book_prefix_article as preceding
from build_pictured_relation_article import build
from inspect_grounded_proposition import inspect

paper, base, previous, prior = (
    preceding.paper,
    preceding.base,
    preceding.previous,
    preceding.prior,
)
study = "verse_canvas"
work = paper.ROOT / "artifacts/verse_canvas_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "dbc5aea7"
prior.SPECS[study] = (
    "Figure_136_verse_canvas",
    "Verse_canvas",
    "fig:verse-canvas",
)
prior.configure(study, "v1")
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
literal = base.literal_figures
base.literal_figures = lambda: literal().replace(
    "build_book_prefix_article.py", "build_verse_canvas_article.py"
)


def figure_data():
    result = json.loads((paper.SOURCE / "confirmation.json").read_text())
    scores = result['summary']
    arms = ('ink_blank_original','ink-bound_blank_original','byte-bound_blank_original','aligned_blank_original')
    series = []
    for task in ('blank','gap'):
        rows = [scores[arm.replace('blank',task)] for arm in arms]
        series.append([round(100 * r['hits']/r['cells'], 2) for r in rows])
    table = [['Ink source','Blank line (glyphs)','Two gaps (glyphs)','Both gaps exact']]
    for cond,label in [('original','Original'),('exchanged','Exchanged'),('droid','Held typeface')]:
        full = scores['ink-bound_blank_'+cond]
        gap = scores['ink-bound_gap_'+cond]
        table.append([label, f"{full['hits']:,} / {full['cells']:,}",
                      f"{gap['hits']:,} / {gap['cells']:,}",f"{gap['exact']:,} / {gap['lines']:,}"])
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

    paper.promote(check, "verse_canvas_article_build.json")


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
