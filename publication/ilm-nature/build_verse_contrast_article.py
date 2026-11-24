"""Build the paired verse study and verify its measured figure."""

import argparse
import fcntl
import json
import build_verse_canvas_article as preceding
from build_pictured_relation_article import build
from inspect_grounded_proposition import inspect

paper, base, previous, prior = (
    preceding.paper,
    preceding.base,
    preceding.previous,
    preceding.prior,
)
study = "verse_contrast"
work = paper.ROOT / "artifacts/verse_contrast_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "67ad666e"
prior.SPECS[study] = (
    "Figure_137_verse_contrast",
    "Verse_contrast",
    "fig:verse-contrast",
)
prior.configure(study, "v1")
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
literal = base.literal_figures
base.literal_figures = lambda: literal().replace(
    "build_verse_canvas_article.py", "build_verse_contrast_article.py"
)


def figure_data():
    result = json.loads((paper.SOURCE / "results.json").read_text())
    scores = result["summary"]
    arms = ("ink-bound","ink-paired","ink-contrast","byte-contrast")
    series = [[scores[f"{arm}/gap/{policy}"]["exact"] for arm in arms]
              for policy in ("left","pair")]
    table = [["Writer","Original assignment","Two-cell repair","Exchanged source"]]
    for arm,label in zip(arms, ("Frozen image","Paired image","Contrastive image","Contrastive byte")):
        p = result["source_preference"][arm]
        table.append([label, f"{p['reciprocal_win']:,} / {p['reciprocal_scorable']:,}",
                      str(scores[arm+"/gap/pair"]["exact"]),str(scores[arm+"/gap/exchanged-pair"]["exact"])])
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

    paper.promote(check, "verse_contrast_article_build.json")


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
