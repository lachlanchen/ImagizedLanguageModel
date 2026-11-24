"""Build the paired verse study and verify its measured figure."""

import argparse
import fcntl
import json
import build_verse_contrast_article as preceding
from build_pictured_relation_article import build
from inspect_grounded_proposition import inspect

paper, base, previous, prior = (
    preceding.paper,
    preceding.base,
    preceding.previous,
    preceding.prior,
)
study = "glyph_atlas"
work = paper.ROOT / "artifacts/glyph_atlas_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "fcc30d55"
prior.SPECS[study] = (
    "Figure_138_glyph_atlas",
    "Glyph_atlas",
    "fig:glyph-atlas",
)
prior.configure(study, "v1")
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
literal = base.literal_figures
base.literal_figures = lambda: literal().replace(
    "build_verse_contrast_article.py", "build_glyph_atlas_article.py"
)


def figure_data():
    result = json.loads((paper.SOURCE / "wording-results.json").read_text())
    scores = result["summary"]
    arms = ("ink","ink-wording","byte","byte-wording")
    series = [[round(100*scores[f"{arm}/fresh/{condition}"]["novel"]["exact"]/6360,1) for arm in arms]
              for condition in ("retrieved","supplied")]
    membership = json.loads((paper.SOURCE/"membership.json").read_text())["all"]
    table = [["Headword rule","Book present","Wrong record after deletion"]]
    for label,key,error in [("Read then look up","unconstrained","false_original"),
                            ("Choose from book","constrained","false_constrained")]:
        table.append([label,f"{membership[key]:,} / {membership['n']:,}",f"{membership[error]:,} / {membership['n']:,}"])
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

    paper.promote(check, "glyph_atlas_article_build.json")


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
