"""Build the bilingual card-writing study and verify both new figures."""

import argparse
import fcntl
import json
import build_glyph_atlas_article as preceding
from build_pictured_relation_article import build
from inspect_grounded_proposition import inspect

paper, base, previous, prior = (
    preceding.paper,
    preceding.base,
    preceding.previous,
    preceding.prior,
)
study = "semantic_page"
work = paper.ROOT / "artifacts/semantic_page_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "fcc30d55"
prior.SPECS[study] = (
    "Figure_139_semantic_page",
    "Semantic_page",
    "fig:semantic-page",
)
prior.configure(study, "v1")
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
literal = base.literal_figures
base.literal_figures = lambda: literal().replace(
    "build_glyph_atlas_article.py", "build_semantic_page_article.py"
)


def figure_data():
    scores = json.loads((paper.SOURCE / "listener-results.json").read_text())
    arms = ("ink-en", "ink-zh", "byte-en", "byte-zh")
    series = [[round(100*scores[a][decoder]["original"]["equivalent"]/1847,1) for a in arms]
              for decoder in ("greedy","beam","readback")]
    table = [["Chinese card", "Exact facts", "Edited fact correct twice", "Exact after exchange"]]
    for key,label in (("greedy","Greedy"),("beam","Beam"),("readback","Visual readback")):
        row=scores["ink-zh"][key]
        table.append([label, f'{row["original"]["strict"]:,} / 1,847',
                      f'{row["polarity_both"]:,} / 1,291', f'{row["exchanged"]} / 1,847'])
    return dict(series=series, table=table)


def verify():
    if json.loads((paper.SOURCE / "figure.json").read_text()) != figure_data():
        raise ValueError("Different measured values")
    return prior.verify(study, "v1")


def promote():
    prior.verify("glyph_atlas","v1")
    prior.configure(study,"v1")
    expected = verify()

    def check(table):
        if table != expected:
            raise ValueError("Different native table")

    paper.promote(check, "semantic_page_article_build.json")


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
