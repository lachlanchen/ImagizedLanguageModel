"""Build the bilingual card-writing study and verify both new figures."""

import argparse
import fcntl
import json
import build_semantic_page_article as preceding
from build_pictured_relation_article import build
from inspect_grounded_proposition import inspect

paper, base, previous, prior = (
    preceding.paper,
    preceding.base,
    preceding.previous,
    preceding.prior,
)
study = "visual_fact_reply"
work = paper.ROOT / "artifacts/visual_fact_reply_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "25e19af8"
prior.SPECS[study] = (
    "Figure_140_visual_fact_reply",
    "Visual_fact_reply",
    "fig:visual-fact-reply",
)
prior.configure(study, "v1")
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
literal = base.literal_figures
base.literal_figures = lambda: literal().replace(
    "build_semantic_page_article.py", "build_visual_fact_reply_article.py"
)


def figure_data():
    evidence=paper.ROOT/"references/evidence"
    chinese=json.loads((evidence/"chinese_reply_v1/results.json").read_text())
    names=json.loads((evidence/"variable_ink_v1/renaming.json").read_text())
    series=[[round(100*chinese[arm][group]["clean_value"]/chinese[arm][group]["n"],1)
             for group in ("all","unseen_question","unseen_entity","both_unseen")]
            for arm in ("ink","ink-erased","byte")]
    table=[["Novel name / paired cards","Frozen writer","Frozen + surrogates","Trained copy actions"]]
    for kind in ("latin","historical"):
        table.append([kind.capitalize()+" pairs"]+[f'{names[arm+"-"+kind]["complete_pair"]} / 64'
                      for arm in ("parent","surrogate","variable")])
    return dict(series=series,table=table)


def verify():
    if json.loads((paper.SOURCE / "figure.json").read_text()) != figure_data():
        raise ValueError("Different measured values")
    return prior.verify(study, "v1")


def promote():
    expected = verify()

    def check(table):
        if table != expected:
            raise ValueError("Different native table")

    paper.promote(check, "visual_fact_reply_article_build.json")


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
