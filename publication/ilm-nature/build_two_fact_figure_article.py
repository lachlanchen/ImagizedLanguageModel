"""Build and verify the ordered two-fact question study."""

import argparse
import fcntl
import json
import build_visual_fact_reply_article as preceding
from build_pictured_relation_article import build
from inspect_grounded_proposition import inspect

paper, base, previous, prior = (
    preceding.paper,
    preceding.base,
    preceding.previous,
    preceding.prior,
)
study = "field_cursor"
work = paper.ROOT / "artifacts/two_fact_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "87523297"
prior.SPECS[study] = (
    "Figure_141_two_fact_figure",
    "Two_fact_figure",
    "fig:two-fact-figure",
)
prior.configure(study, "v1")
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
literal = base.literal_figures
base.literal_figures = lambda: literal().replace(
    "build_visual_fact_reply_article.py", "build_two_fact_figure_article.py"
)


def figure_data():
    evidence=paper.ROOT/"references/evidence"
    root=evidence/"chinese_two_fact_v1"
    def load(name):
        return json.loads((root/(name+".json")).read_text())
    original=load("results")["test"]
    aligned=load("alignment-ink")
    byte=load("alignment-byte")
    cursor=load("cursor-results")["ink"]
    attachment=load("attachment")["passed"]
    series=[[original["ink-joined"]["2"]["clean"],aligned["test"]["groups"]["2"]["clean"],
             cursor["test"]["groups"]["2"]["clean"],sum(r["score"]["clean"] for r in load("qwen-answer")["outputs"])],
            [attachment[k] for k in ("ink-joined","aligned","cursor-ink","qwen")]]
    table=[["Requested order (18 pairs per construction)","Image input","Byte input"]]
    for group,label in (("swap","First A, then report B"),("fresh-list","Report the list in order"),
                        ("fresh-delay","Leave B until second, answer A first")):
        table.append([label,str(cursor[group]["paired_order"])+" / 18",str(byte[group]["paired_order"])+" / 18"])
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

    paper.promote(check, "two_fact_article_build.json")


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
