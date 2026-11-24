"""Build the pictured English passage experiment in all three manuscript editions."""

import argparse
import fcntl
import hashlib
import json
from zipfile import ZipFile

import build_questioned_events_article as preceding
from build_pictured_relation_article import build
from inspect_grounded_proposition import inspect

paper, base, previous, prior = (
    preceding.paper,
    preceding.base,
    preceding.previous,
    preceding.prior,
)
study = "optical_passage_qa"
work = paper.ROOT / "artifacts/optical_passage_qa_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "8e7cf541"
prior.SPECS[study] = (
    "Figure_127_optical_passage_qa",
    "Optical_passage_qa",
    "fig:optical-passage-qa",
)
prior.configure(study, "v1")
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
literal = base.literal_figures
base.literal_figures = lambda: literal().replace(
    "build_questioned_events_article.py", "build_optical_passage_qa_article.py"
)


def figure_data():
    groups = json.loads((paper.SOURCE / "result.json").read_text())["groups"]
    rows = [groups["reserved"], groups["dev"]]
    series = [
        [round(100 * row[key] / row["paired_total"], 2) for row in rows]
        for key in ("both_exact", "both_exchanged")
    ]
    table = [["Evaluation", "Questions", "Exact answer", "Word F1", "Exact ink"]]
    for name, row in zip(("Reserved articles", "Development articles"), rows):
        table.append([name, f"{row['queries']:,}", f"{row['exact']:,}",
                      f"{100 * row['mean_f1']:.1f}%", f"{row['exact_pixels']:,}"])
    return dict(series=series, table=table)


def verify_ink():
    expected = figure_data()
    if json.loads((paper.SOURCE / "figure.json").read_text()) != expected:
        raise ValueError("Figure values differ from measured questions")
    with ZipFile(paper.HERE / "figures" / paper.DECK) as package:
        embedded = {hashlib.sha256(package.read(n)).hexdigest() for n in package.namelist()
                    if n.startswith("ppt/media/")}
    for name in ("reserved-correct-1", "reserved-correct-2", "dev-miss-2"):
        image = paper.SOURCE / "examples" / (name + "-answer.png")
        if base.sha(image) not in embedded:
            raise ValueError(f"Missing original answer {name}")
    return prior.verify(study, "v1")


def promote():
    expected = verify_ink()

    def verify(table):
        if table != expected:
            raise ValueError("Native table differs")

    paper.promote(verify, "optical_passage_qa_article_build.json")


if __name__ == "__main__":
    actions = dict(
        baseline=previous.baseline,
        figure=lambda: prior.export(study, "v1"),
        build=build,
        inspect=lambda: inspect(base, verify_existing_pages=True),
        promote=promote,
        verify=verify_ink,
        captions=lambda: (paper.HERE / "complete_continuation_figures.tex").write_text(
            base.literal_figures()
        ),
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=actions)
    with (paper.ROOT / "artifacts/publication_cpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        actions[parser.parse_args().action]()
