"""Build the paragraph selection experiment in all three manuscript editions."""

import argparse
import fcntl
import json

import build_page_evidence_article as preceding
from build_pictured_relation_article import build
from inspect_grounded_proposition import inspect

paper, base, previous, prior = (
    preceding.paper,
    preceding.base,
    preceding.previous,
    preceding.prior,
)
study = "paragraph_evidence"
work = paper.ROOT / "artifacts/paragraph_evidence_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "a5e98815"
prior.SPECS[study] = (
    "Figure_99_paragraph_evidence",
    "Paragraph_evidence",
    "fig:paragraph-evidence",
)
prior.configure(study, "20261124")
base.FIGURES.append((paper.FIG, study + "_caption.tex", prior.SPECS[study][2]))
second = "mixed_visual_answers"
prior.SPECS[second] = (
    "Figure_100_mixed_visual_answers",
    "Mixed_visual_answers",
    "fig:mixed-visual-answers",
)
base.FIGURES.append(
    (prior.SPECS[second][0], second + "_caption.tex", prior.SPECS[second][2])
)
literal = base.literal_figures
base.literal_figures = lambda: literal().replace(
    "build_page_evidence_article.py", "build_paragraph_evidence_article.py"
)


def promote():
    mixed_table = prior.verify(second, "20261124")
    mixed_deck = base.sha(paper.HERE / "figures" / paper.DECK)
    mixed_sources = {
        str(p.relative_to(paper.SOURCE)): base.sha(p)
        for p in paper.SOURCE.rglob("*")
        if p.is_file()
    }
    expected = prior.verify(study, "20261124")

    def verify(table):
        if table != expected:
            raise ValueError("Native table differs")

    name = "paragraph_evidence_article_build.json"
    paper.promote(verify, name)
    receipt = paper.HERE / "evidence" / name
    content = base.load(receipt)
    content["mixed_visual_answers"] = dict(
        table=mixed_table, deck_sha256=mixed_deck, source_hashes=mixed_sources
    )
    tmp = receipt.with_suffix(".tmp")
    tmp.write_text(json.dumps(content, indent=2) + "\n")
    tmp.replace(receipt)


if __name__ == "__main__":
    actions = dict(
        baseline=previous.baseline,
        figure=lambda: prior.export(study, "20261124"),
        mixed=lambda: prior.export(second, "20261124"),
        build=build,
        inspect=lambda: inspect(base, verify_existing_pages=True),
        promote=promote,
        captions=lambda: (paper.HERE / "complete_continuation_figures.tex").write_text(
            base.literal_figures()
        ),
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=actions)
    with (paper.ROOT / "artifacts/publication_cpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        actions[parser.parse_args().action]()
