"""Build the phrase coverage and causal feedback experiments into the three editions."""

import argparse
import fcntl
import hashlib
import json
from xml.etree import ElementTree as ET
from zipfile import ZipFile
import build_block_translation_article as prior
from scripts.diagnose_visual_context_gradients import require

paper, base, previous = prior.paper, prior.base, prior.previous
work = paper.ROOT / "artifacts/clause_support_publication_v1"
paper.WORK = previous.WORK = work
paper.STAGE = previous.STAGE = base.STAGE = work / "article"
previous.BASELINE = "2eae37e2"
SPECS = {
    "clause_support": (
        "Figure_58_clause_support",
        "Clause_support",
        "fig:clause-support",
    ),
    "visual_feedback": (
        "Figure_59_visual_feedback",
        "Visual_feedback",
        "fig:visual-feedback",
    ),
}


def configure(study, date="20261118"):
    paper.SOURCE = paper.ROOT / "references/evidence" / (study + "_" + date)
    paper.FIG, deck, _ = SPECS[study]
    paper.DECK = deck + ".pptx"


for name, (figure, _, label) in SPECS.items():
    base.FIGURES.append((figure, name + "_caption.tex", label))
prior_captions = base.literal_figures
base.literal_figures = lambda: prior_captions().replace(
    "build_block_translation_article.py", "build_clause_support_article.py"
)


def export(study, date="20261118"):
    configure(study, date)
    paper.figure(study + "_figure.mjs")


def verify(study, date="20261118"):
    configure(study, date)
    ref = base.load(paper.SOURCE / "figure.json")
    ns = {
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    }
    with ZipFile(paper.HERE / "figures" / paper.DECK) as z:
        slide = ET.fromstring(z.read("ppt/slides/slide1.xml"))
        cells = [
            [c.find(".//a:t", ns).text for c in r.findall("a:tc", ns)]
            for r in slide.findall(".//a:tbl/a:tr", ns)
        ]
        require(cells == ref["table"], "Native table differs from source")
        values = []
        for n in sorted(z.namelist()):
            if "/charts/chart" in n and n.endswith(".xml"):
                chart = ET.fromstring(z.read(n))
                values.extend(
                    [
                        [float(p.text) for p in series.findall("c:val//c:pt/c:v", ns)]
                        for series in chart.findall(".//c:ser", ns)
                    ]
                )
        require(
            values == ref["series"]
            and len([n for n in z.namelist() if n.endswith(".xlsx")]) == 1,
            "Native chart values differ",
        )
        elements = {
            hashlib.sha256(z.read(n)).hexdigest()
            for n in z.namelist()
            if n.startswith("ppt/media/")
        }
        require(
            all(base.sha(p) in elements for p in paper.SOURCE.glob("*.png")),
            "A measured strip is missing",
        )
    return cells


def promote():
    feedback_table = verify("visual_feedback")
    feedback_deck = base.sha(paper.HERE / "figures" / paper.DECK)
    feedback_sources = {
        str(p.relative_to(paper.SOURCE)): base.sha(p)
        for p in paper.SOURCE.rglob("*")
        if p.is_file()
    }
    expected = verify("clause_support")

    def compare(cells):
        require(cells == expected, "Changed primary table")

    receipt = paper.HERE / "evidence/clause_support_article_build.json"
    paper.promote(compare, receipt.name)
    record = base.load(receipt)
    record["visual_feedback"] = dict(
        table=feedback_table, deck_sha256=feedback_deck, source_hashes=feedback_sources
    )
    tmp = receipt.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(receipt)


if __name__ == "__main__":
    actions = dict(
        baseline=previous.baseline,
        figure=lambda: export("clause_support"),
        feedback=lambda: export("visual_feedback"),
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
