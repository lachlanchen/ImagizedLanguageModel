#!/usr/bin/env python3
"""Activate the bounded completed-result text after its figure is rendered."""
import hashlib
import json
from pathlib import Path

from state_alignment_figures import BASE, HERE, QA, ROOT, FIGURE, baseline, sha, verify
from build_paired_writing_publication import build_slot


def main():
    controller = ROOT / "artifacts/native_reader_attention/controller_status.json"
    if controller.exists():
        status = json.loads(controller.read_text())
        if status["state"] == "running" or status.get("child_pid"):
            raise RuntimeError("wait for the current controller and every child to be terminal")
    with build_slot():
        record = verify()
        rendered = json.loads((HERE / "evidence/native_state_alignment_figure_render.json").read_text())
        if rendered["new_slide"] != 35 or len(rendered["preserved_prior_slide_pixels"]) != 34 or sha(HERE / "ILM_Figures_editable.pptx") != record["pptx_sha256"]:
            raise ValueError("rendered current 35-slide figure required")
        if not (HERE / "figures" / (FIGURE + ".pdf")).exists():
            raise ValueError("rendered standalone figure required")
        names = ("frontmatter.tex", "article_content.tex", "supplementary_information.tex")
        before = {name: baseline(name).decode() for name in names}
        if any((HERE / name).read_text() != words for name, words in before.items()):
            raise ValueError("active source differs from the declared baseline; preserve and reconcile")
        front = before["frontmatter.tex"]
        start = front.index("A full-page curriculum through a frozen auxiliary text head")
        end = front.index("The experiments identify failures", start)
        replacement = ("A source-preserving visual-state learner raises English QA from 11/32 to 16/32 and English copying from 7/16 to 16/16, "
            "but Chinese QA remains 0/32 and copying 0/16. Four of six reader requirements fail; all eight tested seen Chinese copies change characters. "
            "A separate fixed-weight control reduces copying from 29/32 to 7/32 by adding only white input padding. "
            "Preserving visible marks therefore does not guarantee reading invariance. A separate contextual-attention adaptation also fails natural QA. "
            "These are auxiliary text readouts, not generated answer images. ")
        front = front[:start] + replacement + front[end:]
        article = before["article_content.tex"]
        marker = "\\section*{Discussion}"
        paragraph = ("A subsequent source-preserving visual-state correction improves English QA from 11/32 to 16/32 and English copying from 7/16 to 16/16, "
            "but Chinese QA remains 0/32 and copying 0/16 (Supplementary Note 43 and Fig.~S20). Four of six reader requirements fail. "
            "A separate seen-prompt check corrupts all eight Chinese strings despite previous training exposure. "
            "The earlier dense checkpoint retains 13/16 Chinese copies on its original 1024-by-256 strips; adding only white bottom padding reduces copying "
            "from 29/32 to 7/32 overall, with every ink pixel and model weight fixed. Tight whole-request cropping also fails. "
            "All 480 layout-control calls are retained, including exact replay of the 160 original responses. "
            "A separate contextual-attention learner retains 29/32 copies but leaves QA at 0/32 in both languages. "
            "These observations identify geometric brittleness and failed Chinese fit, without isolating a unique internal cause or qualifying image-answer generation.\n\n")
        if article.count(marker) != 1:
            raise ValueError("unique Results/Discussion boundary required")
        article = article.replace(marker, paragraph + marker)
        supplement = before["supplementary_information.tex"]
        marker = "\\clearpage\n\\section*{Reproduction}"
        if supplement.count(marker) != 1:
            raise ValueError("unique pre-Reproduction insertion point required")
        supplement = supplement.replace(marker, "\\input{state_alignment_note.tex}\n\n" + marker)
        after = dict(zip(names, (front, article, supplement)))
        for name, words in after.items():
            (HERE / name).write_text(words)
        QA.mkdir(parents=True, exist_ok=True)
        receipt = dict(baseline_commit=BASE, sources={name: dict(before_sha256=hashlib.sha256(before[name].encode()).hexdigest(),
            after_sha256=sha(HERE / name)) for name in names}, note_sha256=sha(HERE / "state_alignment_note.tex"),
            figure_pdf_sha256=sha(HERE / "figures" / (FIGURE + ".pdf")), build_and_visual_review_required=True)
        (QA / "activation.json").write_text(json.dumps(receipt,indent=2)+"\n")
        print("Activated bounded completed-result text; compile and visually review before delivery.")


if __name__ == "__main__":
    main()
