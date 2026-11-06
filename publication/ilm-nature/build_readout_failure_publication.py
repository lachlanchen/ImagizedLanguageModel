#!/usr/bin/env python3
"""Build and verify the completed readout-failure manuscript package."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import fitz

from readout_failure_figures import RESULT, evidence, verify as verify_deck

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
QA = ROOT/"artifacts/readout_failure_publication_20261107"
BASELINE = "1f182c4"
DEST = Path("/home/lachlan/Nutstore Files/Projects/ILM")
STEMS = ("native_chinese_v80_result", "native_corpus_fit_v80_result")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    QA.mkdir(parents=True,exist_ok=True)
    evidence();verify_deck()
    rendered = json.loads((HERE/"evidence/readout_failure_figure_render.json").read_text())
    if (rendered["pptx_sha256"] != sha(HERE/"ILM_Figures_editable.pptx")
            or rendered["review_pdf_sha256"] != sha(HERE/"ILM_Figures_review.pdf")
            or len(rendered["preserved_prior_slide_pixels"]) != 29):
        raise ValueError("new/current figures do not match their render receipt")
    shutil.copy2(RESULT,HERE/"evidence/adaptive_readout_v81_result.json")
    for stem in STEMS:
        shutil.copy2(ROOT/"references/evidence"/(stem+".json"),HERE/"evidence"/(stem+".json"))
    names = subprocess.check_output(["git","ls-tree","-r","--name-only",BASELINE,
        "--","publication/ilm-nature/figures"],cwd=ROOT,text=True).splitlines()
    preserved = {}
    for name in names:
        raw = subprocess.check_output(["git","show",BASELINE+":"+name],cwd=ROOT)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != sha(ROOT/name):
            raise ValueError("old individual figure changed: "+name)
        preserved[name] = digest
    outputs = {}
    for name in ("manuscript","manuscript_review","supplementary_information"):
        result = subprocess.run(["latexmk","-pdf","-interaction=nonstopmode","-halt-on-error",
            "-outdir=."+name,name+".tex"],cwd=HERE,capture_output=True,text=True)
        (QA/(name+".build.log")).write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError("compilation failed: "+name)
        folder = HERE/("."+name)
        log = (folder/(name+".log")).read_text()
        if any(s in log for s in ("Overfull","undefined references","undefined citations","LaTeX Error")):
            raise ValueError("layout/reference failure: "+name)
        shutil.copy2(folder/(name+".pdf"),HERE/(name+".pdf"))
        text = subprocess.check_output(["pdftotext",str(HERE/(name+".pdf")),"-"],text=True)
        (QA/(name+".txt")).write_text(text)
        normalized = " ".join(text.split())
        for required in ("72.748", "71.177", "79.629", "77.372", "99.9953", "0/32"):
            if required not in normalized:
                raise ValueError("compiled PDF lacks completed readout result: "+name+" / "+required)
        if name == "supplementary_information" and "Supplementary Note 38" not in normalized:
            raise ValueError("missing completed-result supplement note")
        outputs[name+".pdf"] = sha(HERE/(name+".pdf"))
    result = subprocess.run(["python","build_semantic_transfer_redlines.py"],cwd=HERE,
        capture_output=True,text=True)
    (QA/"redlines.build.log").write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError("native failure redline build failed")
    for name in ("redlined_manuscript","redlined_supplementary_information"):
        text = subprocess.check_output(["pdftotext",str(HERE/(name+".pdf")),"-"],text=True)
        if "72.748" not in text or "79.629" not in text or "V68 conditional-writing baseline" not in text:
            raise ValueError("redline lacks completed results or baseline notice")
        outputs[name+".pdf"] = sha(HERE/(name+".pdf"))
    for name in ("ILM_Figures_review.pdf","figures/Figure_S15_readout_failure.pdf"):
        outputs[name] = sha(HERE/name)
    for name in outputs:
        result = subprocess.run(["qpdf","--check",str(HERE/name)],capture_output=True,text=True)
        (QA/(name.replace('/','_')+".qpdf.log")).write_text(result.stdout+result.stderr)
        if result.returncode:
            raise ValueError("PDF structural check failed: "+name)
    receipt = dict(baseline_commit=BASELINE,pdfs_sha256=outputs,
        preserved_individual_figures_sha256=preserved,preserved_native_slides=29,current_native_slides=30,
        source_receipts_sha256={**{s:sha(ROOT/"references/evidence"/(s+".json")) for s in STEMS},"adaptive_readout_v81_result":sha(RESULT)},
        pdf_pages={name:len(fitz.open(HERE/name)) for name in outputs},
        all_compiled_without_overfull_or_undefined=True,visual_review_required=True,
        language_quality_claim=False,general_ilm_goal_complete=False)
    (HERE/"evidence/readout_failure_publication_build.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(dict(pdfs=outputs,pages=receipt["pdf_pages"],preserved_figures=len(preserved)),indent=2))


def sync():
    build = json.loads((HERE/"evidence/readout_failure_publication_build.json").read_text())
    review = json.loads((HERE/"evidence/readout_failure_visual_review.json").read_text())
    if (not review["visual_review_completed"] or review["pdfs_sha256"] != build["pdfs_sha256"]
            or review["figure_png_sha256"] != sha(HERE/"figures/Figure_S15_readout_failure.png")):
        raise ValueError("review of the current complete package required before sync")
    for name,digest in build["pdfs_sha256"].items():
        if sha(HERE/name) != digest:
            raise ValueError("reviewed PDF changed")
    package = DEST/"ILM_NatureStyle_20261107_ReadoutFailure"
    files = {}
    for source in sorted(HERE.rglob('*')):
        if not source.is_file() or any(p.startswith('.') or p in ('__pycache__','baselines') for p in source.relative_to(HERE).parts):
            continue
        relative = source.relative_to(HERE);target = package/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        if sha(source) != sha(target):
            raise ValueError("local package copy differs")
        files[str(relative)] = sha(target)
    aliases = {"manuscript.pdf":"ILM_Manuscript_LATEST.pdf",
        "manuscript_review.pdf":"ILM_Manuscript_Review_LATEST.pdf",
        "supplementary_information.pdf":"ILM_Supplementary_LATEST.pdf",
        "ILM_Figures_editable.pptx":"ILM_Figures_editable_LATEST.pptx",
        "ILM_Figures_review.pdf":"ILM_Figures_review_LATEST.pdf"}
    for source,name in aliases.items():
        target = DEST/name
        if target.exists() and sha(target) != sha(HERE/source):
            backup = DEST/(target.stem+"_before_readout_failure"+target.suffix)
            if backup.exists() and sha(backup) != sha(target):
                raise ValueError("alias baseline backup already differs")
            if not backup.exists():shutil.copy2(target,backup)
        shutil.copy2(HERE/source,target)
        if sha(HERE/source) != sha(target):raise ValueError("latest local alias differs")
    receipt = dict(package=str(package),files_sha256=files,aliases=aliases,
        local_copy_verified=True,cloud_propagation_verified=False,visual_review_completed=True)
    (QA/"sync.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("Verified local Nutstore package and five current aliases.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('build','sync'),required=True)
    args = parser.parse_args();dict(build=build,sync=sync)[args.stage]()
