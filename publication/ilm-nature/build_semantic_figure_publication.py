#!/usr/bin/env python3
"""Build, verify and locally sync the visual-semantics figures update."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
QA = ROOT/"artifacts/semantic_figures_v72_20261106/publication"
DEST = Path("/home/lachlan/Nutstore Files/Projects/ILM")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    QA.mkdir(parents=True,exist_ok=True)
    from verify_semantic_transfer_figures import verify
    verify()
    baseline = json.loads((HERE/"baselines/v72-visual-semantics/baseline_manifest.json").read_text())
    unchanged = {n:d for n,d in baseline["files_sha256"].items()
        if n.startswith("figures/") or n.endswith(".sty")}
    for name,digest in unchanged.items():
        if sha(HERE/name)!=digest:
            raise ValueError("preserved figure/deck/style changed: "+name)
    evidence = ROOT/"references/evidence/native_visual_transfer_v71_result.json"
    reader = json.loads(evidence.read_text())
    if (not reader["offline_checks_pass"] or reader["helper_projection_argmax_agreements"]!=742
            or reader["summaries"]["native_budget/normal"]["exact"]!=4
            or reader["summaries"]["native_budget/blank"]["exact"]!=0):
        raise ValueError("source-bound reader evidence differs")
    shutil.copy2(evidence,HERE/"evidence/native_visual_transfer_v71_result.json")
    output = {}
    for name in ("manuscript","manuscript_review","supplementary_information"):
        build = subprocess.run(["latexmk","-pdf","-interaction=nonstopmode","-halt-on-error",
            "-outdir=."+name,name+".tex"],cwd=HERE,capture_output=True,text=True)
        (QA/(name+".build.log")).write_text(build.stdout+build.stderr)
        if build.returncode:
            raise RuntimeError("PDF compilation failed: "+name)
        folder = HERE/("."+name)
        log = (folder/(name+".log")).read_text()
        if any(x in log for x in ("Overfull","undefined references","undefined citations","LaTeX Error")):
            raise ValueError("PDF layout/reference failure: "+name)
        shutil.copy2(folder/(name+".pdf"),HERE/(name+".pdf"))
        text = subprocess.check_output(["pdftotext",str(HERE/(name+".pdf")),"-"],text=True)
        (QA/(name+".txt")).write_text(text)
        required = ("742","38.652M","prospective") if name!="supplementary_information" else (
            "Supplementary Note 29","Supplementary Note 30","742","232.533","38,651,649","ongoing training")
        if any(value not in " ".join(text.split()) for value in required):
            raise ValueError("current PDF does not contain the registered new claims: "+name)
        output[name+".pdf"] = sha(HERE/(name+".pdf"))
    subprocess.run(["python","build_semantic_transfer_redlines.py"],cwd=HERE,check=True)
    for name in ("redlined_manuscript","redlined_supplementary_information"):
        text = subprocess.check_output(["pdftotext",str(HERE/(name+".pdf")),"-"],text=True)
        if "V68 conditional-writing baseline" not in text or "742" not in text:
            raise ValueError("redline lacks visible current changes")
        output[name+".pdf"] = sha(HERE/(name+".pdf"))
    receipt = dict(baseline_commit=baseline["source_commit"],pdfs_sha256=output,
        source_reader_receipt_sha256=sha(evidence),unchanged_prior_figure_style_files=len(unchanged),native_prior_slides_preserved=24,new_supplementary_figures=2,
        all_compiled_without_overfull_or_undefined=True,visual_review_required=True,
        image_language_success_claim=False)
    (HERE/"evidence/semantic_figures_publication_build.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


def sync_after_visual_review():
    """Call only after visually inspecting the freshly rendered changed pages."""
    package = DEST/"ILM_NatureStyle_20261106_V72_VisualSemantics_Figures"
    copied = {}
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or any(part.startswith(".") or part=="__pycache__" for part in path.relative_to(HERE).parts):
            continue
        relative = path.relative_to(HERE)
        target = package/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        if sha(path)!=sha(target):
            raise ValueError("Nutstore local copy differs")
        copied[str(relative)] = sha(target)
    aliases = {"manuscript.pdf":"ILM_Manuscript_LATEST.pdf","manuscript_review.pdf":"ILM_Manuscript_Review_LATEST.pdf",
        "supplementary_information.pdf":"ILM_Supplementary_LATEST.pdf",
        "ILM_Figures_editable.pptx":"ILM_Figures_editable_LATEST.pptx","ILM_Figures_review.pdf":"ILM_Figures_review_LATEST.pdf"}
    for source,name in aliases.items():
        destination = DEST/name
        if destination.exists() and sha(destination)!=sha(HERE/source):
            backup = DEST/(destination.stem+"_before_v72_figures"+destination.suffix)
            if backup.exists() and sha(backup)!=sha(destination):
                raise ValueError("alias backup already contains a different baseline")
            if not backup.exists():
                shutil.copy2(destination,backup)
        shutil.copy2(HERE/source,destination)
        if sha(destination)!=sha(HERE/source):
            raise ValueError("current local Nutstore alias differs")
    receipt = dict(package=str(package),files_sha256=copied,aliases=aliases,local_copy_verified=True,
        cloud_propagation_verified=False,visual_review_completed=True)
    (ROOT/"artifacts/semantic_figures_v72_20261106/publication/sync.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("Verified local Nutstore package and five current aliases.")


if __name__=="__main__":
    main()
