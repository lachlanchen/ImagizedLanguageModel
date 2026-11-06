#!/usr/bin/env python3
"""Build the verified reader-content diagnostic and append-only evidence figure."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
QA=ROOT/"artifacts/native_context_publication_v80_20261106"
BASELINE="71e0fc4"
DEST=Path("/home/lachlan/Nutstore Files/Projects/ILM")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    QA.mkdir(parents=True,exist_ok=True)
    for stem,run,report in (
        ("clean_corpus_v75_result","clean_corpus_v75_20261106","development/report.json"),
        ("copy_transfer_v75_result","copy_transfer_v75_20261106","report.json"),
        ("corpus_fit_v75_result","corpus_fit_v75_20261106","report.json")):
        source=ROOT/"references/evidence"/(stem+".json")
        value=json.loads(source.read_text())
        raw=ROOT/"artifacts"/run
        report_sha=value.get("report_sha256",value.get("source_report_sha256"))
        verified_sha=value.get("verification_sha256",value.get("source_verification_sha256"))
        if sha(raw/report)!=report_sha or sha(raw/"verification.json")!=verified_sha:
            raise ValueError("completed diagnosis receipt changed: "+stem)
        if value["language_quality_claim"] or value["general_ilm_goal_complete"]:
            raise ValueError("failed result acquired a success claim")
        shutil.copy2(source,HERE/"evidence"/source.name)
    repair=ROOT/"references/evidence/writing_times_v76_result.json"
    value=json.loads(repair.read_text())
    if (sha(ROOT/value["source_run"]/"verification.json")!=value["verification_sha256"]
            or value["verification"]["training_fit_repair_pass"]):
        raise ValueError("matched repair no longer binds the rejected outcome")
    shutil.copy2(repair,HERE/"evidence"/repair.name)
    reader=ROOT/"references/evidence/reader_content_v77_result.json"
    value=json.loads(reader.read_text())
    if sha(ROOT/value["source_run"]/"verification.json")!=value["verification_sha256"]:
        raise ValueError("reader-content evidence changed")
    shutil.copy2(reader,HERE/"evidence"/reader.name)
    native=ROOT/"references/evidence/native_context_v80.json"
    receipt=json.loads(native.read_text())
    if (sha(ROOT/"artifacts/suffix_kv_v78_20261106/verification.json")!=receipt["suffix_visibility"]["verification_sha256"]
            or sha(ROOT/"artifacts/native_image_execution_v79r2_20261106/supplemental_verification.json")
                !=receipt["native_execution"]["supplemental_verification_sha256"]
            or receipt["native_execution"]["trained"]):
        raise ValueError("native-context evidence changed or acquired a training claim")
    shutil.copy2(native,HERE/"evidence"/native.name)
    from native_context_figures import verify_native
    verify_native()
    rendered=json.loads((HERE/"evidence/native_context_figure_render.json").read_text())
    if (sha(HERE/"ILM_Figures_editable.pptx")!=rendered["pptx_sha256"]
            or sha(HERE/"ILM_Figures_review.pdf")!=rendered["review_pdf_sha256"]):
        raise ValueError("native/current figure rendering changed")
    names=subprocess.check_output(["git","ls-tree","-r","--name-only",BASELINE,
        "--","publication/ilm-nature/figures"],cwd=ROOT,text=True).splitlines()
    figures={}
    for name in names:
        raw=subprocess.check_output(["git","show",BASELINE+":"+name],cwd=ROOT)
        if hashlib.sha256(raw).hexdigest()!=sha(ROOT/name):
            raise ValueError("unrelated native figure or canonical appearance changed: "+name)
        figures[name]=sha(ROOT/name)
    outputs={}
    for name in ("manuscript","manuscript_review","supplementary_information"):
        build=subprocess.run(["latexmk","-pdf","-interaction=nonstopmode","-halt-on-error",
            "-outdir=."+name,name+".tex"],cwd=HERE,capture_output=True,text=True)
        (QA/(name+".build.log")).write_text(build.stdout+build.stderr)
        if build.returncode:
            raise RuntimeError("compilation failed: "+name)
        folder=HERE/("."+name)
        log=(folder/(name+".log")).read_text()
        if any(x in log for x in ("Overfull","undefined references","undefined citations","LaTeX Error")):
            raise ValueError("layout/reference error: "+name)
        shutil.copy2(folder/(name+".pdf"),HERE/(name+".pdf"))
        extracted=subprocess.check_output(["pdftotext",str(HERE/(name+".pdf")),"-"],text=True)
        (QA/(name+".txt")).write_text(extracted)
        normalized=" ".join(extracted.split())
        if any(x not in normalized for x in ("37.631","58.045","99.9953")):
            raise ValueError("compiled PDF lacks the completed diagnosis: "+name)
        if name=="supplementary_information" and "Supplementary Note 33" not in normalized:
            raise ValueError("compiled supplement lacks the new result note")
        if "45.028" not in normalized or (name=="supplementary_information" and "Supplementary Note 34" not in normalized):
            raise ValueError("compiled PDF lacks the rejected matched repair")
        if "19/32" not in normalized or (name=="supplementary_information" and "Supplementary Note 35" not in normalized):
            raise ValueError("compiled PDF lacks the verified reader comparison")
        if "2/32" not in normalized or (name=="supplementary_information" and "Supplementary Note 36" not in normalized):
            raise ValueError("compiled PDF lacks context-visibility result")
        outputs[name+".pdf"]=sha(HERE/(name+".pdf"))
    subprocess.run(["python","build_semantic_transfer_redlines.py"],cwd=HERE,check=True)
    for name in ("redlined_manuscript","redlined_supplementary_information"):
        text=subprocess.check_output(["pdftotext",str(HERE/(name+".pdf")),"-"],text=True)
        if "37.631" not in text or "V68 conditional-writing baseline" not in text:
            raise ValueError("redline lacks the new result or preserved comparison baseline")
        outputs[name+".pdf"]=sha(HERE/(name+".pdf"))
    outputs["ILM_Figures_review.pdf"]=sha(HERE/"ILM_Figures_review.pdf")
    outputs["figures/Figure_S13_native_context.pdf"]=sha(HERE/"figures/Figure_S13_native_context.pdf")
    for name in outputs:
        check=subprocess.run(["qpdf","--check",str(HERE/name)],capture_output=True,text=True)
        (QA/(name.replace("/","_")+".qpdf.log")).write_text(check.stdout+check.stderr)
        if check.returncode:
            raise ValueError("PDF structural check failed: "+name)
    receipt=dict(baseline_commit=BASELINE,pdfs_sha256=outputs,unchanged_figures_sha256=figures,
        preserved_native_slides=27,current_native_slides=28,new_figure="S13",all_compiled_without_overfull_or_undefined=True,
        source_receipts_sha256={stem:sha(ROOT/"references/evidence"/(stem+".json"))
            for stem in ("clean_corpus_v75_result","copy_transfer_v75_result","corpus_fit_v75_result","writing_times_v76_result","reader_content_v77_result","native_context_v80")},
        visual_review_required=True,language_quality_claim=False,general_ilm_goal_complete=False)
    (HERE/"evidence/native_context_publication_build.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(dict(pdfs=outputs,preserved_figure_files=len(figures)),indent=2))


def sync_after_visual_review():
    package=DEST/"ILM_NatureStyle_20261106_V80_NativeContext"
    copied={}
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or any(p.startswith(".") or p in ("__pycache__","baselines") for p in path.relative_to(HERE).parts):
            continue
        relative=path.relative_to(HERE)
        target=package/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        if sha(path)!=sha(target):
            raise ValueError("local package copy differs")
        copied[str(relative)]=sha(target)
    aliases={"manuscript.pdf":"ILM_Manuscript_LATEST.pdf","manuscript_review.pdf":"ILM_Manuscript_Review_LATEST.pdf",
        "supplementary_information.pdf":"ILM_Supplementary_LATEST.pdf",
        "ILM_Figures_editable.pptx":"ILM_Figures_editable_LATEST.pptx","ILM_Figures_review.pdf":"ILM_Figures_review_LATEST.pdf"}
    for source,name in aliases.items():
        target=DEST/name
        if target.exists() and sha(target)!=sha(HERE/source):
            backup=DEST/(target.stem+"_before_v80_native_context"+target.suffix)
            if backup.exists() and sha(backup)!=sha(target):
                raise ValueError("alias baseline backup already differs")
            if not backup.exists():
                shutil.copy2(target,backup)
        shutil.copy2(HERE/source,target)
        if sha(HERE/source)!=sha(target):
            raise ValueError("current Nutstore local alias differs")
    receipt=dict(package=str(package),files_sha256=copied,aliases=aliases,
        local_copy_verified=True,cloud_propagation_verified=False,visual_review_completed=True)
    (QA/"sync.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("Verified local Nutstore package and five current aliases.")


if __name__=="__main__":
    main()
