#!/usr/bin/env python3
"""Build and locally sync the completed decoder-loss failure manuscript unit."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import fitz

from decoder_metric_figures import BASE, FIGURE, HERE, QA, RESULT, ROOT, baseline, evidence, sha, verify

DEST = Path("/home/lachlan/Nutstore Files/Projects/ILM")


@contextmanager
def build_slot():
    memory = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if line.split()[1].isdigit()}
    if memory["MemAvailable"] < 24 * 1024**2 or (memory["SwapTotal"] and 1 - memory["SwapFree"] / memory["SwapTotal"] > .75):
        raise RuntimeError("shared workstation memory policy prevents the heavy build")
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def build():
    with build_slot():
        return _build()


def _build():
    QA.mkdir(parents=True, exist_ok=True)
    evidence()
    verify()
    rendered = json.loads((HERE / "evidence/decoder_metric_figure_render.json").read_text())
    if (rendered["pptx_sha256"] != sha(HERE / "ILM_Figures_editable.pptx")
            or rendered["review_pdf_sha256"] != sha(HERE / "ILM_Figures_review.pdf")
            or len(rendered["preserved_prior_slide_pixels"]) != 30):
        raise ValueError("current figure render binding differs")
    sources = {"decoder_metric_v82_result": RESULT,
               "decoder_metric_v82_position_diagnosis": ROOT / "references/evidence/decoder_metric_v82_position_diagnosis.json",
               "decoder_metric_v82_runtime_replay": ROOT / "references/evidence/decoder_metric_v82_runtime_replay.json"}
    for name, source in sources.items():
        shutil.copy2(source, HERE / "evidence" / (name + ".json"))
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASE,
                                     "--", "publication/ilm-nature/figures"], cwd=ROOT, text=True).splitlines()
    preserved = {}
    for name in names:
        raw = subprocess.check_output(["git", "show", BASE + ":" + name], cwd=ROOT)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != sha(ROOT / name):
            raise ValueError("prior individual figure changed: " + name)
        preserved[name] = digest
    outputs = {}
    for name in ("manuscript", "manuscript_review", "supplementary_information"):
        result = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                                 "-outdir=." + name, name + ".tex"], cwd=HERE, capture_output=True, text=True)
        (QA / (name + ".build.log")).write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError("compilation failed: " + name)
        folder = HERE / ("." + name)
        log = (folder / (name + ".log")).read_text()
        if any(s in log for s in ("Overfull", "undefined references", "undefined citations", "LaTeX Error")):
            raise ValueError("layout/reference failure: " + name)
        shutil.copy2(folder / (name + ".pdf"), HERE / (name + ".pdf"))
        text = subprocess.check_output(["pdftotext", str(HERE / (name + ".pdf")), "-"], text=True)
        (QA / (name + ".txt")).write_text(text)
        normalized = " ".join(text.split())
        for required in ("85.984", "70.009", "92.430", "75.116", "99.9953", "0/32"):
            if required not in normalized:
                raise ValueError("compiled PDF lacks completed metric result: " + name + " / " + required)
        if name == "supplementary_information" and "Supplementary Note 39" not in normalized:
            raise ValueError("new supplement note absent")
        outputs[name + ".pdf"] = sha(HERE / (name + ".pdf"))
    result = subprocess.run(["python", "build_semantic_transfer_redlines.py"], cwd=HERE, capture_output=True, text=True)
    (QA / "redlines.build.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("redline compilation failed")
    for name in ("redlined_manuscript", "redlined_supplementary_information"):
        text = subprocess.check_output(["pdftotext", str(HERE / (name + ".pdf")), "-"], text=True)
        if "70.009" not in text or "85.984" not in text or "V68 conditional-writing baseline" not in text:
            raise ValueError("redline lacks completed result or baseline notice")
        outputs[name + ".pdf"] = sha(HERE / (name + ".pdf"))
    for name in ("ILM_Figures_review.pdf", "figures/" + FIGURE + ".pdf"):
        outputs[name] = sha(HERE / name)
    for name in outputs:
        result = subprocess.run(["qpdf", "--check", str(HERE / name)], capture_output=True, text=True)
        (QA / (name.replace("/", "_") + ".qpdf.log")).write_text(result.stdout + result.stderr)
        if result.returncode:
            raise ValueError("PDF structural failure: " + name)
    old = fitz.open(stream=baseline("supplementary_information.pdf"), filetype="pdf")
    new = fitz.open(HERE / "supplementary_information.pdf")
    preserved_si = []
    for i in range(32):
        left, right = old[i].get_pixmap(alpha=False), new[i].get_pixmap(alpha=False)
        if left.samples != right.samples:
            raise ValueError("supplement before new note changed: page " + str(i + 1))
        preserved_si.append(hashlib.sha256(right.samples).hexdigest())
    record = dict(baseline_commit=BASE, pdfs_sha256=outputs,
                  source_receipts_sha256={name: sha(path) for name, path in sources.items()},
                  preserved_individual_figures_sha256=preserved, preserved_supplement_pages=preserved_si,
                  preserved_native_slides=30, current_native_slides=31,
                  pdf_pages={name: len(fitz.open(HERE / name)) for name in outputs},
                  all_compiled_without_overfull_or_undefined=True, visual_review_required=True,
                  artifact_marker_available=False, language_quality_claim=False, general_ilm_goal_complete=False)
    (HERE / "evidence/decoder_metric_publication_build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(pages=record["pdf_pages"], prior_figures=len(preserved), preserved_si_pages=32), indent=2))


def sync():
    built = json.loads((HERE / "evidence/decoder_metric_publication_build.json").read_text())
    reviewed = json.loads((HERE / "evidence/decoder_metric_visual_review.json").read_text())
    if (not reviewed["visual_review_completed"] or reviewed["pdfs_sha256"] != built["pdfs_sha256"]
            or reviewed["figure_png_sha256"] != sha(HERE / "figures" / (FIGURE + ".png"))):
        raise ValueError("visual review of the current package required")
    for name, digest in built["pdfs_sha256"].items():
        if sha(HERE / name) != digest:
            raise ValueError("reviewed PDF changed")
    package = DEST / "ILM_NatureStyle_20261107_DecoderMetric"
    files = {}
    for source in sorted(HERE.rglob("*")):
        if not source.is_file() or any(p.startswith(".") or p in ("__pycache__", "baselines") for p in source.relative_to(HERE).parts):
            continue
        relative = source.relative_to(HERE)
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha(source) != sha(target):
            raise ValueError("local package copy differs")
        files[str(relative)] = sha(target)
    aliases = {"manuscript.pdf": "ILM_Manuscript_LATEST.pdf", "manuscript_review.pdf": "ILM_Manuscript_Review_LATEST.pdf",
               "supplementary_information.pdf": "ILM_Supplementary_LATEST.pdf", "ILM_Figures_editable.pptx": "ILM_Figures_editable_LATEST.pptx",
               "ILM_Figures_review.pdf": "ILM_Figures_review_LATEST.pdf"}
    for source, name in aliases.items():
        target = DEST / name
        if target.exists() and sha(target) != sha(HERE / source):
            backup = DEST / (target.stem + "_before_decoder_metric" + target.suffix)
            if backup.exists() and sha(backup) != sha(target):
                raise ValueError("prior alias backup differs; preserve both before continuing")
            if not backup.exists():
                shutil.copy2(target, backup)
        shutil.copy2(HERE / source, target)
        if sha(HERE / source) != sha(target):
            raise ValueError("latest local alias differs")
    record = dict(package=str(package), files_sha256=files, aliases=aliases, local_copy_verified=True,
                  cloud_propagation_verified=False, visual_review_completed=True)
    (QA / "sync.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Verified the local Nutstore package and five current aliases.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("build", "sync"), required=True)
    args = parser.parse_args()
    {"build": build, "sync": sync}[args.stage]()
