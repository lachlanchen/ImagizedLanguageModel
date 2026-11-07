#!/usr/bin/env python3
"""Build and locally deliver completed visual-state and canvas observations."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess

import fitz
from PIL import Image

from state_alignment_figures import BASE, FIGURE, HERE, QA, RESULT, SOURCE, FIT, ATTENTION, ROOT, baseline, evidence, sha, verify
from build_paired_writing_publication import build_slot

DEST = Path("/home/lachlan/Nutstore Files/Projects/ILM")
FACTORIZATION = ROOT / "references/evidence/native_canvas_factorization_result/result.json"
DELIVERY = "ILM_NatureStyle_20261107_NativeCanvasFactorization"


def build():
    with build_slot():
        QA.mkdir(parents=True, exist_ok=True)
        evidence()
        verify()
        rendered = json.loads((HERE / "evidence/native_state_alignment_figure_render.json").read_text())
        if rendered["review_pdf_sha256"] != sha(HERE / "ILM_Figures_review.pdf") or len(rendered["preserved_prior_slide_pixels"]) != 34:
            raise ValueError("current 35-slide render with preserved baseline required")
        sources = {"native_state_alignment_result": RESULT, "native_canvas_control_result": SOURCE, "native_state_fit_result": FIT, "native_reader_attention_result": ATTENTION, "native_canvas_factorization_result": FACTORIZATION}
        factor = json.loads(FACTORIZATION.read_text())
        if factor["actual_calls"] != 128 or factor["execution"]["source_decision_replays_exact"] != 64:
            raise ValueError("completed source-bound feature/context intervention required")
        expected = {"strip_short": (13, 16), "square_long": (0, 7), "square_short": (0, 6), "strip_long": (2, 16)}
        for arm, counts in expected.items():
            if tuple(factor["arms"][arm]["languages"][lang]["exact_stopped"] for lang in ("zh", "en")) != counts:
                raise ValueError("feature/context table differs from scored evidence")
        for name, digest in factor["files_sha256"].items():
            if sha(FACTORIZATION.parent / name) != digest:
                raise ValueError("feature/context evidence changed")
        for name, path in sources.items():
            shutil.copy2(path, HERE / "evidence" / (name + ".json"))
        names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASE, "--", "publication/ilm-nature/figures"], cwd=ROOT, text=True).splitlines()
        preserved = {}
        for name in names:
            original = subprocess.check_output(["git", "show", BASE + ":" + name], cwd=ROOT)
            digest = hashlib.sha256(original).hexdigest()
            if sha(ROOT / name) != digest:
                raise ValueError("earlier figure changed: " + name)
            preserved[name] = digest
        outputs = {}
        for name in ("manuscript", "manuscript_review", "supplementary_information"):
            result = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-outdir=." + name, name + ".tex"], cwd=HERE, capture_output=True, text=True)
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
            for required in ("6/32", "10/32", "auxiliary"):
                if required not in normalized:
                    raise ValueError("completed evidence missing from PDF: " + name)
            if name == "supplementary_information" and "Supplementary Note 43" not in normalized:
                raise ValueError("new supplement note absent")
            if "128-call" not in normalized:
                raise ValueError("completed feature/context intervention absent: " + name)
            outputs[name + ".pdf"] = sha(HERE / (name + ".pdf"))
        result = subprocess.run(["python", "build_semantic_transfer_redlines.py"], cwd=HERE, capture_output=True, text=True)
        (QA / "redlines.build.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError("redline compilation failed")
        for name in ("redlined_manuscript", "redlined_supplementary_information"):
            text = subprocess.check_output(["pdftotext", str(HERE / (name + ".pdf")), "-"], text=True)
            if "auxiliary" not in text or "V68 conditional-writing baseline" not in text:
                raise ValueError("redline lacks new result or baseline notice")
            outputs[name + ".pdf"] = sha(HERE / (name + ".pdf"))
        for name in ("ILM_Figures_review.pdf", "figures/" + FIGURE + ".pdf"):
            outputs[name] = sha(HERE / name)
        for name in outputs:
            result = subprocess.run(["qpdf", "--check", str(HERE / name)], capture_output=True, text=True)
            (QA / (name.replace("/", "_") + ".qpdf.log")).write_text(result.stdout + result.stderr)
            if result.returncode:
                raise ValueError("PDF structural failure: " + name)
        expected_pixels = set()
        for picture in verify()["pictures"]:
            with Image.open(ROOT / picture["source"]) as image:
                image = image.convert("RGB")
                expected_pixels.add((image.size, hashlib.sha256(image.tobytes()).hexdigest()))
        preserved_pdf_images = {}
        for name in ("supplementary_information.pdf", "redlined_supplementary_information.pdf", "ILM_Figures_review.pdf", "figures/" + FIGURE + ".pdf"):
            document = fitz.open(HERE / name)
            seen, pixels = set(), set()
            for page in document:
                for item in page.get_images(full=True):
                    if item[0] in seen:
                        continue
                    seen.add(item[0])
                    with Image.open(io.BytesIO(document.extract_image(item[0])["image"])) as image:
                        image = image.convert("RGB")
                        pixels.add((image.size, hashlib.sha256(image.tobytes()).hexdigest()))
            if not expected_pixels <= pixels:
                raise ValueError("PDF changed original complete input-page pixels: " + name)
            preserved_pdf_images[name] = 1
        old = fitz.open(stream=baseline("supplementary_information.pdf"), filetype="pdf")
        new = fitz.open(HERE / "supplementary_information.pdf")
        preserved_si = []
        for i in range(40):
            a, b = old[i].get_pixmap(alpha=False), new[i].get_pixmap(alpha=False)
            if a.samples != b.samples:
                raise ValueError("supplement before Note 43 changed: page " + str(i + 1))
            preserved_si.append(hashlib.sha256(b.samples).hexdigest())
        record = dict(baseline_commit=BASE, pdfs_sha256=outputs,
            current_revision_baseline="db370a4", current_revision="native_canvas_factorization",
            source_receipts_sha256={name: sha(path) for name, path in sources.items()},
            preserved_individual_figures_sha256=preserved, preserved_supplement_pages=preserved_si,
            preserved_native_slides=34, current_native_slides=35,
            preserved_complete_input_images_in_pdfs=preserved_pdf_images, builder_sha256=sha(__file__),
            pdf_pages={name: len(fitz.open(HERE / name)) for name in outputs},
            all_compiled_without_overfull_or_undefined=True, visual_review_required=True,
            artifact_tool_available=False, language_quality_claim=False, general_ilm_goal_complete=False)
        (HERE / "evidence/native_state_alignment_publication_build.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(dict(pages=record["pdf_pages"], prior_figures=len(preserved), preserved_si_pages=len(preserved_si)), indent=2))


def sync():
    built = json.loads((HERE / "evidence/native_state_alignment_publication_build.json").read_text())
    reviewed = json.loads((HERE / "evidence/native_state_alignment_visual_review.json").read_text())
    if not reviewed["visual_review_completed"] or reviewed["pdfs_sha256"] != built["pdfs_sha256"] or reviewed["figure_png_sha256"] != sha(HERE / "figures" / (FIGURE + ".png")):
        raise ValueError("current visual review required")
    for name, digest in built["pdfs_sha256"].items():
        if sha(HERE / name) != digest:
            raise ValueError("reviewed PDF changed")
    package = DEST / DELIVERY
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
            backup = DEST / (target.stem + "_before_native_canvas_factorization_" + sha(target)[:12] + target.suffix)
            if backup.exists() and sha(backup) != sha(target):
                raise ValueError("prior alias backup differs; preserve before continuing")
            if not backup.exists():
                shutil.copy2(target, backup)
        shutil.copy2(HERE / source, target)
        if sha(HERE / source) != sha(target):
            raise ValueError("latest alias differs")
    record = dict(package=str(package), files_sha256=files, aliases=aliases,
        local_copy_verified=True, cloud_propagation_verified=False, visual_review_completed=True)
    (QA / "sync.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Verified local Nutstore package and five current aliases.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("build", "sync"), required=True)
    args = p.parse_args()
    {"build": build, "sync": sync}[args.stage]()
