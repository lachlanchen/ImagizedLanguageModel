#!/usr/bin/env python3
"""Build and locally deliver the assessed control/balanced-writing revision."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess

import fitz
from PIL import Image

from balanced_writing_figures import BASE, FIGURES, HERE, QA, ROOT, SOURCES, baseline, evidence, sha, verify
from build_paired_writing_publication import build_slot
from build_balanced_writing_redlines import prepare_baseline

DEST = Path("/home/lachlan/Nutstore Files/Projects/ILM")
DELIVERY = "ILM_NatureStyle_20261108_BalancedWriting"


def build():
    with build_slot():
        QA.mkdir(parents=True, exist_ok=True)
        evidence()
        deck = verify()
        rendered = json.loads((HERE / "evidence/balanced_writing_figure_render.json").read_text())
        if (rendered["review_pdf_sha256"] != sha(HERE / "ILM_Figures_review.pdf")
                or rendered["pptx_sha256"] != sha(HERE / "ILM_Figures_editable.pptx")
                or len(rendered["preserved_prior_slide_pixels"]) != 40):
            raise ValueError("current 41-slide render with preserved baseline required")
        for name, path in SOURCES.items():
            shutil.copy2(path, HERE / "evidence" / ("balanced_writing_" + name + "_source.json"))
        manifest = prepare_baseline()
        preserved = {}
        for name, digest in manifest["files_sha256"].items():
            if name.startswith("figures/") or name == "nature_reading.sty":
                if sha(HERE / name) != digest:
                    raise ValueError("earlier figure/style changed: " + name)
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
            for required in ("15/32", "13/32", "recognition", "balanced"):
                if required.lower() not in normalized.lower():
                    raise ValueError("completed evidence missing from PDF: " + name + ": " + required)
            if name == "supplementary_information":
                for required in ("Supplementary Note 49", "Fig. S26"):
                    if required not in normalized:
                        raise ValueError("new note/figure absent: " + required)
            outputs[name + ".pdf"] = sha(HERE / (name + ".pdf"))
        result = subprocess.run(["python", "build_balanced_writing_redlines.py"], cwd=HERE, capture_output=True, text=True)
        (QA / "redlines.build.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError("redline compilation failed")
        redlines = json.loads((HERE / "evidence/balanced_writing_redline_build.json").read_text())
        inputs = {}
        for row in redlines["documents"].values():
            inputs.update(row["new_inputs_sha256"])
        inputs["nature_reading.sty"] = sha(HERE / "nature_reading.sty")
        for name, digest in inputs.items():
            if sha(HERE / name) != digest:
                raise ValueError("compiled TeX input changed: " + name)
        for name in ("redlined_manuscript", "redlined_supplementary_information"):
            text = subprocess.check_output(["pdftotext", str(HERE / (name + ".pdf")), "-"], text=True)
            normalized = " ".join(text.split())
            if "13/32" not in normalized or "d8b2aa5 40-slide writing baseline" not in normalized:
                raise ValueError("redline lacks current result or baseline notice")
            outputs[name + ".pdf"] = sha(HERE / (name + ".pdf"))
        for name in ("ILM_Figures_review.pdf", *("figures/" + f + ".pdf" for f in FIGURES)):
            outputs[name] = sha(HERE / name)
        for name in outputs:
            result = subprocess.run(["qpdf", "--check", str(HERE / name)], capture_output=True, text=True)
            (QA / (name.replace("/", "_") + ".qpdf.log")).write_text(result.stdout + result.stderr)
            if result.returncode:
                raise ValueError("PDF structural failure: " + name)
        expected_pixels = {}
        for picture in deck["pictures"]:
            with Image.open(ROOT / picture["source"]) as image:
                image = image.convert("RGB")
                expected_pixels.setdefault(picture["slide"], set()).add((image.size, hashlib.sha256(image.tobytes()).hexdigest()))
        preserved_pdf_images = {}
        for name in ("supplementary_information.pdf", "redlined_supplementary_information.pdf", "ILM_Figures_review.pdf", *("figures/" + f + ".pdf" for f in FIGURES)):
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
            required = expected_pixels[41 + FIGURES.index(Path(name).stem)] if name.startswith("figures/") else set().union(*expected_pixels.values())
            if not required <= pixels:
                raise ValueError("PDF changed original reference/answer pixels: " + name)
            preserved_pdf_images[name] = True
        old = fitz.open(stream=baseline("supplementary_information.pdf"), filetype="pdf")
        new = fitz.open(HERE / "supplementary_information.pdf")
        boundary = next(i for i, page in enumerate(old) if "Reproduction\nFrom the repository root" in page.get_text())
        if boundary != 57:
            raise ValueError("unexpected baseline supplement boundary")
        preserved_si = []
        for i in range(boundary):
            a, b = old[i].get_pixmap(alpha=False), new[i].get_pixmap(alpha=False)
            if a.samples != b.samples:
                raise ValueError("supplement before Note 49 changed: page " + str(i + 1))
            preserved_si.append(hashlib.sha256(b.samples).hexdigest())
        record = dict(baseline_commit=BASE, pdfs_sha256=outputs,
            pptx_sha256=sha(HERE / "ILM_Figures_editable.pptx"),
            current_revision="control_balanced_writing",
            source_receipts_sha256={name: sha(path) for name, path in SOURCES.items()},
            preserved_individual_figures_and_style_sha256=preserved, preserved_supplement_pages=preserved_si,
            preserved_native_slides=40, current_native_slides=41, native_chart_count=49,
            preserved_raw_answer_images_in_pdfs=preserved_pdf_images, builder_sha256=sha(__file__),
            tex_inputs_sha256=inputs,
            redline_builder_sha256=sha(HERE / "build_balanced_writing_redlines.py"),
            pdf_pages={name: len(fitz.open(HERE / name)) for name in outputs},
            all_compiled_without_overfull_or_undefined=True, visual_review_required=True,
            artifact_tool_available=False, language_quality_claim=False, general_ilm_goal_complete=False)
        (HERE / "evidence/balanced_writing_publication_build.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(dict(pages=record["pdf_pages"], prior_figures=len(preserved) - 1, preserved_si_pages=len(preserved_si)), indent=2))


def sync():
    built = json.loads((HERE / "evidence/balanced_writing_publication_build.json").read_text())
    reviewed = json.loads((HERE / "evidence/balanced_writing_visual_review.json").read_text())
    if not reviewed["visual_review_completed"] or reviewed["pdfs_sha256"] != built["pdfs_sha256"] or reviewed["figure_png_sha256"] != {f: sha(HERE / "figures" / (f + ".png")) for f in FIGURES}:
        raise ValueError("current visual review required")
    if reviewed["pptx_sha256"] != built["pptx_sha256"] or sha(HERE / "ILM_Figures_editable.pptx") != built["pptx_sha256"]:
        raise ValueError("reviewed editable deck changed")
    if sha(__file__) != built["builder_sha256"] or sha(HERE / "build_balanced_writing_redlines.py") != built["redline_builder_sha256"]:
        raise ValueError("publication builder changed")
    for name, digest in built["tex_inputs_sha256"].items():
        if sha(HERE / name) != digest:
            raise ValueError("reviewed TeX input changed")
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
            backup = DEST / (target.stem + "_before_balanced_writing_" + sha(target)[:12] + target.suffix)
            if backup.exists() and sha(backup) != sha(target):
                raise ValueError("prior alias backup differs")
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
