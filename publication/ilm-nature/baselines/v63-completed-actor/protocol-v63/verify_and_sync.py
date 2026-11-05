#!/usr/bin/env python3
"""Verify source pixels/native shapes and deliver the reviewed methods companion."""

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import fitz
import numpy as np
from PIL import Image, ImageFont
from pptx import Presentation


HERE = Path(__file__).resolve().parent
PUB = HERE.parent
ROOT = PUB.parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixel_key(array):
    return array.shape, hashlib.sha256(array.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", default=str(ROOT/"artifacts/v63_method_qa_20261105"))
    parser.add_argument("--destination", default=str(Path.home()/"Nutstore Files/Projects/ILM/ILM_V63_Training_Method_20261105"))
    args = parser.parse_args()
    qa, destination = Path(args.qa), Path(args.destination)
    review = json.loads((qa/"visual_review.json").read_text())
    if not review["all_pages_and_figure_inspected"]:
        raise ValueError("completed rendered visual review required")
    for name, expected in review["reviewed_sha256"].items():
        if digest(ROOT/name) != expected:
            raise ValueError("reviewed artifact changed")
    log = (qa/"methods.log").read_text()
    if any(word in log for word in ("Overfull", "undefined", "Undefined", "LaTeX Warning", "Package natbib Warning")):
        raise ValueError("unresolved TeX diagnostics")
    baseline = json.loads((ROOT/"references/evidence/v63_method_publication_baseline.json").read_text())
    if any(digest(PUB/name) != value for name, value in baseline["sha256"].items()):
        raise ValueError("an existing manuscript/deck/figure changed")
    manifest = json.loads((HERE/"figure_sources.json").read_text())
    if manifest["generator_sha256"] != digest(HERE/"generate_method_figure.py"):
        raise ValueError("figure generator changed after generation")
    for name, expected in manifest["inputs_sha256"].items():
        if digest(ROOT/name) != expected:
            raise ValueError("frozen figure source changed")
    for name, expected in manifest["artifacts"].items():
        if digest(HERE/name) != expected:
            raise ValueError("figure/deck changed after generation")
    expected = Counter()
    for row in manifest["displayed_images"]:
        path = HERE/"images"/(row["name"]+".png")
        image = np.asarray(Image.open(path).convert("L"))
        if digest(path) != row["png_sha256"] or pixel_key(image)[1] != row["pixel_sha256"]:
            raise ValueError("display source pixels changed")
        expected[pixel_key(image)] += 1
    pdf_counts = {}
    for path, pages in ((HERE/"question_assignment_method.pdf", 1), (qa/"methods.pdf", 2),
                        (qa/"question_assignment_method_editable.pdf", 1)):
        pdf = fitz.open(path)
        if len(pdf) != pages:
            raise ValueError("unexpected reviewed page count")
        actual = Counter()
        for page in pdf:
            for item in page.get_images(full=True):
                xref = item[0]
                placements = page.get_image_rects(xref, transform=True)
                if len(placements) != 1:
                    raise ValueError("unexpected source image reuse")
                matrix = placements[0][1]
                if matrix.b or matrix.c or not matrix.a or not matrix.d:
                    raise ValueError("rotated or degenerate evidence bitmap")
                image = np.asarray(Image.open(io.BytesIO(pdf.extract_image(xref)["image"])).convert("L"))
                if matrix.d < 0:
                    image = image[::-1]
                if matrix.a < 0:
                    image = image[:, ::-1]
                actual[pixel_key(image)] += 1
        if actual != expected:
            raise ValueError(f"PDF changed displayed source pixels: {path.name}")
        pdf_counts[path.name] = sum(actual.values())
    deck = Presentation(HERE/"question_assignment_method_editable.pptx")
    if len(deck.slides) != 1 or (deck.slide_width, deck.slide_height) != (15240000, 8572500):
        raise ValueError("native figure slide geometry differs")
    actual = Counter()
    native_texts = 0
    fontpath = subprocess.check_output(["fc-match", "-f", "%{file}", "Arial"], text=True).strip()
    for shape in deck.slides[0].shapes:
        if not (shape.left >= 0 and shape.top >= 0 and shape.left+shape.width <= deck.slide_width
                and shape.top+shape.height <= deck.slide_height):
            raise ValueError("native object outside slide bounds")
        if shape.shape_type == 13:
            if any((shape.crop_left, shape.crop_right, shape.crop_top, shape.crop_bottom)):
                raise ValueError("evidence picture cropped")
            actual[pixel_key(np.asarray(Image.open(io.BytesIO(shape.image.blob)).convert("L")))] += 1
        if shape.has_text_frame and shape.text_frame.text:
            native_texts += 1
            for paragraph in shape.text_frame.paragraphs:
                if paragraph.text:
                    font = ImageFont.truetype(fontpath, round(paragraph.font.size.pt*96/72))
                    if font.getlength(paragraph.text) > shape.width/9525 + 2:
                        raise ValueError("native text exceeds its explicit width")
    if actual != expected or native_texts < 20:
        raise ValueError("editable text/source image evidence differs")
    with zipfile.ZipFile(PUB/"baselines/v62-primary/ILM_Figures_editable.pptx") as old, \
            zipfile.ZipFile(HERE/"question_assignment_method_editable.pptx") as new:
        if old.read("ppt/theme/theme1.xml") != new.read("ppt/theme/theme1.xml"):
            raise ValueError("publication theme changed")
    for name in ("methods.pdf", "question_assignment_method_editable.pdf"):
        shutil.copyfile(qa/name, HERE/name)
    receipt = dict(scope="Methods and completed implementation preflight, not autonomous-language outcomes",
        baseline_commit=baseline["baseline_commit"], existing_publication_files_unchanged=len(baseline["sha256"]),
        pdf_rasters_exact_in_display_coordinates=pdf_counts, ppt_rasters_exact=4,
        editable_text_objects=native_texts, slide_count=1, native_bounds_and_text_widths_checked=True,
        source_theme_xml_unchanged=True, reviewed_pages=4, visual_review_sha256=digest(qa/"visual_review.json"),
        tex_log_sha256=digest(qa/"methods.log"), tex_diagnostics_clean=True,
        no_previous_companion_tex_baseline=True, compiled_redline_claim=False,
        verifier_sha256=digest(__file__), model_generation_calls=0, language_quality_claim=False)
    (HERE/"verification.json").write_text(json.dumps(receipt, indent=2)+"\n")
    names = [str(p.relative_to(HERE)) for p in HERE.rglob("*") if p.is_file()
             and p.suffix in (".md", ".py", ".tex", ".pdf", ".png", ".svg", ".pptx", ".json")
             and "__pycache__" not in p.parts and p.name != "delivery_manifest.json"]
    hashes = {name: digest(HERE/name) for name in sorted(names)}
    aliases = [destination.parent/name for name in ("ILM_Manuscript_LATEST.pdf", "ILM_Manuscript_Review_LATEST.pdf",
        "ILM_Supplementary_LATEST.pdf", "ILM_Figures_editable_LATEST.pptx", "ILM_Figures_review_LATEST.pdf")]
    before_aliases = {p.name: digest(p) for p in aliases}
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected_hash in hashes.items():
        target = destination/name
        if target.exists() and digest(target) != expected_hash:
            raise FileExistsError("existing delivery differs; select a new --destination")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE/name, target)
        if digest(target) != expected_hash:
            raise ValueError("copied asset differs")
    if before_aliases != {p.name: digest(p) for p in aliases}:
        raise ValueError("a main manuscript LATEST alias changed")
    delivery = dict(destination_folder=destination.name, assets=hashes, byte_verified_files=len(hashes),
        main_latest_aliases_preserved_sha256=before_aliases, remote_cloud_propagation_verified=False)
    (HERE/"delivery_manifest.json").write_text(json.dumps(delivery, indent=2)+"\n")
    shutil.copyfile(HERE/"delivery_manifest.json", destination/"delivery_manifest.json")
    if digest(HERE/"delivery_manifest.json") != digest(destination/"delivery_manifest.json"):
        raise ValueError("delivery manifest copy differs")
    print(json.dumps(receipt, indent=2))
    print(json.dumps(dict(byte_verified_assets=len(hashes)+1, destination_folder=destination.name)))


if __name__ == "__main__":
    main()
