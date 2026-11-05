#!/usr/bin/env python3
"""Verify the V62 change-marked supplement and optionally copy its review bundle."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DESTINATION = Path.home()/"Nutstore Files/Projects/ILM"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_hashes(document):
    result = []
    for page in document:
        for item in page.get_image_info(xrefs=True):
            pixels = fitz.Pixmap(document, item["xref"])
            result.append((pixels.width, pixels.height, hashlib.sha256(pixels.samples).hexdigest()))
    return Counter(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-dir", type=Path, default=ROOT/"artifacts/v62_supp_redline_qa_20261105")
    parser.add_argument("--sync", action="store_true")
    args = parser.parse_args()
    baseline = json.loads((args.qa_dir/"baseline.json").read_text())
    visual = json.loads((args.qa_dir/"visual_review.json").read_text())
    pdf = HERE/"redlined_supplementary_information.pdf"
    if (visual["pdf_sha256"] != digest(pdf) or not visual["all_pages_inspected"]
            or visual["pages"] != list(range(1, 16)) or visual["high_resolution_changed_pages"] != [13]):
        raise ValueError("hash-bound complete visual review required")
    for name, value in visual["render_sha256"].items():
        if digest(args.qa_dir/name) != value:
            raise ValueError("reviewed render changed")
    changed = [name for name, value in baseline["publication_files_sha256"].items() if digest(ROOT/name) != value]
    if changed != ["publication/ilm-nature/README.md"]:
        raise ValueError("unplanned existing publication change: "+str(changed))
    build = json.loads((HERE/"evidence/supplement_redline_build.json").read_text())
    for name, value in build["output_sha256"].items():
        if digest(HERE/name) != value:
            raise ValueError("built artifact changed")
    if digest(HERE/"build_supplement_redline.py") != build["builder_source_sha256"]:
        raise ValueError("builder source changed")
    clean, marked = fitz.open(HERE/"supplementary_information.pdf"), fitz.open(pdf)
    if len(marked) != 15 or image_hashes(clean) != image_hashes(marked):
        raise ValueError("page count or raw image pixels changed")
    blue = 0
    for page in marked:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    blue += span["color"] == 255
                    x0, y0, x1, y1 = span["bbox"]
                    if x0 < 0 or y0 < 0 or x1 > page.rect.width+.1 or y1 > page.rect.height+.1:
                        raise ValueError("text outside page")
    if blue < 10:
        raise ValueError("visible revision markup missing")
    new_text = marked[12].get_text()
    numbers = sorted(set(re.findall(r"\d+\.\d{3}", (HERE/"transcript_bounds_table.tex").read_text())))
    if any(number not in new_text for number in numbers):
        raise ValueError("transcript table numbers differ in the review PDF")
    log = (args.qa_dir/"redlined_supplementary_information.log").read_text()
    if any(token in log for token in ("Overfull", "Warning", "undefined", "Undefined control", "LaTeX Error")):
        raise ValueError("redline compile warnings remain")
    for name, value in baseline["primary_nutstore_aliases_sha256"].items():
        if digest(DESTINATION/name) != value:
            raise ValueError("a primary Nutstore alias changed")
    receipt = dict(baseline_commit=build["baseline_commit"], current_source_commit=baseline["source_commit"],
        pages=len(marked), all_pages_visually_reviewed=True, high_resolution_changed_pages=[13],
        unchanged_existing_publication_files=len(baseline["publication_files_sha256"])-1,
        allowed_existing_changes=changed, raw_image_occurrences_preserved=sum(image_hashes(marked).values()),
        marked_blue_text_spans=blue, transcript_table_numbers_verified=numbers,
        no_compile_warnings=True, primary_nutstore_aliases_preserved=True,
        file_sha256={p.name: digest(p) for p in [pdf, pdf.with_suffix(".tex"), HERE/"build_supplement_redline.py", Path(__file__)]},
        baseline_manifest_sha256=digest(args.qa_dir/"baseline.json"), visual_review_sha256=digest(args.qa_dir/"visual_review.json"),
        primary_manuscript_changed=False, new_language_result=False)
    evidence = HERE/"evidence/supplement_redline_verification.json"
    evidence.write_text(json.dumps(receipt, indent=2)+"\n")
    if args.sync:
        target = DESTINATION/"ILM_V62_SupplementRedline_20261105"
        target.mkdir(exist_ok=True)
        names = [pdf.name, pdf.with_suffix(".tex").name, "nature_reading.sty",
            "evidence/supplement_redline_build.json", "evidence/supplement_redline_verification.json"]
        source = pdf.with_suffix(".tex").read_text()
        names.extend(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source))
        names.extend("figures/"+name+".pdf" for name in re.findall(r"\\plate\{([^}]+)\}", source))
        files = {}
        for name in sorted(set(names)):
            dst = target/name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(HERE/name, dst)
            if digest(dst) != digest(HERE/name):
                raise ValueError("Nutstore copy differs")
            files[name] = digest(dst)
        readme = target/"README.md"
        readme.write_text("# Change-marked V62 supplement\n\nReview copy relative to the preserved V62 primary baseline. Additions are blue.\nThe clean article, supplement and editable figure deck remain the primary outputs.\n\nThe flattened TeX, shared style and figure PDFs permit local XeLaTeX compilation.\nBuild and visual/numeric integrity receipts accompany the PDF.\n")
        files[readme.name] = digest(readme)
        delivery = dict(files_sha256=files, total_files=len(files)+1, manifest_excludes_its_own_hash=True,
            locally_byte_verified=True, remote_cloud_propagation_verified=False,
            primary_aliases_unchanged=baseline["primary_nutstore_aliases_sha256"])
        (target/"delivery_manifest.json").write_text(json.dumps(delivery, indent=2)+"\n")
        for name, value in baseline["primary_nutstore_aliases_sha256"].items():
            if digest(DESTINATION/name) != value:
                raise ValueError("primary alias changed while copying")
        print("Locally byte-verified Nutstore review files:", len(files)+1)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
