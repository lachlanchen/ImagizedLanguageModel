#!/usr/bin/env python3
"""Validate the complete article package and copy it into Nutstore Projects/ILM."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import openpyxl

HERE = Path(__file__).resolve().parent
DESTINATION = Path("/home/lachlan/Nutstore Files/Projects/ILM")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = json.loads((HERE / "evidence/v53_development_report.json").read_text())
    for name in ("manuscript", "manuscript_review", "supplementary_information"):
        pdf = HERE / f"{name}.pdf"
        subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True)
        extracted = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True)
        if "0.586" not in extracted or "361.6" not in extracted:
            raise ValueError(f"{name} lacks current measured results")
        log = (HERE / f".{name}/{name}.log").read_text()
        if any(
            text in log
            for text in ("Overfull", "undefined references", "undefined citations")
        ):
            raise ValueError(f"{name} has unresolved layout/references")
    with zipfile.ZipFile(HERE / "ILM_Figures_editable.pptx") as deck:
        for name in deck.namelist():
            if name.endswith((".xml", ".rels")):
                ET.fromstring(deck.read(name))
        chart = ET.fromstring(deck.read("ppt/charts/chart1.xml"))
        namespace = {"c": "http://schemas.openxmlformats.org/drawingml/2006/chart"}
        chart_series = chart.findall(".//c:barChart/c:ser", namespace)
        if len(chart_series) != 3:
            raise ValueError("missing native evidence chart")
        workbook = openpyxl.load_workbook(
            io.BytesIO(deck.read("ppt/embeddings/evidence.xlsx")), data_only=True
        )
        for index, route in enumerate(("ordered", "shuffled", "zero_condition")):
            for position, value in enumerate(
                report["generation"][route]["identity_top1_by_position"]
            ):
                cell = workbook.active.cell(position + 2, index + 2).value
                if abs(cell - 100 * value) > 1e-8:
                    raise ValueError("editable workbook differs from audit")
    target = DESTINATION / "ILM_NatureStyle_20261104_V53"
    target.mkdir(parents=True, exist_ok=True)
    copied = {}
    for path in sorted(HERE.rglob("*")):
        relative = path.relative_to(HERE)
        if not path.is_file() or any(
            part.startswith(".") or part == "__pycache__" for part in relative.parts
        ):
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if digest(destination) != digest(path):
            raise IOError(f"copy mismatch: {relative}")
        copied[str(relative)] = digest(path)
    aliases = {
        "manuscript.pdf": "ILM_Manuscript_LATEST.pdf",
        "manuscript_review.pdf": "ILM_Manuscript_Review_LATEST.pdf",
        "supplementary_information.pdf": "ILM_Supplementary_LATEST.pdf",
        "ILM_Figures_editable.pptx": "ILM_Figures_editable_LATEST.pptx",
    }
    for source, name in aliases.items():
        destination = DESTINATION / name
        if destination.exists() and digest(destination) != digest(HERE / source):
            previous = target / ("previous_" + name)
            if not previous.exists():
                shutil.copy2(destination, previous)
        shutil.copy2(HERE / source, destination)
        if digest(destination) != digest(HERE / source):
            raise IOError(f"alias mismatch: {name}")
    receipt = {
        "destination": target.name,
        "files_sha256": copied,
        "aliases": aliases,
        "local_copy_verified": True,
        "cloud_delivery_verified": False,
        "pptx_validation": "OOXML and embedded evidence workbook; LibreOffice PDF inspected separately",
    }
    (target / "sync_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(target)
    print(f"Verified {len(copied)} files and {len(aliases)} current aliases")


if __name__ == "__main__":
    main()
