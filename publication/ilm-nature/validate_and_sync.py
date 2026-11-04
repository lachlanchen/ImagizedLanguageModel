#!/usr/bin/env python3
"""Validate the complete article package and copy it into Nutstore Projects/ILM."""

from __future__ import annotations

import hashlib
import io
import json
import posixpath
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import openpyxl

HERE = Path(__file__).resolve().parent
DESTINATION = Path("/home/lachlan/Nutstore Files/Projects/ILM")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = json.loads((HERE / "evidence/v53_development_report.json").read_text())
    density = json.loads((HERE / "evidence/v54_development_report.json").read_text())
    joint_reports = {
        arm: json.loads(
            (HERE / f"evidence/v55_{arm}_development_report.json").read_text()
        )
        for arm in ("frozen", "joint")
    }
    beam = json.loads((HERE / "evidence/v56_report.json").read_text())
    forms = json.loads((HERE / "evidence/v55_form_validity.json").read_text())
    qa = json.loads((HERE / "evidence/v57_development_report.json").read_text())
    reading = {version: json.loads((HERE / f"evidence/{version}_development_report.json").read_text())
               for version in ("v58", "v59", "v60")}
    native = json.loads((HERE / "evidence/v61_development_report.json").read_text())
    correlated = json.loads((HERE / "evidence/v62_development_report.json").read_text())
    transcripts = json.loads((HERE / "evidence/v62_transcript_partial_bounds.json").read_text())
    original_reader = json.loads((HERE / "evidence/v62_vision_transcript_diagnostic.json").read_text())
    if not transcripts["post_hoc"] or transcripts["model_calls"] != 0 or original_reader["decision"] != "invalid_actor_transcription":
        raise ValueError("transcript scope or original decision changed")
    table = (HERE / "transcript_bounds_table.tex").read_text()
    def formatted_interval(values):
        a, b = values
        def number(value):
            return str(Decimal(str(100*value)).quantize(Decimal(".001"), rounding=ROUND_HALF_UP))
        return number(a) if a == b else number(a) + "--" + number(b)
    for route, label in (("k1", "K=1"), ("k1_blank", "K=1, blank Q"), ("k8", "K=8"), ("k8_blank", "K=8, blank Q")):
        row = transcripts["measurements"][route]
        expected = f"{label} & {row['valid_examples']} / {row['unknown_examples']} & " + " & ".join(
            formatted_interval(row["metrics"][metric]["identification_interval"])
            for metric in ("text_exact", "bounded_character_accuracy"))
        if expected not in table or row["valid_examples"] + row["unknown_examples"] != 256:
            raise ValueError("transcript table differs from full-case interval evidence")
    render = json.loads((HERE / "evidence/figure_render_receipt.json").read_text())
    if render["pptx_sha256"] != digest(HERE / "ILM_Figures_editable.pptx"):
        raise ValueError("editable deck has changed since rendering")
    if render["rendered_pdf_sha256"] != digest(HERE / "ILM_Figures_review.pdf"):
        raise ValueError("figure PDF differs from inspected render")
    figure_info = subprocess.check_output(
        ["pdfinfo", str(HERE / "ILM_Figures_review.pdf")], text=True
    )
    if not any(line.split() == ["Pages:", "17"] for line in figure_info.splitlines()):
        raise ValueError("figure render must contain seventeen slides")
    for name in ("manuscript", "manuscript_review", "supplementary_information"):
        pdf = HERE / f"{name}.pdf"
        subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True)
        extracted = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True)
        if any(
            value not in extracted
            for value in (
                "0.586",
                "15.219",
                "0.99949",
                "207.8",
                "7.026",
                "7.579",
                "342.4",
                "11.621",
                "25.446",
                "3.906",
                "15.718",
                "9.766",
                "27.033",
                "11.588",
                "25.340",
                "0.684156" if name == "supplementary_information" else "0.6842",
                "0.1548",
                "0.149",
                "9.912",
                f"{correlated['likelihood']['k1']:.4f}",
                f"{correlated['likelihood']['k8']:.4f}",
                f"{100*correlated['measurements']['k8']['raster_f1']:.3f}",
                "10.174",
            )
        ):
            raise ValueError(f"{name} lacks current measured results")
        if name == "supplementary_information" and any(value not in extracted for value in
                ("122/128", "0.781", "2.806", "2.945", "post-hoc")):
            raise ValueError("supplement lacks calibrated, explicitly post-hoc transcript evidence")
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
        density_book = openpyxl.load_workbook(
            io.BytesIO(deck.read("ppt/embeddings/density.xlsx")), data_only=True
        )
        for row, section, route in (
            (2, "motor", "copy_greedy"),
            (3, "motor", "copy_sample"),
            (4, "generation", "ordered"),
        ):
            item = density[section][route]
            for column, key in ((2, "pixel_f1"), (3, "identity_top1")):
                if (
                    abs(density_book.active.cell(row, column).value - 100 * item[key])
                    > 1e-8
                ):
                    raise ValueError("density workbook differs from audit")
        chart2 = ET.fromstring(deck.read("ppt/charts/chart2.xml"))
        if len(chart2.findall(".//c:barChart/c:ser", namespace)) != 2:
            raise ValueError("missing native density chart")
        slides = ET.fromstring(deck.read("ppt/presentation.xml"))
        pns = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main"}
        if len(slides.findall("p:sldIdLst/p:sldId", pns)) != 17:
            raise ValueError("missing current figure slides")
        rels = ET.fromstring(deck.read("ppt/charts/_rels/chart3.xml.rels"))
        relation_ns = "{http://schemas.openxmlformats.org/package/2006/relationships}"
        package = next(
            rel
            for rel in rels.findall(relation_ns + "Relationship")
            if rel.get("Type").endswith("/package")
        )
        workbook_path = posixpath.normpath(
            posixpath.join("ppt/charts", package.get("Target"))
        )
        matched_book = openpyxl.load_workbook(
            io.BytesIO(deck.read(workbook_path)), data_only=True
        )
        for row, (arm, route) in enumerate(
            (("frozen", "ordered"), ("joint", "ordered"), ("joint", "shuffled")),
            start=2,
        ):
            item = joint_reports[arm]["generation"][route]
            for column, key in ((2, "pixel_f1"), (3, "identity_top1")):
                if (
                    abs(matched_book.active.cell(row, column).value - 100 * item[key])
                    > 1e-8
                ):
                    raise ValueError("matched evidence workbook differs from audit")
        for chart_index, expected in [
            (12, [[correlated["likelihood"][r]] for r in ("k1", "k8")]),
            (13, [[100*correlated["measurements"][r][key]
                   for key in ("raster_f1", "pixel_foreground_f1")]
                  for r in ("k1", "k1_blank", "k8", "k8_blank")]),
            (10, [[native["likelihood"][r]] for r in ("base", "trained")]),
            (11, [[100*native["measurements"][r][key]
                   for key in ("raster_f1", "pixel_foreground_f1")]
                  for r in ("trained", "trained_blank", "base")]),
            (7, [[100 * reading["v58"]["measurements"][r][key]
                  for key in ("exact_answer", "raster_f1")]
                 for r in ("block", "block_blank_question", "glyph")]),
            (8, [[100 * reading["v59"]["measurements"][r][key]
                  for key in ("exact_answer", "raster_f1")]
                 for r in ("student", "student_blank", "mixar", "mixar_blank")]),
            (9, [[100 * reading["v60"]["measurements"][r][key]
                  for key in ("exact_answer", "raster_f1")]
                 for r in ("context", "context_blank", "question", "question_blank")]),
            (
                6,
                [
                    [
                        100 * qa["measurements"][r][key]
                        for key in ("exact_answer", "raster_f1")
                    ]
                    for r in ("copy", "copy_blank_question", "no_copy")
                ],
            ),
            (
                4,
                [
                    [
                        100 * forms["summaries"][n]["generation"]["ordered"][key]
                        for key in (
                            "exact_known_raster_fraction",
                            "exact_target_fraction",
                        )
                    ]
                    for n in ("v54", "v55_frozen", "v55_joint")
                ],
            ),
            (
                5,
                [
                    [
                        100 * beam["measurements"]["generation"][r]["exact_target"],
                        100
                        * beam["measurements"]["feedback"][r][
                            "adjacent_repeated_raster_fraction"
                        ],
                    ]
                    for r in ("beam8", "beam8_shuffled", "greedy")
                ],
            ),
        ]:
            relationships = ET.fromstring(
                deck.read(f"ppt/charts/_rels/chart{chart_index}.xml.rels")
            )
            package = next(
                rel
                for rel in relationships.findall(relation_ns + "Relationship")
                if rel.get("Type").endswith("/package")
            )
            name = posixpath.normpath(
                posixpath.join("ppt/charts", package.get("Target"))
            )
            book = openpyxl.load_workbook(io.BytesIO(deck.read(name)), data_only=True)
            for row, values in enumerate(expected, start=2):
                for column, value in enumerate(values, start=2):
                    if abs(book.active.cell(row, column).value - value) > 1e-8:
                        raise ValueError(
                            f"chart {chart_index} workbook differs from audit"
                        )
    target = DESTINATION / "ILM_NatureStyle_20261105_V62_TranscriptBounds"
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
        "ILM_Figures_review.pdf": "ILM_Figures_review_LATEST.pdf",
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
