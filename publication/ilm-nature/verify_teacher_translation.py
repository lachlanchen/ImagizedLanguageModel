"""Audit the two releases' pixels, editable chart values and measured weights."""

from collections import Counter
from io import BytesIO
from pathlib import Path
import json
from zipfile import ZipFile
from lxml import etree
from openpyxl import load_workbook
import fitz
from verify_raw_ilm import pixels, digest

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[1]
EVIDENCE = ROOT / "references/evidence/teacher_translation_v1"


def read(name):
    return json.loads((EVIDENCE / name).read_text())


def verify():
    expected = {}
    all_pixels = {}
    for index, name in ((55, "natural-rasters"), (57, "limits-rasters")):
        expected[index] = Counter()
        for it in read(name + ".json")["items"]:
            if it["type"] == "image":
                f = PAPER / "figures/teacher_translation" / it["name"]
                assert digest(f) == it["sha256"]
                value = pixels(f.read_bytes())
                expected[index][value] += 1
                all_pixels[f.name] = value
    reading = PAPER / "visual_language_paper.pdf"
    cumulative = PAPER / "manuscript.pdf"
    with fitz.open(reading) as a, fitz.open(cumulative) as b:
        assert (len(a), len(b)) == (58, 307)
        for i, p in enumerate(a):
            assert "".join(p.get_text().split()) == "".join(b[i].get_text().split()), i
        for doc in (a, b):
            for i, wanted in expected.items():
                actual = Counter()
                for im in doc[i].get_image_info(xrefs=True):
                    value = pixels(doc.extract_image(im["xref"])["image"])
                    rect = fitz.Rect(im["bbox"])
                    assert (
                        abs(rect.width / rect.height * value[0][1] / value[0][0] - 1)
                        < 0.001
                    )
                    actual[value] += 1
                assert wanted == actual
    deck = PAPER / "figures/Teaching_a_visual_translator.pptx"
    with ZipFile(deck) as z:
        stored = {pixels(z.read(p)) for p in z.namelist() if p.startswith("ppt/media/")}
        assert set(all_pixels.values()) == stored
        books = sorted(n for n in z.namelist() if n.endswith(".xlsx"))
        charts = sorted(
            n for n in z.namelist() if "/charts/chart" in n and n.endswith(".xml")
        )
        for panel, x, c in zip(
            read("charts.json")["panels"], books, charts, strict=True
        ):
            wb = load_workbook(BytesIO(z.read(x)), read_only=True, data_only=True)
            rows = list(wb.active.values)
            wb.close()
            assert list(rows[0][1::2]) == list(panel["series"])
            assert [str(r[0]) for r in rows[1:]] == panel["labels"]
            expected_values = list(panel["series"].values())
            assert [
                list(col) for col in zip(*(r[1::2] for r in rows[1:]))
            ] == expected_values
            xml = etree.fromstring(z.read(c))
            assert [
                [
                    float(v)
                    for v in s.xpath(
                        './/*[local-name()="val"]//*[local-name()="pt"]/*[local-name()="v"]/text()'
                    )
                ]
                for s in xml.xpath('//*[local-name()="ser"]')
            ] == expected_values
    for arm in ("corpus", "distilled"):
        model = ROOT / "models/teacher_translation" / arm
        fit = json.loads((model / "fit.json").read_text())
        assert digest(model / "weights.safetensors") == fit["weights"]
        assert all(
            digest(ROOT / name) == value for name, value in fit["binding"].items()
        )
    run = ROOT / "artifacts/teacher_translation_v1"
    result = read("results.json")
    for field in ("generated", "graded"):
        assert digest(run / (field + ".json")) == result[field]
    review = read("fresh-source-review.json")
    assert digest(run / "all-fresh-packet.json") == review["board"]
    assert digest(run / "all-fresh-reading.json") == review["readings"]
    assert result["graded"] == review["source"]
    forks = read("forks.json")
    for name, path in {
        "scores": run / "fork-scores.json",
        "program": ROOT / "scripts/measure_translation_forks.py",
        "proof": ROOT / "formal/span_list/TeacherSurfaceFork.lean",
        "protocol": ROOT / "references/teacher_surface_fork_protocol.md",
    }.items():
        assert digest(path) == forks[name]
    with fitz.open(run / "office/Teaching_a_visual_translator.pdf") as export:
        assert len(export) == 3
    return dict(
        reading_pages=58,
        cumulative_pages=307,
        native_charts=len(charts),
        native_workbooks=len(books),
        distinct_raw_png_files=len(all_pixels),
        all_raw_pixels_and_aspect_ratios_verified=True,
        entire_reading_text_verified_in_archive=True,
        files={
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                reading,
                cumulative,
                deck,
                EVIDENCE / "results.json",
                EVIDENCE / "fresh-source-review.json",
                EVIDENCE / "forks.json",
                EVIDENCE / "neighbors.json",
                ROOT / "formal/span_list/TeacherSurfaceFork.lean",
            )
        },
    )


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
