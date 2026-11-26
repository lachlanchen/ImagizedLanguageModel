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
EVIDENCE = ROOT / "references/evidence/library_diversity_v1"


def read(name):
    return json.loads((EVIDENCE / name).read_text())


def verify():
    expected = {}
    all_pixels = {}
    for index, name in ((58, "natural-rasters"), (60, "contrast-rasters"), (61, "memory-rasters")):
        expected[index] = Counter()
        for it in read(name + ".json")["items"]:
            if it["type"] == "image":
                f = PAPER / "figures/library_diversity" / it["name"]
                assert digest(f) == it["sha256"]
                value = pixels(f.read_bytes())
                expected[index][value] += 1
                all_pixels[f.name] = value
    reading = PAPER / "visual_language_paper.pdf"
    cumulative = PAPER / "manuscript.pdf"
    with fitz.open(reading) as a, fitz.open(cumulative) as b:
        assert (len(a), len(b)) == (62, 311)
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
    deck = PAPER / "figures/Bilingual_writing_on_unseen_books.pptx"
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
    for arm in ("repeat", "broad"):
        model = ROOT / "models/library_diversity" / arm
        fit = json.loads((model / "fit.json").read_text())
        assert digest(model / "weights.safetensors") == fit["weights"]
        assert all(
            digest(ROOT / name) == value for name, value in fit["binding"].items()
        )
    run = ROOT / "artifacts/library_diversity_v1"
    evidence = ROOT / "references/evidence"
    memory = ROOT / "artifacts/library_memory_v1"
    for folder, record in ((run, read("evaluation.json")), (run, read("contrast.json")),
                           (memory, json.loads((evidence / "library_memory_v1/evaluation.json").read_text()))):
        assert all(digest(folder / name) == value for name,value in record["files"].items())
    raw = json.loads((run / "generated.json").read_text())
    con = json.loads((run / "contrast-generated.json").read_text())
    mem = json.loads((memory / "generated.json").read_text())
    for check in read("illustrated.json")["checks"]:
        i, m = check["request"], check["method"]
        data = mem if m == "associative" else con if m == "contrast" else raw
        src = memory if m == "associative" else run
        path = src / "pixels" / f"{i:04d}" / (m + ".png")
        out = data["answers"][i][m]
        assert check["cpu_matches"] and check["original"] == check["cpu"] == digest(path) == out["png"]
        assert check["original_stopped"] == check["cpu_stopped"] == out["stop"]
        packaged = PAPER / "figures/library_diversity" / f"request-{i}-{m}.png"
        assert digest(packaged) == out["png"]
    cal = json.loads((evidence / "library_memory_v1/calibration.json").read_text())
    assert all(digest(memory / name) == value for name,value in cal["files"].items())
    support = json.loads((evidence / "library_memory_v1/phrase-support.json").read_text())
    assert support["calibration"] == digest(evidence / "library_memory_v1/calibration.json")
    alignment = json.loads((evidence / "library_roles_v1/frozen-attention.json").read_text())
    assert all(digest(ROOT / name) == value for name, value in alignment["binding"].items())
    with fitz.open(reading) as doc:
        paragraph = doc[-1].get_text().split("Which source glyphs received attention?")[1]
        for group in ("total", "en/predicate"):
            measure = alignment["results"]["broad"][group]
            assert format(measure["positions"], ",") in paragraph
            assert all(format(measure[k], ".2%") in paragraph for k in ("observed", "expected"))
    with fitz.open(run / "office/Bilingual_writing_on_unseen_books.pdf") as export:
        assert len(export) == 4
    return dict(
        reading_pages=62,
        cumulative_pages=311,
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
                EVIDENCE / "evaluation.json",
                EVIDENCE / "contrast.json",
                evidence / "library_memory_v1/evaluation.json",
                evidence / "library_memory_v1/phrase-support.json",
                evidence / "library_memory_v1/library-coverage.json",
                ROOT / "formal/span_list/PairedLibrarySampling.lean",
                ROOT / "formal/span_list/BoundedGlyphEvidence.lean",
                ROOT / "formal/span_list/RoleAttentionNull.lean",
                evidence / "library_roles_v1/frozen-attention.json",
            )
        },
    )


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
