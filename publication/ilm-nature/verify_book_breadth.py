"""Verify publication copies against retained real image and chart data."""

from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZipFile
import fitz
from lxml import etree
from openpyxl import load_workbook
from verify_raw_ilm import pixels, digest

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[1]
EVIDENCE = ROOT / "references/evidence/book_breadth_v1"
FIG = PAPER / "figures"


def verify():
    decks = FIG / "Reading_bilingual_books.pptx"
    assets = {}
    filenames = {}
    presentations = {}
    for page, manifest in ((47, "rasters"), (48, "credited")):
        scene = json.loads((EVIDENCE / (manifest + ".json")).read_text())
        expected = []
        for elem in scene["items"]:
            if elem["type"] != "image":
                continue
            file = FIG / "book_breadth" / elem["name"]
            assert digest(file) == elem["sha256"]
            assets[file.name] = pixels(file.read_bytes())
            expected.append(file.name)
        filenames[page + 1] = expected
        presentations[page] = Counter(assets[name] for name in expected)
    with fitz.open(PAPER / "visual_language_paper.pdf") as reading, fitz.open(
        PAPER / "manuscript.pdf"
    ) as archive:
        assert (len(reading), len(archive)) == (49, 298)
        for p, page in enumerate(reading):
            assert "".join(page.get_text().split()) == "".join(
                archive[p].get_text().split()
            ), p
        for doc in [reading, archive]:
            for p, expected in presentations.items():
                found = Counter()
                for im in doc[p].get_image_info(xrefs=True):
                    item = pixels(doc.extract_image(im["xref"])["image"])
                    found[item] += 1
                    if item in expected:
                        rect = fitz.Rect(im["bbox"])
                        assert (
                            abs(rect.width / rect.height * item[0][1] / item[0][0] - 1)
                            < 0.001
                        )
                assert not expected - found
    panels = json.loads((EVIDENCE / "charts.json").read_text())["panels"]
    with ZipFile(decks) as z:
        image_counts = Counter(
            pixels(z.read(k)) for k in z.namelist() if k.startswith("ppt/media/")
        )
        assert not Counter(assets.values()) - image_counts
        books = sorted(k for k in z.namelist() if k.endswith(".xlsx"))
        charts = sorted(
            k for k in z.namelist() if re.search(r"/charts/chart\d+\.xml$", k)
        )
        for p, f, ch in zip(panels, books, charts, strict=True):
            expected = list(p["series"].values())
            wb = load_workbook(BytesIO(z.read(f)), read_only=True, data_only=True)
            rows = list(wb.active.values)
            assert list(rows[0][1::2]) == list(p["series"])
            assert [str(r[0]) for r in rows[1:]] == p["labels"]
            assert [list(x) for x in zip(*(r[1::2] for r in rows[1:]))] == expected
            wb.close()
            xml = etree.fromstring(z.read(ch))
            actual = [
                [
                    float(v)
                    for v in ser.xpath(
                        './/*[local-name()="val"]//*[local-name()="pt"]/*[local-name()="v"]/text()'
                    )
                ]
                for ser in xml.xpath('//*[local-name()="ser"]')
            ]
            assert actual == expected
    for mode in ("broad", "narrow"):
        folder = ROOT / "models/book_breadth" / mode
        receipt = json.loads((folder / "fit.json").read_text())
        assert receipt["weights"] == digest(folder / "weights.safetensors")
        assert all(digest(ROOT / n) == h for n, h in receipt["binding"].items())
    paths = [
        PAPER / "visual_language_paper.pdf",
        PAPER / "manuscript.pdf",
        decks,
        PAPER / "reading/book_breadth.tex",
        EVIDENCE / "result.json",
        EVIDENCE / "script-result.json",
        EVIDENCE / "exposure.json",
    ]
    for n in (195, 196, 197):
        paths.extend(sorted(FIG.glob(f"Figure_{n}_*")))
    return dict(
        reading_pages=49,
        cumulative_pages=298,
        reading_nonwhitespace_text_matches=True,
        embedded_raw_files_by_page=filenames,
        embedded_pixels_and_aspect_verified=True,
        native_chart_workbooks_verified=len(charts),
        frozen_weights_and_fit_sources_verified=True,
        files={str(p.relative_to(ROOT)): digest(p) for p in paths},
    )


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
