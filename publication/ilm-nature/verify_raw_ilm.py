"""Verify embedded glyph pixels, proportions and editable chart data in the paper."""

from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZipFile

import fitz
from lxml import etree
from openpyxl import load_workbook
from PIL import Image

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[1]
FIGURES = PAPER / "figures"
EVIDENCE = ROOT / "references/evidence/paired_ink_v1"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def pixels(payload):
    with Image.open(BytesIO(payload)) as im:
        return im.size, sha256(im.convert("L").tobytes()).hexdigest()


def verify():
    raster = json.loads((EVIDENCE / "raw_rasters.json").read_text())
    source = {}
    for name, entry in raster["files"].items():
        p = FIGURES / "raw_ilm" / name
        assert digest(p) == entry["sha256"], name
        source[name] = pixels(p.read_bytes())
    scene = json.loads((EVIDENCE / "scene.json").read_text())
    pages = {0: ["role-54-input.png", "role-54-output.png"],
             44: [item["name"] for item in scene["items"] if item["type"] == "image"]}
    for name in ("visual_language_paper.pdf", "manuscript.pdf"):
        with fitz.open(PAPER / name) as article:
            for page, names in pages.items():
                matches = Counter()
                for img in article[page].get_image_info(xrefs=True):
                    image = pixels(article.extract_image(img["xref"])["image"])
                    if image in source.values():
                        rect = fitz.Rect(img["bbox"])
                        w, h = image[0]
                        assert abs((rect.width / rect.height) / (w / h) - 1) < 0.001
                        matches[image] += 1
                assert not (Counter(source[n] for n in names) - matches), (name, page)
    with fitz.open(PAPER / "visual_language_paper.pdf") as reading, fitz.open(
        PAPER / "manuscript.pdf"
    ) as archive:
        assert (len(reading), len(archive)) == (46, 295)
        for index, page in enumerate(reading):
            assert "".join(page.get_text().split()) == "".join(archive[index].get_text().split()), index

    deck = FIGURES / "Raw_ILM_images.pptx"
    chart_data = json.loads((EVIDENCE / "charts.json").read_text())
    with ZipFile(deck) as z:
        media = Counter(pixels(z.read(k)) for k in z.namelist() if k.startswith("ppt/media/"))
        assert not (Counter(source.values()) - media)
        charts = sorted(k for k in z.namelist() if re.search(r"/charts/chart\d+\.xml$", k))
        books = sorted(k for k in z.namelist() if k.endswith(".xlsx"))
        for panel, chart, book in zip(chart_data["panels"], charts, books, strict=True):
            expect = list(panel["series"].values())
            series = etree.fromstring(z.read(chart)).xpath('//*[local-name()="ser"]')
            values = './/*[local-name()="val"]//*[local-name()="pt"]/*[local-name()="v"]/text()'
            assert [[float(v) for v in s.xpath(values)] for s in series] == expect
            wb = load_workbook(BytesIO(z.read(book)), read_only=True, data_only=True)
            rows = list(wb.active.values)
            assert list(rows[0][1::2]) == list(panel["series"])
            assert [r[0] for r in rows[1:]] == panel["labels"]
            assert list(map(list, zip(*(r[1::2] for r in rows[1:])))) == expect
            wb.close()
    for mode in ("likelihood", "assignment"):
        folder = ROOT / "models/paired_ink" / mode
        fit = json.loads((folder / "fit.json").read_text())
        assert digest(folder / "weights.safetensors") == fit["weights"]
        assert all(digest(ROOT / f) == h for f, h in fit["binding"].items())
    assert digest(ROOT / "formal/span_list/PairedInkOdds.lean") == raster["formal"]
    paths = [PAPER / "reading/raw_ilm.tex", PAPER / "reading/visual_language_paper.tex",
             PAPER / "visual_language_paper.pdf", PAPER / "manuscript.pdf", deck,
             PAPER / "raw_ilm_figure.mjs", EVIDENCE / "raw_rasters.json"]
    for prefix in ("Figure_193_", "Figure_194_"):
        paths.extend(sorted(FIGURES.glob(prefix + "*")))
    return dict(reading_pages=46, cumulative_pages=295, reading_nonwhitespace_text_matches_cumulative=True,
                full_rasters_in_pdf_by_page={k+1: v for k, v in pages.items()},
                source_and_emitted_pixel_content_verified=True, native_charts=len(charts),
                literal_chart_workbooks_verified=len(books), slide_source_rasters_verified=len(source),
                matched_fit_bindings_verified=True,
                files={str(p.relative_to(ROOT)): digest(p) for p in paths})


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
