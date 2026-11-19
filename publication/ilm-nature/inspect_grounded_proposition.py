"""Inspect the three compiled editions and index embedded figures with one page scan."""
from collections import defaultdict
import hashlib
import io
import json
import os
from time import monotonic

import fitz
from PIL import Image, ImageDraw


def embedded_image(pdf, xref):
    with Image.open(io.BytesIO(pdf.extract_image(xref)["image"])) as source:
        return source.convert("RGB")


def rgb(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


def inspect(base):
    stage = base.STAGE
    base.preserved()
    base.model_sources()
    before = base.load(stage / "before_manifest.json")
    build = base.load(stage / "build_receipt.json")
    started = monotonic()
    result = dict(documents={}, figure_assets_preserved=len(before["figures_sha256"]),
                  automated_checks_are_not_visual_review=True)
    sources = {}
    for stem, _, label in base.FIGURES:
        with Image.open(base.HERE / "figures" / (stem + ".png")) as source:
            sources[label] = (rgb(source.convert("RGB")), source.size,
                              source.getchannel("A").tobytes() if source.mode == "RGBA" else None)
    for folder, name in base.DOCUMENTS:
        path = stage / folder / (name + ".pdf")
        if base.sha(path) != build["documents"][name]:
            raise ValueError(f"Compiled document changed: {name}")
        target = stage / "inspection" / name
        target.mkdir(parents=True, exist_ok=False)
        pages_for = defaultdict(set)
        masks = {}
        texts = []
        with fitz.open(path) as pdf:
            for index, page in enumerate(pdf):
                if base.host()["available_gib"] < base.CPU_PUBLICATION_RESERVE_GIB or base.tree_rss(os.getpid()) > 2:
                    raise RuntimeError("Insufficient memory for publication inspection")
                if monotonic() - started > 180:
                    raise TimeoutError("Image inspection exceeded three minutes")
                texts.append(page.get_text())
                for b in page.get_text("blocks"):
                    if b[6] == 0 and (b[0] < -1 or b[1] < -1 or b[2] > page.rect.width+1 or b[3] > page.rect.height+1):
                        raise ValueError(f"Text outside {name} page {index+1}")
                page.get_pixmap(matrix=fitz.Matrix(.65, .65)).save(target / f"page-{index+1:03}.png")
                for xref, mask, *_ in page.get_images(full=True):
                    pages_for[xref].add(index+1)
                    masks[xref] = mask
            digest_to_xrefs = defaultdict(list)
            for xref in pages_for:
                digest_to_xrefs[rgb(embedded_image(pdf, xref))].append(xref)
            figures = {}
            for label, (digest, size, alpha) in sources.items():
                xrefs = digest_to_xrefs.get(digest, [])
                if not xrefs:
                    raise ValueError(f"Figure absent: {name} {label}")
                for xref in xrefs:
                    if embedded_image(pdf, xref).size != size:
                        raise ValueError(f"Figure dimensions changed: {label}")
                    if alpha is not None and embedded_image(pdf, masks[xref]).convert("L").tobytes() != alpha:
                        raise ValueError(f"Figure transparency changed: {label}")
                figures[label] = dict(RGB_and_alpha_exact=True, pages=sorted(set.union(*(pages_for[x] for x in xrefs))))
            with fitz.open(stage / "before" / (name + ".pdf")) as old:
                previous = {x[0] for page in old for x in page.get_images(full=True)}
                old_images = {rgb(embedded_image(old, x)) for x in previous}
            if old_images.difference(digest_to_xrefs):
                raise ValueError(f"An earlier figure was lost: {name}")
            text = "\n".join(texts)
            for anchor in ("970/1,040", "7/32", "17.795", "softplus", "5,005,760", "7,553,448,370",
                           "324", "689", "108", "ranking", "1,002/1,040", "4,609", "12,597", "314",
                           "238", "138,471", "DuIE"):
                if anchor not in text:
                    raise ValueError(f"Missing text in {name}: {anchor}")
            (target / "text.txt").write_text(text)
            for first in range(0, len(pdf), 36):
                sheet = Image.new("RGB", (1020, 1560), "white")
                draw = ImageDraw.Draw(sheet)
                for i in range(first, min(len(pdf), first+36)):
                    with Image.open(target / f"page-{i+1:03}.png") as tile:
                        tile.thumbnail((165, 235))
                        x, y = (i-first)%6*170, (i-first)//6*260
                        sheet.paste(tile, (x, y+20))
                        draw.text((x+4, y+2), str(i+1), fill="black")
                sheet.save(target / f"overview-{first//36+1}.png")
            result["documents"][name] = dict(pages=len(pdf), pdf_sha256=base.sha(path),
                all_text_bounds_inside_page=True, all_prior_image_contents_retained=len(old_images),
                replaced_prior_image_contents=[], figures=figures, overview_sheets=(len(pdf)+35)//36)
    result["inspection_seconds"] = monotonic()-started
    result["build_plus_inspection_seconds"] = base.load(stage/"child_costs.json")["child_seconds"] + result["inspection_seconds"]
    (stage/"inspection.json").write_text(json.dumps(result, indent=2)+"\n")
