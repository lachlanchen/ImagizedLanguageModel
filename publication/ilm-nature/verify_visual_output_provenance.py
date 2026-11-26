"""Authenticate every figure in the current article against its primary experimental records."""
from collections import Counter
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import json
import re

import fitz
from lxml import etree
from openpyxl import load_workbook
from PIL import Image

PAPER=Path(__file__).resolve().parent
ROOT=PAPER.parents[1]
EVIDENCE=ROOT/"references/evidence/print_translation_v1"
ASSETS=PAPER/"figures/print_translation"


def digest(p):
    return sha256(p.read_bytes()).hexdigest()


def load(p):
    return json.loads(p.read_text())


def raster(blob):
    with Image.open(BytesIO(blob)) as img:
        return img.size,sha256(img.convert("RGB").tobytes()).hexdigest()


def extracted(doc,page):
    return Counter(raster(doc.extract_image(im["xref"])["image"]) for im in page.get_image_info(xrefs=True))


def verify_printed_grades(doc, blocks):
    """Read each coloured cell back from the matrices embedded on page two."""
    panels = sorted(doc[1].get_image_info(xrefs=True),
                    key=lambda item: (round(item["bbox"][1], 1), item["bbox"][0]))
    for panel, block in zip(panels, blocks, strict=True):
        blob = doc.extract_image(panel["xref"])["image"]
        with Image.open(BytesIO(blob)) as source:
            img = source.convert("RGB")
        rows = block["values"]
        for r, values in enumerate(rows):
            y = int(img.height * (r + 0.5) / len(rows))
            for c, credit in enumerate(values):
                x = int(img.width * (c + 0.5) / len(values))
                expected = (21, 108, 98) if credit else (227, 232, 230)
                assert img.getpixel((x, y)) == expected, (block["work"], r, c)


def vote_counts(rows,results):
    tally=Counter()
    for i,(row,credits) in enumerate(zip(rows,results["primary"]["records"],strict=True)):
        votes=results["primary_votes"][i*2:i*2+2]
        answers={**row["answers"],**results["comparators"][i]}
        for mode,a in answers.items():
            got=bool(a["stop"] and all(v[mode] for v in votes))
            assert got==credits[mode]
            tally[mode]+=got
    assert all(tally[m]==results["primary"]["total"].get(m,0) for m in tally)
    return dict(tally)


def main():
    paper=PAPER/"manuscript.pdf"
    assert digest(paper)==digest(PAPER/"visual_language_paper.pdf")
    study=load(EVIDENCE/"assessment.json")
    replay=load(EVIDENCE/"replay.json")
    timed=load(EVIDENCE/"inference.json")
    archive=PAPER/"evidence/print_translation_answers.zip"
    assert digest(archive)==study["archive"]==timed["archive"]==replay["archive"]
    assert replay["matches"]==dict(guided=192,likelihood=192,serif=192,adapted=192,exchange=192) and not replay["differences"]
    assert timed["archived_mismatches"]==dict(direct=[],compiled=[])
    deck=PAPER/"figures/Glyph_stream_experiments.pptx"
    names=["Print_translation_outputs","Print_translation_outcomes","Print_translation_credits","Print_translation_time"]
    descriptors=["Paired print inputs and original writer emissions","Complete per-request assessment",
                 "Original sources and every credited canonical emission","Five paired on-device inference timings"]
    results=[]
    with ZipFile(archive) as raw, fitz.open(paper) as doc, ZipFile(deck) as slides:
        rows=json.loads(raw.read("outcomes.json"))
        assert len(rows)==192 and len(raw.namelist())==1536+1 and len(doc)==4
        for r in rows:
            for mode,a in r["answers"].items():
                assert a["png"]==sha256(raw.read(f'{r["id"]:04d}/{mode}.png')).hexdigest()
        totals=vote_counts(rows,study)
        all_tiles=Counter()
        for j,(name,kind) in enumerate(zip(names,descriptors,strict=True),1):
            page=doc[j-1]
            txt=" ".join(page.get_text().split())
            assert re.search(rf'Fig\.\s*{j}\s*\|',txt) and 'RESEARCH ARTICLE' in txt
            fig=PAPER/"figures"/(name+".pdf")
            with fitz.open(fig) as standalone:
                source=extracted(standalone,standalone[0])
            assert extracted(doc,page)==source
            if j in (1,3):
                scene=load(EVIDENCE/("scene.json" if j==1 else "credits.json"))
                tiles=Counter(); indices=set()
                for item in scene["items"]:
                    if item["type"]!="strip":continue
                    prefix,filename=item["name"].split("-",1)
                    blob=raw.read(prefix+"/"+filename)
                    assert blob==(ASSETS/item["name"]).read_bytes() and sha256(blob).hexdigest()==item["sha256"]
                    tiles[raster(blob)]+=1;indices.add(int(prefix))
                if j==3:
                    assert indices=={i for i,c in enumerate(study["primary"]["records"]) if c["guided"]}
                assert tiles==source
                all_tiles+=tiles
            results.append(dict(number=j,page=j,figure=name,kind=kind,source_sha256=digest(fig),embedded_images=sum(source.values())))
        assert Counter(raster(slides.read(n)) for n in slides.namelist() if n.startswith("ppt/media/"))==all_tiles
        matrix=load(EVIDENCE/"blocks.json")
        verify_printed_grades(doc, matrix["blocks"])
        xml=etree.fromstring(slides.read("ppt/slides/slide2.xml"))
        tables=xml.xpath('//*[local-name()="tbl"]')
        assert len(tables)==8
        for table,block in zip(tables,matrix["blocks"],strict=True):
            tr=table.xpath('./*[local-name()="tr"]')
            for line,m,values in zip(tr,matrix["methods"],block["values"],strict=True):
                assert values==[int(study["primary"]["records"][i][m]) for i in block["ids"]]
                for cell,value in zip(line.xpath('./*[local-name()="tc"]'),values,strict=True):
                    color=cell.xpath('./*[local-name()="tcPr"]/*[local-name()="solidFill"]/*[local-name()="srgbClr"]/@val')
                    assert [v.lower() for v in color]==["156c62" if value else "e3e8e6"]
        books=[n for n in slides.namelist() if n.endswith(".xlsx")]
        assert len(books)==1
        wb=load_workbook(BytesIO(slides.read(books[0])),read_only=True,data_only=True)
        values=list(wb.active.values);wb.close()
        expected=[[round(p[k]*1000,3) for p in timed["samples"]] for k in ("direct","compiled")]
        assert [list(row[1::2]) for row in values[1:]]==expected
        charts = [n for n in slides.namelist() if re.search(r"/charts/chart[0-9]+\.xml$", n)]
        assert len(charts) == 1
        series = etree.fromstring(slides.read(charts[0])).xpath('//*[local-name()="ser"]')
        for pair, measurement in zip(series, zip(*expected, strict=True), strict=True):
            points = pair.xpath('./*[local-name()="val"]//*[local-name()="pt"]/*[local-name()="v"]/text()')
            assert tuple(map(float, points)) == measurement
    evidence=dict(paper_sha256=digest(paper),figures=results,primary_counts=totals,raw_archive_sha256=digest(archive),
                  deck_sha256=digest(deck),replay_sha256=digest(EVIDENCE/"replay.json"),timing_sha256=digest(EVIDENCE/"inference.json"),
                  scope="Four current article figures; original input/output images, paired votes and measured GPU timings.")
    (PAPER/"evidence/visual_output_figure_index.json").write_text(json.dumps(evidence,indent=2)+"\n")
    print("Verified four article figures, 960 neural answers, primary grades, native slide tables and timing workbook.")


if __name__=="__main__":
    main()
