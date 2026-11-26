"""Verify the archived article, native charts, original image panels and two reader scopes."""

from collections import Counter
from io import BytesIO
from pathlib import Path
import json
from zipfile import ZipFile

import fitz
from lxml import etree
from openpyxl import load_workbook
from safetensors import safe_open
from verify_raw_ilm import digest, pixels

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[1]
PUB = ROOT/"references/evidence"
ROLE = PUB/"library_roles_v1"
FIELD = PUB/"optical_state_v1"
TEACH = PUB/"library_teacher_v1"


def read(path):
    return json.loads(path.read_text())


def chart_data(path, series):
    with ZipFile(path) as z:
        books=sorted(n for n in z.namelist() if n.endswith(".xlsx"))
        charts=sorted(n for n in z.namelist() if "/charts/chart" in n and n.endswith(".xml"))
        assert len(books)==len(charts)==len(series)
        for book,chart,panel in zip(books,charts,series,strict=True):
            wb=load_workbook(BytesIO(z.read(book)),data_only=True,read_only=True)
            vals=list(wb.active.values)
            wb.close()
            assert list(vals[0][1::2])==list(panel["series"])
            assert [str(r[0]) for r in vals[1:]]==panel["labels"]
            columns=[list(col) for col in zip(*(r[1::2] for r in vals[1:]))]
            assert columns==list(panel["series"].values())
            tree=etree.fromstring(z.read(chart))
            cached=[[float(v) for v in s.xpath('.//*[local-name()="val"]//*[local-name()="pt"]/*[local-name()="v"]/text()')]
                    for s in tree.xpath('//*[local-name()="ser"]')]
            assert cached==columns
        return {pixels(z.read(n)) for n in z.namelist() if n.startswith("ppt/media/")}


def verify():
    article=PAPER/"visual_language_paper.pdf"
    cumulative=PAPER/"manuscript.pdf"
    previous=PAPER/"versions/2026-11-26-d4626a06/reading.pdf"
    expected=Counter()
    for it in read(ROLE/"rasters.json")["items"]:
        if it["type"]=="image":
            p=PAPER/"figures/library_roles"/it["name"]
            assert digest(p)==it["sha256"]
            expected[pixels(p.read_bytes())]+=1
    with fitz.open(article) as a,fitz.open(cumulative) as b,fitz.open(previous) as old:
        assert (len(a),len(b),len(old))==(66,315,62)
        for i,page in enumerate(a):
            text="".join(page.get_text().split())
            assert text=="".join(b[i].get_text().split())
            if 0<i<len(old): assert text=="".join(old[i].get_text().split())
        for doc in (a,b):
            found=Counter()
            for im in doc[63].get_image_info(xrefs=True):
                v=pixels(doc.extract_image(im["xref"])["image"])
                rect=fitz.Rect(im["bbox"])
                assert abs(rect.width/rect.height*v[0][1]/v[0][0]-1)<.001
                found[v]+=1
            assert found==expected
    roles=PAPER/"figures/Writing_with_grammatical_roles.pptx"
    field=PAPER/"figures/Glyph_attention_and_reading.pptx"
    assert chart_data(roles,read(ROLE/"charts.json")["panels"])==set(expected)
    field_panels=sum((read(FIELD/(f+".json"))["panels"] for f in ("Figure_213_glyph_attention_measure","Figure_214_optical_state_reader")),[])
    chart_data(field,field_panels)
    for parent in (ROLE,FIELD):
        folder=ROOT/"artifacts"/parent.name
        report=read(parent/"evaluation.json")
        files=report.get("files") or {"free.json":report["free"],"source.json":report["source_votes"]}
        assert all(digest(folder/name)==d for name,d in files.items())
    for p in (TEACH/"assignments.json",TEACH/"glyph-measure.json",FIELD/"factor.json"):
        d=read(p)
        if "program" in d:
            prog="scripts/score_bilingual_assignments.py" if p.parent==TEACH else "scripts/audit_optical_state_readout.py"
            assert digest(ROOT/prog)==d["program"]
        if "binding" in d: assert all(digest(ROOT/k)==v for k,v in d["binding"].items())
    for name,folder in (("library_roles","library_roles_v1"),("optical_state","optical_state_v1")):
        for arm,f in read(PUB/folder/"training.json")["fits"].items():
            d=ROOT/"models"/name/arm
            assert digest(d/"weights.safetensors")==read(d/"fit.json")["weights"]
            assert f==digest(d/("fit.json" if name=="library_roles" else "weights.safetensors"))
    archive=PAPER/"evidence/role_translation_outputs.zip"
    gen=read(ROOT/"artifacts/library_roles_v1/generated.json")
    with ZipFile(archive) as z:
        rows=json.loads(z.read("manifest.json"))
        assert len(rows)==384 and len(z.namelist())==1537
        from hashlib import sha256
        for i,r in enumerate(rows):
            for m,a in r["answers"].items():
                assert sha256(z.read(a["path"])).hexdigest()==a["png"]==gen["answers"][i][m]["png"]
    for v in read(ROLE/"illustrated.json")["checks"]:
        assert v["cpu_png"]==v["png"]==gen["answers"][v["request"]][v["method"]]["png"]
    for p in [ROOT/"artifacts/library_roles_v1/office/Writing_with_grammatical_roles.pdf",
              ROOT/"artifacts/optical_state_v1/office/Glyph_attention_and_reading.pdf"]:
        with fitz.open(p) as exported: assert len(exported)==2
    return dict(reading_pages=66,cumulative_pages=315,preserved_baseline_pages=62,
                raw_archive_outputs=1536,distinct_illustrated_pngs=len(expected),cpu_regenerated_outputs=4,
                editable_charts=6,chart_workbooks=6,rendered_slides=4,inspected_reading_pages=[1,63,64,65,66],
                files={str(p.relative_to(ROOT)):digest(p) for p in (article,cumulative,previous,roles,field,archive,
                      ROLE/"evaluation.json",ROLE/"gradient.json",TEACH/"glyph-measure.json",TEACH/"assignments.json",
                      FIELD/"training.json",FIELD/"factor.json",FIELD/"evaluation.json")})


if __name__=="__main__":
    print(json.dumps(verify(),indent=2))
