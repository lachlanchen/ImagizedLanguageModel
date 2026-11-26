"""Trace the article's answer images, case-level grades and interface data to the frozen records."""
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from statistics import median
from zipfile import ZipFile
import fitz
from lxml import etree
from openpyxl import load_workbook
from PIL import Image

PAPER=Path(__file__).resolve().parent
PUB=PAPER.parents[1]/"references/evidence/optical_span_rollout_v1"
FIG=PAPER/"figures"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def raster(blob):
    with Image.open(BytesIO(blob)) as img:
        return img.size,sha256(img.convert("RGB").tobytes()).hexdigest()


def pixels(doc,page):
    return Counter(raster(doc.extract_image(a["xref"])["image"]) for a in page.get_image_info(xrefs=True))


def recount(answers,votes,records):
    for i,(row,credit) in enumerate(zip(answers,records,strict=True)):
        for m,a in row.items():
            assert (a["stop"] and votes[i*2][m] and votes[i*2+1][m])==credit[m]


def main():
    main_pdf=PAPER/"manuscript.pdf"
    assert sha(main_pdf)==sha(PAPER/"visual_language_paper.pdf")
    archive=PAPER/"evidence/span_rollout_answers.zip"
    assessed=read(PUB/"assessment.json")
    controls=read(PUB/"interfaces.json")
    second=read(PUB/"interface-votes.json")
    gen=read(PUB/"generation.json")
    replay=read(PUB/"replay.json")
    assert sha(archive)==gen["archive"]==replay["archive"]==assessed["archive"]==controls["binding"]["archive"]
    assert second["inputs"]==sha(PUB/"interfaces.json")
    assert replay["same"]=={"single":192,"span":192,"exchange":192} and not replay["differences"]
    cost = read(PUB/"gpu-cost.json")
    assert cost["archive"] == sha(archive) and cost["cases"] == 192
    assert cost["program"] == sha(PAPER.parents[1]/"scripts/measure_span_gpu_cost.py")
    assert cost["protocol"] == sha(PAPER.parents[1]/"references/span_gpu_cost_protocol.md")
    assert Counter(t["size"] for t in cost["trials"]) == {1:3, 12:3, 48:3, 96:3}
    for size in (1, 12, 48, 96):
        trials = [t for t in cost["trials"] if t["size"] == size]
        assert all(t["differences"] == [] for t in trials)
        group = cost["summary"][str(size)]
        assert group["exact"]
        assert group["requests_per_second"] == median(cost["cases"]/t["seconds"] for t in trials)
        assert group["board_joules_per_request"] == median(t["joules"]/cost["cases"] for t in trials)
    charts=read(PUB/"interface-chart.json")
    deck=FIG/"Autonomous_span_translation.pptx"
    standalone=[]
    with ZipFile(archive) as z, ZipFile(deck) as slides, fitz.open(main_pdf) as doc:
        rows=json.loads(z.read("records.json"))
        assert len(doc)==4 and len(rows)==192 and len(z.namelist())==769
        grades=assessed["primary"]["records"]
        for row in rows:
            for mode,a in row["answers"].items():
                assert sha256(z.read(f'{row["case"]:04d}/{mode}.png')).hexdigest()==a["sha256"]
        recount([{**r["answers"],**a} for r,a in zip(rows,assessed["controls"],strict=True)],assessed["votes"],grades)
        recount(controls["answers"],second["votes"],second["records"]["records"])
        for i,name in enumerate(("output","cases","interfaces","origins")):
            fig=FIG/f"Span_translation_{name}.pdf"
            with fitz.open(fig) as source: assert pixels(doc,doc[i])==pixels(source,source[0])
            assert f"Fig. {i+1}" in " ".join(doc[i].get_text().split())
            standalone.append(dict(page=i+1,source=fig.name,sha256=sha(fig)))
        examples=read(PUB/"plate.json")
        displayed=Counter()
        ids=set()
        for item in examples["items"]:
            if item["type"]!="strip": continue
            n,part=item["name"].split("-",1)
            content=z.read(n+"/"+part)
            assert sha256(content).hexdigest()==item["sha256"] and content==(FIG/"span_rollout"/item["name"]).read_bytes()
            ids.add(int(n))
            displayed[raster(content)]+=1
        assert ids=={next(i for i,v in enumerate(grades) if v["span"]==flag) for flag in (True,False)}
        assert displayed==pixels(doc,doc[0])
        extra = archive.with_name("word_origin_probes.zip")
        probe_scores = read(PUB/"word-origin-scores.json")
        assert sha(extra) == probe_scores["archive"] == read(PUB/"word-origin-images.json")["archive"]
        extra_replay = read(PUB/"word-origin-replay.json")
        assert extra_replay["archive"] == sha(extra) and not extra_replay["differences"]
        assert extra_replay["same"] == {"single":38, "span":38}
        probe_tiles = Counter()
        with ZipFile(extra) as probes:
            probes_rows = json.loads(probes.read("records.json"))
            assert len(probes_rows)==38
            translations = [r for r in probes_rows if r["kind"]=="translation"]
            recount([{m:r[m] for m in ("span","single")} for r in translations],
                    probe_scores["votes"],probe_scores["contrast_grades"])
            for item in read(PUB/"origin-plate.json")["items"]:
                if item["type"]!="strip": continue
                _,n,part = item["name"].split("-",2)
                blob = probes.read(n+"/"+part)
                assert sha256(blob).hexdigest()==item["sha256"] and blob==(FIG/"span_rollout"/item["name"]).read_bytes()
                probe_tiles[raster(blob)]+=1
        assert probe_tiles==pixels(doc,doc[3])
        displayed+=probe_tiles
        assert Counter(raster(slides.read(n)) for n in slides.namelist() if n.startswith("ppt/media/"))==displayed
        matrices=read(PUB/"blocks.json")
        panels=sorted(doc[1].get_image_info(xrefs=True),key=lambda im:(round(im["bbox"][1],1),im["bbox"][0]))
        tables=etree.fromstring(slides.read("ppt/slides/slide2.xml")).xpath('//*[local-name()="tbl"]')
        assert len(panels)==len(tables)==8
        for block,panel,table in zip(matrices["blocks"],panels,tables,strict=True):
            with Image.open(BytesIO(doc.extract_image(panel["xref"])["image"])) as im: matrix=im.convert("RGB")
            tr=table.xpath('./*[local-name()="tr"]')
            for r,(method,values,line) in enumerate(zip(matrices["methods"],block["values"],tr,strict=True)):
                assert values==[int(grades[i][method]) for i in block["ids"]]
                for c,(value,cell) in enumerate(zip(values,line.xpath('./*[local-name()="tc"]'),strict=True)):
                    color="156c62" if value else "e3e8e6"
                    assert bytes(matrix.getpixel((int((c+.5)*matrix.width/24),int((r+.5)*matrix.height/5)))).hex()==color
                    assert cell.xpath('./*[local-name()="tcPr"]/*[local-name()="solidFill"]/*[local-name()="srgbClr"]/@val')[0].lower()==color
        series=[charts["series"][lang] for lang in ("en","zh")]
        for lang,values in zip(("en","zh"),series,strict=True):
            for key,value in zip(charts["keys"],values,strict=True):
                assert value==sum(g[key] for r,g in zip(rows,second["records"]["records"],strict=True) if r["direction"]==lang)
        chart=etree.fromstring(slides.read(next(x for x in slides.namelist() if x.endswith("/charts/chart1.xml"))))
        for native,values in zip(chart.xpath('//*[local-name()="ser"]'),series,strict=True):
            assert list(map(float,native.xpath('./*[local-name()="val"]//*[local-name()="pt"]/*[local-name()="v"]/text()')))==values
        wb=load_workbook(BytesIO(slides.read(next(x for x in slides.namelist() if x.endswith(".xlsx")))),read_only=True,data_only=True)
        values=list(wb.active.values);wb.close()
        # Office stores one pair of category and value columns per series.
        assert [list(row[1::2]) for row in values[1:]]==[list(k) for k in zip(*series,strict=True)]
    index=dict(paper=sha(main_pdf),figures=standalone,archive=sha(archive),editable=sha(deck),requests=192,
               gpu_answers=576,reproduced=576,original_image_instances=11,additional_prompts=38,additional_answers=76,probe_archive=sha(extra),verified_matrix_cells=960,
               additional_reproduced=sum(extra_replay["same"].values()),
               timed_identical_answers=len(cost["trials"])*cost["cases"],gpu_measurement=sha(PUB/"gpu-cost.json"),
               primary=assessed["primary"]["total"],controlled=second["records"]["total"])
    (PAPER/"evidence/visual_output_figure_index.json").write_text(json.dumps(index,indent=2)+"\n")
    print("Verified raw answers, 960 plotted decisions, both sets of votes, native chart and workbook.")


if __name__=="__main__":
    main()
