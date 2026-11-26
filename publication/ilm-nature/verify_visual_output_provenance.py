"""Authenticate the pictured-source article's raw image panels and editable scores."""
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import fitz
from lxml import etree
from openpyxl import load_workbook
from PIL import Image

PAPER = Path(__file__).resolve().parent
FIG = PAPER / "figures"
PUB = PAPER.parents[1] / "references/evidence/pictured_path_translation_v1"
ARCHIVE = PAPER / "evidence/pictured_path_answers.zip"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def bitmap(blob):
    with Image.open(BytesIO(blob)) as im:
        return im.size, sha256(im.convert("RGB").tobytes()).hexdigest()


def images(pdf, page):
    return Counter(bitmap(pdf.extract_image(x["xref"])["image"]) for x in page.get_image_info(xrefs=True))


def main():
    report = read(PUB/"assessment.json")
    replay = read(PUB/"replay.json")
    audit = read(PUB/"repair-audit.json")
    assert sha(ARCHIVE)==report["archive"]==replay["archive"]==audit["archive"]
    assert audit["assessment"]==sha(PUB/"assessment.json")
    assert audit["proof"]==sha(PAPER.parents[1]/"formal/span_list/SourcePathRepair.lean")
    deck = FIG/"Sentence_reconstruction.pptx"
    paper = PAPER/"manuscript.pdf"
    assert sha(paper)==sha(PAPER/"visual_language_paper.pdf")
    assert sum(replay["same"].values())+len(replay["differences"])==960
    grades=report["primary"]["records"]
    chart=read(PUB/"chart.json")
    media=Counter()
    figures=[]
    with fitz.open(paper) as doc, ZipFile(ARCHIVE) as archive, ZipFile(deck) as ppt:
        records=json.loads(archive.read("records.json"))
        assert len(doc)==6 and len(records)==len(grades)==192
        fresh=PUB.parent/"emitted_history_v1"
        fresh_archive=PAPER/"evidence/sentence_reconstruction_answers.zip"
        new_report=read(fresh/"assessment.json")
        new_replay=read(fresh/"replay.json")
        routing=read(fresh/"routing.json")
        plate=read(fresh/"plate.json")
        assert sha(fresh_archive)==plate["archive"]==new_report["archive"]==new_replay["archive"]
        assert routing["assessment"]==sha(fresh/"assessment.json")
        assert routing["proof"]==sha(PAPER.parents[1]/"formal/span_list/DraftReconstruction.lean")
        assert sum(new_replay["same"].values())+len(new_replay["differences"])==768
        fresh_figure=FIG/"Sentence_reconstruction.pdf"
        expected=Counter()
        with ZipFile(fresh_archive) as raw, fitz.open(fresh_figure) as pdf:
            for item in plate["items"]:
                if item["type"]!="strip": continue
                blob=raw.read(item["member"])
                assert blob==(FIG/"sentence_reconstruction"/item["name"]).read_bytes()
                assert sha256(blob).hexdigest()==item["sha256"]
                expected[bitmap(blob)]+=1
            assert expected==images(pdf,pdf[0])==images(doc,doc[0])
            assert "Fig. 1" in " ".join(doc[0].get_text().split())
        media.update(expected)
        figures.append(dict(page=1,file=fresh_figure.name,sha256=sha(fresh_figure)))
        for p,name in enumerate(("answers","scores","reading","origins"),start=1):
            fig=FIG/f"Pictured_path_{name}.pdf"
            figures.append({"page":p+1,"file":fig.name,"sha256":sha(fig)})
            assert f"Fig. {p+1}" in " ".join(doc[p].get_text().split())
            with fitz.open(fig) as single:
                if name=="scores":
                    values=[v for lang in ("en","zh") for v in chart["series"][lang]]
                    for lang in ("en","zh"):
                        assert chart["series"][lang]==[sum(g[m] for g,r in zip(grades,records,strict=True) if r["direction"]==lang) for m in chart["keys"]]
                    bars=[d["rect"].width for d in single[0].get_drawings() if d["type"]=="f"]
                    assert len(bars)==len(values) and all(abs(w-1.53*v)<.001 for w,v in zip(bars,values))
                    scaled=[d["rect"].width for d in doc[p].get_drawings() if d["type"]=="f" and d["fill"] and
                            any(max(abs(a-b) for a,b in zip(d["fill"],color))<.002 for color in ((19/255,107/255,98/255),(150/255,100/255,66/255)))]
                    assert len(scaled)==len(bars)
                    factor=max(scaled)/max(bars)
                    assert all(abs(a-factor*b)<.02 for a,b in zip(scaled,bars))
                    continue
                rows=read(PUB/f"{name}.json")["items"]
                expected=Counter(); selected=[]
                for item in rows:
                    if item["type"]!="strip": continue
                    if name=="origins":
                        with ZipFile(PAPER/"evidence"/item["archive"]) as historical:
                            raw=historical.read(item["member"])
                        part=item["member"].split("/")[-1]
                        idx=item["member"].split("/")[0]
                    else:
                        idx,part=item["name"].split("-",1)
                        raw=archive.read(idx+"/"+part)
                    assert raw==(FIG/"pictured_path"/item["name"]).read_bytes()
                    assert sha256(raw).hexdigest()==item["sha256"]
                    expected[bitmap(raw)]+=1
                    if part=="source.png": selected.append(int(idx))
                if name != "origins":
                    first_gain=next(i for i,g in enumerate(grades) if g["path"] and not g["field"])
                    fail=next(i for i,g in enumerate(grades) if not g["path"])
                    if name=="answers":
                        english=next(i for i,(r,g) in enumerate(zip(records,grades)) if r["direction"]=="en" and g["path"] and not g["field"])
                        assert selected==[first_gain,english,fail]
                    else:
                        assert selected==[next(i for i,(r,g) in enumerate(zip(records,grades)) if r["answers"]["path"]["native_path_equal"] and not g["path"] and i!=fail)]
                assert expected==images(single,single[0])==images(doc,doc[p])
                media.update(expected)
        order_pub=PUB.parent/"visual_order_v1"
        order_raw=PAPER/"evidence/visual_order_answers.zip"
        order_plate=read(order_pub/"plate.json")
        order_eval=read(order_pub/"evaluation.json")
        assert order_plate["archive"]==sha(order_raw)==order_eval["archive"]==read(order_pub/"replay.json")["archive"]
        assert order_eval["counts"]["new_pairs"]["ordered"]==332
        order_fig=FIG/"Origin_instruction_order.pdf"
        expected=Counter()
        with ZipFile(order_raw) as data, fitz.open(order_fig) as single:
            for item in order_plate["items"]:
                if item["type"]!="strip": continue
                blob=data.read(item["member"])
                assert blob==(FIG/"visual_order"/item["name"]).read_bytes()
                assert sha256(blob).hexdigest()==item["sha256"]
                expected[bitmap(blob)]+=1
            assert expected==images(doc,doc[5])==images(single,single[0])
            assert "Fig. 6" in " ".join(doc[5].get_text().split())
        media.update(expected)
        figures.append(dict(page=6,file=order_fig.name,sha256=sha(order_fig)))
        # A package may deduplicate identical bitmaps used in two places.
        packaged={bitmap(ppt.read(n)) for n in ppt.namelist() if n.startswith("ppt/media/")}
        assert packaged==set(media)
        native=etree.fromstring(ppt.read(next(n for n in ppt.namelist() if n.endswith("/charts/chart1.xml"))))
        series=[chart["series"][k] for k in ("en","zh")]
        for s,expected in zip(native.xpath('//*[local-name()="ser"]'),series,strict=True):
            assert list(map(float,s.xpath('./*[local-name()="val"]//*[local-name()="pt"]/*[local-name()="v"]/text()')))==expected
        wb=load_workbook(BytesIO(ppt.read(next(n for n in ppt.namelist() if n.endswith(".xlsx")))),read_only=True,data_only=True)
        table=list(wb.active.values);wb.close()
        assert [list(r[1::2]) for r in table[1:]]==[list(pair) for pair in zip(*series,strict=True)]
    index=dict(paper=sha(paper),figures=figures,archive=sha(ARCHIVE),editable=sha(deck),
               original_image_instances=sum(media.values()),requests=192,answers=960,replayed=replay["same"],
               independent_trial=dict(requests=192,answers=768,archive=sha(fresh_archive),replayed=new_replay["same"],
                   cpu_differences=new_replay["differences"],assessment=sha(fresh/"assessment.json"),routing=sha(fresh/"routing.json")),
               order_experiment=dict(archive=sha(order_raw),evaluation=sha(order_pub/"evaluation.json"),replay=sha(order_pub/"replay.json")),
               dictionary_grades=sha(PUB.parent/"origin_value_reader_v1/score-audit.json"),grade_audit=sha(PUB/"repair-audit.json"),transfer=sha(PUB/"transfer.json"),cost=sha(PUB/"gpu-cost.json"))
    (PAPER/"evidence/visual_output_figure_index.json").write_text(json.dumps(index,indent=2)+"\n")
    print("Verified original image panels, both rendered score series and the editable chart workbook.")


if __name__=="__main__":
    main()
