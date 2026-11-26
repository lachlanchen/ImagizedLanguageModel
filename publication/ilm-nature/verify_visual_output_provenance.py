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
    deck = FIG/"Full_image_role_writing.pptx"
    paper = PAPER/"manuscript.pdf"
    assert sha(paper)==sha(PAPER/"visual_language_paper.pdf")
    assert sum(replay["same"].values())+len(replay["differences"])==960
    grades=report["primary"]["records"]
    chart=read(PUB/"chart.json")
    media=Counter()
    figures=[]
    with fitz.open(paper) as doc, ZipFile(ARCHIVE) as archive, ZipFile(deck) as ppt:
        records=json.loads(archive.read("records.json"))
        assert len(doc)==9 and len(records)==len(grades)==192
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
            assert expected==images(pdf,pdf[0])==images(doc,doc[3])
            assert "Fig. 4" in " ".join(doc[3].get_text().split())
        media.update(expected)
        figures.append(dict(page=4,file=fresh_figure.name,sha256=sha(fresh_figure)))
        for p,name in enumerate(("answers","scores","reading","origins"),start=4):
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
            assert expected==images(doc,doc[8])==images(single,single[0])
            assert "Fig. 9" in " ".join(doc[8].get_text().split())
        media.update(expected)
        figures.append(dict(page=9,file=order_fig.name,sha256=sha(order_fig)))
        actions = PUB.parent/"independent_optical_actions_v1"
        raw_actions = PAPER/"evidence/independent_optical_actions.zip"
        assessment = read(actions/"assessment.json")
        assert sha(raw_actions)==assessment["archive"]==read(actions/"replay.json")["archive"]
        assert "Fig. 3" in " ".join(doc[2].get_text().split())
        for manifest,stem,indices in (("plate","Independent_optical_actions",[0,49]+[i for i,r in enumerate(assessment["primary"]["records"]) if r["strip"]][:1]),
                                      ("other-works","Optical_actions_other_works",[96,145])):
            plate=read(actions/(manifest+".json"))
            assert plate["archive"]==sha(raw_actions)
            expected=Counter()
            with ZipFile(raw_actions) as raw, fitz.open(FIG/(stem+".pdf")) as pdf:
                outputs=json.loads(raw.read("records.json"))
                members=[item for item in plate["items"] if item["type"]=="strip"]
                assert [v["member"] for v in members]==[f"{i:04d}/{m}.png" for i in indices for m in ("source","cell","strip")]
                for item in members:
                    blob=raw.read(item["member"])
                    assert blob==(FIG/"independent_optical_actions"/item["name"]).read_bytes()
                    assert sha256(blob).hexdigest()==item["sha256"]
                    number,mode=item["member"].removesuffix(".png").split("/")
                    if mode!="source":
                        assert item["sha256"]==outputs[int(number)]["answers"][mode]["sha256"]
                    expected[bitmap(blob)]+=1
                assert expected==images(pdf,pdf[0])
                if manifest=="plate":
                    assert expected==images(doc,doc[2])
                    figures.insert(0,dict(page=3,file=stem+".pdf",sha256=sha(FIG/(stem+".pdf"))))
            media.update(expected)
        supported=PUB.parent/"source_supported_writing_v1"
        leading=Counter()
        for manifest,stem,archive_name,ranking in (("plate","Source_supported_writing","source_supported_writing","assessment"),
                ("opus-plate","Source_supported_opus","supported_opus_outputs","opus-assessment")):
            plate=read(supported/(manifest+".json"))
            grades=read(supported/(ranking+".json"))
            source_zip=PAPER/"evidence"/(archive_name+".zip")
            assert sha(source_zip)==plate["archive"]==grades["archive"]
            decisions=grades["primary"]["records"]
            if manifest=="plate":
                picked=[0,49]+[i for i,g in enumerate(decisions) if i not in (0,49) and g["aligned"] and not g["parent"]][:1]
            else:
                picked=[next((i for i,g in enumerate(decisions) if g["aligned"]),0)]
            items=[v for v in plate["items"] if v["type"]=="strip"]
            assert [v["member"] for v in items]==[f"{i:04d}/{m}.png" for i in picked for m in ("source","parent","aligned")]
            expected=Counter()
            with ZipFile(source_zip) as raw, fitz.open(FIG/(stem+".pdf")) as pdf:
                rec=json.loads(raw.read("records.json"))
                for v in items:
                    blob=raw.read(v["member"])
                    assert blob==(FIG/"source_supported"/v["name"]).read_bytes()
                    assert sha256(blob).hexdigest()==v["sha256"]
                    idx,mode=v["member"].removesuffix(".png").split("/")
                    if mode!="source":
                        assert v["sha256"]==rec[int(idx)]["answers"][mode]["sha256"]
                    expected[bitmap(blob)]+=1
                assert expected==images(pdf,pdf[0])
            leading.update(expected)
            media.update(expected)
        with fitz.open(PAPER/"versions/2026-11-27-53dbe94a/reading.pdf") as prior:
            assert leading==images(prior,prior[0])
        restoration=PUB.parent/"visual_strip_restoration_v1"
        current=Counter()
        sources=(("Free_strip_restoration","book","book-generation"),("Supplied_strip_restoration","supplied","supplied"))
        for stem,kind,receipt in sources:
            plate=read(restoration/(stem+".json"))
            raw_file=PAPER/"evidence"/("strip_restoration_"+kind+".zip")
            assert sha(raw_file)==plate["archive"]==read(restoration/(receipt+".json"))["archive"]
            expected=Counter()
            with ZipFile(raw_file) as raw, fitz.open(FIG/(stem+".pdf")) as pdf:
                for item in plate["items"]:
                    if item["type"]!="strip":
                        continue
                    blob=raw.read(item["member"])
                    assert blob==(FIG/"strip_restoration"/item["name"]).read_bytes()
                    assert sha256(blob).hexdigest()==item["sha256"]
                    expected[bitmap(blob)]+=1
                assert expected==images(pdf,pdf[0])
            current.update(expected)
            media.update(expected)
        assert current==images(doc,doc[1])
        assert "Fig. 2" in " ".join(doc[1].get_text().split())
        figures.insert(0,dict(page=2,files={n:sha(FIG/n) for n in ("Free_strip_restoration.pdf","Supplied_strip_restoration.pdf")}))
        role_pub=PUB.parent/"natural_pictured_roles_v1"
        ref_pub=PUB.parent/"visual_referents_v1"
        full_pub=PUB.parent/"full_image_roles_v1"
        role_archive=PAPER/"evidence/full_image_roles_plate.zip"
        role_plate=read(full_pub/"plate.json")
        assert sha(role_archive)==role_plate["archive"]
        lex_pub=PUB.parent/"optical_lexicon_v1"
        origins={"full_image_roles.zip":read(full_pub/"generation.json")["archive"],"optical_lexicon.zip":read(lex_pub/"generation.json")["archive"]}
        for name,digest in origins.items():
            assert sha(PAPER/"evidence"/name)==digest
        expected=Counter()
        with ZipFile(role_archive) as raw, fitz.open(FIG/"Full_image_role_writing.pdf") as pdf:
            for item in role_plate["items"]:
                if item["type"]!="strip":continue
                blob=raw.read(item["member"])
                assert blob==(FIG/"full_image_roles"/item["name"]).read_bytes()
                assert item["upstream"] in origins
                with ZipFile(PAPER/"evidence"/item["upstream"]) as original:
                    assert blob==original.read(item["original"])
                assert sha256(blob).hexdigest()==item["sha256"]
                expected[bitmap(blob)]+=1
            assert expected==images(pdf,pdf[0])==images(doc,doc[0])
        assert "Fig. 1" in " ".join(doc[0].get_text().split())
        media.update(expected)
        figures.insert(0,dict(page=1,file="Full_image_role_writing.pdf",sha256=sha(FIG/"Full_image_role_writing.pdf")))
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
               optical_lexicon={k:sha(lex_pub/(k+".json")) for k in ("corpus","generation","audit","assessment","calibration","retrieval","pixel-search")},
               full_image_roles={k:sha(full_pub/(k+".json")) for k in ("corpus","generation","audit","assessment","plate","influence","external-assessment")},
               visual_referents={k:sha(ref_pub/(k+".json")) for k in ("corpus","outcomes","audit","plate","external-assessment")},
               natural_roles={k:sha(role_pub/(k+".json")) for k in ("corpus","assessment","audit","spacing","pud","localization")},
               strip_restoration={k:sha(restoration/(k+".json")) for k in ("book-assessment","opus-assessment","supplied","book-replay","opus-replay")},
               source_support={k:sha(supported/(k+".json")) for k in ("assessment","replay","roles","prefixes","opus-assessment","oracle-inventory","assignment-bias","independent-choices","book-candidate-selection","opus-candidate-selection")},
               visual_actions=dict(archive=sha(raw_actions),assessment=sha(actions/"assessment.json"),cost=sha(actions/"cost.json"),assignment=sha(actions/"source-assignment.json")),
               order_experiment=dict(archive=sha(order_raw),evaluation=sha(order_pub/"evaluation.json"),replay=sha(order_pub/"replay.json")),
               dictionary_grades=sha(PUB.parent/"origin_value_reader_v1/score-audit.json"),grade_audit=sha(PUB/"repair-audit.json"),transfer=sha(PUB/"transfer.json"),cost=sha(PUB/"gpu-cost.json"))
    (PAPER/"evidence/visual_output_figure_index.json").write_text(json.dumps(index,indent=2)+"\n")
    print("Verified original image panels, both rendered score series and the editable chart workbook.")


if __name__=="__main__":
    main()
