"""Authenticate the pictured-source article's raw image panels and editable scores."""
import re
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
    deck = FIG/"Visual_training_breadth.pptx"
    paper = PAPER/"manuscript.pdf"
    assert sha(paper)==sha(PAPER/"visual_language_paper.pdf")
    assert sum(replay["same"].values())+len(replay["differences"])==960
    grades=report["primary"]["records"]
    chart=read(PUB/"chart.json")
    media=Counter()
    figures=[]
    with fitz.open(paper) as doc, ZipFile(ARCHIVE) as archive, ZipFile(deck) as ppt:
        records=json.loads(archive.read("records.json"))
        assert len(doc)==15 and len(records)==len(grades)==192
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
            assert expected==images(pdf,pdf[0])==images(doc,doc[9])
            assert "Fig. 10" in " ".join(doc[9].get_text().split())
        media.update(expected)
        figures.append(dict(page=7,file=fresh_figure.name,sha256=sha(fresh_figure)))
        for p,name in enumerate(("answers","scores","reading","origins"),start=7):
            fig=FIG/f"Pictured_path_{name}.pdf"
            figures.append({"page":p+1,"file":fig.name,"sha256":sha(fig)})
            assert f"Fig. {p+4}" in " ".join(doc[p+3].get_text().split())
            with fitz.open(fig) as single:
                if name=="scores":
                    values=[v for lang in ("en","zh") for v in chart["series"][lang]]
                    for lang in ("en","zh"):
                        assert chart["series"][lang]==[sum(g[m] for g,r in zip(grades,records,strict=True) if r["direction"]==lang) for m in chart["keys"]]
                    bars=[d["rect"].width for d in single[0].get_drawings() if d["type"]=="f"]
                    assert len(bars)==len(values) and all(abs(w-1.53*v)<.001 for w,v in zip(bars,values))
                    scaled=[d["rect"].width for d in doc[p+3].get_drawings() if d["type"]=="f" and d["fill"] and
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
                assert expected==images(single,single[0])==images(doc,doc[p+3])
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
            assert expected==images(doc,doc[14])==images(single,single[0])
            assert "Fig. 15" in " ".join(doc[14].get_text().split())
        media.update(expected)
        figures.append(dict(page=12,file=order_fig.name,sha256=sha(order_fig)))
        actions = PUB.parent/"independent_optical_actions_v1"
        raw_actions = PAPER/"evidence/independent_optical_actions.zip"
        assessment = read(actions/"assessment.json")
        assert sha(raw_actions)==assessment["archive"]==read(actions/"replay.json")["archive"]
        assert "Fig. 9" in " ".join(doc[8].get_text().split())
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
                    assert expected==images(doc,doc[8])
                    figures.insert(0,dict(page=6,file=stem+".pdf",sha256=sha(FIG/(stem+".pdf"))))
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
        assert current==images(doc,doc[7])
        assert "Fig. 8" in " ".join(doc[7].get_text().split())
        figures.insert(0,dict(page=5,files={n:sha(FIG/n) for n in ("Free_strip_restoration.pdf","Supplied_strip_restoration.pdf")}))
        role_pub=PUB.parent/"natural_pictured_roles_v1"
        ref_pub=PUB.parent/"visual_referents_v1"
        full_pub=PUB.parent/"full_image_roles_v1"
        lex_pub=PUB.parent/"optical_lexicon_v1"
        aligned_pub=PUB.parent/"ordered_alignment_v1"
        phrase_pub=PUB.parent/"covered_phrases_v1"
        editor_pub=PUB.parent/"glyph_draft_editor_v1"
        seq_pub=PUB.parent/"visual_sequence_preference_v1"
        raw_path=PAPER/"evidence/visual_sequence_preference.zip"
        plate=read(seq_pub/"plate.json")
        assert sha(raw_path)==plate["archive"]==read(seq_pub/"assessment.json")["archive"]
        expected=Counter()
        with ZipFile(raw_path) as raw, ZipFile(PAPER/"evidence/visual_preference_training_example.zip") as train, fitz.open(FIG/"Visual_sequence_preference.pdf") as pdf:
            responses=json.loads(raw.read("records.json"))
            example=json.loads(train.read("record.json"))
            train_check=read(seq_pub/"training-transfer.json")
            assert example["source_archive"]==train_check["archive"]
            assert len(responses)==568
            for item in plate["items"]:
                if item["type"]!="strip":
                    continue
                training=item["origin"]=="training"
                blob=(train if training else raw).read(item["member"])
                assert blob==(FIG/"visual_sequence_preference"/item["name"]).read_bytes()
                assert sha256(blob).hexdigest()==item["sha256"]
                i,kind=item["member"].removesuffix(".png").split("/")
                if kind!="source":
                    record=example["record"] if training else responses[int(i)]
                    assert record["answers"][kind]["png"]==item["sha256"]
                    if training:
                        assert int(i)==example["index"] and train_check["rows"][int(i)]["png"][kind]==item["sha256"]
                expected[bitmap(blob)]+=1
            assert expected==images(pdf,pdf[0])==images(doc,doc[5])
        assert "Fig. 6" in " ".join(doc[5].get_text().split())
        media.update(expected)
        figures.insert(0,dict(page=3,file="Visual_sequence_preference.pdf",sha256=sha(FIG/"Visual_sequence_preference.pdf")))
        retrieved=PUB.parent/"retrieved_glyph_writer_v1"
        rplate=read(retrieved/"plate.json")
        rarchive=PAPER/"evidence/retrieved_glyph_writer.zip"
        assert sha(rarchive)==rplate["archive"]==read(retrieved/"assessment.json")["archive"]
        expected=Counter()
        with ZipFile(rarchive) as rz,fitz.open(FIG/"Retrieved_glyphs.pdf") as figraw:
            rr=json.loads(rz.read("records.json"))
            for part in rplate["items"]:
                if part["type"]!="strip":continue
                blob=rz.read(part["member"])
                assert sha256(blob).hexdigest()==part["sha256"] and blob==(FIG/"retrieved_glyphs"/part["name"]).read_bytes()
                i,m=part["member"].removesuffix(".png").split("/")
                if m!="source":assert rr[int(i)]["answers"][m]["png"]==part["sha256"]
                expected[bitmap(blob)]+=1
            assert expected==images(figraw,figraw[0])==images(doc,doc[6])
            assert "Fig. 7" in " ".join(doc[6].get_text().split())
        media.update(expected)
        figures.insert(1,dict(page=4,file="Retrieved_glyphs.pdf",sha256=sha(FIG/"Retrieved_glyphs.pdf")))
        bp=PUB.parent/"visual_training_breadth_v1"
        ba=PAPER/"evidence/visual_training_breadth.zip"
        assert sha(ba)==read(bp/"generation.json")["archive"]==read(bp/"assessment.json")["archive"]==read(bp/"audit.json")["archive"]
        expected=Counter()
        with ZipFile(ba) as raw,ZipFile(PAPER/"evidence/breadth_pretrained_images.zip") as alternate,fitz.open(FIG/"Visual_training_breadth.pdf") as fp:
            for item in read(bp/"plate.json")["items"]:
                if item["type"]!="strip":continue
                blob=(alternate if item["member"].endswith("/pretrained.png") else raw).read(item["member"])
                assert sha256(blob).hexdigest()==item["sha256"] and blob==(FIG/"visual_training_breadth"/item["name"]).read_bytes()
                expected[bitmap(blob)]+=1
            assert expected==images(fp,fp[0])==images(doc,doc[3])
        media.update(expected)
        for page,file in enumerate(("Visual_training_breadth.pdf","Training_breadth_curves.pdf"),start=1):
            assert f"Fig. {page+3}" in " ".join(doc[page+2].get_text().split())
            figures.insert(page-1,dict(page=page,file=file,sha256=sha(FIG/file)))
        curves=read(bp/"curve.json")["series"]
        for direction,lang in enumerate(("English","Chinese")):
            for arm in ("broad","narrow"):
                fit=read(PAPER.parents[1]/"models/visual_training_breadth"/arm/"fit.json")
                assert curves[lang+" / "+arm]==[read(bp/"starting-development.json")[direction]["mean_glyph_f2"]]+[e["development"][direction]["mean_glyph_f2"] for e in fit["epochs"]]
        with fitz.open(FIG/"Training_breadth_curves.pdf") as curve_pdf:
            assert "Into English" in doc[4].get_text() and "Into Chinese" in curve_pdf[0].get_text()
            assert len([d for d in doc[4].get_drawings() if d["type"]=="s"])>=len([d for d in curve_pdf[0].get_drawings() if d["type"]=="s"])
        phase_count=Counter()
        with fitz.open(FIG/"Phase_feedback_example.pdf") as fig:
            for item in read(bp/"phase-plate.json")["items"]:
                if item["type"]!="strip":continue
                with ZipFile(PAPER/"evidence"/item["file"]) as archive:
                    blob=archive.read(item["member"])
                assert sha256(blob).hexdigest()==item["sha256"] and blob==(FIG/"visual_training_breadth"/item["name"]).read_bytes()
                phase_count[bitmap(blob)]+=1
            assert phase_count==images(fig,fig[0])==images(doc,doc[4])
        media.update(phase_count)
        figures[1]["companion"]=sha(FIG/"Phase_feedback_example.pdf")
        # A package may deduplicate identical bitmaps used in two places.
        packaged={bitmap(ppt.read(n)) for n in ppt.namelist() if n.startswith("ppt/media/")}
        paired=PUB.parent/"paired_translation_teacher_v1"
        for item in read(paired/"plate.json")["items"]:
            if item["type"]=="strip": media[bitmap((FIG/"paired_visual_answers"/item["name"]).read_bytes())]+=1
        assert packaged==set(media)
        expected_charts=[list(curves.values()),[chart["series"][k] for k in ("en","zh")]]
        actual_charts=[]
        workbooks=[]
        for n in ppt.namelist():
            if re.search(r"/charts/chart[0-9]+[.]xml$",n):
                node=etree.fromstring(ppt.read(n))
                actual_charts.append([list(map(float,ser.xpath('./*[local-name()="val"]//*[local-name()="pt"]/*[local-name()="v"]/text()'))) for ser in node.xpath('//*[local-name()="ser"]')])
            if n.endswith(".xlsx"):
                wb=load_workbook(BytesIO(ppt.read(n)),read_only=True,data_only=True)
                rows=list(wb.active.values);wb.close()
                workbooks.append(list(map(list,zip(*[r[1::2] for r in rows[1:]]))))
        assert len(actual_charts)==len(workbooks)==2
        for values in expected_charts:
            for collection in (actual_charts,workbooks):
                matched=[v for v in collection if len(v)==len(values) and len(v[0])==len(values[0])]
                assert len(matched)==1 and all(abs(a-b)<1e-12 for x,y in zip(values,matched[0],strict=True) for a,b in zip(x,y,strict=True))
        paired=PUB.parent/"paired_translation_teacher_v1"
        with ZipFile(PAPER/"evidence/paired_visual_answers.zip") as archive, fitz.open(FIG/"Paired_visual_answers.pdf") as source:
            items=read(paired/"plate.json")["items"]
            expected=Counter()
            for item in items:
                if item["type"]=="strip":
                    blob=archive.read(item["member"])
                    assert blob==(FIG/"paired_visual_answers"/item["name"]).read_bytes() and sha256(blob).hexdigest()==item["sha256"]
                    expected[bitmap(blob)]+=1
            assert expected==images(source,source[0])==images(doc,doc[1])
            assert "Fig. 2" in " ".join(doc[1].get_text().split())
            assert set(expected)<=packaged
        for fig in figures:
            fig["page"]+=2
        figures.insert(0,dict(page=1,file="Paired_visual_answers.pdf",sha256=sha(FIG/"Paired_visual_answers.pdf")))
        news = PUB.parent/"human_translation_calibration_v1"
        source = read(news/"figure.json")
        chart_path = FIG/"Expert_news_translation.pptx"
        with ZipFile(PAPER/"evidence/wmt_news_visual_answers.zip") as archive, ZipFile(chart_path) as chart_deck:
            expected_news = Counter()
            for key in ("source", "answer"):
                blob = archive.read(f"{source['index']}/{key}.png")
                assert sha256(blob).hexdigest() == source[key]
                expected_news[bitmap(blob)] += 1
            assert expected_news == images(doc,doc[2])
            assert set(expected_news) == {bitmap(chart_deck.read(n)) for n in chart_deck.namelist() if n.startswith("ppt/media/")}
            xml = etree.fromstring(chart_deck.read("ppt/charts/chart1.xml"))
            series = [[int(p.text) for p in z.xpath('.//*[local-name()="val"]//*[local-name()="pt"]/*[local-name()="v"]')]
                      for z in xml.xpath('.//*[local-name()="ser"]')]
            assert series == list(map(list,zip(*source["counts"])))
            workbook = load_workbook(BytesIO(chart_deck.read(next(n for n in chart_deck.namelist() if n.endswith(".xlsx")))),read_only=True)
            assert [list(row[1:]) for row in list(workbook.active.values)[1:]] == source["counts"]
            assert "Fig. 3" in " ".join(doc[2].get_text().split())
            media.update(expected_news)
        figures.insert(1,dict(page=2,file="Expert_news_translation.pdf",sha256=sha(FIG/"Expert_news_translation.pdf"),
                              editable=sha(chart_path),source=sha(news/"figure.json")))
        for fig in figures:
            fig["page"] += 1
        fresh_news=PUB.parent/"open_news_external_v1"
        details=read(fresh_news/"figure.json")
        archive_news=PAPER/"evidence/reviewed_news_writers.zip"
        assert sha(archive_news)==details["archive"] and sha(fresh_news/"meaning.json")==details["meaning"]
        expected=Counter()
        with ZipFile(archive_news) as archive, ZipFile(FIG/"Reviewed_news_writer.pptx") as slides, fitz.open(FIG/"Reviewed_news_writer.pdf") as plate:
            for item in details["items"]:
                blob=archive.read(f"{item['index']}/{item['key']}.png")
                assert sha256(blob).hexdigest()==item["sha256"]
                expected[bitmap(blob)]+=1
            assert expected==images(doc,doc[0])==images(plate,plate[0])
            assert set(expected)=={bitmap(slides.read(n)) for n in slides.namelist() if n.startswith("ppt/media/")}
            assert "Fig. 1" in " ".join(doc[0].get_text().split())
        media.update(expected)
        figures.insert(0,dict(page=1,file="Reviewed_news_writer.pdf",sha256=sha(FIG/"Reviewed_news_writer.pdf"),
                              editable=sha(FIG/"Reviewed_news_writer.pptx"),source=sha(fresh_news/"figure.json")))
    index=dict(paired_translations={k:sha(paired/(k+".json")) for k in ("selection","provenance","teacher","verification","writer","free-meaning","branches","draft-copy","plate")},paper=sha(paper),figures=figures,archive=sha(ARCHIVE),editable=sha(deck),
               original_image_instances=sum(media.values()),requests=192,answers=960,replayed=replay["same"],
               independent_trial=dict(requests=192,answers=768,archive=sha(fresh_archive),replayed=new_replay["same"],
                   cpu_differences=new_replay["differences"],assessment=sha(fresh/"assessment.json"),routing=sha(fresh/"routing.json")),
               training_breadth={k:sha(bp/(k+".json")) for k in ("selection","drafts","assessment","generation","audit","curve","plate","transfer","transfer-generation","surface-coverage","phase-generation","phase","phase-plate")},
               retrieved_glyphs={k:sha(retrieved/(k+".json")) for k in ("selection","generation","assessment","audit","calibration","plate")},
               visual_sequence_preference={k:sha(seq_pub/(k+".json")) for k in ("selection","mined","verification","confirmed","label-inspection","generation","assessment","paths","audit","prefix-weights","training-transfer","passage-memory","plate")},
               glyph_draft_editor={k:sha(editor_pub/(k+".json")) for k in ("selection","drafts","generation","assessment","audit","source-probe","path-comparison","plate")},
               covered_phrases={k:sha(phrase_pub/(k+".json")) for k in ("corpus","generation","links-0","links-1","assessment","role-meaning","audit","name-probes","plate","editing")},
               ordered_alignment={k:sha(aligned_pub/(k+".json")) for k in ("corpus","generation","kernel-check","assessment","audit","repetition","retrieval","plate")},
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
