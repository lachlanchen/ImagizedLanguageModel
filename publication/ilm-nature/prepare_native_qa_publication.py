#!/usr/bin/env python3
"""Stage completed QA evidence as native slides and draft TeX, without Office.

This CPU preparation never changes active publication files. Rendering,
visual review, manuscript inclusion and delivery are separate required stages.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

import openpyxl
from PIL import Image

from conditional_learning_figures import multi_chart
from generate_assignment_editable_figures import NS, Slide, element, q, relpath, resolve, xml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = "daef4bf"
RESULT = ROOT / "references/evidence/native_paired_qa_result/result.json"
CALIBRATION = ROOT / "references/evidence/native_paired_qa_final_normal_cohort.json"
FIGURES = ("Figure_S29_native_qa_learning", "Figure_S30_native_qa_pages")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline(name):
    return subprocess.check_output(["git", "show", BASE + ":publication/ilm-nature/" + name], cwd=ROOT)


def active_manifest():
    """Guard current publication bytes, including PDFs and the editable deck."""
    return {str(p.relative_to(HERE)): sha(p) for p in HERE.rglob("*") if p.is_file()
        and not any(v.startswith(".") or v == "__pycache__" for v in p.relative_to(HERE).parts)}


def evidence():
    r = json.loads(RESULT.read_text())
    for key in ("files_sha256", "reports_sha256"):
        for name, digest in r[key].items():
            if sha(RESULT.parent / name) != digest:
                raise ValueError("completed evidence changed: " + name)
    if (len(r["rows"]) != 192 or r["complete_output_images"] != 384
            or r["complete_training_images"] != 128 or r["actual_new_native_assessment_calls"] != 320
            or r["reused_initial_copying_outputs"] != 64 or not r["all_retained_outputs_equal_actual_tensors"]
            or not r["raw_review_completed"] or not r["all_generated_outputs_complete_and_unrepaired"]):
        raise ValueError("complete authenticated QA/copy result required")
    if any(r[k] for k in ("exploratory_rule_met", "semantic_learning_validated", "general_ilm_goal_complete")):
        raise ValueError("failed language qualification changed")
    gates = r["prospective_exploratory_gates"]
    if len(gates) != 6 or any(gates.values()):
        raise ValueError("all six registered conditions must remain failed")
    for arm in ("initial", "final"):
        for variant in ("normal", "question_erased", "passage_erased"):
            for split in (("train", "validation") if variant == "normal" else ("validation",)):
                rows = [x for x in r["rows"] if x["task"] == "qa" and x["variant"] == variant and x["split"] == split]
                groups = {}
                for row in rows:
                    groups.setdefault(row["pair"], []).append(row)
                observed = dict(cases=len(rows), canonical_exact=sum(x["observations"][arm]["canonical_exact"] for x in rows),
                    accepted_exact=sum(x["observations"][arm]["accepted_exact"] for x in rows), pairs=len(groups),
                    both_correct_pairs=sum(all(x["observations"][arm]["accepted_exact"] for x in pair) for pair in groups.values()))
                if observed != r["summaries"][arm][variant][split] or observed != dict(cases=32, canonical_exact=0,
                        accepted_exact=0, pairs=16, both_correct_pairs=0) or any(len(pair) != 2 for pair in groups.values()):
                    raise ValueError("complete paired QA counts differ")
        for split, expected in zip(("train", "validation"), (18, 19) if arm == "initial" else (19, 16)):
            rows = [x for x in r["rows"] if x["task"] == "copy" and x["split"] == split]
            count = sum(x["observations"][arm]["canonical_exact"] for x in rows)
            if count != expected or r["copying_summaries"][arm][split] != dict(cases=32, ocr_exact=count):
                raise ValueError("complete copying counts differ")
    reviewed = json.loads((RESULT.parent / "raw_visual_review.json").read_text())
    if sha(RESULT.parent / "raw_visual_review.json") != r["raw_visual_review_sha256"] or not reviewed["all_original_answer_PNGs_covered"]:
        raise ValueError("bound complete raw review required")
    training = json.loads((RESULT.parent / "reports/training_report.json").read_text())
    verification = json.loads((RESULT.parent / "training_verification.json").read_text())
    if (training["updates"] != 128 or training["checkpoint_sha256"] != r["final_checkpoint_sha256"]
            or verification["observed_updates"] != 128 or not verification["all_optimizer_steps_match"]
            or not verification["first_native_and_differentiable_actual_tensors_equal"]
            or not verification["finite_nonzero_early_middle_final_gradients"]):
        raise ValueError("actual training verification required")
    calibration = json.loads(CALIBRATION.read_text())["reference_calibration_summary"]
    if calibration != {"train": dict(cases=32, ocr_exact=23), "validation": dict(cases=32, ocr_exact=26)}:
        raise ValueError("reference OCR limitation changed")
    return r, training, calibration


def tex_fragments(output, r, training, calibration):
    seconds = training["elapsed_seconds_this_invocation"]
    memory = training["peak_allocated_vram_gib_this_invocation"]
    paragraph = r"""A complete natural-QA adaptation of the image writer fails all six prospective
exploratory conditions (Supplementary Note 52 and Fig.~S29). Four permutations
of 32 training questions give 128 updates. Initial and final models each
produce 0/32 source-accepted OCR-exact seen answers, 0/32 held-out answers
and 0/16 both-correct held-out passage pairs. Both erased-input controls
also give 0/32. Held-out copying falls from 19/32 to 16/32, exceeding the
allowed loss of two exact strings. Training takes SECONDS\,s at MEMORY\,GiB
peak allocation. All 384 original assessment pages are inspected, with
malformed writing, extra fragments and omissions retained (Fig.~S30).
Imperfect reference OCR remains material. These observations reject scaling
the unchanged recipe and do not establish an independent language model.
""".replace("SECONDS", f"{seconds:,.3f}").replace("MEMORY", f"{memory:.3f}")
    (output / "results_paragraph.tex").write_text(paragraph)
    (output / "abstract_result_sentence.tex").write_text(
        "A subsequent 128-update image-answer adaptation yields no exact seen or held-out Chinese answers "
        "and reduces held-out copying from 19/32 to 16/32.\n")
    note = r"""\section*{Supplementary Note 52 | Complete natural-question image-answer learning}
\label{note:native-qa-learning}
The continuous image writer starts from the fixed ink-mass checkpoint selected
before QA adaptation. A fresh rank-16 AdamW optimizer uses rate $10^{-5}$,
decay .01 and gradient clipping at one. Four complete permutations of the
first 32 training questions from sixteen passages give 128 updates. All 32
held-out questions come from sixteen disjoint articles. The public cases have
been reused during development, so this is a developmental comparison with
one schedule rather than a pristine benchmark or seed replication.

The entire $1024\times1024$ passage/question image conditions a native
$512\times256$ answer image. The same generic condition applies to every
example. Generation begins from independent noise. No varying text embedding,
answer crop, transcription, teacher or font lookup enters deployment. The
target is loaded only after generation during training. Gradients traverse
all fifty native steps and the frozen decoder. The objective is frozen-reader
recognition cross-entropy plus $.4035679875746343$ times target-mass Bernoulli
KL at temperature .05. Recognition vocabulary and answer text supply training
supervision only. External image/language pretraining remains inherited.

The first actual differentiable output matches native generation exactly at
the same seed. Finite nonzero gradients reach early, middle and final steps.
All 206 adapter tensors change, actual optimizer counters reach 128 and the
complete schedule, RNG states and all 128 pre-update images authenticate.
An earlier save-path failure occurs before any update. Its preserved attempt
is an execution correction, not a restart selected by answer quality. Only
the final fixed-budget checkpoint enters assessment. The initial checkpoint
is assessed afterwards under the same prospectively fixed inputs and seeds.

\begin{center}\small
\begin{tabular}{lrr}
\toprule
Complete independent OCR observation & Initial & Final\\
\midrule
Seen Chinese answers & 0/32 & 0/32\\
Held-out Chinese answers & 0/32 & 0/32\\
Held-out pairs, both answers correct & 0/16 & 0/16\\
Held out, question erased & 0/32 & 0/32\\
Held out, passage erased & 0/32 & 0/32\\
Seen Chinese/English copies & 18/32 & 19/32\\
Held-out Chinese/English copies & 19/32 & 16/32\\
\bottomrule
\end{tabular}
\end{center}

Canonical and source-accepted QA exact counts agree at zero. Matching uses
NFC/whitespace normalization, preserving case and punctuation. Each passage
pair has disjoint source-accepted answer sets. All 32 question-erased pairs
have identical inputs and seeds and yield bit-identical tensors at each
checkpoint. This execution equality is not evidence of answering.

All six prospective conditions fail: seen fit of at least 24/32; at least
12/32 held-out answers and a gain of four; at least 4/16 both-correct pairs
and a gain of two; a normal-answer advantage of four over question-erased
control; the same advantage over passage-erased control; and loss of at most
two held-out copies from the initial 19/32. The observed copy loss is three.
The one-case seen-copy gain does not establish semantic learning.

Reference OCR recognizes only 23/32 seen and 26/32 held-out canonical QA
images exactly. The seen requirement of 24/32 therefore exceeds recognition
of these clean reference images. That limitation is not a strict ceiling on
another output style. It remains disclosed without changing the registered
rule or repairing scores. Raw review covers every original answer: natural
QA contains malformed writing, unwanted passage/question fragments and extra
labels. Copies retain omissions, substitutions, repeats and unwanted quotes.
Some final pages are faint. No contrast enhancement, rerendered transcription
or quantitative manual replacement of OCR accuracy is used.

Training takes SECONDS\,s (HOURS\,h) at MEMORY\,GiB peak allocated memory.
The runtime retains a 3,899,903,488-parameter image transformer, including
24,358,912 trained adapter parameters, and an 84,046,115-parameter VAE.
The 560,528,640-parameter recognition reader supervises training only.
The assessment comprises 320 new native calls and 64 authenticated reused
initial-copy outputs. All 384 complete original answers and 128 training
pages remain available. Neither the training time nor the adapter count is
a foundation-training or quality-matched efficiency result. The failure
does not identify a unique causal bottleneck or rule out image-language
learning with a different method.

{\footnotesize\raggedright
Protocol: \path{references/native_paired_qa_learning_protocol.md}.
Complete evidence: \path{references/evidence/native_paired_qa_result/README.md}.
Training: \path{scripts/train_native_paired_qa.py}.
Assessment: \path{scripts/assess_native_paired_qa.py}.
Export: \path{scripts/publish_native_qa_evidence.py}.\par}
""".replace("SECONDS", f"{seconds:,.3f}").replace("HOURS", f"{seconds/3600:.2f}").replace("MEMORY", f"{memory:.3f}")
    (output / "supplementary_note_52.tex").write_text(note)


def stage(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("hide CUDA for CPU-only publication preparation")
    active_before = active_manifest()
    r, training, calibration = evidence()
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(baseline("ILM_Figures_editable.pptx"))) as archive:
        original = {name: archive.read(name) for name in archive.namelist()}
    parts = dict(original)
    types = ET.fromstring(parts["[Content_Types].xml"])
    presentation = ET.fromstring(parts["ppt/presentation.xml"])
    relationships = ET.fromstring(parts["ppt/_rels/presentation.xml.rels"])
    slides = presentation.find("p:sldIdLst", NS)
    if len(slides) != 43:
        raise ValueError("immutable 43-slide baseline required")
    source = "ppt/slides/slide43.xml"
    source_rels = ET.fromstring(parts[relpath(source)])
    pictures, charts = [], []

    def override(name, kind):
        if name in parts or any(n.get("PartName") == "/"+name for n in types):
            raise ValueError("new part collision: " + name)
        element("ct:Override", types, PartName="/"+name, ContentType=kind)

    def append(slide, number, notes):
        slide.relation("slideLayout", next(n.get("Target") for n in source_rels if n.get("Type").endswith("/slideLayout")))
        note = f"ppt/notesSlides/native_qa_notes{number}.xml"
        part = f"ppt/slides/slide{number}.xml"
        slide.relation("notesSlide", "../notesSlides/"+Path(note).name)
        override(part, "application/vnd.openxmlformats-officedocument.presentationml.slide+xml")
        parts[part], parts[relpath(part)] = xml(slide.root), xml(slide.rels)
        old_note = next(resolve(source, n.get("Target")) for n in source_rels if n.get("Type").endswith("/notesSlide"))
        node = ET.fromstring(parts[old_note])
        for shape in node.findall("p:cSld/p:spTree/p:sp", NS):
            ph = shape.find("p:nvSpPr/p:nvPr/p:ph", NS)
            if ph is not None and ph.get("type") == "body":
                texts = shape.findall(".//a:t", NS)
                if not texts: raise ValueError("missing native note text")
                for i, text in enumerate(texts): text.text = notes if i == 0 else ""
        rels = ET.fromstring(parts[relpath(old_note)])
        for n in rels:
            if n.get("Type").endswith("/slide"): n.set("Target", f"../slides/slide{number}.xml")
        override(note, "application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml")
        parts[note], parts[relpath(note)] = xml(node), xml(rels)
        used = {n.get("Id") for n in relationships}
        rid = next(f"rId{i}" for i in range(1, 1000) if f"rId{i}" not in used)
        element("pkg:Relationship", relationships, Id=rid, Type=NS["r"]+"/slide", Target=f"slides/slide{number}.xml")
        element("p:sldId", slides, id=max(int(s.get("id")) for s in slides)+1).set(q("r:id"), rid)

    def picture(slide, row, key, name, x, y, width, crop=None):
        path = RESULT.parent / row["images"][key]
        image = Image.open(path).convert("RGB")
        if crop is not None: image = image.crop(crop)
        buffer = io.BytesIO()
        if crop is None: raw = path.read_bytes()
        else: image.save(buffer, format="PNG"); raw = buffer.getvalue()
        part = "ppt/media/native_qa_"+name+".png"
        if part in parts: raise ValueError("duplicate staged picture")
        parts[part] = raw
        slide.picture(slide.relation("image", "../media/"+Path(part).name), name, x, y, width, width*image.height/image.width)
        pictures.append(dict(part=part, original=row["images"][key], original_sha256=sha(path),
            sha256=hashlib.sha256(raw).hexdigest(), crop_box=crop, size=list(image.size)))

    s = Slide(parts[source])
    s.shape("Natural-QA adaptation does not establish answering", 45, 24, 1510, 60, 31, True)
    s.shape("a  Source-accepted OCR-exact answers", 45, 110, 750, 40, 22, True)
    for x, label in ((45, "Complete cohort"), (530, "Initial"), (665, "Final")):
        s.shape(label, x, 170, 470 if x == 45 else 130, 35, 19, True)
    labels = [("Seen questions", "0/32"), ("Held-out questions", "0/32"),
        ("Held-out pairs, both correct", "0/16"), ("Held out, question erased", "0/32"), ("Held out, passage erased", "0/32")]
    for i, (label, count) in enumerate(labels):
        y = 225+56*i
        s.shape(label, 45, y, 480, 42, 20)
        for x in (530, 665): s.shape(count, x, y, 130, 42, 20)
    s.shape("All six prospective conditions fail.\nHeld-out copying loses three exact strings; at most two allowed.", 45, 555, 740, 100, 21, True)
    s.shape("b  Complete OCR-exact copies (%)", 835, 110, 720, 40, 22, True)
    categories = ["Seen (32)", "Held out (32)"]
    series = [(arm.title(), [100*r["copying_summaries"][arm][split]["ocr_exact"]/32 for split in ("train", "validation")]) for arm in ("initial", "final")]
    counts = [[f'{r["copying_summaries"][arm][split]["ocr_exact"]}/32' for split in ("train", "validation")] for arm in ("initial", "final")]
    node, workbook = multi_chart(parts["ppt/charts/chart51.xml"], categories, series, counts, 100, 25)
    node.find("c:externalData", NS).set(q("r:id"), "rId1")
    part, book = "ppt/charts/chart52.xml", "ppt/embeddings/native_qa_chart52.xlsx"
    rels = element("pkg:Relationships")
    element("pkg:Relationship", rels, Id="rId1", Type=NS["r"]+"/package", Target="../embeddings/"+Path(book).name)
    override(part, "application/vnd.openxmlformats-officedocument.drawingml.chart+xml")
    override(book, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    parts[part], parts[relpath(part)], parts[book] = xml(node), xml(rels), workbook
    s.chart(s.relation("chart", "../charts/chart52.xml"), 835, 170, 710, 450)
    charts.append(dict(part=part, workbook=book, categories=categories, series=series, labels=counts))
    s.shape("128 updates on 32 questions from 16 passages\n3.55 h training; 12.204 GiB allocated peak\n24.359M trained adapters in a 3.900B image transformer", 45, 710, 1510, 105, 20)
    s.shape("384 original answers reviewed; reference OCR: 23/32 seen, 26/32 held. Reused public data; one schedule; inherited pretraining.", 45, 851, 1510, 42, 16)
    append(s, 44, "Figure S29. Complete fixed initial/final native QA learning assessment. "
        "The five zero QA measures use an editable count table, copying uses a native chart and workbook. "
        "QA counts accept all source answers; copying uses canonical matching. All six conditions fail. "
        "320 new native calls plus64 reused initial copies; full384 original PNGs reviewed. No manual score repair. "
        "Single development schedule, imperfect OCR, retained external pretraining, no independent language/efficiency qualification.")

    held = [x for x in r["rows"] if x["task"] == "qa" and x["variant"] == "normal" and x["split"] == "validation"][:2]
    if [x["id"] for x in held] != ["TRAIN_2295_QUERY_0", "TRAIN_2295_QUERY_1"] or held[0]["pair"] != held[1]["pair"]:
        raise ValueError("first registered held-out passage pair changed")
    s = Slide(parts[source])
    s.shape("Fixed held-out pair: complete unrepaired answer pages", 45, 24, 1510, 60, 30, True)
    s.shape("a  Complete question inputs", 45, 108, 500, 35, 21, True)
    for i, row in enumerate(held):
        x = 45+255*i
        s.shape(f"Question {i+1}", x, 155, 240, 30, 17, True)
        picture(s, row, "input", f"q{i}_input", x, 195, 240)
    # Insets retain the last two physical lines, not an OCR reconstruction.
    # Both selected source pages have a one-line question followed by 回答.
    for i, row in enumerate(held):
        s.shape(f"Q{i+1}: source question inset (display only)", 45, 485+150*i, 500, 32, 17, True)
        picture(s, row, "input", f"q{i}_question_inset", 45, 525+150*i, 495, crop=(0,552,1024,648))
    s.shape("b  Original answer pages", 590, 108, 965, 35, 21, True)
    for i, row in enumerate(held):
        x = 590+490*i
        s.shape(f"Q{i+1} reference (whitespace cropped)", x, 154, 470, 26, 16, True)
        picture(s, row, "reference", f"q{i}_reference_inset", x, 189, 460, crop=(0,24,512,80))
        for j, arm in enumerate(("initial", "final")):
            s.shape(arm.title(), x+235*j, 260, 220, 28, 17, True)
    for k, (variant, label) in enumerate((("normal", "Normal"), ("question_erased", "Question erased"), ("passage_erased", "Passage erased"))):
        y = 307+180*k
        s.shape(label, 590, y, 965, 28, 16, True)
        for i, original_row in enumerate(held):
            matches = [v for v in r["rows"] if v["task"] == "qa" and v["variant"] == variant and v["id"] == original_row["id"]]
            if len(matches) != 1: raise ValueError("fixed control coverage differs")
            for j, arm in enumerate(("initial", "final")):
                picture(s, matches[0], arm, f"q{i}_{variant}_{arm}", 590+490*i+235*j, y+34, 220)
    s.shape("First registered held-out pair, not selected by appearance. All 12 model pages are complete; only input/question and reference display insets are cropped.", 45, 852, 1510, 42, 15)
    append(s, 45, "Figure S30. First held-out passage pair TRAIN_2295_QUERY_0/1 in the fixed index. "
        "Complete input pages shown at left. Source question display insets are exact rectangles(0,552,1024,648); "
        "reference display insets remove whitespace at(0,24,512,80). All12 normal/question-erased/passage-erased "
        "initial/final output images shown complete without repair or contrast changes. Original input/reference pages "
        "and all384 generated pages remain in the evidence package. Reference/inputs are not generated answers.")
    for name, node in (("[Content_Types].xml", types), ("ppt/presentation.xml", presentation), ("ppt/_rels/presentation.xml.rels", relationships)):
        parts[name] = xml(node)
    target = output / "draft.pptx"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in parts.items(): archive.writestr(name, raw)
    tex_fragments(output, r, training, calibration)
    mutable = {"[Content_Types].xml", "ppt/presentation.xml", "ppt/_rels/presentation.xml.rels"}
    with zipfile.ZipFile(target) as archive:
        for name, raw in original.items():
            if name not in mutable and archive.read(name) != raw: raise ValueError("baseline part changed: "+name)
        for name in archive.namelist():
            if name.endswith((".xml", ".rels")): ET.fromstring(archive.read(name))
        if len(ET.fromstring(archive.read("ppt/presentation.xml")).find("p:sldIdLst", NS)) != 45:
            raise ValueError("45 slides required")
        for spec in charts:
            node = ET.fromstring(archive.read(spec["part"]))
            book = openpyxl.load_workbook(io.BytesIO(archive.read(spec["workbook"])), data_only=True)
            for j, series_node in enumerate(node.findall("c:chart/c:plotArea/c:barChart/c:ser", NS)):
                values = [float(v.text) for v in series_node.findall("c:val/c:numRef/c:numCache/c:pt/c:v", NS)]
                labels = [v.text for v in series_node.findall("c:dLbls/c:dLbl/c:tx/c:rich/a:p/a:r/a:t", NS)]
                if values != spec["series"][j][1] or labels != spec["labels"][j]: raise ValueError("chart values or labels differ")
                if [book.active.cell(i+2,j+2).value for i in range(len(values))] != values: raise ValueError("chart workbook differs")
        for picture_spec in pictures:
            if hashlib.sha256(archive.read(picture_spec["part"])).hexdigest() != picture_spec["sha256"]:
                raise ValueError("original staged picture bytes differ")
    if active_manifest() != active_before: raise ValueError("active publication files changed during isolated preparation")
    record = dict(baseline_commit=BASE, baseline_pptx_sha256=hashlib.sha256(baseline("ILM_Figures_editable.pptx")).hexdigest(),
        source_result_sha256=sha(RESULT), source_calibration_summary_sha256=sha(CALIBRATION),
        builder_sha256=sha(__file__), output_files_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        old_native_slides_preserved=43, staged_native_slides=45, old_native_charts_preserved=51,
        staged_native_charts=52, pictures=pictures, charts=charts,
        active_publication_files_unchanged=True, active_publication_manifest_sha256=hashlib.sha256(json.dumps(active_before,sort_keys=True).encode()).hexdigest(),
        native_OOXML_and_workbooks_checked=True, source_models_loaded=False, Office_or_LaTeX_started=False,
        rendered=False, visual_review_completed=False, active_manuscript_updated=False, Nutstore_updated=False,
        semantic_learning_validated=False, general_ilm_goal_complete=False)
    (output / "preparation.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({k: record[k] for k in ("staged_native_slides", "staged_native_charts", "active_publication_files_unchanged", "rendered")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    stage(parser.parse_args().output.resolve())
