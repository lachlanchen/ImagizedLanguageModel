#!/usr/bin/env python3
"""Prepare exact-pixel question/passage interventions without a neural model."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.image_edit.data import display_text, font_and_coverage, page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, default=ROOT / "artifacts/image_edit_learning/pages_v2")
    parser.add_argument("--source", type=Path, default=ROOT / "data/visual_qa/cmrc2018_short/train.jsonl")
    parser.add_argument("--audit", type=Path, default=ROOT / "references/evidence/paired_language_data_readiness.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prep = json.loads((args.pages / "preparation.json").read_text())
    audit = json.loads(args.audit.read_text())
    manifest = args.pages / "pairs.jsonl"
    if (sha(manifest) != audit["manifest_sha256"] or sha(args.source) != audit["source_sha256"]
            or prep["manifest_sha256"] != sha(manifest)
            or any(sha(ROOT / p) != v for p,v in prep["implementation_sha256"].items())):
        raise ValueError("unchanged audited real question pages required")
    source = {r["id"]: r for r in map(json.loads,args.source.read_text().splitlines())}
    rows = [r for r in map(json.loads,manifest.read_text().splitlines()) if r["task"] == "qa"]
    cases = []
    for r in rows:
        original = source[r["id"]]
        context, question = display_text(original["context"]), display_text(original["question"])
        prompt = f"资料：{context}\n问题：{question}\n回答："
        if (hashlib.sha256(context.encode()).hexdigest() != r["pair"]
                or original["references"][0]["text"] != r["reference_text"]):
            raise ValueError("source binding differs")
        cases.append(dict(row=r,context=context,question=question,prompt=prompt,reference=r["reference_text"]))
    font, provenance = font_and_coverage(cases)
    for key in ("font_file_sha256","font_face_index","font_size","charset_sha256"):
        if provenance[key] != prep["font"][key]: raise ValueError("source font differs")
    pair_groups = defaultdict(list)
    for case in cases:
        r=case["row"]
        rendered, geometry = page(case["prompt"],(1024,1024),font)
        actual = args.pages/r["input"]
        if sha(actual) != r["input_sha256"] or not np.array_equal(np.asarray(rendered),np.asarray(Image.open(actual).convert("RGB"))):
            raise ValueError("complete normal page does not reproduce exactly")
        if geometry != r["input_geometry"]:raise ValueError("rendered source geometry differs")
        _, cg = page("资料："+case["context"],(1024,1024),font)
        _, qg = page("问题："+case["question"],(1024,1024),font)
        if cg["lines"]+qg["lines"]+1 != geometry["lines"]:raise ValueError("paragraph boundaries differ")
        if any(b[3] > 32+(i+1)*48 for i,b in enumerate(geometry["boxes"])):
            raise ValueError("ink reaches another paragraph row")
        case.update(image=rendered,context_lines=cg["lines"],question_lines=qg["lines"])
        pair_groups[r["pair"]].append(case)
    args.output.mkdir(parents=True,exist_ok=False)
    control_dir=args.output/"controls";control_dir.mkdir()
    indexes = {k:[] for k in ("normal_training","normal_seen","normal_held","question_erased_held","passage_erased_held",
        "preflight_normal","preflight_question_erased","preflight_passage_erased")}
    seen_ids=[r["id"] for r in rows if r["split"]=="train"][:32]
    held_ids=[r["id"] for r in rows if r["split"]=="validation"]
    preflight_ids=seen_ids[:2]+held_ids[:2]
    bindings=[]

    def index_entry(case,path,seed):
        return dict(id=case["row"]["id"],input=os.path.relpath(path,args.output),input_sha256=sha(path),seed=seed)

    for pair_index,(pair,group) in enumerate(pair_groups.items()):
        if len(group)!=2 or len({c["context"] for c in group})!=1 or len({c["row"]["split"] for c in group})!=1:
            raise ValueError("complete same-passage pairs required")
        if {a["text"] for a in source[group[0]["row"]["id"]]["references"]} & {a["text"] for a in source[group[1]["row"]["id"]]["references"]}:
            raise ValueError("paired accepted answers overlap")
        seed=20268701+pair_index
        question_y=32+group[0]["context_lines"]*48
        answer_y=32+(group[0]["context_lines"]+max(c["question_lines"] for c in group))*48
        # Preserve the complete passage. Retain both fixed labels at one common
        # position so erasure cannot leak which question had more wrapped lines.
        erased=group[0]["image"].copy();erased.paste("white",(0,question_y,1024,1024))
        d=ImageDraw.Draw(erased)
        d.text((32,question_y),"问题：",font=font,anchor="lt",fill="black")
        d.text((32,answer_y),"回答：",font=font,anchor="lt",fill="black")
        qpath=control_dir/f"pair_{pair_index:03}_question_erased.png"
        if group[0]["row"]["split"]=="validation":erased.save(qpath)
        for case in group:
            r=case["row"];normal=args.pages/r["input"]
            entry=index_entry(case,normal,seed)
            if r["split"]=="train":indexes["normal_training"].append(entry)
            if r["id"] in seen_ids:indexes["normal_seen"].append(entry)
            if r["id"] in preflight_ids:indexes["preflight_normal"].append(entry)
            if r["split"]!="validation":continue
            pixels=np.asarray(case["image"])
            if not np.array_equal(np.asarray(erased)[:question_y],pixels[:question_y]):raise ValueError("question erasure changed passage")
            passage=case["image"].copy()
            passage.paste("white",(0,0,1024,question_y))
            ImageDraw.Draw(passage).text((32,32),"资料：",font=font,anchor="lt",fill="black")
            if not np.array_equal(np.asarray(passage)[question_y:],pixels[question_y:]):raise ValueError("passage erasure changed question")
            ppath=control_dir/f"{r['id']}_passage_erased.png";passage.save(ppath)
            indexes["normal_held"].append(entry)
            qentry=index_entry(case,qpath,seed);pentry=index_entry(case,ppath,seed)
            indexes["question_erased_held"].append(qentry);indexes["passage_erased_held"].append(pentry)
            if r["id"] in preflight_ids:
                indexes["preflight_question_erased"].append(qentry);indexes["preflight_passage_erased"].append(pentry)
            bindings.append(dict(id=r["id"],pair=pair,seed=seed,normal_sha256=entry["input_sha256"],
                question_erased_sha256=qentry["input_sha256"],passage_erased_sha256=pentry["input_sha256"],
                source_question_y=question_y,source_answer_y=32+(case["context_lines"]+case["question_lines"])*48,
                question_erased_common_answer_y=answer_y,full_passage_pixels_preserved=True,full_question_pixels_preserved=True))
    positions={r["id"]:i for i,r in enumerate(rows)}
    expected=dict(normal_training=256,normal_seen=32,normal_held=32,question_erased_held=32,passage_erased_held=32,
        preflight_normal=4,preflight_question_erased=2,preflight_passage_erased=2)
    index_files={}
    for key,values in indexes.items():
        values.sort(key=lambda r:positions[r["id"]])
        if len(values)!=expected[key] or any(set(r)!={"id","input","input_sha256","seed"} for r in values):raise ValueError("input-only cohort differs")
        save_json(args.output/(key+"_inputs.json"),values);index_files[key+"_inputs.json"]=sha(args.output/(key+"_inputs.json"))
    for i in range(0,32,2):
        a,b=indexes["question_erased_held"][i:i+2]
        if a["input_sha256"]!=b["input_sha256"] or a["seed"]!=b["seed"]:raise ValueError("erased same-passage inputs must be identical")
    report=dict(source_sha256=sha(args.source),manifest_sha256=sha(manifest),audit_sha256=sha(args.audit),
        source_font=provenance,index_files_sha256=index_files,cohort_sizes=expected,held_control_bindings=bindings,
        held_question_erased_pairs_bit_identical=16,complete_normal_pages_reproduced=len(cases),
        question_erasure_preserves_complete_passage_pixels=True,passage_erasure_preserves_complete_question_pixels=True,
        question_erasure_fixed_labels_retained=True,answer_label_y_normalized_within_erased_pair=True,
        answer_aware_cropping=False,model_calls=0,optimizer_updates=0,semantic_learning_validated=False,
        native_large_page_memory_and_representation_unvalidated=True,implementation_sha256=sha(__file__))
    save_json(args.output/"preparation.json",report)
    print(json.dumps(dict(cohorts=expected,complete_normal_pages_reproduced=len(cases),erased_pairs_bit_identical=16),indent=2))


if __name__=="__main__":
    main()
