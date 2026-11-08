#!/usr/bin/env python3
"""Inspect real training prefixes or export a complete native QA experiment.

Neither mode launches a model. Export requires all registered updates and all
assessment outputs. A live prefix is explicitly not a language-quality result.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.image_edit.qa_learning import (CONDITION, CONTROLS, INITIAL, INITIAL_SHA, PAGES,
    SOURCE, dataset, digest, identity, schedule)


def inspect_training(run, *, final):
    import torch
    from scripts.export_balanced_autonomous_writer import tensor_png
    train, _ = dataset(); plan = schedule(train); prepared = json.loads((run/"preparation.json").read_text())
    if identity()!=prepared["code_sha256"] or digest(plan)!=prepared["schedule_sha256"]:
        raise ValueError("unchanged actual implementation and schedule required")
    # The trainer replaces this path atomically. One complete byte read binds
    # the observed checkpoint even when a later update replaces the live file.
    payload=(run/"training/checkpoint.pt").read_bytes();checkpoint_sha=hashlib.sha256(payload).hexdigest()
    state=torch.load(io.BytesIO(payload),map_location="cpu",weights_only=True);del payload
    count=state["step"];history=state["history"];meta=state["metadata"]
    if not 1<=count<=128 or (final and count!=128) or len(history)!=count:
        raise ValueError("valid actual prefix, or complete final training, required")
    if (meta["initial_checkpoint_sha256"]!=INITIAL_SHA or meta["implementation"]!=identity()
            or meta["condition_sha256"]!=sha(CONDITION) or meta["source_qa_sha256"]!=sha(SOURCE)
            or meta["manifest_sha256"]!=sha(PAGES/"pairs.jsonl")
            or meta["schedule_sha256"]!=prepared["schedule_sha256"]
            or not meta["fresh_optimizer"] or meta["optimizer_restored_from_copying"]
            or meta["updates"]!=128 or meta["physical_coefficient"]!=.4035679875746343
            or meta["steps"]!=50 or meta["output_canvas"]!=[512,256] or meta["input_canvas"]!=[1024,1024]
            or any(meta[k] for k in ("target_in_sampler","teacher_at_inference","output_repair"))):
        raise ValueError("registered actual training metadata differs")
    if (not state["optimizer"]["state"] or len(state["optimizer"]["state"])!=len(state["adapters"])
            or any(int(s["step"])!=count for s in state["optimizer"]["state"].values())):
        raise ValueError("every actual adapter optimizer state must match the observed update")
    if (len(state["optimizer"]["param_groups"])!=1 or state["optimizer"]["param_groups"][0]["lr"]!=1e-5
            or state["optimizer"]["param_groups"][0]["weight_decay"]!=.01):
        raise ValueError("actual registered optimizer settings differ")
    by_id={r["id"]:r for r in train}
    for row,planned in zip(history,plan):
        if any(row[k]!=v for k,v in planned.items()) or not all(math.isfinite(row[k]) for k in ("loss","recognition","ink_mass_kl","gradient_norm","elapsed_seconds")) or row["gradient_norm"]<=0:
            raise ValueError("actual finite update and exact schedule prefix required")
        original=by_id[row["id"]]
        if row["input_sha256"]!=original["input_sha256"] or row["target_sha256"]!=original["target_sha256"]:
            raise ValueError("actual supervision binding differs")
        folder=run/"training"/row["folder"];out=row["autonomous_page"]
        tensor_png(folder/"pixels.pt",folder/"answer.png",out["pixels_sha256"],out["png_sha256"])
    first=history[0]
    if (not state["initial_check"]["native_forward_equal"] or state["initial_check"]["id"]!=first["id"]
            or state["initial_check"]["seed"]!=first["noise_seed"]
            or set(first["prediction_gradient_norms"])!={"0","24","49"}
            or not all(v>0 and math.isfinite(v) for v in first["prediction_gradient_norms"].values())):
        raise ValueError("actual native equivalence and full-trajectory gradient checks required")
    # A zero-update interrupted attempt may precede the committed first update.
    # Locate the unique retained native page bound by the checkpoint receipt.
    candidates=[p for p in (run/"training/update_001").glob("attempt_*/native_forward")
        if (p/"answer.png").exists() and (p/"pixels.pt").exists()
        and sha(p/"answer.png")==state["initial_check"]["native_output"]["png_sha256"]
        and sha(p/"pixels.pt")==state["initial_check"]["native_output"]["pixels_sha256"]]
    if not candidates:raise ValueError("bound native first forward is missing")
    native=candidates[0];check=state["initial_check"]["native_output"]
    tensor_png(native/"pixels.pt",native/"answer.png",check["pixels_sha256"],check["png_sha256"])
    if not torch.equal(torch.load(native/"pixels.pt",map_location="cpu",weights_only=True),
            torch.load(run/"training"/first["folder"]/"pixels.pt",map_location="cpu",weights_only=True)):
        raise ValueError("retained first differentiable/native tensors differ")
    if sha(INITIAL)!=INITIAL_SHA:raise ValueError("fixed original checkpoint changed")
    initial=torch.load(INITIAL,map_location="cpu",weights_only=True)
    if state["adapters"].keys()!=initial["adapters"].keys() or not all(torch.isfinite(v).all() for v in state["adapters"].values()):
        raise ValueError("finite complete actual adapter architecture required")
    changed=sum(not torch.equal(v,initial["adapters"][name]) for name,v in state["adapters"].items())
    if not changed:raise ValueError("no actual trained parameter change")
    report=None
    if final:
        report=json.loads((run/"training/report.json").read_text())
        if report["updates"]!=128 or report["checkpoint_sha256"]!=checkpoint_sha or report["metadata"]!=meta or report["changed_adapter_tensors"]!=changed:
            raise ValueError("complete training report does not bind the final weights")
        if Counter(r["id"] for r in history)!=Counter({r["id"]:4 for r in train}):
            raise ValueError("four complete passes required")
    return dict(checkpoint_sha256=checkpoint_sha,observed_updates=count,registered_updates=128,
        verified_actual_training_images=len(history),all_optimizer_steps_match=True,changed_adapter_tensors=changed,
        trainable_parameter_count=sum(v.numel() for v in state["adapters"].values()),
        first_native_and_differentiable_actual_tensors_equal=True,finite_nonzero_early_middle_final_gradients=True,
        schedule_prefix_reproduced=True,checkpoint_read_was_one_atomic_snapshot=True,
        prefix_only=not final,report=report,metadata=meta,history=history,
        verifier_sha256=sha(__file__),tensor_authenticator_sha256=sha(ROOT/"scripts/export_balanced_autonomous_writer.py"),
        new_neural_calls=0,semantic_learning_validated=False,general_ilm_goal_complete=False)


def render_galleries(rows, output, *, columns=("input","reference","initial","final")):
    """Preview inputs to fit; retain reference/model answer pixels unchanged."""
    from PIL import Image,ImageDraw,ImageFont
    if len(columns)!=4 or columns[0]!="input":raise ValueError("input plus three evidence columns required")
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",14)
    names=[]
    for offset in range(0,len(rows),8):
        selected=rows[offset:offset+8];canvas=Image.new("RGB",(2048,len(selected)*568),"white");draw=ImageDraw.Draw(canvas)
        for i,row in enumerate(selected):
            y=i*568;draw.text((8,y+4),f"{row['task']} / {row['variant']} / {row['split']} / {row['id']}",fill='black',font=font)
            for col,key in enumerate(columns):
                x=col*512;draw.text((x+8,y+24),key,fill='black',font=font)
                with Image.open(output/row["images"][key]) as image:
                    shown=image.copy()
                    if key=="input":shown.thumbnail((512,512),Image.Resampling.LANCZOS)
                    elif shown.size!=(512,256):raise ValueError("complete reference/answer geometry required")
                    canvas.paste(shown,(x,y+48))
                    if canvas.crop((x,y+48,x+shown.width,y+48+shown.height)).tobytes()!=shown.tobytes():
                        raise ValueError("contact sheet changed or clipped an evidence image")
        name=f"galleries/cases_{offset:03}_{offset+len(selected)-1:03}.png";canvas.save(output/name);names.append(name)
    return names


def export(run, output):
    import torch
    from scripts.export_balanced_autonomous_writer import tensor_png
    from scripts.evaluate_image_edit_baseline import normalize
    verified=inspect_training(run,final=True);a=json.loads((run/"assessment.json").read_text())
    if (a["actual_qa_assessment_calls"]!=256 or a["actual_final_copying_calls"]!=64
            or a["final_checkpoint_sha256"]!=verified["checkpoint_sha256"]
            or a["initial_checkpoint_sha256"]!=INITIAL_SHA or len(a["rows"])!=256):
        raise ValueError("complete actual registered QA/copying assessment required")
    train,held=dataset()
    source={r["id"]:r for r in map(json.loads,SOURCE.read_text().splitlines())}
    copies=ROOT/"artifacts/image_edit_learning/strip_pages/pairs.jsonl"
    copy_rows={r["id"]:r for r in map(json.loads,copies.read_text().splitlines())}
    output.mkdir(parents=True,exist_ok=False)
    (output/"images").mkdir();(output/"reports").mkdir();(output/"galleries").mkdir()
    reports={};files={};rows=[]

    def retain(path,name,expected=None):
        path=Path(path);checksum=sha(path)
        if expected is not None and checksum!=expected:raise ValueError("source artifact changed")
        relative="images/"+name;target=output/relative
        if target.exists():
            if sha(target)!=checksum:raise ValueError("conflicting export name")
        else:shutil.copyfile(path,target)
        files[relative]=checksum;return relative

    for kind,originals,base,variants in (("qa",train+held,PAGES,("normal","question_erased","passage_erased")),
            ("copy",list(copy_rows.values()),copies.parent,("copy",))):
        if kind=="copy":
            # The complete copying assessment uses the first 32 rows per split.
            originals=[copy_rows[r["id"]] for r in json.loads((ROOT/"artifacts/image_edit_learning/full_strip_inputs.json").read_text())]
        for variant in variants:
            chosen=originals if variant in ("normal","copy") else held
            index_path=(ROOT/"artifacts/image_edit_learning/full_strip_inputs.json" if kind=="copy" else run/(variant+"_inputs.json"))
            provided=json.loads(index_path.read_text())
            if [r["id"] for r in provided]!=[r["id"] for r in chosen]:
                raise ValueError("complete registered input index required")
            calibration=(ROOT/"artifacts/balanced_autonomous_writer_v1/reference_ocr/report.json" if kind=="copy" else CONTROLS/"reference_ocr/report.json")
            cohort={}
            for arm in ("initial","final"):
                folder=(ROOT/"artifacts/ink_mass_comparison_v1/ink_mass_inference" if arm=="initial" and kind=="copy"
                    else run/(arm+"_"+variant+"_inference"))
                score_path=(ROOT/"artifacts/ink_mass_comparison_v1/ink_mass_ocr/report.json" if arm=="initial" and kind=="copy"
                    else run/(arm+"_"+variant+"_ocr/report.json"))
                actual=json.loads((folder/"outputs.json").read_text());scores=json.loads(score_path.read_text())
                native=json.loads((folder/"report.json").read_text())
                expected_checkpoint=INITIAL_SHA if arm=="initial" else verified["checkpoint_sha256"]
                if (native["checkpoint_sha256"]!=expected_checkpoint or native["steps"]!=50 or native["guidance"]!=1.
                        or native["output_canvas"]!=[512,256] or native["inputs_sha256"]!=sha(index_path)
                        or native["condition_sha256"]!=sha(CONDITION) or native["source"]!=verified["metadata"]["source"]
                        or native["runner_sha256"]!=sha(ROOT/"scripts/infer_paired_image_cohort_v2.py")
                        or not native["inference_source_files_unchanged"] or native["complete_generations"]!=len(chosen)
                        or any(native[k] for k in ("reference_answer_used","training_manifest_loaded","question_specific_text_condition","output_repair"))):
                    raise ValueError("unassisted native image inference required")
                if scores["reference_calibration_sha256"]!=sha(calibration) or any(sha(ROOT/p)!=h for p,h in scores["implementation"].items()):
                    raise ValueError("unchanged separate evaluator/calibration required")
                if any([r["id"] for r in items]!=[r["id"] for r in chosen] for items in (actual,scores["rows"])):
                    raise ValueError("complete original-order output observations required")
                cohort[arm]=(folder,actual,scores)
                for suffix,path in (("inference_report",folder/"report.json"),("output_ocr",score_path),("output_index",folder/"outputs.json")):
                    name=f"{arm}_{kind}_{variant}_{suffix}.json";shutil.copyfile(path,output/"reports"/name);reports["reports/"+name]=sha(path)
            if variant=="question_erased":
                for _,(folder,actual,_) in cohort.items():
                    for i in range(0,32,2):
                        given=provided[i:i+2]
                        raw=[torch.load((folder/r["path"]).parent/"pixels.pt",map_location="cpu",weights_only=True) for r in actual[i:i+2]]
                        if given[0]["input_sha256"]!=given[1]["input_sha256"] or given[0]["seed"]!=given[1]["seed"] or not torch.equal(*raw):
                            raise ValueError("identical question-erased inputs/seeds require identical actual tensors")
            for i,item in enumerate(chosen):
                prefix=f"{kind}_{variant}_{i:03}"
                input_path=(base/item["input"] if variant in ("normal","copy") else Path(provided[i]["input"]))
                input_hash=provided[i]["input_sha256"]
                if variant in ("normal","copy") and input_hash!=item["input_sha256"]:
                    raise ValueError("source input and registered request differ")
                images=dict(input=retain(input_path,prefix+"_input.png",input_hash),reference=retain(base/item["target"],prefix+"_reference.png",item["target_sha256"]))
                observed={};common_seed=None
                for arm,(folder,actual,scores) in cohort.items():
                    out,score=actual[i],scores["rows"][i];path=folder/out["path"];evidence=out["result"]
                    if (out["input_sha256"]!=input_hash or out["seed"]!=provided[i]["seed"]
                            or score["png_sha256"]!=out["sha256"] or score["expected"]!=item["reference_text"]):
                        raise ValueError("actual input/target observation differs")
                    if common_seed is not None and common_seed!=out["seed"]:raise ValueError("matched initial/final seeds required")
                    common_seed=out["seed"]
                    tensor_png(path.parent/"pixels.pt",path,evidence["pixels_sha256"],out["sha256"])
                    images[arm]=retain(path,prefix+"_"+arm+".png",out["sha256"])
                    accepted=([r["text"] for r in source[item["id"]]["references"]] if kind=="qa" else [item["reference_text"]])
                    observed[arm]=dict(raw_transcript=score["raw_transcript"],canonical_exact=normalize(score["raw_transcript"],item["language"])==normalize(item["reference_text"],item["language"]),
                        accepted_exact=normalize(score["raw_transcript"],item["language"]) in {normalize(r,item["language"]) for r in accepted},
                        source_accepted_answers=accepted,actual_tensor_sha256=evidence["pixels_sha256"])
                    if observed[arm]["canonical_exact"]!=score["ocr_exact"]:raise ValueError("canonical exact observation does not reproduce")
                    if kind=="qa":
                        old=next(r for r in a["rows"] if r["id"]==item["id"] and r["arm"]==arm and r["variant"]==variant)
                        if any(observed[arm][k]!=old[k] for k in ("raw_transcript","canonical_exact","accepted_exact")):
                            raise ValueError("independently reconstructed assessment observation differs")
                rows.append(dict(task=kind,variant=variant,id=item["id"],split=item["split"],language=item["language"],
                    pair=item.get("pair"),seed=common_seed,canonical_answer=item["reference_text"],images=images,observations=observed))
    if Counter((r["task"],r["variant"]) for r in rows)!=Counter({("qa","normal"):64,("qa","question_erased"):32,("qa","passage_erased"):32,("copy","copy"):64}):
        raise ValueError("all QA/control/copy cases required")
    summaries={}
    for arm in ("initial","final"):
        summaries[arm]={}
        for variant in ("normal","question_erased","passage_erased"):
            summaries[arm][variant]={}
            for split in ("train","validation"):
                selected=[r for r in rows if r["task"]=="qa" and r["variant"]==variant and r["split"]==split]
                pairs={}
                for row in selected:pairs.setdefault(row["pair"],[]).append(row)
                if any(len(rs)!=2 for rs in pairs.values()):raise ValueError("complete passage pairs required")
                summaries[arm][variant][split]=dict(cases=len(selected),canonical_exact=sum(r["observations"][arm]["canonical_exact"] for r in selected),
                    accepted_exact=sum(r["observations"][arm]["accepted_exact"] for r in selected),pairs=len(pairs),
                    both_correct_pairs=sum(all(r["observations"][arm]["accepted_exact"] for r in rs) for rs in pairs.values()))
    copying={arm:{split:dict(cases=32,ocr_exact=sum(r["observations"][arm]["canonical_exact"] for r in rows if r["task"]=="copy" and r["split"]==split))
        for split in ("train","validation")} for arm in ("initial","final")}
    before=summaries["initial"]["normal"]["validation"];after=summaries["final"]["normal"]["validation"]
    gates=dict(seen_fit=summaries["final"]["normal"]["train"]["accepted_exact"]>=24,
        held_answers=after["accepted_exact"]>=max(12,before["accepted_exact"]+4),
        held_pairs=after["both_correct_pairs"]>=max(4,before["both_correct_pairs"]+2),
        question_dependence=after["accepted_exact"]>=summaries["final"]["question_erased"]["validation"]["accepted_exact"]+4,
        passage_dependence=after["accepted_exact"]>=summaries["final"]["passage_erased"]["validation"]["accepted_exact"]+4,
        copying_retention=copying["final"]["validation"]["ocr_exact"]>=copying["initial"]["validation"]["ocr_exact"]-2)
    if summaries!=a["summaries"] or copying!=a["copying_summaries"] or gates!=a["prospective_exploratory_gates"] or all(gates.values())!=a["exploratory_rule_met"]:
        raise ValueError("independently reconstructed complete counts/criteria differ")
    training_images=[]
    for entry in verified["history"]:
        name=retain(run/"training"/entry["folder"]/"answer.png",f"training_{entry['update']:03}_before.png",entry["autonomous_page"]["png_sha256"])
        training_images.append(dict(update=entry["update"],id=entry["id"],image=name))
    galleries=render_galleries(rows,output)
    for name in galleries:files[name]=sha(output/name)
    for name,path in (("assessment.json",run/"assessment.json"),("preparation.json",run/"preparation.json"),("training_report.json",run/"training/report.json"),("training_schedule.json",run/"training/schedule.json"),("training_history.json",run/"training/updates.json")):
        shutil.copyfile(path,output/"reports"/name);reports["reports/"+name]=sha(path)
    save_json(output/"training_verification.json",verified);reports["training_verification.json"]=sha(output/"training_verification.json")
    licence=(SOURCE.parent/"LICENCE").read_bytes()
    if hashlib.sha1(f"blob {len(licence)}\0".encode()+licence).hexdigest()!="3b7b82d0da2db857eda1a798dbd908ea136f07b5":
        raise ValueError("pinned original source licence differs")
    (output/"CMRC2018_LICENCE").write_bytes(licence)
    (output/"SOURCE_ATTRIBUTION.md").write_text(
        "# Source attribution\n\n"
        "Question/passages derive from CMRC 2018: Yiming Cui et al., *A Span-Extraction Dataset for Chinese Machine Reading Comprehension*, EMNLP-IJCNLP 2019, https://aclanthology.org/D19-1600/. "
        "Source: https://github.com/ymcui/cmrc2018/tree/c0eb1b6ba219847457e6af3180da722bbeb656af. "
        "The original Attribution-ShareAlike 4.0 licence is retained verbatim in CMRC2018_LICENCE; see https://creativecommons.org/licenses/by-sa/4.0/. "
        "The local preparation removes whitespace, selects full short paragraphs, pairs questions, and renders images. Erased-input controls and resized contact-sheet input previews are identified separately. "
        "Complete source bindings and normalization are recorded in the experiment's preparation and source reports. Model outputs are unrepaired experimental images.\n")
    reports["SOURCE_ATTRIBUTION.md"]=sha(output/"SOURCE_ATTRIBUTION.md")
    reports["CMRC2018_LICENCE"]=sha(output/"CMRC2018_LICENCE")
    result=dict(rows=rows,training_images=training_images,galleries=galleries,files_sha256=files,reports_sha256=reports,
        summaries=a["summaries"],copying_summaries=a["copying_summaries"],prospective_exploratory_gates=a["prospective_exploratory_gates"],
        exploratory_rule_met=a["exploratory_rule_met"],initial_checkpoint_sha256=INITIAL_SHA,final_checkpoint_sha256=verified["checkpoint_sha256"],
        actual_new_native_assessment_calls=320,reused_initial_copying_outputs=64,complete_output_images=384,complete_training_images=128,
        all_retained_outputs_equal_actual_tensors=True,input_previews_only_resized_in_contact_sheets=True,
        all_individual_inputs_original=True,all_generated_outputs_complete_and_unrepaired=True,exporter_sha256=sha(__file__),
        raw_review_completed=False,semantic_learning_validated=False,reliable_writing_qualified=False,general_ilm_goal_complete=False)
    save_json(output/"result.json",result)
    table=["# Complete native full-page QA assessment", "",
        "Numerical assessment is complete; raw-image review is pending. No reliable language or writing capability is qualified.", "",
        "| Source-accepted OCR observation | Initial | Final |", "| --- | ---: | ---: |"]
    for title,split,key,denominator in (("Seen answers","train","accepted_exact",32),
            ("Held-out answers","validation","accepted_exact",32),
            ("Held-out both-correct passage pairs","validation","both_correct_pairs",16)):
        table.append(f"| {title} | {summaries['initial']['normal'][split][key]}/{denominator} | {summaries['final']['normal'][split][key]}/{denominator} |")
    for variant in ("question_erased","passage_erased"):
        table.append(f"| Held out, {variant.replace('_',' ')} | {summaries['initial'][variant]['validation']['accepted_exact']}/32 | {summaries['final'][variant]['validation']['accepted_exact']}/32 |")
    table.extend(["", "| Canonical OCR copying observation | Initial | Final |", "| --- | ---: | ---: |"])
    for split in ("train","validation"):
        table.append(f"| {split} | {copying['initial'][split]['ocr_exact']}/32 | {copying['final'][split]['ocr_exact']}/32 |")
    table.extend(["", "The prospective exploratory conditions are recorded individually in [result.json](result.json). "
        "They do not establish broad language competence, reliable writing or efficiency at matched quality. "
        "The source preparation uses public development data already examined in earlier work. "
        "Canonical-reference OCR is imperfect: 23/32 seen and 26/32 held-out answers. "
        "The first 32 natural questions are optimized for four complete passes; only final update 128 is assessed. "
        "Initial weights are fixed before QA training and assessed afterwards with the same inputs and seeds.", "",
        "All 384 initial/final answer images and 128 actual pre-update training images are retained, with hashes. "
        "There are 320 new native assessment calls and 64 reused initial copying outputs. "
        "Every original input is preserved; only contact-sheet input previews are resized to fit. "
        "Every displayed model answer is complete and unrepaired. "
        "Galleries are listed in result.json; unmodified source reports and independent training verification accompany them.", "",
        "The runtime retains the externally pretrained transformer and VAE. "
        "Training-only OCR supervision is excluded from the standalone image generator. "
        "Parameter counts, costs, source hashes and the fixed generic condition are in [training verification](training_verification.json) "
        "and [training report](reports/training_report.json). "
        "[Dataset attribution and transformations](SOURCE_ATTRIBUTION.md), [original dataset licence](CMRC2018_LICENCE).", ""])
    (output/"README.md").write_text("\n".join(table))
    print(json.dumps(dict(cases=len(rows),outputs=384,training_images=128,raw_review_completed=False),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage",choices=("inspect-training","export"),required=True)
    args=parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("hide CUDA for the separate offline exporter")
    if args.stage=="inspect-training":
        if args.output.exists():raise ValueError("preserve earlier checkpoint observation")
        observed=inspect_training(args.run.resolve(),final=False);args.output.parent.mkdir(parents=True,exist_ok=True);save_json(args.output,observed)
        print(json.dumps({k:observed[k] for k in ("observed_updates","verified_actual_training_images","changed_adapter_tensors","prefix_only")},indent=2))
    else:export(args.run.resolve(),args.output.resolve())
