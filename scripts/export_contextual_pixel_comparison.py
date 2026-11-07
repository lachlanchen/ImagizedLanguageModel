#!/usr/bin/env python3
"""Independently verify both trained arms, all draws and every answer raster."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from scripts.audit_contextual_pixel_data import audit,pixels,read,sha,trim

ROOT=Path(__file__).absolute().parents[1]
PREPARED=ROOT/"artifacts/contextual_pixel_learning_v1_attempt02/prepared"
PREVIOUS=ROOT/"artifacts/contextual_pixel_spans_v1/prepared"


def training_check(run,arm,schedule,initial,identity):
    folder=run/arm;report=read(folder/"training_report.json")
    if report["identity"]!=identity or sha(folder/"checkpoint.pt")!=report["checkpoint_sha256"]:
        raise ValueError("trained checkpoint identity differs")
    saved=torch.load(folder/"checkpoint.pt",map_location="cpu",weights_only=True)
    if saved["step"]!=3257 or saved["identity"]!=identity or set(saved["model"])!=set(initial):
        raise ValueError("incomplete or incompatible learned state")
    journals=[]
    for name,digest in sorted(report["journals_sha256"].items()):
        path=folder/name
        if sha(path)!=digest:raise ValueError("optimizer journal changed")
        journals.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    if {r["update"] for r in journals}!=set(range(1,3258)):
        raise ValueError("optimizer history does not cover every registered update")
    for row in journals:
        if row["ids"]!=schedule[(row["update"]-1)*4:row["update"]*4] or not all(math.isfinite(row[k]) for k in ("loss","gradient_norm","seconds")):
            raise ValueError("draw schedule or finite optimization differs")
    heads=[n for n in initial if n.startswith("head.")]
    body=[n for n in initial if not n.startswith("head.")]
    expected_groups=[heads,body] if arm=="adapted" else [heads]
    groups=saved["optimizer"]["param_groups"]
    if len(groups)!=len(expected_groups):raise ValueError("unexpected optimizer parameter groups")
    moments=saved["optimizer"]["state"];parameter_rows={}
    optimized=sum(expected_groups,[])
    for names,group in zip(expected_groups,groups):
        if len(names)!=len(group["params"]):raise ValueError("optimizer tensor count differs")
        for name,index in zip(names,group["params"]):
            state=moments[index];weight=saved["model"][name]
            if int(state["step"])!=3257 or state["exp_avg"].shape!=weight.shape or state["exp_avg_sq"].shape!=weight.shape:
                raise ValueError("Adam state does not describe complete training")
            if not all(torch.isfinite(state[k]).all() for k in ("exp_avg","exp_avg_sq")):
                raise ValueError("nonfinite optimizer moments")
    for name,weight in saved["model"].items():
        if weight.dtype!=torch.float32 or not torch.isfinite(weight).all() or weight.shape!=initial[name].shape:
            raise ValueError("invalid learned weight")
        changed=not torch.equal(weight,initial[name])
        if arm=="frozen" and name in body and changed:raise ValueError("control contextual state changed")
        parameter_rows[name]=dict(shape=list(weight.shape),changed=changed,optimized=name in optimized,
            max_absolute_delta=float((weight-initial[name]).abs().max()),
            final_tensor_sha256=hashlib.sha256(weight.contiguous().numpy().tobytes()).hexdigest())
    return dict(updates=3257,draws=13028,all_training_ids_seen_four_times=True,
        optimizer_invocations_including_replays=len(journals),additional_replayed_updates=len(journals)-3257,
        all_optimizer_tensors_finite=True,all_adam_steps_exact=True,
        optimized_parameters=sum(initial[n].numel() for n in optimized),
        changed_head_tensors=sum(parameter_rows[n]["changed"] for n in heads),
        changed_context_tensors=sum(parameter_rows[n]["changed"] for n in body),parameters=parameter_rows)


def score_images(run,arm,expected,refs,retain):
    folder=run/arm/"assessment";report=read(folder/"report.json")
    raw=read(folder/"outputs.json");original=read(run/arm/"result.json")
    trained=read(run/arm/"training_report.json")
    if report["calls"]!=256 or len(raw)!=256 or sha(folder/"outputs.json")!=report["outputs_sha256"] or report["final_checkpoint_sha256"]!=trained["checkpoint_sha256"]:
        raise ValueError("complete final native assessment required")
    cases=[]
    for index,(row,wanted) in enumerate(zip(raw,expected)):
        if any(row[k]!=wanted[k] for k in wanted):raise ValueError("assessment cohort/order changed")
        source=PREVIOUS/row["input"];answer=folder/row["output"]
        if sha(source)!=row["input_sha256"] or sha(answer)!=row["output_sha256"]:raise ValueError("raw image changed")
        a,b=pixels(source),pixels(answer)
        occupied=(a==0).any(0);edge=np.diff(np.r_[False,occupied,False].astype(np.int8))
        left,right=np.flatnonzero(edge==1),np.flatnonzero(edge==-1)
        start,length=row["event"]["start"],row["event"]["length"]
        if not 0<=start<start+length<=len(left):raise ValueError("invalid physical source boundary")
        expected_pixels=np.pad(a[:,left[start]:right[start+length-1]],((0,0),(2,2)),constant_values=255)
        if not np.array_equal(b,expected_pixels):raise ValueError("output is not its declared original source pixels")
        key=row["id"].split("/question_erased")[0].split("/passage_erased")[0]
        targets=[];paths=[]
        for target in refs[key]["targets"]:
            path=PREVIOUS/target["file"]
            if sha(path)!=target["sha256"]:raise ValueError("source reference changed")
            targets.append(pixels(path));paths.append(retain(path,"references/"+target["sha256"]+".png"))
        exact=any(np.array_equal(trim(b),trim(t)) for t in targets)
        prediction=Counter(bytes(c) for c in trim(b).T)
        f1=max(2*sum((prediction&Counter(bytes(c) for c in trim(t).T)).values())/(trim(b).shape[1]+trim(t).shape[1]) for t in targets)
        other=original["cases"][index]
        if other["id"]!=row["id"] or exact!=other["exact"] or abs(f1-other["unordered_column_f1"])>1e-12:
            raise ValueError("independent complete-image score differs")
        cases.append(dict(**{k:row[k] for k in ("id","split","task","language","condition","pair","article","event")},
            input=retain(source,f"inputs/{index:04d}.png"),output=retain(answer,f"{arm}/answers/{index:04d}.png"),
            references=paths,exact=exact,unordered_column_f1=f1))
    groups=defaultdict(list);pairs=defaultdict(list)
    for r in cases:
        groups[f"{r['split']}/{r['task']}/{r['condition']}/{r['language']}"].append(r)
        if r["split"]=="development" and r["task"]=="qa" and r["condition"]=="normal":pairs[r["pair"]].append(r)
    totals={k:dict(cases=len(v),exact=sum(r["exact"] for r in v),unordered_column_f1=sum(r["unordered_column_f1"] for r in v)/len(v)) for k,v in groups.items()}
    both=sum(all(r["exact"] for r in v) for v in pairs.values())
    for condition in ("normal","question_erased","passage_erased"):
        if totals[f"development/qa/{condition}/zh"]["cases"]!=64:raise ValueError("incomplete question/erasure cohort")
    copies=sum(v["exact"] for k,v in totals.items() if k.startswith("development/copy/"))
    qa=totals["development/qa/normal/zh"]["exact"];erased=totals["development/qa/question_erased/zh"]["exact"]
    gates=dict(copying=copies>=30,natural_answers=qa>=48,paired_answers=both>=20,question_dependence=(qa-erased)/64>=.25)
    if totals!=original["groups"] or both!=original["both_correct_pairs"] or gates!=original["gates"]:
        raise ValueError("independent aggregate or registered decision differs")
    return dict(groups=totals,both_correct_pairs=both,gates=gates,component_qualified=all(gates.values()),
        normal_qa_exact=qa,new_copy_exact=copies,cases=cases,all256_outputs_source_reconstructed=True)


def main(args):
    run=args.run
    if read(run/"status.json")["stage"]!="completed":raise ValueError("comparison is still live or incomplete; preserve its runtime")
    identity=read(run/"identity.json")
    for name,digest in identity["sources"].items():
        if sha(ROOT/name)!=digest:raise ValueError("bound executed implementation changed: "+name)
    for path,key in ((PREPARED/"report.json","preparation_sha256"),
                     (PREVIOUS/"inputs.json","development_inputs_sha256"),
                     (PREVIOUS/"references.json","development_references_sha256")):
        if sha(path)!=identity[key]:raise ValueError("bound data changed")
    schedule=read(run/"schedule.json");ids=[r["id"] for r in read(PREPARED/"inputs.json")]
    expected_schedule=[];rng=random.Random(2026110706)
    for _ in range(4):
        order=ids.copy();rng.shuffle(order);expected_schedule.extend(order)
    if schedule!=expected_schedule or set(Counter(schedule).values())!={4}:raise ValueError("complete draw schedule differs")
    initial=torch.load(run/"initial_state.pt",map_location="cpu",weights_only=True)
    if sha(run/"initial_state.pt")!=read(run/"setup.json")["initial_state_sha256"]:raise ValueError("initial state changed")
    earlier=ROOT/"artifacts/contextual_pixel_spans_v1/training/checkpoint.pt"
    if sha(earlier)!=identity["initial_head_sha256"]:raise ValueError("original head changed")
    old=torch.load(earlier,map_location="cpu",weights_only=True)["model"]
    if any(not torch.equal(initial["head."+name],value) for name,value in old.items()):raise ValueError("arms did not start from the registered head")
    if any(bool(v.any()) for n,v in initial.items() if n.endswith(".up")):raise ValueError("contextual adapters were not initially zero")
    checks={arm:training_check(run,arm,schedule,initial,identity) for arm in ("frozen","adapted")}
    data=audit(PREPARED,PREVIOUS)
    features=read(run/"features/report.json")
    if not features["complete"] or len(features["records"])!=3257:raise ValueError("source feature coverage incomplete")
    for r in features["records"]:
        if sha(ROOT/r["file"])!=r["sha256"]:raise ValueError("source feature payload changed")
    old_rows=read(PREVIOUS/"inputs.json")
    expected=[r for r in old_rows if r["split"]=="development"]+[r for r in old_rows if r["split"]=="train" and r["task"]=="qa"][:32]
    refs={r["id"]:r for r in read(PREVIOUS/"references.json")}
    for arm in ("frozen","adapted"):
        if not (run/arm/"standalone_replay/report.json").is_file():
            raise ValueError("complete the four standalone PNG calls per arm before exporting")
    args.output.mkdir(parents=True,exist_ok=False);files={}
    def retain(source,name):
        destination=args.output/name;destination.parent.mkdir(parents=True,exist_ok=True)
        digest=sha(source)
        if destination.exists() and sha(destination)!=digest:raise ValueError("retained filename collision")
        if not destination.exists():shutil.copy2(source,destination)
        files[name]=digest;return name
    results={arm:score_images(run,arm,expected,refs,retain) for arm in ("frozen","adapted")}
    standalone=[]
    for arm in ("frozen","adapted"):
        folder=run/arm/"standalone_replay";report=read(folder/"report.json")
        if report["checkpoint_sha256"]!=read(run/arm/"training_report.json")["checkpoint_sha256"] or len(report["outputs"])!=4:
            raise ValueError("four standalone calls from the correct final checkpoint required per arm")
        chosen=[r for r in results[arm]["cases"] if r["split"]=="development" and r["task"]=="qa" and r["condition"]=="normal"][:2]
        chosen += [r for r in results[arm]["cases"] if r["split"]=="development" and r["task"]=="copy"][:2]
        for index,(raw,wanted) in enumerate(zip(report["outputs"],chosen)):
            if raw["input_sha256"]!=files[wanted["input"]] or raw["event"]!=wanted["event"]:
                raise ValueError("standalone input or selected event differs")
            output=folder/raw["output"]
            if sha(output)!=raw["output_sha256"] or not np.array_equal(pixels(output),pixels(args.output/wanted["output"])):
                raise ValueError("standalone answer pixels differ")
            retain(output,f"{arm}/standalone_replay/{index:03d}.png")
            standalone.append(dict(arm=arm,id=wanted["id"],answer_pixels_exact=True,events_exact=True,
                original_head_checkpoint_opened=report["original_head_checkpoint_opened"],
                training_manifests_opened=report["training_manifests_opened"],
                feature_cache_opened=report["feature_cache_opened"]))
        for path in (run/arm).glob("updates_from_*.jsonl"):retain(path,f"{arm}/reports/{path.name}")
        for path in (run/arm/"training_report.json",run/arm/"result.json",run/arm/"assessment/report.json",run/arm/"assessment/outputs.json",folder/"report.json"):
            retain(path,f"{arm}/reports/"+str(path.relative_to(run/arm)).replace("/","_"))
    for path in (run/"identity.json",run/"schedule.json",run/"setup.json",run/"matched_forward_check.json",run/"features/report.json",PREPARED/"report.json"):
        retain(path,"reports/"+str(path.relative_to(run) if path.is_relative_to(run) else Path("preparation.json")).replace("/","_"))
    a,c=results["adapted"],results["frozen"]
    gains=dict(normal_qa=a["normal_qa_exact"]-c["normal_qa_exact"],both_correct_pairs=a["both_correct_pairs"]-c["both_correct_pairs"],new_copies=a["new_copy_exact"]-c["new_copy_exact"])
    useful=gains["normal_qa"]>=8 and gains["both_correct_pairs"]>=4 and gains["new_copies"]>=-2
    original=read(run/"result.json")
    if gains!=original["adaptation_minus_control"] or useful!=original["useful_improvement_over_control"]:raise ValueError("comparison decision differs")
    result=dict(arms=results,adaptation_minus_control=gains,useful_improvement_over_control=useful,
        optimizer_audit=checks,data_audit=data,source_feature_files_authenticated=3257,
        native_assessment_calls=512,standalone_native_replays=standalone,files_sha256=files,
        source_result_sha256=sha(run/"result.json"),export_script_sha256=sha(__file__),
        all512_raw_answers_retained=True,independent_png_scores_exact=True,
        source_pixel_copy_only=True,external_pretraining_retained=True,reused_public_development=True,
        free_generation_demonstrated=False,general_ilm_goal_complete=False,
        copyright="CMRC2018 Cui et al. (2019), CC-BY-SA-4.0; Qwen3-VL Alibaba, Apache-2.0; synthetic copies.")
    (args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print("Independently verified both optimizers, all512 answer rasters and eight standalone native calls.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4);main(args)
