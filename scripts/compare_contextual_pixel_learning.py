#!/usr/bin/env python3
"""Run the registered frozen-versus-adapted physical answer-image comparison."""
import argparse
from collections import Counter, defaultdict
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.contextual_pixel_learning import LearningPixelReader
from ilm.visual_lm.contextual_pixel_spans import units, normalized, matches
from scripts.cache_semantic_images_v72 import ROOT, CONTROLS, CONTROL_SHA, atomic_json, sha, committed_identity
from scripts.check_contextual_pixel_learning import PREVIOUS, HEAD_SHA, SOURCES as CHECK_SOURCES, cpu_tree, exact
from scripts.run_contextual_pixel_spans import array, read
from scripts.probe_native_unified_source import gpu_slot
from scripts.train_clean_corpus_v75 import setup

PREPARED_RUN = ROOT / "artifacts/contextual_pixel_learning_v1_attempt02"
SEED = 2026110706
SOURCES = (*CHECK_SOURCES, "scripts/compare_contextual_pixel_learning.py",
           "references/contextual_pixel_adaptation_protocol.md",
           "scripts/train_clean_corpus_v75.py", "scripts/cache_semantic_images_v72.py",
           "scripts/probe_native_unified_source.py")


def status(root, stage, **fields):
    value = dict(stage=stage, pid=os.getpid()); value.update(fields)
    atomic_json(root/"status.json",value)
    if fields.get("print_event",True): print(json.dumps(value),flush=True)


def pixel_matches(image, spans, targets):
    """Same complete-image marginal, pruning by exact physical width only."""
    endings = {int(end):i for i,(_,end) in enumerate(spans)}
    found = set()
    for target in targets:
        target = normalized(target); width = target.shape[1]
        for start,(left,_) in enumerate(spans):
            end = int(left)+width; last = endings.get(end)
            if last is not None and 0 <= last-start < 128 and np.array_equal(image[:,left:end],target):
                found.add((start,last-start))
    if not found: raise ValueError("reference has no complete visible source span")
    return sorted(found)


def inventory():
    prep = read(PREPARED_RUN/"prepared/report.json")
    check = read(PREPARED_RUN/"preflight/report.json")
    if not check["passed"] or check["preparation_sha256"] != sha(PREPARED_RUN/"prepared/report.json"):
        raise ValueError("completed authenticated backward preflight required")
    for name,digest in check["identity"]["sources"].items():
        if sha(ROOT/name)!=digest: raise ValueError("qualified source changed: "+name)
    for name,key in (("inputs.json","inputs_sha256"),("references.json","references_sha256")):
        if sha(PREPARED_RUN/"prepared"/name)!=prep[key]: raise ValueError("prepared index changed")
    rows = read(PREPARED_RUN/"prepared/inputs.json")
    refs = {r["id"]:r for r in read(PREPARED_RUN/"prepared/references.json")}
    if len(rows)!=3257 or prep["training_qa"]!=3193 or prep["training_copies"]!=64 or len(prep["rejected"])!=18:
        raise ValueError("registered complete training inventory required")
    labels = {}
    compare = set(prep["gradient_case_ids"]+list(prep["resource_case_ids"].values()))
    for row in rows:
        p = ROOT/row["input"]
        if sha(p)!=row["input_sha256"]: raise ValueError("training image changed")
        image=array(p); targets=[]
        for target in refs[row["id"]]["targets"]:
            p=ROOT/target["file"]
            if sha(p)!=target["sha256"]: raise ValueError("training reference changed")
            targets.append(array(p))
        u=units(image); label=pixel_matches(image,u,targets)
        if row["id"] in compare and label!=matches(image,u,targets):
            raise ValueError("width-pruned matching differs from exhaustive image marginal")
        labels[row["id"]]=label
    return rows,labels


def feature_logits(model, value):
    return model.head(value["visual"][None].to(model.device).float(),
        value["memory"][None].to(model.device).float(),value["geometry"][None].to(model.device),
        torch.tensor([len(value["visual"])],device=model.device))[0]


def cache_features(model, rows, root):
    folder=root/"features";folder.mkdir(exist_ok=True)
    manifest=read(folder/"report.json") if (folder/"report.json").exists() else {}
    complete={r["id"]:r for r in manifest.get("records",[])}
    old={r["id"]:r for r in read(PREVIOUS/"features/report.json")["records"]}
    started=time.monotonic(); extra=0; reused=0; model.eval()
    for i,row in enumerate(rows):
        if row["id"] in complete:
            record=complete[row["id"]]
            if sha(ROOT/record["file"])!=record["sha256"] or record["input_sha256"]!=row["input_sha256"]:
                raise ValueError("resumed feature changed")
            continue
        previous=old.get(row["id"])
        if previous:
            p=PREVIOUS/"features"/previous["file"]
            if sha(p)!=previous["sha256"]: raise ValueError("earlier feature changed")
            value=torch.load(p,map_location="cpu",weights_only=True)
            if value["input_sha256"]!=row["input_sha256"]: raise ValueError("earlier input geometry differs")
            reused+=1
        else:
            p=ROOT/row["input"]
            if sha(p)!=row["input_sha256"]: raise ValueError("cache input changed")
            with torch.no_grad(): value=model.encode(array(p))
            value={k:(v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v) for k,v in value.items()}
            value.update(id=row["id"],input_sha256=row["input_sha256"])
            p=folder/f"{i:04d}.pt";temporary=p.with_suffix(".partial")
            torch.save(value,temporary)
            if p.exists() and sha(p)!=sha(temporary): raise ValueError("unindexed feature differs; preserve it")
            temporary.replace(p);extra+=1
        complete[row["id"]]=dict(id=row["id"],file=str(p.relative_to(ROOT)),sha256=sha(p),input_sha256=row["input_sha256"])
        if (i+1)%16==0 or i+1==len(rows):
            atomic_json(folder/"report.json",dict(records=list(complete.values()),total=len(rows),complete=len(complete)==len(rows),
                elapsed_seconds=manifest.get("elapsed_seconds",0)+time.monotonic()-started,
                reused_prior_payloads=manifest.get("reused_prior_payloads",0)+reused,
                new_native_encodings=manifest.get("new_native_encodings",0)+extra,
                cache_function_receives_references=False,initial_model_only=True))
            status(root,"caching_frozen_features",completed=len(complete),total=len(rows),print_event=(i+1)%128==0)
        del value
    return complete


def check_head_gradients(model,row,record,label,root):
    value=torch.load(ROOT/record["file"],map_location="cpu",weights_only=True)
    model.train();model.zero_grad(set_to_none=True)
    logp,features=model(array(ROOT/row["input"]))
    for name in ("visual","memory","geometry","source_units"):
        if not torch.equal(features[name].detach().cpu(),value[name]): raise ValueError("native/cache features differ")
    loss=model.head.loss(logp[None],[label]);loss.backward()
    reference=dict(loss=float(loss.detach()),logp=logp.detach().cpu(),
        head_gradients={n:p.grad.detach().cpu().clone() for n,p in model.head.named_parameters()})
    del logp,features,loss;model.zero_grad(set_to_none=True)
    logp=feature_logits(model,value);loss=model.head.loss(logp[None],[label]);loss.backward()
    actual=dict(loss=float(loss.detach()),logp=logp.detach().cpu(),
        head_gradients={n:p.grad.detach().cpu().clone() for n,p in model.head.named_parameters()})
    if not exact(reference,actual): raise ValueError("native-versus-cached head logits/loss/gradients differ")
    model.zero_grad(set_to_none=True)
    atomic_json(root/"matched_forward_check.json",dict(id=row["id"],native_cached_logits_exact=True,
        native_cached_loss_exact=True,all_head_gradients_exact=True,loss=actual["loss"],optimizer_updates=0))
    del reference,actual,logp,loss,value;gc.collect();torch.cuda.empty_cache()


def save_checkpoint(path,model,optimizer,step,elapsed,identity):
    temporary=path.with_suffix(".partial")
    torch.save(dict(model=model.trained_state(),optimizer=cpu_tree(optimizer.state_dict()),step=step,
        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(model.device),
        elapsed_seconds=elapsed,identity=identity),temporary)
    if path.exists(): path.replace(path.with_name("checkpoint_previous.pt"))
    temporary.replace(path)


def train_arm(model,arm,rows,labels,features,schedule,root,identity):
    folder=root/arm;folder.mkdir(exist_ok=True);checkpoint=folder/"checkpoint.pt"
    report=folder/"training_report.json"
    if report.exists():
        value=read(report)
        if value["updates"]!=3257 or sha(checkpoint)!=value["checkpoint_sha256"]: raise ValueError("completed arm changed")
        return
    initial=torch.load(root/"initial_state.pt",map_location="cpu",weights_only=True)
    model.restore_trained_state(initial);del initial
    model.train(arm=="adapted")
    optimizer=model.optimizer() if arm=="adapted" else torch.optim.AdamW(model.head.parameters(),lr=1e-4,
        betas=(.9,.95),weight_decay=.01,foreach=False)
    params=[p for group in optimizer.param_groups for p in group["params"]]
    if len({id(p) for p in params})!=len(params): raise ValueError("duplicate optimized parameter")
    peaks=[1e-4,1e-5] if arm=="adapted" else [1e-4]
    step,elapsed=0,0.
    if checkpoint.exists():
        value=torch.load(checkpoint,map_location="cpu",weights_only=True)
        if value["identity"]!=identity or not 0<=value["step"]<=3257: raise ValueError("checkpoint identity differs")
        model.restore_trained_state(value["model"]);optimizer.load_state_dict(value["optimizer"])
        torch.set_rng_state(value["cpu_rng"]);torch.cuda.set_rng_state(value["cuda_rng"],model.device)
        step,elapsed=value["step"],value["elapsed_seconds"];del value
    index={r["id"]:r for r in rows};began=time.monotonic();torch.cuda.reset_peak_memory_stats()
    journal=folder/f"updates_from_{step:05d}_{time.time_ns()}.jsonl"
    with journal.open("w") as log:
        while step<3257:
            ids=schedule[step*4:(step+1)*4]
            factor=(step+1)/100 if step<100 else .1+.9*.5*(1+math.cos(math.pi*(step-100)/(3257-100)))
            for group,peak in zip(optimizer.param_groups,peaks):group["lr"]=peak*factor
            optimizer.zero_grad(set_to_none=True);losses=[];update_began=time.monotonic()
            for key in ids:
                if arm=="adapted":
                    row=index[key];path=ROOT/row["input"]
                    if sha(path)!=row["input_sha256"]: raise ValueError("active training input changed")
                    logp,values=model(array(path))
                else:
                    record=features[key];path=ROOT/record["file"]
                    if sha(path)!=record["sha256"]: raise ValueError("frozen feature changed")
                    values=torch.load(path,map_location="cpu",weights_only=True);logp=feature_logits(model,values)
                loss=model.head.loss(logp[None],[labels[key]])
                if not torch.isfinite(loss):raise FloatingPointError("nonfinite conditional image-span loss")
                (loss/4).backward();losses.append(float(loss.detach()));del loss,logp,values
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
                raise FloatingPointError("missing or nonfinite optimized gradient")
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));optimizer.step();step+=1
            torch.cuda.synchronize()
            record=dict(update=step,ids=ids,loss=sum(losses)/4,example_losses=losses,gradient_norm=norm,
                head_lr=optimizer.param_groups[0]["lr"],seconds=time.monotonic()-update_began,
                active_elapsed_seconds=elapsed+time.monotonic()-began)
            log.write(json.dumps(record)+"\n");log.flush()
            status(root,"training",arm=arm,update=step,total_updates=3257,loss=record["loss"],
                active_elapsed_seconds=record["active_elapsed_seconds"],print_event=step==1 or step%32==0)
            if step%32==0 or step==3257:
                save_checkpoint(checkpoint,model,optimizer,step,elapsed+time.monotonic()-began,identity)
    invocations=[]
    for path in sorted(folder.glob("updates_from_*.jsonl")):
        invocations.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    for row in invocations:
        if row["ids"]!=schedule[(row["update"]-1)*4:row["update"]*4]: raise ValueError("executed draw schedule differs")
    atomic_json(report,dict(arm=arm,identity=identity,updates=3257,draws=13028,epochs=4,examples=len(rows),
        checkpoint_sha256=sha(checkpoint),optimized_parameters=sum(p.numel() for p in params),
        full_runtime_parameters=sum(p.numel() for p in model.parameters()),
        elapsed_seconds=elapsed+time.monotonic()-began,all_invoked_update_seconds=sum(r["seconds"] for r in invocations),
        optimizer_invocations_including_replays=len(invocations),
        journals_sha256={p.name:sha(p) for p in folder.glob("updates_from_*.jsonl")},
        peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
        peak_reserved_vram_gib=torch.cuda.max_memory_reserved()/1024**3,
        cache_and_external_pretraining_excluded_from_arm_time=True,development_used_for_selection=False,
        free_generation_demonstrated=False,general_ilm_goal_complete=False))
    optimizer.zero_grad(set_to_none=True);del optimizer;gc.collect();torch.cuda.empty_cache()


def assess(model,arm,root,identity):
    folder=root/arm/"assessment";folder.mkdir(exist_ok=True)
    checkpoint=root/arm/"checkpoint.pt";training=read(root/arm/"training_report.json")
    if sha(checkpoint)!=training["checkpoint_sha256"]:raise ValueError("final checkpoint changed")
    value=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if value["step"]!=3257 or value["identity"]!=identity:raise ValueError("only final registered checkpoint may be assessed")
    model.restore_trained_state(value["model"]);model.eval();del value
    rows=read(PREVIOUS/"prepared/inputs.json")
    chosen=[r for r in rows if r["split"]=="development"]+[r for r in rows if r["split"]=="train" and r["task"]=="qa"][:32]
    outputs=read(folder/"outputs.json") if (folder/"outputs.json").exists() else []
    for index,row in enumerate(outputs):
        if row["id"]!=chosen[index]["id"] or sha(folder/row["output"])!=row["output_sha256"]:
            raise ValueError("resumed autonomous output changed")
    began=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for row in chosen[len(outputs):]:
        p=PREVIOUS/"prepared"/row["input"]
        if sha(p)!=row["input_sha256"]:raise ValueError("autonomous input changed")
        image=array(p)
        with torch.inference_mode():
            logp,features=model(image)
            pixels,event=model.head.select(logp,image,features["source_units"].numpy())
        path=folder/f"{len(outputs):04d}.png"
        if path.exists() and not np.array_equal(array(path),pixels):raise ValueError("unindexed autonomous answer differs")
        if not path.exists():Image.fromarray(pixels).save(path)
        outputs.append(dict(**row,output=path.name,output_sha256=sha(path),event=event,
            full_native_image_forward=True,image_input_image_output=True))
        atomic_json(folder/"outputs.json",outputs)
        status(root,"native_assessment",arm=arm,calls=len(outputs),total=256,print_event=len(outputs)%32==0)
        del logp,features
    if len(outputs)!=256:raise ValueError("all fixed autonomous cases required")
    atomic_json(folder/"report.json",dict(calls=256,outputs_sha256=sha(folder/"outputs.json"),
        final_checkpoint_sha256=sha(checkpoint),current_invocation_seconds=time.monotonic()-began,
        peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
        peak_reserved_vram_gib=torch.cuda.max_memory_reserved()/1024**3,
        assessment_function_opens_references=False,full_native_image_path=True,
        external_pretraining=True,tokenizer=False,ocr=False,language_vocabulary=False,
        image_answers_are_source_pixel_copies=True,free_generation_demonstrated=False,general_ilm_goal_complete=False))


def score_arm(root,arm):
    folder=root/arm/"assessment";rows=read(folder/"outputs.json")
    refs={r["id"]:r for r in read(PREVIOUS/"prepared/references.json")}
    cases=[]
    for row in rows:
        path=folder/row["output"]
        if sha(path)!=row["output_sha256"]:raise ValueError("raw output changed")
        output=array(path);source=array(PREVIOUS/"prepared"/row["input"])
        edge=np.diff(np.r_[False,(source==0).any(0),False].astype(np.int8))
        left,right=np.flatnonzero(edge==1),np.flatnonzero(edge==-1)
        start,length=row["event"]["start"],row["event"]["length"]
        if not 0<=start<start+length<=len(left):raise ValueError("invalid source boundary")
        reconstructed=np.pad(source[:,left[start]:right[start+length-1]],((0,0),(2,2)),constant_values=255)
        if not np.array_equal(output,reconstructed):raise ValueError("answer differs from declared original source pixels")
        key=row["id"].split("/question_erased")[0].split("/passage_erased")[0];targets=[]
        for target in refs[key]["targets"]:
            p=PREVIOUS/"prepared"/target["file"]
            if sha(p)!=target["sha256"]:raise ValueError("assessment reference changed")
            targets.append(array(p))
        prediction=Counter(bytes(c) for c in normalized(output).T)
        f1=[]
        for target in targets:
            truth=Counter(bytes(c) for c in normalized(target).T)
            f1.append(2*sum((prediction&truth).values())/(sum(prediction.values())+sum(truth.values())))
        cases.append(dict(**row,exact=any(np.array_equal(normalized(output),normalized(t)) for t in targets),
            unordered_column_f1=max(f1)))
    groups=defaultdict(list);pairs=defaultdict(list)
    for r in cases:
        groups[f"{r['split']}/{r['task']}/{r['condition']}/{r['language']}"].append(r)
        if r["split"]=="development" and r["task"]=="qa" and r["condition"]=="normal":pairs[r["pair"]].append(r)
    totals={k:dict(cases=len(v),exact=sum(r["exact"] for r in v),unordered_column_f1=sum(r["unordered_column_f1"] for r in v)/len(v)) for k,v in groups.items()}
    qa=totals["development/qa/normal/zh"]["exact"];erased=totals["development/qa/question_erased/zh"]["exact"]
    if any(totals[f"development/qa/{condition}/zh"]["cases"]!=64 for condition in ("normal","question_erased","passage_erased")):
        raise ValueError("complete 64-case question/erasure cohorts required")
    if sum(v["cases"] for k,v in totals.items() if k.startswith("development/copy/"))!=32:
        raise ValueError("all32 fixed new copies required")
    copies=sum(v["exact"] for k,v in totals.items() if k.startswith("development/copy/"))
    if len(cases)!=256 or len(pairs)!=32 or any(len(v)!=2 for v in pairs.values()):raise ValueError("complete registered cohort required")
    both=sum(all(r["exact"] for r in v) for v in pairs.values())
    gates=dict(copying=copies>=30,natural_answers=qa>=48,paired_answers=both>=20,question_dependence=(qa-erased)/64>=.25)
    result=dict(arm=arm,groups=totals,both_correct_pairs=both,gates=gates,component_qualified=all(gates.values()),
        cases=cases,normal_qa_exact=qa,new_copy_exact=copies,all_answers_independently_reconstructed=True,
        training=read(root/arm/"training_report.json"),assessment=read(folder/"report.json"),
        reused_public_development=True,free_generation_demonstrated=False,general_ilm_goal_complete=False)
    atomic_json(root/arm/"result.json",result);return result


def main(args):
    if args.run.exists()!=args.resume:raise ValueError("use --resume exactly for an existing run")
    args.run.mkdir(parents=True,exist_ok=args.resume)
    identity=dict(**committed_identity(SOURCES),preparation_sha256=sha(PREPARED_RUN/"prepared/report.json"),
        preflight_sha256=sha(PREPARED_RUN/"preflight/report.json"),initial_head_sha256=HEAD_SHA,
        development_inputs_sha256=sha(PREVIOUS/"prepared/inputs.json"),
        development_references_sha256=sha(PREVIOUS/"prepared/references.json"),
        seed=SEED,epochs=4,updates_per_arm=3257,examples=3257,batch_size=4)
    if args.resume:
        old=read(args.run/"identity.json")
        if {k:v for k,v in old.items() if k!="source_commit"}!={k:v for k,v in identity.items() if k!="source_commit"}:
            raise ValueError("registered code/data identity changed")
        identity=old
    else:atomic_json(args.run/"identity.json",identity)
    args.owns_run=True
    status(args.run,"checking_training_inventory")
    rows,labels=inventory();rng=random.Random(SEED);schedule=[]
    for _ in range(4):
        order=[r["id"] for r in rows];rng.shuffle(order);schedule.extend(order)
    if len(schedule)!=13028 or set(Counter(schedule).values())!={4}:raise ValueError("complete four-pass schedule required")
    path=args.run/"schedule.json"
    if path.exists() and read(path)!=schedule:raise ValueError("resumed schedule differs")
    if not path.exists():atomic_json(path,schedule)
    with gpu_slot():
        device=setup();torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        local=ROOT.parent/"LocalLLM/.local";status(args.run,"loading_initial_reader")
        began=time.monotonic()
        model=LearningPixelReader(checkpoint=PREVIOUS/"training/checkpoint.pt",checkpoint_sha256=HEAD_SHA,
            controls=CONTROLS,controls_sha256=CONTROL_SHA,ollama_root=local/"models/ollama",
            gguf_source=local/"tools/llama.cpp/gguf-py",device=device)
        initial=args.run/"initial_state.pt"
        if initial.exists():
            setup_report=read(args.run/"setup.json")
            if sha(initial)!=setup_report["initial_state_sha256"]:raise ValueError("initial tensors changed")
            model.restore_trained_state(torch.load(initial,map_location="cpu",weights_only=True))
        else:
            torch.save(model.trained_state(),initial)
            atomic_json(args.run/"setup.json",dict(initial_state_sha256=sha(initial),setup_seconds=time.monotonic()-began,
                full_runtime_parameters=sum(p.numel() for p in model.parameters()),source_pretraining_retained=True))
        features=cache_features(model,rows,args.run)
        if not (args.run/"matched_forward_check.json").exists():
            check_head_gradients(model,rows[0],features[rows[0]["id"]],labels[rows[0]["id"]],args.run)
        for arm in ("frozen","adapted"):train_arm(model,arm,rows,labels,features,schedule,args.run,identity)
        for arm in ("frozen","adapted"):assess(model,arm,args.run,identity)
        results={arm:score_arm(args.run,arm) for arm in ("frozen","adapted")}
        a,c=results["adapted"],results["frozen"]
        gains=dict(normal_qa=a["normal_qa_exact"]-c["normal_qa_exact"],
            both_correct_pairs=a["both_correct_pairs"]-c["both_correct_pairs"],new_copies=a["new_copy_exact"]-c["new_copy_exact"])
        useful=gains["normal_qa"]>=8 and gains["both_correct_pairs"]>=4 and gains["new_copies"]>=-2
        atomic_json(args.run/"result.json",dict(identity=identity,arms={k:{x:v[x] for x in ("groups","both_correct_pairs","gates","component_qualified")} for k,v in results.items()},
            adaptation_minus_control=gains,useful_improvement_over_control=useful,
            full_native_assessment_calls=512,final_checkpoints_only=True,
            feature_cache=read(args.run/"features/report.json"),
            free_generation_demonstrated=False,general_ilm_goal_complete=False))
    status(args.run,"completed",pid=None)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args();args.run=args.run.absolute()
    args.owns_run=False
    with (ROOT/"artifacts/contextual_pixel_comparison.lock").open("a") as controller_lock:
        # Reject a duplicate controller before it can mutate a live run's status.
        fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main(args)
        except BaseException as error:
            if args.owns_run:
                atomic_json(args.run/"failure.json",dict(error=repr(error),traceback=traceback.format_exc(),pid=None))
                status(args.run,"failed",pid=None,error=repr(error))
            raise
