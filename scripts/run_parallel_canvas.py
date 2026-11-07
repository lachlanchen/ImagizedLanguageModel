#!/usr/bin/env python3
"""Prepare, train, assess, verify and independently infer whole-canvas writing."""
import argparse
from collections import Counter
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.parallel_canvas import CanvasConfig, ParallelCanvasWriter, load_canvas_actor
from ilm.visual_lm.codec_block_writer import load_qualified_writing_codec, CODEC_SHA
from ilm.visual_lm.clean_image_actor import tensor_digest
from ilm.visual_lm.semantic_writing_data import ExactEpochSampler
from scripts.cache_semantic_images_v72 import ROOT, CODEC, CONTROLS, CONTROL_SHA, sha, atomic_json, committed_identity, pixels_sha

SEED = 2026110801
UPDATES = 2000
HELPER = ROOT / "artifacts/reader_content_v77_20261106"
COPY = ROOT / "data/general_instructions/copy_transfer_v75_20261106"
DATA = ROOT / "data/general_instructions/codec_language_audit_v68_20261105"
SOURCES = ("scripts/run_parallel_canvas.py", "ilm/visual_lm/parallel_canvas.py",
    "references/parallel_canvas_writer_protocol.md", "ilm/visual_lm/codec_block_writer.py",
    "ilm/visual_lm/continuous_glyph_codec.py", "ilm/visual_lm/semantic_writing_data.py",
    "ilm/visual_lm/grounded_image_actor.py", "ilm/visual_lm/grounded_image_weights.py",
    "ilm/visual_lm/semantic_image_memory.py", "ilm/visual_lm/image_context_interface.py",
    "ilm/visual_lm/codec_language_audit.py", "scripts/probe_native_corpus_fit_v80.py",
    "scripts/cache_semantic_images_v72.py", "scripts/train_clean_corpus_v75.py")


def read(path):
    return json.loads(Path(path).read_text())


def status(run, stage, **fields):
    value = dict(stage=stage, pid=os.getpid(), **fields)
    atomic_json(run/"status.json", value)
    print(json.dumps(value), flush=True)


def png(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)


def image_tensor(path):
    a = np.asarray(Image.open(path).convert("L")).copy()
    if a.ndim != 2 or min(a.shape) < 32 or any(n % 32 for n in a.shape):
        raise ValueError("physical image dimensions must be multiples of 32")
    return torch.from_numpy(a).float().div(127.5).sub(1)[None,None]


def prepare(run):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("CPU-only preparation required")
    from scripts.probe_native_corpus_fit_v80 import cases_and_data, prompt, targets
    torch.set_num_threads(4)
    if run.exists() and any(run.iterdir()): raise ValueError("preserve existing run; choose a new output")
    run.mkdir(parents=True); (run/"records").mkdir()
    identity = dict(**committed_identity(SOURCES), purpose="parallel_canvas_fit", seed=SEED,
        updates=UPDATES, config=CanvasConfig().__dict__, codec_sha256=CODEC_SHA,
        controls_sha256=CONTROL_SHA, cache_only_training=True, general_ilm_goal_complete=False)
    if sha(CONTROLS) != CONTROL_SHA: raise ValueError("source controls changed")
    cases, data, indices = cases_and_data()
    helper = read(HELPER/"generation_report.json")
    verified = read(HELPER/"verification.json")
    if sha(HELPER/"generation_report.json") != verified["generation_report_sha256"]:
        raise ValueError("source-helper cache has no verified identity")
    codec = load_qualified_writing_codec(CODEC, "cpu")
    white = codec.encode(torch.ones(1,1,32,32))[0].cpu()
    torch.save(white, run/"white.pt")
    rows, files = [], {"white.pt":sha(run/"white.pt")}
    for i,c in enumerate(cases):
        index = indices[c["source"]]
        physical = prompt(data,index,c)
        memory, codes, reference = targets(data,index,c)
        if len(codes)*2 > CanvasConfig().slots: raise ValueError("complete target exceeds declared canvas capacity")
        path = run/"records"/f"training_{i:02}.pt"
        torch.save(dict(memory=memory, codes=codes.reshape(-1,768), prompt=physical,
                        references=[reference], source=c["source"]), path)
        p = path.with_suffix(".png"); png(p,((physical[0,0]+1)*127.5).numpy())
        files[str(path.relative_to(run))]=sha(path); files[str(p.relative_to(run))]=sha(p)
        rows.append(dict(group="training", index=i, pair=i//2, source=c["source"],
            file=str(path.relative_to(run)), input=str(p.relative_to(run)),
            input_sha256=sha(p), memory_sha256=tensor_digest(memory[None]), tiles=len(codes)*2,
            original_cache_sha256=c["cache_record_sha256"], reference_sha256=c["target_sha256"]))
    prep = read(HELPER/"preparation.json"); cr = read(COPY/"report.json")
    for i,row in enumerate(r for r in prep["cases"] if r["group"]=="copy"):
        name=f"outputs/copy_{i:02}_normal.pt"; path=HELPER/name
        if sha(path)!=helper["raw_files_sha256"][name]: raise ValueError("copy memory source changed")
        saved=torch.load(path,map_location="cpu",weights_only=True); memory=saved["prefill_memory"][0]
        original=cr["cases"][i]
        for key in ("input_file","reference_file"):
            name=original[key]
            if sha(COPY/name)!=cr["files_sha256"][name]: raise ValueError("original copy image changed")
        physical=torch.load(COPY/original["input_file"],map_location="cpu",weights_only=True)
        reference=torch.from_numpy(np.asarray(Image.open(COPY/original["reference_file"]).convert("L")).copy())
        if pixels_sha(physical)!=row["physical_tensor_sha256"] or tensor_digest(memory[None])!=row["expected_memory_sha256"]:
            raise ValueError("copy memory does not belong to the actual prompt")
        path=run/"records"/f"copy_{i:02}.pt"; torch.save(dict(memory=memory,prompt=physical,references=[reference]),path)
        p=path.with_suffix(".png");png(p,((physical[0,0]+1)*127.5).numpy())
        files[str(path.relative_to(run))]=sha(path);files[str(p.relative_to(run))]=sha(p)
        rows.append(dict(group="copy",index=i,file=str(path.relative_to(run)),input=str(p.relative_to(run)),input_sha256=sha(p),
            memory_sha256=tensor_digest(memory[None]), original_source_sha256=sha(HELPER/f"outputs/copy_{i:02}_normal.pt")))
    if Counter(r["group"] for r in rows)!=Counter(training=32,copy=12): raise ValueError("all fixed cases required")
    identity.update(helper_report_sha256=sha(HELPER/"generation_report.json"), audit_report_sha256=sha(DATA/"report.json"))
    atomic_json(run/"identity.json",identity)
    atomic_json(run/"prepared.json",dict(rows=rows,files_sha256=files,identity_sha256=sha(run/"identity.json"),
        actual_source_model_calls=0, source_memories_reused=44, target_capacity_tiles=64))
    status(run,"prepared",training=32,new_copies=12)


def prepared(run):
    identity, report = read(run/"identity.json"), read(run/"prepared.json")
    if sha(run/"identity.json")!=report["identity_sha256"]: raise ValueError("prepared identity changed")
    for name,digest in identity["sources"].items():
        if sha(ROOT/name)!=digest: raise ValueError("producer source changed: "+name)
    for name,digest in report["files_sha256"].items():
        if sha(run/name)!=digest: raise ValueError("prepared physical data changed")
    return identity, report


def values(run, row):
    return torch.load(run/row["file"], map_location="cpu", weights_only=True)


def tree(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:tree(v) for k,v in value.items()}
    if isinstance(value,list): return [tree(v) for v in value]
    if isinstance(value,tuple): return tuple(tree(v) for v in value)
    return value


def initialize(white, device):
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model=ParallelCanvasWriter(white).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.01,foreach=False)
    return model,optimizer,ExactEpochSampler(32,SEED+1)


def update(model, optimizer, sampler, data, step, *, gradients=False):
    chosen=sampler.take(4);batch=[data[i] for i in chosen]
    memory=torch.stack([r["memory"] for r in batch]).to(model.device)
    target=model.white_code.expand(4,model.config.slots,768).clone()
    lengths=torch.tensor([len(r["codes"]) for r in batch],device=model.device)
    for i,r in enumerate(batch): target[i,:len(r["codes"])]=r["codes"].to(model.device)
    factor=(step+1)/100 if step<100 else .1+.9*.5*(1+math.cos(math.pi*(step-100)/(UPDATES-100)))
    optimizer.param_groups[0]["lr"]=3e-4*factor
    optimizer.zero_grad(set_to_none=True)
    loss,metrics=model.training_loss(memory,target,lengths)
    if not torch.isfinite(loss): raise FloatingPointError("nonfinite canvas loss")
    loss.backward()
    checks={}
    if gradients:
        for group in ("memory.","queries","layers.","output."):
            gs=[p.grad for n,p in model.named_parameters() if n.startswith(group)]
            checks[group]=bool(gs and all(g is not None and torch.isfinite(g).all() for g in gs) and any(g.abs().max()>0 for g in gs))
        if not all(checks.values()): raise AssertionError("missing/nonfinite gradient group: "+str(checks))
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    return dict(update=step+1,indices=chosen,loss=float(loss.detach()),gradient_norm=float(norm),
                **{k:float(v) for k,v in metrics.items()},gradient_checks=checks)


def snapshot(model,optimizer,sampler,step,identity,elapsed):
    return dict(model=tree(model.state_dict()),optimizer=tree(optimizer.state_dict()),sampler=tree(sampler.state_dict()),
        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),update=step,identity=identity,elapsed_seconds=elapsed)


def restore(saved,model,optimizer,sampler):
    model.load_state_dict(saved["model"],strict=True);optimizer.load_state_dict(saved["optimizer"])
    sampler.load_state_dict(saved["sampler"]);torch.set_rng_state(saved["cpu_rng"]);torch.cuda.set_rng_state(saved["cuda_rng"])


def save_output(folder, image, coordinates, metadata):
    folder.mkdir(parents=True,exist_ok=False)
    png(folder/"answer.png",image.numpy());torch.save(coordinates,folder/"coordinates.pt")
    atomic_json(folder/"header.json",dict(**metadata, answer_sha256=sha(folder/"answer.png"),coordinates_sha256=sha(folder/"coordinates.pt")))


@torch.no_grad()
def cached_outputs(run,model,codec,rows,stage):
    for row in rows:
        p=run/stage/f"{row['group']}_{row['index']:02}"
        if p.exists():
            h=read(p/"header.json")
            if h["answer_sha256"]!=sha(p/"answer.png") or h["coordinates_sha256"]!=sha(p/"coordinates.pt"): raise ValueError("retained canvas changed")
            continue
        value=values(run,row)
        image,coordinates=model.generate(value["memory"][None].to(model.device),codec)
        save_output(p,image[0],coordinates[0],dict(group=row["group"],index=row["index"],memory_sha256=row["memory_sha256"],cached_prompt_only=True))


def train(run):
    from scripts.train_clean_corpus_v75 import setup
    identity,prep=prepared(run);rows=[r for r in prep["rows"] if r["group"]=="training"]
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);device=setup()
        white=torch.load(run/"white.pt",map_location="cpu",weights_only=True)
        codec=load_qualified_writing_codec(CODEC,device); data=[values(run,r) for r in rows]
        model,optimizer,sampler=initialize(white,device)
        if not (run/"preflight.json").exists():
            for step in range(3): checks=update(model,optimizer,sampler,data,step,gradients=step==2)
            saved=snapshot(model,optimizer,sampler,3,identity,0.)
            expected=update(model,optimizer,sampler,data,3);state=tree(model.state_dict())
            restore(saved,model,optimizer,sampler);actual=update(model,optimizer,sampler,data,3)
            exact=actual==expected and all(torch.equal(v,model.state_dict()[n].cpu()) for n,v in state.items())
            if not exact: raise AssertionError("actual optimizer/sampler resume differs")
            atomic_json(run/"preflight.json",dict(passed=True,exact_resume=True,gradient_checks=checks["gradient_checks"],
                distinct_update_positions=4, optimizer_invocations=5, reset_before_production=True,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3))
            del saved,state;del model,optimizer,sampler;gc.collect();torch.cuda.empty_cache()
            model,optimizer,sampler=initialize(white,device)
        latest=run/"checkpoint_latest.pt";start=0;elapsed=0.
        if latest.exists():
            saved=torch.load(latest,map_location="cpu",weights_only=True)
            if saved["identity"]!=identity: raise ValueError("checkpoint identity changed")
            restore(saved,model,optimizer,sampler);start=saved["update"];elapsed=saved["elapsed_seconds"];del saved
            journal_path=run/"updates.jsonl"
            records=journal_path.read_text().splitlines()
            if [json.loads(s)["update"] for s in records[:start]]!=list(range(1,start+1)):
                raise ValueError("checkpointed journal prefix is incomplete")
            if len(records)>start:
                events=read(run/"resume_events.json") if (run/"resume_events.json").exists() else []
                archived=run/f"interrupted_updates_{len(events):02}.jsonl"
                if archived.exists(): raise ValueError("preserve the interrupted journal")
                archived.write_text("\n".join(records[start:])+"\n")
                events.append(dict(checkpoint_update=start,discarded_updates=len(records)-start,
                    archived=archived.name,sha256=sha(archived),discarded_stage_seconds_unmeasured=True))
                atomic_json(run/"resume_events.json",events)
                journal_path.write_text("\n".join(records[:start])+"\n")
        else: cached_outputs(run,model,codec,rows,"before")
        torch.cuda.reset_peak_memory_stats();begun=time.monotonic()
        with (run/"updates.jsonl").open("a") as journal:
            for step in range(start,UPDATES):
                result=update(model,optimizer,sampler,data,step)
                journal.write(json.dumps(result)+"\n");journal.flush()
                if (step+1)%50==0: status(run,"training",**result)
                if (step+1)%500==0 or step+1==UPDATES:
                    tmp=run/"checkpoint.partial";torch.save(snapshot(model,optimizer,sampler,step+1,identity,elapsed+time.monotonic()-begun),tmp)
                    if latest.exists(): latest.replace(run/"checkpoint_previous.pt")
                    tmp.replace(latest)
        active=elapsed+time.monotonic()-begun
        bundle=run/"model_final.pt"
        if not bundle.exists():
            torch.save(dict(purpose="parallel_canvas_fit",training_complete=True,updates=UPDATES,
                config=model.configuration(),model=tree(model.state_dict()),identity=identity),bundle)
        cached_outputs(run,model,codec,prep["rows"],"final")
        atomic_json(run/"training.json",dict(identity_sha256=sha(run/"identity.json"),updates=UPDATES,
            optimized_parameters=sum(p.numel() for p in model.parameters()),active_training_stage_seconds=active,
            training_time_includes_checkpoint_io=True, discarded_invocation_time_complete=not (run/"resume_events.json").exists(),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,bundle_sha256=sha(bundle),
            checkpoint_sha256=sha(latest),external_reader_pretraining_excluded=True,source_encoding_reused=True,
            no_source_body_resident_during_head_training=True, general_ilm_goal_complete=False))
        status(run,"training_finished",updates=UPDATES)


def actor(run,device):
    local=ROOT.parent/"LocalLLM/.local"
    return load_canvas_actor(run/"model_final.pt",controls=CONTROLS,codec_path=CODEC,
        ollama_root=local/"models/ollama",gguf_source=local/"tools/llama.cpp/gguf-py",device=device)


def evaluate(run):
    from scripts.train_clean_corpus_v75 import setup
    from ilm.visual_lm.codec_language_audit import CodecLanguageAuditData
    identity,prep=prepared(run);training=read(run/"training.json")
    if sha(run/"model_final.pt")!=training["bundle_sha256"]: raise ValueError("completed bundle changed")
    data=CodecLanguageAuditData(DATA)
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);device=setup();started=time.monotonic()
        model,provenance=actor(run,device);setup_seconds=time.monotonic()-started
        replays=[]
        for group in ("training","copy"):
            for row in [r for r in prep["rows"] if r["group"]==group][:2]:
                value=values(run,row);result=model.generate(value["prompt"])
                original=np.asarray(Image.open(run/"final"/f"{group}_{row['index']:02}"/"answer.png").convert("L"))
                same=np.array_equal(result["image"].numpy(),original)
                memory=tensor_digest(result["memory"])==row["memory_sha256"]
                if not same or not memory: raise AssertionError("native prompt pixel/memory replay differs")
                replays.append(dict(group=group,index=row["index"],pixels_exact=same,memory_exact=memory))
        for i,row in enumerate(data.cases):
            if row["source"]!="cmrc": continue
            for route in ("normal","blank"):
                destination=run/"development"/f"{i:03}_{route}"
                if destination.exists():
                    h=read(destination/"header.json")
                    if h["answer_sha256"]!=sha(destination/"answer.png") or h["bundle_sha256"]!=training["bundle_sha256"]: raise ValueError("retained assessment differs")
                    continue
                physical=data.prompt(i,blank=route=="blank")
                result=model.generate(physical)
                save_output(destination,result["image"],result["coordinates"],dict(index=i,id=row["id"],pair=i//2,
                    route=route,prompt_sha256=pixels_sha(physical),memory_sha256=tensor_digest(result["memory"]),
                    bundle_sha256=training["bundle_sha256"],actual_prompt_only_image_call=True))
                status(run,"development",index=i,route=route)
        atomic_json(run/"evaluation.json",dict(provenance=provenance,native_replays=replays,
            native_development_calls=256,setup_seconds=setup_seconds,invocation_seconds=time.monotonic()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            timing_scope="current invocation including setup and only newly computed/replay calls; not complete on resume",
            general_ilm_goal_complete=False))
        status(run,"evaluated_awaiting_verification")


def scores(prediction,references):
    """NumPy scorer, full common-origin canvas, with no foreground removed."""
    output=[]
    for reference in references:
        width=max(prediction.shape[1],reference.shape[1])
        p=np.pad(prediction<128,((0,0),(0,width-prediction.shape[1])))
        t=np.pad(reference<128,((0,0),(0,width-reference.shape[1])))
        denom=int(p.sum()+t.sum());intersection=int(np.logical_and(p,t).sum())
        output.append(dict(exact=bool(np.array_equal(p,t)),foreground_f1=2*intersection/denom if denom else 1.,
                           reference_exceeds_capacity=reference.shape[1]>prediction.shape[1]))
    return dict(exact=any(r["exact"] for r in output),foreground_f1=max(r["foreground_f1"] for r in output),
                blank=not bool((prediction<128).any()),references=output)


def verify(run):
    from ilm.visual_lm.codec_language_audit import CodecLanguageAuditData
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized(): raise ValueError("CPU verification required")
    torch.set_num_threads(4);identity,prep=prepared(run);codec=load_qualified_writing_codec(CODEC,"cpu")
    training=read(run/"training.json");evaluation=read(run/"evaluation.json")
    if sha(run/"model_final.pt")!=training["bundle_sha256"]: raise ValueError("bundle changed")
    steps=[json.loads(s) for s in (run/"updates.jsonl").read_text().splitlines()]
    if [r["update"] for r in steps]!=list(range(1,UPDATES+1)): raise ValueError("missing/replayed optimizer updates")
    draws=Counter(i for r in steps for i in r["indices"])
    if draws!=Counter({i:250 for i in range(32)}): raise ValueError("training draws differ")
    saved=torch.load(run/"checkpoint_latest.pt",map_location="cpu",weights_only=True,mmap=True)
    bundle=torch.load(run/"model_final.pt",map_location="cpu",weights_only=True,mmap=True)
    if saved["update"]!=UPDATES or saved["sampler"]["draws"]!=8000: raise ValueError("final optimizer/cursor differs")
    for state in saved["optimizer"]["state"].values():
        if float(state["step"])!=UPDATES or not all(torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor)):
            raise ValueError("optimizer not complete/finite")
    for n,v in saved["model"].items():
        if not torch.equal(v,bundle["model"][n]): raise ValueError("bundle/checkpoint differs")
    results=[];files={}
    def score_folder(folder,references,metadata):
        h=read(folder/"header.json")
        if sha(folder/"answer.png")!=h["answer_sha256"] or sha(folder/"coordinates.pt")!=h["coordinates_sha256"]: raise ValueError("raw output changed")
        a=np.asarray(Image.open(folder/"answer.png").convert("L"))
        z=torch.load(folder/"coordinates.pt",map_location="cpu",weights_only=True)
        if z.shape!=(64,768) or a.shape!=(32,2048): raise ValueError("complete fixed canvas required")
        with torch.no_grad(): tiles=(codec.decode(z)>=0).to(torch.uint8).mul(255).reshape(64,32,32)
        reconstructed=tiles.permute(1,0,2).reshape(32,2048).numpy()
        if not np.array_equal(a,reconstructed): raise ValueError("retained coordinates do not decode to raw pixels")
        for p in folder.iterdir(): files[str(p.relative_to(run))]=sha(p)
        results.append(dict(**metadata,output=str((folder/"answer.png").relative_to(run)),**scores(a,references)))
    for row in prep["rows"]:
        value=values(run,row);refs=[r.numpy() for r in value["references"]]
        for stage in (["before","final"] if row["group"]=="training" else ["final"]):
            score_folder(run/stage/f"{row['group']}_{row['index']:02}",refs,dict(group=row["group"],index=row["index"],stage=stage))
    data=CodecLanguageAuditData(DATA)
    for i,row in enumerate(data.cases):
        if row["source"]!="cmrc": continue
        for route in ("normal","blank"):
            score_folder(run/"development"/f"{i:03}_{route}",[r.numpy() for r in data.references(i)],
                dict(group="development",index=i,id=row["id"],pair=i//2,stage=route))
    def aggregate(group,stage):
        rows=[r for r in results if r["group"]==group and r["stage"]==stage]
        return dict(cases=len(rows),exact=sum(r["exact"] for r in rows),foreground_f1=float(np.mean([r["foreground_f1"] for r in rows])))
    groups={f"{g}/{s}":aggregate(g,s) for g,s in (("training","before"),("training","final"),("copy","final"),("development","normal"),("development","blank"))}
    fit=[r for r in results if r["group"]=="training" and r["stage"]=="final"]
    dev=[r for r in results if r["group"]=="development" and r["stage"]=="normal"]
    both=lambda rows:sum(all(r["exact"] for r in rows[j:j+2]) for j in range(0,len(rows),2))
    fit_pairs,dev_pairs=both(fit),both(dev)
    gates=dict(fit_exact=groups["training/final"]["exact"]>=24,fit_f1=groups["training/final"]["foreground_f1"]>=.98,fit_pairs=fit_pairs>=12)
    gain=groups["development/normal"]["foreground_f1"]-groups["development/blank"]["foreground_f1"]
    reader=dict(exact=groups["development/normal"]["exact"]>=64,pairs=dev_pairs>=23,question_gain=gain>=.1)
    report=dict(identity=identity,groups=groups,training_both_correct_pairs=fit_pairs,development_both_correct_pairs=dev_pairs,
        fit_gates=gates,fit_pass=all(gates.values()),reader_screen=reader,reader_screen_pass=all(reader.values()),
        additional_training_optimizer_invocations=sum(r["discarded_updates"] for r in read(run/"resume_events.json")) if (run/"resume_events.json").exists() else 0,
        question_erasure_f1_gain=gain,results=results,files_sha256=files,optimizer_updates_verified=UPDATES,
        each_training_case_draws=250,coordinate_images_replayed=len(results),evaluation=evaluation,
        reused_public_development=True,fixed_canvas_without_learned_termination=True,
        language_quality_claim=False,general_ilm_goal_complete=False)
    atomic_json(run/"verification.json",report)
    status(run,"verified",groups=groups,fit_pass=report["fit_pass"],reader_screen_pass=report["reader_screen_pass"])


def infer(run, images, out):
    from scripts.train_clean_corpus_v75 import setup
    if not images or out is None: raise ValueError("--images PNG... and --output fresh directory required")
    forbidden=[ROOT/"data",ROOT/"artifacts/clean_corpus_v75_20261106/cache",run/"records"]
    import sys
    opened=[]
    def audit(event,args):
        if event=="open" and isinstance(args[0],(str,bytes)):
            path=Path(os.fsdecode(args[0])).absolute()
            if any(path.is_relative_to(root) for root in forbidden): raise RuntimeError("training/reference path entered standalone image inference")
            opened.append(str(path))
    sys.addaudithook(audit)
    out.mkdir(parents=True,exist_ok=False)
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);device=setup();started=time.monotonic()
        model,provenance=actor(run,device);setup_seconds=time.monotonic()-started
        for i,path in enumerate(images):
            physical=image_tensor(path);result=model.generate(physical)
            save_output(out/f"{i:02}",result["image"],result["coordinates"],dict(input_sha256=sha(path),memory_sha256=tensor_digest(result["memory"])))
        atomic_json(out/"report.json",dict(provenance=provenance,inputs=[dict(file=str(p),sha256=sha(p)) for p in images],
            setup_seconds=setup_seconds,total_seconds=time.monotonic()-started,calls=len(images),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,python_open_audit=True,
            forbidden_training_paths_opened=False,observed_open_paths=sorted(set(opened)),general_ilm_goal_complete=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("prepare","train","evaluate","verify","infer"),required=True)
    p.add_argument("--run",type=Path,default=ROOT/"artifacts/parallel_canvas_v1")
    p.add_argument("--images",type=Path,nargs="+");p.add_argument("--output",type=Path)
    a=p.parse_args()
    if a.stage=="infer": infer(a.run,a.images,a.output)
    else: {"prepare":prepare,"train":train,"evaluate":evaluate,"verify":verify}[a.stage](a.run)
