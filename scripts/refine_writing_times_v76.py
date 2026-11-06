#!/usr/bin/env python3
"""Run a matched head-only time-law intervention on the failed V75 writer."""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from ilm.visual_lm.clean_writing_flow import CleanSemanticWritingModel, CleanWritingConfig
from ilm.visual_lm.codec_block_writer import CodecBlockWriter, load_qualified_writing_codec
from ilm.visual_lm.codec_language_audit import reference_scores
from ilm.visual_lm.complete_writing_data import CompleteWritingImageData
from ilm.visual_lm.semantic_writing_data import SemanticWritingData, ExactEpochSampler
from scripts.cache_semantic_images_v72 import ROOT, TRAIN, TRAIN_SHA, CODEC, sha, atomic_json, committed_identity
from scripts.probe_corpus_fit_v75 import fixed_cases, raster
from scripts.train_clean_corpus_v75 import setup
from scripts.verify_codec_language_v68 import physical_scores

BASE=ROOT/"artifacts/clean_corpus_v75_20261106"
FIT=ROOT/"artifacts/corpus_fit_v75_20261106"
BOUNDARY=ROOT/"artifacts/corpus_boundary_v75_20261106"
BASE_SHA="d0cb3aa7923db450dc904269c753e0a4157e704a6da1be487507b168d7612249"
SOURCES=("scripts/refine_writing_times_v76.py","references/writing_time_coverage_v76_protocol.md",
    "ilm/visual_lm/clean_writing_flow.py","ilm/visual_lm/full_rank_writing_flow.py",
    "ilm/visual_lm/semantic_writing_model.py","ilm/visual_lm/semantic_writing_data.py",
    "ilm/visual_lm/complete_writing_data.py","ilm/visual_lm/codec_block_writer.py",
    "ilm/visual_lm/codec_language_audit.py",
    "scripts/probe_corpus_fit_v75.py","scripts/train_clean_corpus_v75.py",
    "scripts/verify_codec_language_v68.py")


def digest_tensors(named):
    digest=hashlib.sha256()
    for name,value in sorted(named):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape),value.dtype)).encode())
        digest.update(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def frozen_digest(model):
    return digest_tensors((n,p) for n,p in model.state_dict().items() if not n.startswith("flow."))


def construct(device):
    value=torch.load(BASE/"training/checkpoint_latest.pt",map_location="cpu",weights_only=True,mmap=True)
    model=CleanSemanticWritingModel(CleanWritingConfig(**value["model_config"]))
    model.load_state_dict(value["model"],strict=True)
    model=model.to(device).eval().requires_grad_(False)
    model.flow.requires_grad_(True)
    optimizer=torch.optim.AdamW(model.flow.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=.01,foreach=False)
    sampler=ExactEpochSampler(12976,seed=20267601)
    generator=torch.Generator(device=device).manual_seed(20267602)
    return model,optimizer,sampler,generator


def update(model,optimizer,sampler,generator,data,step,law,device):
    indices=sampler.take(16)
    memory,codes,lengths=data.batch(indices,device)
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        states=model.conditions(memory,codes,lengths)
    batch,position,weights=[],[],[]
    for b,n in enumerate(lengths.tolist()):
        selected=torch.randperm(n,device=device,generator=generator)[:16]
        batch.append(torch.full_like(selected,b))
        position.append(selected)
        weights.append(torch.full((len(selected),),1/(16*len(selected)),device=device))
    batch,position=torch.cat(batch),torch.cat(position)
    target=codes[batch,position].float().flatten(1)
    noise=torch.randn(target.shape,device=device,generator=generator)
    variate=torch.randn((len(target),),device=device,generator=generator)
    t=((-0.8+0.8*variate).sigmoid() if law=="logit_normal"
        else (.5*(1+torch.erf(variate/math.sqrt(2)))).clamp(1e-6,1-1e-6))
    factor=min((step+1)/100,1.) if step<100 else .1+.9*.5*(1+math.cos(math.pi*(step-100)/3900))
    optimizer.param_groups[0]["lr"]=1e-4*factor
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        predicted=model.flow((1-t[:,None])*noise+t[:,None]*target,t,states[batch,position])
        error=(predicted-target).square().mean(-1)
        loss=(error/(1-t).clamp_min(.05).square()*torch.cat(weights)).sum()
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite writing loss")
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.flow.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    row=dict(update=step+1,loss=float(loss.detach()),clean_mse=float(error.mean().detach()),
        gradient_norm=float(norm),sampled_blocks=len(target),draws=sampler.draws,
        high_signal_count=int((t>=31/32).sum()),lr=optimizer.param_groups[0]["lr"])
    if step==0 or (step+1)%500==0:
        row["common_draw_sha256"]=digest_tensors([("indices",torch.tensor(indices)),
            ("batch",batch),("position",position),("noise",noise),("time_variate",variate)])
    return row


def save(path,model,optimizer,sampler,generator,step,identity,elapsed):
    partial=path.with_suffix(".partial")
    torch.save(dict(flow={n:p.detach().cpu() for n,p in model.flow.state_dict().items()},
        optimizer=optimizer.state_dict(),sampler=sampler.state_dict(),generator=generator.get_state(),
        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),update=step,
        identity=identity,active_elapsed_seconds=elapsed),partial)
    partial.replace(path)


def restore(path,model,optimizer,sampler,generator,identity):
    value=torch.load(path,map_location="cpu",weights_only=True)
    if value["identity"]!=identity:
        raise ValueError("resume identity differs")
    model.flow.load_state_dict(value["flow"],strict=True)
    optimizer.load_state_dict(value["optimizer"])
    sampler.load_state_dict(value["sampler"])
    generator.set_state(value["generator"])
    torch.set_rng_state(value["cpu_rng"])
    torch.cuda.set_rng_state(value["cuda_rng"])
    return value


def preflight(out,data,device,identity):
    out.mkdir()
    model,optimizer,sampler,generator=construct(device)
    before=frozen_digest(model)
    for step in range(3):
        update(model,optimizer,sampler,generator,data,step,"uniform",device)
    path=out/"resume_check.pt"
    save(path,model,optimizer,sampler,generator,3,identity,0.)
    expected=update(model,optimizer,sampler,generator,data,3,"uniform",device)
    expected_weights=digest_tensors(model.flow.state_dict().items())
    restore(path,model,optimizer,sampler,generator,identity)
    actual=update(model,optimizer,sampler,generator,data,3,"uniform",device)
    passed=expected==actual and expected_weights==digest_tensors(model.flow.state_dict().items())
    if not passed or before!=frozen_digest(model):
        raise AssertionError("exact resume or frozen-body preservation failed")
    atomic_json(out/"report.json",dict(identity=identity,updates_exercised=4,
        exact_optimizer_rng_cursor_resume=True,frozen_body_unchanged=True,weights_discarded=True,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        language_quality_claim=False))


def train_arm(out,data,device,identity):
    out.mkdir(exist_ok=True)
    completed=out/"training_report.json"
    if completed.exists():
        report=json.loads(completed.read_text())
        if report["identity"]!=identity or sha(out/"head_latest.pt")!=report["head_sha256"]:
            raise ValueError("completed head changed")
        return
    model,optimizer,sampler,generator=construct(device)
    before=frozen_digest(model)
    step,elapsed=0,0.
    path=out/"head_latest.pt"
    if path.exists():
        saved=restore(path,model,optimizer,sampler,generator,identity)
        step,elapsed=saved["update"],saved["active_elapsed_seconds"]
    log_path=out/"metrics.jsonl"
    if log_path.exists():
        retained=[json.loads(line) for line in log_path.read_text().splitlines()
            if line.strip() and json.loads(line)["update"]<=step]
        if [row["update"] for row in retained]!=list(range(1,step+1)):
            raise ValueError("committed optimizer cursor has a discontinuous update log")
        log_path.write_text("".join(json.dumps(row)+"\n" for row in retained))
    begun=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with log_path.open("a") as log:
        while step<4000:
            row=update(model,optimizer,sampler,generator,data,step,identity["law"],device)
            step+=1
            # Log every update so the full realized time-tail count is available.
            log.write(json.dumps(row)+"\n")
            if step==1 or step%100==0:
                log.flush()
                atomic_json(out/"status.json",dict(stage="training",pid=os.getpid(),**row))
                print(json.dumps(dict(law=identity["law"],**row)),flush=True)
            if step%500==0:
                save(path,model,optimizer,sampler,generator,step,identity,elapsed+time.perf_counter()-begun)
    after=frozen_digest(model)
    if after!=before:
        raise AssertionError("frozen condition/stop body changed")
    atomic_json(completed,dict(identity=identity,updates=step,draws=sampler.draws,
        active_elapsed_seconds=elapsed+time.perf_counter()-begun,
        peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen_body_sha256=before,frozen_body_unchanged=True,head_sha256=sha(path),
        language_quality_claim=False))


@torch.inference_mode()
def evaluate_arm(out,data,device,identity):
    report_path=out/"evaluation.json"
    if report_path.exists():
        return
    training=json.loads((out/"training_report.json").read_text())
    if training["identity"]!=identity or sha(out/"head_latest.pt")!=training["head_sha256"]:
        raise ValueError("completed matched training required")
    model,optimizer,sampler,generator=construct(device)
    del optimizer,sampler,generator
    saved=torch.load(out/"head_latest.pt",map_location="cpu",weights_only=True)
    model.flow.load_state_dict(saved["flow"],strict=True)
    del saved
    model.requires_grad_(False).eval()
    if frozen_digest(model)!=training["frozen_body_sha256"]:
        raise ValueError("frozen inference body differs")
    writer=CodecBlockWriter(load_qualified_writing_codec(CODEC,device))
    source=CompleteWritingImageData(TRAIN,purpose="train",expected_report_sha256=TRAIN_SHA)
    source_index={name:i for i,name in enumerate(source.names)}
    cache_index={name:i for i,name in enumerate(data.names)}
    values=torch.load(FIT/"conditioning.pt",map_location="cpu",weights_only=True)
    predictions=out/"predictions"
    predictions.mkdir(exist_ok=True)
    rows,conditions=[],[]
    for i,case in enumerate(fixed_cases()):
        folder=predictions/f"{i:02}"
        folder.mkdir(exist_ok=True)
        memory,codes=data[cache_index[case["cache_record"]]]
        memory=memory[None].to(device)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            answer=model.generate(memory,writer,generator=torch.Generator(device=device).manual_seed(20267420+i//2))
            condition=model.conditions(memory,torch.empty(1,0,2,768,device=device))[:,-1]
        torch.save(answer,folder/"answer.pt")
        pixels=raster(answer["images"])
        if len(answer["images"]):
            Image.fromarray(pixels.numpy()).save(folder/"answer.png")
        target=((source.target(source_index[case["source"]])[0,0,:,:-128]+1)*127.5).to(torch.uint8)
        Image.fromarray(target.numpy()).save(folder/"reference.png")
        score=reference_scores(pixels,[target],stopped=answer["stopped"])
        first=reference_scores(raster(answer["images"][:1]),[target[:,:64]],stopped=True)
        rows.append(dict(position=i,free=score,first_block_f1=first["foreground_f1"]))
        conditions.append(condition)
        atomic_json(folder/"header.json",dict(identity=identity,position=i,source=case["source"],
            memory_sha256=digest_tensors([("memory",memory)]),target_supplied_to_free_generation=False,
            files_sha256={p.name:sha(p) for p in folder.iterdir() if p.name!="header.json"}))
    near_clean={"times":[.96875,1.],"predictions":[]}
    target,noise=values["target"].to(device),values["noise"].to(device)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        conditions=torch.cat(conditions)
        for t in near_clean["times"]:
            near_clean["predictions"].append(model.flow((1-t)*noise+t*target,
                torch.full((32,),t,device=device),conditions).cpu())
    near_clean["predictions"]=torch.stack(near_clean["predictions"])
    torch.save(near_clean,out/"near_clean.pt")
    atomic_json(report_path,dict(identity=identity,training_report_sha256=sha(out/"training_report.json"),
        head_sha256=training["head_sha256"],near_clean_sha256=sha(out/"near_clean.pt"),rows=rows,
        cached_prompt_feature_calls=32,actual_inherited_reader_calls=0,training_cases_only=True,
        language_quality_claim=False,general_ilm_goal_complete=False))


def verify(out,identity):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized():
        raise ValueError("independent CPU scoring required")
    torch.set_num_threads(4)
    original=CompleteWritingImageData(TRAIN,purpose="train",expected_report_sha256=TRAIN_SHA)
    cache=SemanticWritingData(BASE/"cache")
    cache_index={name:i for i,name in enumerate(cache.names)}
    index={name:i for i,name in enumerate(original.names)}
    writer=CodecBlockWriter(load_qualified_writing_codec(CODEC,torch.device("cpu")))
    base=torch.load(BASE/"training/checkpoint_latest.pt",map_location="cpu",weights_only=True,mmap=True)
    body=digest_tensors((n,p) for n,p in base["model"].items() if not n.startswith("flow."))
    summaries,common={},{}
    for law in ("logit_normal","uniform"):
        arm=out/law
        training=json.loads((arm/"training_report.json").read_text())
        evaluation=json.loads((arm/"evaluation.json").read_text())
        saved=torch.load(arm/"head_latest.pt",map_location="cpu",weights_only=True,mmap=True)
        if (training["identity"]!=dict(identity,law=law) or saved["identity"]!=training["identity"]
                or saved["update"]!=4000 or saved["sampler"]["draws"]!=64000
                or training["frozen_body_sha256"]!=body
                or sha(arm/"head_latest.pt")!=training["head_sha256"]
                or sha(arm/"training_report.json")!=evaluation["training_report_sha256"]
                or sha(arm/"near_clean.pt")!=evaluation["near_clean_sha256"]):
            raise ValueError("final training/evaluation identity changed")
        rows=[]
        for i,case in enumerate(fixed_cases()):
            folder=arm/"predictions"/f"{i:02}"
            header=json.loads((folder/"header.json").read_text())
            if header["identity"]!=training["identity"] or header["source"]!=case["source"]:
                raise ValueError("prediction identity changed")
            memory=cache[cache_index[case["cache_record"]]][0][None]
            if header["memory_sha256"]!=digest_tensors([("memory",memory)]):
                raise ValueError("prompt-derived inference memory changed")
            for name,digest in header["files_sha256"].items():
                if sha(folder/name)!=digest:
                    raise ValueError("prediction changed")
            target=((original.target(index[case["source"]])[0,0,:,:-128]+1)*127.5).to(torch.uint8)
            if not np.array_equal(np.array(Image.open(folder/"reference.png")),target.numpy()):
                raise ValueError("reference differs from original source")
            answer=torch.load(folder/"answer.pt",map_location="cpu",weights_only=True)
            pixels=raster(answer["images"])
            if len(answer["images"]) and not np.array_equal(np.array(Image.open(folder/"answer.png")),pixels.numpy()):
                raise ValueError("answer PNG differs from tensor")
            score=physical_scores(pixels.numpy()==0,[target],answer["stopped"])
            if score!=evaluation["rows"][i]["free"]:
                raise ValueError("free physical score differs")
            first=physical_scores(raster(answer["images"][:1]).numpy()==0,[target[:,:64]],True)
            if first["foreground_f1"]!=evaluation["rows"][i]["first_block_f1"]:
                raise ValueError("first-block score differs")
            rows.append(score)
        near=torch.load(arm/"near_clean.pt",map_location="cpu",weights_only=True)
        if near["times"]!=[.96875,1.] or near["predictions"].shape!=(2,32,1536):
            raise ValueError("near-clean diagnostic geometry differs")
        high=[]
        for predicted in near["predictions"]:
            with torch.inference_mode():
                normalized=F.layer_norm(predicted.reshape(-1,768),(768,),
                    eps=writer.codec.config.layer_norm_epsilon).reshape(-1,2,768)
                decoded=torch.where(writer.predict(normalized)<0,0,255).to(torch.uint8)[:,0]
            scores=[]
            for i,case in enumerate(fixed_cases()):
                target=((original.target(index[case["source"]])[0,0,:,:64]+1)*127.5).to(torch.uint8)
                scores.append(physical_scores(decoded[i].numpy()==0,[target],True))
            high.append(dict(exact=sum(r["exact"] for r in scores),
                foreground_f1=float(np.mean([r["foreground_f1"] for r in scores]))))
        log=[json.loads(line) for line in (arm/"metrics.jsonl").read_text().splitlines()]
        if [row["update"] for row in log]!=list(range(1,4001)):
            raise ValueError("complete unique production update log required")
        common[law]={row["update"]:row["common_draw_sha256"] for row in log if "common_draw_sha256" in row}
        summaries[law]=dict(free_exact=sum(r["exact"] for r in rows),
            free_f1=float(np.mean([r["foreground_f1"] for r in rows])),
            stopped=sum(r["stopped"] for r in rows),near_clean=high,
            realized_high_signal_training_blocks=sum(r["high_signal_count"] for r in log),
            sampled_training_blocks=sum(r["sampled_blocks"] for r in log),
            training_seconds=training["active_elapsed_seconds"],
            trainable_parameters=training["trainable_parameters"])
    if common["logit_normal"]!=common["uniform"] or len(common["uniform"])!=9:
        raise ValueError("matched common random draws differ")
    uniform,control=summaries["uniform"],summaries["logit_normal"]
    gates=dict(high_signal_f1=uniform["near_clean"][0]["foreground_f1"]>=.999,
        free_f1=uniform["free_f1"]>=.8,free_exact=uniform["free_exact"]>=16,
        free_gain=uniform["free_f1"]-control["free_f1"]>=.05)
    receipt=dict(identity=identity,summaries=summaries,common_draw_checks=common["uniform"],
        frozen_body_sha256=body,matched_common_draws=True,all_64_streams_independently_scored=True,
        near_clean_predictions_redecoded_on_cpu=True,gates=gates,training_fit_repair_pass=all(gates.values()),
        actual_inherited_reader_calls=0,full_neural_sampler_replayed=False,
        source_report_sha256={law:sha(out/law/"evaluation.json") for law in summaries},
        language_quality_claim=False,general_ilm_goal_complete=False)
    atomic_json(out/"verification.json",receipt)
    print(json.dumps(receipt,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"artifacts/writing_times_v76_20261106")
    parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    if sha(BASE/"training/checkpoint_latest.pt")!=BASE_SHA:
        raise ValueError("completed V75 base changed")
    fit=json.loads((FIT/"verification.json").read_text())
    if fit["report_sha256"]!=sha(FIT/"report.json"):
        raise ValueError("verified fit diagnosis changed")
    identity=dict(**committed_identity(SOURCES),base_checkpoint_sha256=BASE_SHA,
        base_training_report_sha256=sha(BASE/"training/training_report.json"),
        source_fit_verification_sha256=sha(FIT/"verification.json"),
        source_boundary_report_sha256=sha(BOUNDARY/"report.json"),planned_updates=4000,batch=16,
        purpose="matched_head_time_refinement",sample_seed=20267601,noise_seed=20267602)
    path=args.out/"identity.json"
    if path.exists():
        prior=json.loads(path.read_text())
        if {k:v for k,v in prior.items() if k!="source_commit"}!={k:v for k,v in identity.items() if k!="source_commit"}:
            raise ValueError("registered experiment identity changed")
        identity=prior
    elif args.verify:
        raise ValueError("no registered run")
    else:
        args.out.mkdir(parents=True,exist_ok=True)
        atomic_json(path,identity)
    if args.verify:
        verify(args.out,identity)
        return
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        device=setup()
        data=SemanticWritingData(BASE/"cache")
        if data.report_sha256!="1da155a5476f59f0c6cff1d0e897534f371dba7e8bf7e07188157176204b4aeb":
            raise ValueError("complete training cache differs")
        if not (args.out/"preflight/report.json").exists():
            preflight(args.out/"preflight",data,device,identity)
        for law in ("logit_normal","uniform"):
            train_arm(args.out/law,data,device,dict(identity,law=law))
            evaluate_arm(args.out/law,data,device,dict(identity,law=law))
        atomic_json(args.out/"status.json",dict(stage="completed_awaiting_cpu_verification",pid=None))


if __name__=="__main__":
    main()
