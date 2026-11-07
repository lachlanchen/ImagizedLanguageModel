#!/usr/bin/env python3
"""Learn clean-image reading; preserve the base generator and full ILM goal."""
import argparse
import json
import math
from pathlib import Path
import random
import time
import traceback
from types import SimpleNamespace

import torch

from ilm.native_unified.showo2_source import load, assets, source_modules, image_latent, reading_prefix, sha, MODEL_SHA256
from ilm.native_unified.reading import (add_reading_adapters, reader_gate, checkpoint_reading,
    supervised_loss, cached_decisions, trainable_state, restore)
from probe_native_unified_source import gpu_slot, save, ANSWER_INSTRUCTION

ROOT=Path(__file__).resolve().parents[1]
SEED=20268921


def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def code_binding():
    names=("scripts/train_native_visual_access.py","ilm/native_unified/reading.py","ilm/native_unified/showo2_source.py")
    return {name:sha(ROOT/name) for name in names}


def cache(args):
    original=rows(args.manifest);strips={r["id"]:r for r in rows(args.copy_manifest)}
    diagnostic=[r for split in ("train","validation") for r in
                [x for x in original if x["task"]=="copy" and x["split"]==split][:4]]
    diagnostic += [r for r in original if r["task"]=="qa" and r["split"]=="validation"][:4]
    fixed={r["id"]:20268521+i for i,r in enumerate(diagnostic)}
    native=source_modules(args.assets);files=assets(args.assets)
    vae=native.vae.WanVAE(vae_pth=str(files.vae),dtype=torch.float32,device="cuda:0")
    b=SimpleNamespace(vae=vae,device=torch.device("cuda:0"))
    records=[];(args.output/"samples").mkdir()
    for i,r in enumerate(original):
        row=strips[r["id"]] if r["task"]=="copy" else r
        folder=args.copy_manifest.parent if r["task"]=="copy" else args.manifest.parent
        path=folder/row["input"]
        if sha(path)!=row["input_sha256"]:raise ValueError("input source changed")
        seed=fixed.get(r["id"],SEED+i)
        z,_=image_latent(b,path,seed)
        name=f"samples/{i:04}.pt";torch.save(z.cpu(),args.output/name)
        records.append(dict(id=r["id"],split=r["split"],task=r["task"],expected=r["reference_text"],
            input_sha256=row["input_sha256"],latent=name,latent_sha256=sha(args.output/name),
            seed=seed,shape=list(z.shape),input_characters=row["input_geometry"]["non_newline_characters"]))
        if (i+1)%32==0:print("cached",i+1,"/",len(original),flush=True)
        save(args.output/"manifest.partial.json",dict(records=records))
    save(args.output/"manifest.json",dict(records=records,source_manifest_sha256=sha(args.manifest),
        copy_manifest_sha256=sha(args.copy_manifest),assets_lock_sha256=sha(ROOT/"references/native_unified_assets.lock.json"),
        codec_dtype="float32",source_pixels_resized=False,model_calls=len(records),optimizer_updates=0,
        implementation=code_binding()))


def selected_cache(args):
    data=json.loads((args.cache/"manifest.json").read_text())
    selected=[r for r in data["records"] if r["split"]=="train"]
    if len(selected)!=384 or sum(r["task"]=="copy" for r in selected)!=128 or sum(r["task"]=="qa" for r in selected)!=256:
        raise ValueError("complete declared 128-copy/256-QA training set required")
    return data,selected


def latent(args,row,device):
    path=args.cache/row["latent"]
    if sha(path)!=row["latent_sha256"]:raise ValueError("cached image changed")
    return torch.load(path,map_location=device,weights_only=True)


def schedule(selected,budget):
    rng=random.Random(SEED);groups={task:[r for r in selected if r["task"]==task] for task in ("copy","qa")}
    pools={task:[] for task in groups};result=[]
    for i in range(budget):
        task="copy" if i%2==0 else "qa"
        if not pools[task]:pools[task]=rng.sample(groups[task],len(groups[task]))
        result.append(pools[task].pop())
    return result


def run_learning(args):
    _,selected=selected_cache(args)
    budget=2 if args.stage=="preflight" else args.updates
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    b=load(args.assets,dtype=torch.bfloat16)
    b.vae=None  # cached-input optimization does not retain an unused GPU decoder
    layers=add_reading_adapters(b,16)
    parameters=[p for p in b.model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(parameters,lr=1e-4,betas=(.9,.95),weight_decay=.01)
    meta=dict(source_model_sha256=MODEL_SHA256,
        cache_manifest_sha256=sha(args.cache/"manifest.json"),rank=16,budget=budget,seed=SEED,
        training_ids=[r["id"] for r in selected],implementation=code_binding())
    if args.stage=="preflight":
        chosen=[max((r for r in selected if r["task"]=="copy"),key=lambda r:len(r["expected"])),
                max((r for r in selected if r["task"]=="qa"),key=lambda r:r["input_characters"])]
    else:chosen=schedule(selected,budget)
    completed=0
    if args.resume:
        saved=torch.load(args.resume,map_location="cpu",weights_only=True)
        if saved["metadata"]!=meta:raise ValueError("resume source/data/code/budget differs")
        restore(b.model,saved["adapters"]);opt.load_state_dict(saved["optimizer"]);completed=saved["completed_updates"]
        torch.set_rng_state(saved["cpu_rng"]);torch.cuda.set_rng_state_all(saved["cuda_rng"])
    before=None;zero_error=None;probe=None
    if args.stage=="preflight":
        b.model.eval();probe=latent(args,chosen[0],b.device)
        with torch.autocast("cuda",dtype=torch.bfloat16),torch.no_grad():
            before=reading_prefix(b,probe,ANSWER_INSTRUCTION)[0].clone()
            with reader_gate(b.model):at_zero=reading_prefix(b,probe,ANSWER_INSTRUCTION)[0]
        zero_error=float((before-at_zero).abs().max())
        if zero_error!=0:raise ValueError("zero-residual source prefix differs")
    checkpoint_reading(b)
    start=time.monotonic();torch.cuda.reset_peak_memory_stats();journal=[]

    def snapshot():
        value=dict(metadata=meta,completed_updates=completed,adapters=trainable_state(b.model),optimizer=opt.state_dict(),
            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        target=args.output/"checkpoint.pt";temporary=args.output/"checkpoint.pending.pt"
        torch.save(value,temporary);temporary.replace(target)

    with (args.output/"updates.jsonl").open("a",buffering=1) as log:
        for index in range(completed,budget):
            row=chosen[index];z=latent(args,row,b.device);opt.zero_grad(set_to_none=True)
            if args.stage=="preflight":rate=1e-4
            elif index<100:rate=1e-4*(index+1)/100
            else:rate=1e-5+.5*(1e-4-1e-5)*(1+math.cos(math.pi*(index-100)/max(1,budget-100)))
            for group in opt.param_groups:group["lr"]=rate
            tick=time.monotonic()
            with reader_gate(b.model):
                with torch.autocast("cuda",dtype=torch.bfloat16):loss,observation=supervised_loss(b,z,row["expected"],ANSWER_INSTRUCTION)
                if not bool(torch.isfinite(loss)):raise ValueError("nonfinite native reading loss")
                loss.backward()
            gradients={kind:[p.grad for name,p in b.model.named_parameters() if p.requires_grad and name.endswith("."+kind)] for kind in ("down","up")}
            stats={kind:dict(total=len(values),nonzero=sum(g is not None and bool(g.abs().sum()) for g in values),
                finite=all(g is not None and bool(torch.isfinite(g).all()) for g in values)) for kind,values in gradients.items()}
            if not all(x["finite"] for x in stats.values()):raise ValueError("missing/nonfinite adapter gradients")
            if any(p.grad is not None for p in b.model.parameters() if not p.requires_grad):raise ValueError("frozen native weight acquired gradients")
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True)
            opt.step();completed=index+1
            item=dict(update=completed,id=row["id"],task=row["task"],loss=float(loss.detach()),lr=rate,
                gradient_norm=float(norm),gradients=stats,**observation,elapsed_seconds=time.monotonic()-tick)
            journal.append(item);log.write(json.dumps(item)+"\n")
            if completed%32==0 or args.stage=="preflight":print(completed,budget,row["id"],item["loss"],flush=True)
            if completed%128==0 or completed==budget:snapshot()
            save(args.output/"status.json",dict(stage=args.stage,state="running",completed_updates=completed,budget=budget))
    disabled_error=None
    if before is not None:
        b.model.eval()
        with torch.autocast("cuda",dtype=torch.bfloat16):after=reading_prefix(b,probe,ANSWER_INSTRUCTION)[0]
        disabled_error=float((before-after).abs().max())
        if disabled_error!=0:raise ValueError("disabled native visual path changed")
        if not all(x["nonzero"]==x["total"] for x in journal[-1]["gradients"].values()):raise ValueError("second-update gradient coverage incomplete")
    save(args.output/"report.json",dict(updates=completed,training_examples=len(selected),adapted_layers=len(layers),
        trainable_parameters=sum(p.numel() for p in parameters),complete_model_parameters=sum(p.numel() for p in b.model.parameters()),
        codec_parameters=b.counts["vae"],zero_adapter_prefix_error=zero_error,disabled_prefix_error_after_updates=disabled_error,
        elapsed_seconds=time.monotonic()-start,peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
        checkpoint_sha256=sha(args.output/"checkpoint.pt"),metadata=meta,actual_optimizer_updates=True,
        auxiliary_unicode_head=True,image_answer_generation_demonstrated=False,language_quality_qualified=False,general_ilm_goal_complete=False))


def evaluate(args):
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    b=load(args.assets,dtype=torch.bfloat16);add_reading_adapters(b,16)
    if args.checkpoint:
        value=torch.load(args.checkpoint,map_location="cpu",weights_only=True);restore(b.model,value["adapters"])
    # Match the original FP32 VAE samples; only the language/visual body uses BF16.
    b.vae=None;torch.cuda.empty_cache()
    b.vae=b.native.vae.WanVAE(vae_pth=str(b.files.vae),dtype=torch.float32,device=b.device)
    b.model.eval();records=json.loads(args.inputs.read_text());results=[];start=time.monotonic()
    with torch.inference_mode(),reader_gate(b.model):
        for i,row in enumerate(records):
            path=args.inputs.parent/row["input"]
            if sha(path)!=row["input_sha256"]:raise ValueError("evaluation input changed")
            z,_=image_latent(b,path,row["seed"])
            with torch.autocast("cuda",dtype=torch.bfloat16):
                prefix,mask=reading_prefix(b,z,ANSWER_INSTRUCTION)
                generated=cached_decisions(b,prefix,mask,seed=20268621+i,cap=96)
            result=dict(id=row["id"],ids=generated,text=b.tokenizer.decode(generated,skip_special_tokens=True),
                stopped=bool(generated and generated[-1]==b.tokenizer.eos_token_id),input_sha256=row["input_sha256"],student_answer=False)
            results.append(result);save(args.output/"outputs.json",results);print(row["id"],repr(result["text"]),flush=True)
    save(args.output/"report.json",dict(cases=len(results),elapsed_seconds=time.monotonic()-start,
        inputs_sha256=sha(args.inputs),checkpoint_sha256=sha(args.checkpoint) if args.checkpoint else None,
        implementation=code_binding(),auxiliary_unicode_head=True,language_quality_qualified=False,general_ilm_goal_complete=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("cache","preflight","train","evaluate"),required=True)
    p.add_argument("--assets",type=Path,default=Path("artifacts/external/native_unified_source"))
    p.add_argument("--manifest",type=Path,default=Path("artifacts/image_edit_learning/pages_v2/pairs.jsonl"))
    p.add_argument("--copy-manifest",type=Path,default=Path("artifacts/image_edit_learning/strip_pages/pairs.jsonl"))
    p.add_argument("--cache",type=Path,default=Path("artifacts/native_visual_access/cache_v1"))
    p.add_argument("--inputs",type=Path,default=Path("artifacts/native_unified_probe/prepare_v1/inputs.json"))
    p.add_argument("--checkpoint",type=Path);p.add_argument("--resume",type=Path)
    p.add_argument("--updates",type=int,default=4096);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.updates<2:raise ValueError("at least two updates required")
    args.output.mkdir(parents=True,exist_ok=False);save(args.output/"status.json",dict(stage=args.stage,state="running"))
    torch.set_num_threads(8)
    try:
        with gpu_slot():
            if args.stage=="cache":cache(args)
            elif args.stage=="evaluate":evaluate(args)
            else:run_learning(args)
        save(args.output/"status.json",dict(stage=args.stage,state="completed"))
    except BaseException as e:
        save(args.output/"status.json",dict(stage=args.stage,state="failed",error=repr(e),traceback=traceback.format_exc()));raise


if __name__=="__main__":main()
