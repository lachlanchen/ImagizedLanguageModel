#!/usr/bin/env python3
"""Separate image-information and objective failures in the actual ILM writer.

All fitting/evaluation images are from training. This does not qualify language.
"""

import argparse
import copy
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.nn import functional as F

from ilm.visual_lm.blind_vision_transcription import check_host_memory,gpu_memory
from ilm.visual_lm.complete_writing_data import CompleteWritingImageData
from ilm.visual_lm.continuous_glyph_codec import ContinuousGlyphCodec,ContinuousGlyphCodecConfig
from ilm.visual_lm.grounded_image_actor import ContinuousBlockWriter,GroundedImageConfig
from ilm.visual_lm.grounded_image_weights import load_grounded_image_actor,file_sha
from scripts.check_grounded_image_actor import ROOT,WRITING_SHA,SOURCE_NAMES,atomic_json,cpu_tree,exact
from scripts.train_grounded_writer import save_checkpoint,repair_journal,foreground_f1


ARMS = ("inherited_flow","inherited_direct","pixels_flow","pixels_direct")
UPDATES = 2000
CODEC_SHA = "a138c9cb3b0502e43d1227f689c020893d56b468742c32e1840e44d299662f33"
NAMES = sorted(set(SOURCE_NAMES)|{"scripts/diagnose_grounded_writer_v65.py",
    "references/writer_factorial_v65_protocol.md","scripts/train_grounded_writer.py",
    "ilm/visual_lm/continuous_glyph_codec.py"})


def save_image(path,pixels):
    path.parent.mkdir(parents=True,exist_ok=True)
    image = ((pixels.detach().cpu().reshape(32,64)+1)*127.5).numpy().astype(np.uint8)
    Image.fromarray(image).save(path)


def raw_condition(images):
    if images.ndim!=4 or images.shape[1:]!=(1,32,64):
        raise ValueError("physical blocks required")
    patches = torch.stack((images[...,:32],images[...,32:]),dim=1).flatten(2)
    return torch.cat((patches,patches,torch.ones_like(patches),-torch.ones_like(patches)),dim=-1)


def raw_inverse(condition):
    patches = condition[...,:1024].reshape(-1,2,1,32,32)
    return torch.cat((patches[:,0],patches[:,1]),dim=-1)


def score(pixels,reference):
    return dict(foreground_f1=foreground_f1(pixels,reference),
        exact=bool(torch.equal(pixels,reference)),blank=bool((pixels==1).all()))


def features_summary(features):
    flat = F.layer_norm(features.float(),(4096,)).flatten(1).cpu().numpy()
    singular = np.linalg.svd(flat-flat.mean(0),compute_uv=False)
    mass = singular**2
    mass = mass/mass.sum() if mass.sum() else mass
    entropy = -float(np.sum(mass[mass>0]*np.log(mass[mass>0])))
    return dict(centered_effective_rank=math.exp(entropy),
        mean_between_case_std=float(np.std(flat,axis=0).mean()),
        mean_per_case_std=float(np.std(flat,axis=1).mean()),
        minimum_pairwise_l2=float(torch.pdist(torch.from_numpy(flat)).min()))


@torch.no_grad()
def codec_reference(images,out,device):
    path = ROOT/"artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt"
    if file_sha(path)!=CODEC_SHA:
        raise ValueError("qualified codec checkpoint changed")
    saved = torch.load(path,map_location="cpu",weights_only=False)
    model = ContinuousGlyphCodec(ContinuousGlyphCodecConfig(**saved["model_config"]))
    model.load_state_dict(saved["ema"]["shadow"])
    model = model.to(device).requires_grad_(False).eval()
    targets = images.to(device)
    ink = (1-torch.cat((targets[...,:32],targets[...,32:]),dim=0))/2
    logits = model.decode(model.encode(ink))
    bits = torch.where(logits>=0,-1.,1.)
    pixels = torch.cat((bits[:64],bits[64:]),dim=-1).cpu()
    rows=[]
    for i,(prediction,reference) in enumerate(zip(pixels,images)):
        save_image(out/"codec"/f"{i:03}.png",prediction)
        rows.append(score(prediction[None],reference[None]))
    del saved,model
    result=dict(checkpoint_sha256=CODEC_SHA,measurements={key:float(np.mean([r[key] for r in rows])) for key in rows[0]},
        cases=rows,scope="Existing continuous-codec reconstruction on the same 64 training-only blocks")
    atomic_json(out/"codec_report.json",result)


def prepare(out,sources,device,status):
    target = out/"conditions.pt"
    if target.exists():
        receipt = json.loads((out/"conditions_receipt.json").read_text())
        if receipt["sources"]!=sources or receipt["sha256"]!=file_sha(target):
            raise ValueError("fixed diagnostic conditions changed")
        return torch.load(target,map_location="cpu",weights_only=True)
    data = CompleteWritingImageData(ROOT/"data/general_instructions/complete_lines_train_20261105",
        purpose="train",expected_report_sha256=WRITING_SHA)
    generator = torch.Generator().manual_seed(20266510)
    pools = [[i for i,name in enumerate(data.names) if name.startswith(prefix)] for prefix in ("cmrc/","dolly/")]
    selected = [[pool[i] for i in torch.randperm(len(pool),generator=generator)[:32].tolist()] for pool in pools]
    indices = [index for pair in zip(*selected) for index in pair]
    images,manifest=[],[]
    for index in indices:
        strip = data.target(index)[...,:-128]
        block = int(torch.randint(strip.shape[-1]//64,(),generator=generator))
        image = strip[...,block*64:(block+1)*64].clone()
        images.append(image)
        manifest.append(dict(index=index,file=data.names[index],block=block,
            source_sha256=data.report["files_sha256"][data.names[index]]))
    images = torch.cat(images)
    for i,image in enumerate(images):
        save_image(out/"reference"/f"{i:03}.png",image)
    atomic_json(out/"manifest.json",manifest)
    actor,provenance = load_grounded_image_actor(ollama_root=ROOT.parent/"LocalLLM/.local/models/ollama",
        gguf_source=ROOT.parent/"LocalLLM/.local/tools/llama.cpp/gguf-py",device=device,
        progress=lambda loaded,total:status("extracting_visual_conditions",loaded=loaded,total=total))
    actor.requires_grad_(False).eval()
    features=[]
    for start in range(0,64,16):
        value,_,_ = actor.encode_images([image[None].clone() for image in images[start:start+16]])
        features.extend(item.detach().cpu() for item in value)
    inherited = torch.stack(features)
    extraction_peak = torch.cuda.max_memory_allocated()/1024**3
    del actor
    gc.collect()
    torch.cuda.empty_cache()
    pixels = raw_condition(images)
    if not torch.equal(raw_inverse(pixels),images):
        raise AssertionError("raw conditional control dropped pixels")
    # The two fixed anchors recover the original affine scale after LayerNorm.
    normalized = F.layer_norm(pixels,(4096,))
    lower,upper = normalized[...,-1024:].mean(-1,keepdim=True),normalized[...,-2048:-1024].mean(-1,keepdim=True)
    restored = 2*(normalized[...,:1024]-lower)/(upper-lower)-1
    if not torch.allclose(restored,pixels[...,:1024],atol=1e-5,rtol=0):
        raise AssertionError("raw control loses distinctions under conditioning normalization")
    schedule = torch.randint(64,(UPDATES,16),generator=torch.Generator().manual_seed(20266521))
    value = dict(images=images,inherited=inherited,pixels=pixels,schedule=schedule)
    temporary = target.with_suffix(".partial")
    torch.save(value,temporary);temporary.replace(target)
    atomic_json(out/"conditions_receipt.json",dict(sha256=file_sha(target),sources=sources,
        extraction_peak_allocated_vram_gib=extraction_peak,initialization=provenance,
        writing_report_sha256=data.report_sha256,training_only=True,examples=64,
        normalized_raw_inverse_max_error=float((restored-pixels[...,:1024]).abs().max()),
        feature_geometry={name:features_summary(value[name]) for name in ("inherited","pixels")}))
    codec_reference(images,out,device)
    return value


@torch.no_grad()
def evaluate(writer,condition,images,arm,out,device):
    rows=[]
    for i in range(64):
        row=dict(index=i,scores={})
        for route,source in (("aligned",i),("shuffled",(i+2)%64)):
            value=condition[source:source+1]
            with torch.autocast("cuda",dtype=torch.bfloat16):
                if arm.endswith("flow"):
                    continuous=writer.sample(value,steps=16,
                        generator=torch.Generator(device=device).manual_seed(20266530+i))
                else:
                    continuous=writer(torch.zeros((1,1,32,64),device=device),torch.zeros(1,device=device),value)
            pixels=torch.where(continuous>=0,1.,-1.).cpu()
            row["scores"][route]=score(pixels,images[i:i+1])
            save_image(out/arm/route/f"{i:03}.png",pixels)
        rows.append(row)
    measured={route:{key:float(np.mean([r["scores"][route][key] for r in rows]))
        for key in ("foreground_f1","exact","blank")} for route in ("aligned","shuffled")}
    gain=measured["aligned"]["foreground_f1"]-measured["shuffled"]["foreground_f1"]
    gates=dict(foreground_f1=measured["aligned"]["foreground_f1"]>=.95,conditioning_gain=gain>=.50)
    atomic_json(out/arm/"cases.json",rows)
    return dict(measurements=measured,conditioning_gain=gain,gates=gates,
        fitting_criterion_pass=all(gates.values()),training_only=True,language_quality_claim=False)


def galleries(out):
    for route in ("aligned","shuffled"):
        image=Image.new("RGB",(1080,8*96+50),"white");draw=ImageDraw.Draw(image)
        for j,name in enumerate(("reference",*ARMS)):
            draw.text((65+j*200,12),name,fill="black")
            for i in range(8):
                path=out/"reference"/f"{i:03}.png" if j==0 else out/name/route/f"{i:03}.png"
                image.paste(Image.open(path).resize((192,96),Image.Resampling.NEAREST),(65+j*200,45+i*96))
                draw.text((5,65+i*96),f"{i:03}",fill="black")
        image.save(out/("first_eight_"+route+".png"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"artifacts/writer_factorial_v65_20261105")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args();out=args.out
    if (os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or os.environ.get("CUDA_DEVICE_ORDER")!="PCI_BUS_ID"
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8"):
        raise ValueError("explicit physical GPU 0 and deterministic cuBLAS required")
    out.mkdir(parents=True,exist_ok=True)
    if (out/"report.json").exists():
        raise FileExistsError("completed diagnosis is immutable")
    if (out/"configuration.json").exists()!=args.resume:
        raise ValueError("use --resume exactly for an existing run")
    lock=(out/"operation.lock").open("a+");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources={name:file_sha(ROOT/name) for name in NAMES}
    config=dict(sources=sources,arms=list(ARMS),updates_per_arm=UPDATES,
        training_only=True,source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    if args.resume:
        old=json.loads((out/"configuration.json").read_text())
        if old["sources"]!=sources or old["arms"]!=list(ARMS):
            raise ValueError("frozen diagnosis changed")
        config=old
    else:
        atomic_json(out/"configuration.json",config)
    begun=time.perf_counter();stopping=False
    prior_runtime=json.loads((out/"status.json").read_text()).get("cumulative_elapsed_seconds",0.) if (out/"status.json").exists() else 0.
    def interrupted(_signal,_frame):
        nonlocal stopping
        stopping=True
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
        signal.signal(sig,interrupted)
    def status(stage,**values):
        row=dict(stage=stage,pid=os.getpid(),physical_gpu=0,elapsed_this_process=time.perf_counter()-begun,
            cumulative_elapsed_seconds=prior_runtime+time.perf_counter()-begun,**values)
        atomic_json(out/"status.json",row);print(json.dumps(row),flush=True)
    try:
        check_host_memory()
        if gpu_memory()>256:
            raise RuntimeError("GPU 0 occupied; do not overlap another job")
        torch.set_num_threads(4);torch.manual_seed(20266520);torch.cuda.manual_seed_all(20266520)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
        device=torch.device("cuda:0")
        status("preparing_training_only_conditions")
        cache=prepare(out,sources,device,status)
        if stopping:
            status("interrupted",resumable=True,reason="handled signal before fitting");return
        initial=out/"initial_writer.pt"
        torch.manual_seed(20266520)
        common=ContinuousBlockWriter(GroundedImageConfig())
        if not initial.exists():
            temporary=initial.with_suffix(".partial")
            torch.save(common.state_dict(),temporary);temporary.replace(initial)
        elif not exact(common.state_dict(),torch.load(initial,map_location="cpu",weights_only=True)):
            raise ValueError("common initialization differs from the fixed seed")
        del common
        atomic_json(out/"initial_writer_receipt.json",dict(sha256=file_sha(initial),seed=20266520,sources=sources))
        if not (out/"codec_report.json").exists():
            codec_reference(cache["images"],out,device)
        results={}
        for arm in ARMS:
            folder=out/arm;folder.mkdir(exist_ok=True)
            report_path=folder/"report.json"
            if report_path.exists():
                result=json.loads(report_path.read_text())
                if result["sources"]!=sources or result["final_sha256"]!=file_sha(folder/"final_ema.pt"):
                    raise ValueError("completed arm changed")
                results[arm]=result;continue
            condition=cache["inherited" if arm.startswith("inherited") else "pixels"].to(device)
            images=cache["images"].to(device)
            writer=ContinuousBlockWriter(GroundedImageConfig()).to(device)
            writer.load_state_dict(torch.load(initial,map_location="cpu",weights_only=True));writer.train()
            ema=copy.deepcopy(writer).requires_grad_(False).eval()
            optimizer=torch.optim.AdamW(writer.parameters(),lr=2e-4,betas=(.9,.95),weight_decay=.01,fused=True)
            noise_rng=torch.Generator(device=device).manual_seed(20266522)
            start=0;prior_elapsed=0.;arm_begun=time.perf_counter()
            checkpoint=out/"checkpoint_latest.pt"
            def snapshot(step):
                return dict(arm=arm,update=step,sources=sources,writer=cpu_tree(writer.state_dict()),
                    ema=cpu_tree(ema.state_dict()),optimizer=cpu_tree(optimizer.state_dict()),
                    noise_rng=noise_rng.get_state().cpu(),cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(device),elapsed_seconds=prior_elapsed+time.perf_counter()-arm_begun,
                    conditions_sha256=file_sha(out/"conditions.pt"),initial_sha256=file_sha(initial))
            def restore(saved):
                if (saved["arm"]!=arm or saved["sources"]!=sources
                        or saved["conditions_sha256"]!=file_sha(out/"conditions.pt")
                        or saved["initial_sha256"]!=file_sha(initial)):
                    raise ValueError("checkpoint identity differs")
                writer.load_state_dict(saved["writer"]);ema.load_state_dict(saved["ema"])
                optimizer.load_state_dict(saved["optimizer"]);noise_rng.set_state(saved["noise_rng"])
                torch.set_rng_state(saved["cpu_rng"]);torch.cuda.set_rng_state(saved["cuda_rng"],device)
            if checkpoint.exists():
                saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
                if saved["arm"]==arm:
                    restore(saved);start=saved["update"];prior_elapsed=saved["elapsed_seconds"]
                del saved
            repair_journal(folder/"metrics.jsonl",start)
            torch.cuda.reset_peak_memory_stats();status("fitting_writer",arm=arm,update=start)
            def update(step):
                chosen=cache["schedule"][step-1].to(device)
                target=images[chosen];features=condition[chosen]
                noise=torch.randn(target.shape,device=device,generator=noise_rng)
                times=torch.rand((len(target),),device=device,generator=noise_rng)
                if arm.endswith("flow"):
                    inputs=(1-times[:,None,None,None])*noise+times[:,None,None,None]*target
                    desired=target-noise
                else:
                    inputs=torch.zeros_like(target);times=torch.zeros_like(times);desired=target
                scale=step/100 if step<=100 else .1+.9*.5*(1+math.cos(math.pi*(step-100)/(UPDATES-100)))
                for group in optimizer.param_groups:group["lr"]=2e-4*scale
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    predicted=writer(inputs,times,features)
                    loss=F.mse_loss(predicted.float(),desired)
                loss.backward()
                if not bool(torch.isfinite(loss)) or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in writer.parameters()):
                    raise FloatingPointError("nonfinite writer gradients")
                norm=float(torch.nn.utils.clip_grad_norm_(writer.parameters(),1.))
                optimizer.step()
                with torch.no_grad():
                    for a,b in zip(ema.parameters(),writer.parameters()):a.mul_(.99).add_(b,alpha=.01)
                return dict(update=step,loss=float(loss.detach()),gradient_norm=norm,
                    indices=chosen.cpu().tolist(),learning_rate=2e-4*scale)
            def persist(step):
                if sources!={name:file_sha(ROOT/name) for name in NAMES}:
                    raise ValueError("frozen diagnosis sources changed")
                save_checkpoint(checkpoint,snapshot(step))
            for step in range(start+1,UPDATES+1):
                row=update(step)
                if step==3 and not (folder/"resume_verification.json").exists():
                    reference=snapshot(step);saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
                    if saved["update"]!=2:raise ValueError("early replay checkpoint missing")
                    restore(saved);replay=update(step);recovered=snapshot(step)
                    checks={key:exact(reference[key],recovered[key]) for key in ("writer","ema","optimizer","noise_rng","cpu_rng","cuda_rng")}
                    checks["metrics_and_indices"]=row==replay
                    atomic_json(folder/"resume_verification.json",dict(checks=checks,passed=all(checks.values()),extra_optimizer_replays=1))
                    if not all(checks.values()):raise AssertionError(checks)
                    del reference,saved,recovered
                row.update(elapsed_seconds=prior_elapsed+time.perf_counter()-arm_begun,
                    peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3)
                with (folder/"metrics.jsonl").open("a") as stream:stream.write(json.dumps(row,sort_keys=True)+"\n")
                if step<=3 or step%50==0:status("fitting_writer",arm=arm,**row)
                reason=None
                if step%250==0:
                    try:check_host_memory()
                    except RuntimeError as error:reason=str(error)
                if torch.cuda.max_memory_allocated()/1024**3>=22:reason="CUDA memory ceiling"
                if prior_runtime+time.perf_counter()-begun>=3600:reason="cumulative run time ceiling"
                if step==2 or step%250==0 or stopping or reason:persist(step)
                if stopping or reason:
                    status("interrupted",arm=arm,update=step,resumable=True,reason=reason or "handled signal");return
            if not json.loads((folder/"resume_verification.json").read_text())["passed"]:
                raise ValueError("exact early replay required")
            torch.save(ema.state_dict(),folder/"final_ema.pt")
            status("evaluating_fixed_fitting_images",arm=arm)
            result=evaluate(ema,condition,cache["images"],arm,out,device)
            result.update(sources=sources,update=UPDATES,initial_sha256=file_sha(initial),
                conditions_sha256=file_sha(out/"conditions.pt"),final_sha256=file_sha(folder/"final_ema.pt"),
                trainable_parameters=sum(p.numel() for p in writer.parameters()),extra_optimizer_replays=1,
                elapsed_seconds=prior_elapsed+time.perf_counter()-arm_begun,
                peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3)
            atomic_json(report_path,result);results[arm]=result
            del writer,ema,optimizer,condition,images
            gc.collect();torch.cuda.empty_cache()
        galleries(out)
        report=dict(arms=results,codec=json.loads((out/"codec_report.json").read_text()),
            conditions=json.loads((out/"conditions_receipt.json").read_text()),
            source_commit=config["source_commit"],sources=sources,all_arms_complete=True,
            scope="Training-only fitting diagnosis; not language generation or held-out generalization",
            language_quality_claim=False,general_ilm_goal_complete=False)
        atomic_json(out/"report.json",report);status("complete",general_ilm_goal_complete=False)
    except BaseException as error:
        atomic_json(out/f"failure_{time.time_ns()}.json",dict(error=repr(error),sources=sources))
        status("failed",error=repr(error));raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=="__main__":main()
