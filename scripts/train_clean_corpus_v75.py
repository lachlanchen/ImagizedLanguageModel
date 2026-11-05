#!/usr/bin/env python3
"""Train/resume the complete causal continuous image writer on frozen memory."""

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

from ilm.visual_lm.blind_vision_transcription import check_host_memory, gpu_memory
from ilm.visual_lm.codec_block_writer import CodecBlockWriter, load_qualified_writing_codec
from ilm.visual_lm.semantic_writing_data import ExactEpochSampler, SemanticWritingData
from ilm.visual_lm.clean_training_objective import CleanTrainingModel as SemanticWritingModel
from scripts.cache_semantic_images_v72 import ROOT, CODEC, atomic_json, committed_identity, sha

SOURCES = ("scripts/train_clean_corpus_v75.py", "scripts/run_clean_corpus_v75.py",
    "scripts/audit_clean_corpus_v75.py", "scripts/verify_semantic_ilm_v72.py",
    "ilm/visual_lm/clean_training_objective.py",
    "ilm/visual_lm/clean_writing_flow.py", "ilm/visual_lm/full_rank_writing_flow.py",
    "ilm/visual_lm/semantic_writing_model.py", "ilm/visual_lm/semantic_writing_data.py",
    "ilm/visual_lm/codec_block_writer.py", "references/clean_corpus_v75_protocol.md",
    "references/evidence/clean_learning_v74_result.json")
SEED = 20267500



def setup():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("explicit physical GPU 0 selection required")
    check_host_memory()
    if gpu_memory()>256:
        raise RuntimeError("GPU 0 is occupied")
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    return torch.device("cuda:0")


def components(data,device):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = SemanticWritingModel().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=.01,foreach=False)
    sampler = ExactEpochSampler(len(data), seed=20267501)
    generator = torch.Generator(device=device).manual_seed(20267502)
    return model,optimizer,sampler,generator


def update(model,optimizer,sampler,generator,data,step,device,*,schedule=True):
    indices = sampler.take(16)
    memory,codes,lengths = data.batch(indices,device)
    if schedule:
        factor = min((step+1)/500,1.) if step<500 else .1+.9*.5*(1+math.cos(math.pi*(step-500)/(40000-500)))
        for group in optimizer.param_groups:
            group["lr"] = 1e-4*factor
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        loss,metrics = model.training_loss(memory,codes,lengths,generator=generator,writer_blocks=16)
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite writing loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    return dict(update=step+1,loss=float(loss.detach()),flow=float(metrics["flow"]),
        clean_mse=float(metrics["clean"]),
        stop=float(metrics["stop"]),gradient_norm=float(norm),sampled_blocks=metrics["sampled_blocks"],
        draws=sampler.draws,epoch=sampler.epoch,lr=optimizer.param_groups[0]["lr"],indices=indices)


def checkpoint(path,model,optimizer,sampler,generator,step,identity,elapsed):
    temporary = path.with_suffix(".partial")
    torch.save(dict(model_config=model.configuration(),model=model.state_dict(),optimizer=optimizer.state_dict(),
        sampler=sampler.state_dict(),generator=generator.get_state(),cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(),update=step,identity=identity,active_elapsed_seconds=elapsed),temporary)
    temporary.replace(path)


def restore(path,model,optimizer,sampler,generator,identity):
    value = torch.load(path,map_location="cpu",weights_only=True)
    if value["identity"]!=identity or value["model_config"]!=model.configuration():
        raise ValueError("training checkpoint identity changed")
    model.load_state_dict(value["model"],strict=True)
    optimizer.load_state_dict(value["optimizer"])
    sampler.load_state_dict(value["sampler"])
    generator.set_state(value["generator"])
    torch.set_rng_state(value["cpu_rng"])
    torch.cuda.set_rng_state(value["cuda_rng"])
    return value


def save_answer(path,result):
    blocks = result["images"]
    strip = torch.cat(list(blocks),dim=-1) if len(blocks) else torch.ones(1,32,64)
    array = ((strip[0]+1)*127.5).round().clamp(0,255).to(torch.uint8).numpy()
    Image.fromarray(array).save(path)


def preflight(args,data,device,identity):
    out = args.output
    if len(data)<128:
        raise ValueError("preflight requires 128 actual full training examples")
    model,optimizer,sampler,generator = components(data,device)
    memory,codes,lengths = data.batch(list(range(16)),device)
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        original = model.conditions(memory,codes,lengths)
        changed = codes.clone()
        changed[:,1:] = torch.randn_like(changed[:,1:])
        alternate = model.conditions(memory,changed,lengths)
        future_exclusion = torch.equal(original[:,:2],alternate[:,:2])
    if not future_exclusion:
        raise AssertionError("future answer pixels entered a writing query")
    rows = []
    begun = time.perf_counter()
    for step in range(32):
        rows.append(update(model,optimizer,sampler,generator,data,step,device,schedule=False))
    snapshot = out/"resume_check.pt"
    checkpoint(snapshot,model,optimizer,sampler,generator,32,identity,time.perf_counter()-begun)
    expected = update(model,optimizer,sampler,generator,data,32,device,schedule=False)
    expected_state = {n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
    restored,restored_optimizer,restored_sampler,restored_generator = components(data,device)
    restore(snapshot,restored,restored_optimizer,restored_sampler,restored_generator,identity)
    actual = update(restored,restored_optimizer,restored_sampler,restored_generator,data,32,device,schedule=False)
    exact_resume = expected==actual and all(torch.equal(expected_state[n],v.cpu()) for n,v in restored.state_dict().items())
    if not exact_resume:
        raise AssertionError("optimizer/RNG/data cursor resume is not bit-exact")
    del expected_state,restored,restored_optimizer,restored_sampler,restored_generator
    writer = CodecBlockWriter(load_qualified_writing_codec(CODEC,device))
    model.eval()
    sample_memory = data.batch([0],device)[0]
    with torch.autocast("cuda",dtype=torch.bfloat16):
        natural = model.generate(sample_memory,writer,generator=torch.Generator(device=device).manual_seed(72),maximum_blocks=8)
    save_answer(out/"preflight_unforced_answer.png",natural)
    # Execution-only coverage of physical feedback, even if the untrained
    # stop head immediately stops. This forced branch receives no quality credit.
    saved_bias = model.stop.bias.detach().clone()
    with torch.no_grad():
        model.stop.bias.fill_(-30)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        rendered = model.generate(sample_memory,writer,generator=torch.Generator(device=device).manual_seed(73),maximum_blocks=2)
    with torch.no_grad():
        model.stop.bias.copy_(saved_bias)
    hashes = [hashlib.sha256(writer.encode_targets(image[None].to(device)).cpu().numpy().tobytes()).hexdigest()
        for image in rendered["images"]]
    feedback = len(hashes)==2 and hashes==rendered["feedback_sha256"]
    if not feedback:
        raise AssertionError("history did not reread actual emitted pixels")
    # The fixed codec must reconstruct the same cached image labels.
    target_codes = codes[0,:int(lengths[0])]
    supplied = writer.predict(target_codes).cpu()
    save_answer(out/"preflight_supplied_code_render.png",dict(images=supplied))
    atomic_json(out/"preflight.json",dict(identity=identity,real_cached_examples=len(data),
        trainable_parameters=sum(p.numel() for p in model.parameters()),updates_exercised=33,
        future_target_exclusion_exact=future_exclusion,exact_optimizer_rng_resume=exact_resume,
        physical_feedback_reencoded_exact=feedback,forced_feedback_branch_quality_claim=False,
        unforced_answer_blocks=len(natural["images"]),unforced_stopped=natural["stopped"],
        unforced_image_sha256=sha(out/"preflight_unforced_answer.png"),
        peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
        elapsed_seconds=time.perf_counter()-begun,first_loss=rows[0],last_loss=rows[-1],
        passed=True,language_quality_claim=False,general_ilm_goal_complete=False))
    atomic_json(out/"status.json",dict(stage="preflight_passed",pid=None))


def train(args,data,device,identity):
    completed = args.output/"training_report.json"
    if completed.exists():
        report = json.loads(completed.read_text())
        if (report["updates"]!=40000 or report["draws"]!=640000
                or report["identity"]!=identity
                or sha(args.output/"checkpoint_latest.pt")!=report["checkpoint_sha256"]):
            raise ValueError("completed training state changed")
        return
    if not args.preflight_receipt or not args.preflight_receipt.exists():
        raise ValueError("production requires the real-example execution receipt")
    pre = json.loads(args.preflight_receipt.read_text())
    if (not pre["passed"] or pre["identity"]["recipe_sha256"]!=data.recipe_sha256
            or pre["identity"]["sources"]!=identity["sources"]):
        raise ValueError("preflight implementation/recipe differs")
    model,optimizer,sampler,generator = components(data,device)
    step,accumulated = 0,0.
    latest = args.output/"checkpoint_latest.pt"
    if latest.exists():
        saved = restore(latest,model,optimizer,sampler,generator,identity)
        step,accumulated = saved["update"],saved["active_elapsed_seconds"]
    begun = time.perf_counter()
    try:
        with (args.output/"metrics.jsonl").open("a") as log:
            while step<40000:
                metrics = update(model,optimizer,sampler,generator,data,step,device)
                step += 1
                if step==1 or step%20==0:
                    metrics.pop("indices")
                    metrics.update(active_elapsed_seconds=accumulated+time.perf_counter()-begun,
                        peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3)
                    log.write(json.dumps(metrics)+"\n")
                    log.flush()
                    atomic_json(args.output/"status.json",dict(stage="training",pid=os.getpid(),**metrics))
                    print(json.dumps(metrics),flush=True)
                if step%500==0 or step==40000:
                    checkpoint(latest,model,optimizer,sampler,generator,step,identity,accumulated+time.perf_counter()-begun)
            atomic_json(args.output/"training_report.json",dict(identity=identity,updates=step,
                draws=sampler.draws,completed_epochs=sampler.epoch,checkpoint_sha256=sha(latest),
                trainable_parameters=sum(p.numel() for p in model.parameters()),
                active_elapsed_seconds=accumulated+time.perf_counter()-begun,
                peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
                vocabulary_head=False,teacher_service=False,language_quality_claim=False))
            atomic_json(args.output/"status.json",dict(stage="training_complete_awaiting_audit",pid=None,update=step))
    except (KeyboardInterrupt, SystemExit):
        checkpoint(latest,model,optimizer,sampler,generator,step,identity,accumulated+time.perf_counter()-begun)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--preflight-receipt",type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        device = setup()
        data = SemanticWritingData(args.cache,preflight=args.preflight)
        identity = dict(**committed_identity(SOURCES),cache_report_sha256=data.report_sha256,
            recipe_sha256=data.recipe_sha256,seed=SEED,batch=16,planned_updates=40000,
            purpose="full_corpus_clean", time_distribution=dict(kind="logit_normal", mean=-.8, std=.8),
            paired_control_weights_used=False, prior_velocity_weights_used=False)
        if (data.report_sha256!="1da155a5476f59f0c6cff1d0e897534f371dba7e8bf7e07188157176204b4aeb"
                or data.recipe_sha256!="8156258a7a708556c37db4320d54966adadd5d7192dcb3bd615cd6995e14482e"):
            raise ValueError("full admitted cache identity changed")
        # A later commit may add inference/reporting code without changing this
        # registered optimizer implementation. Preserve original resume identity.
        if (args.output/"identity.json").exists():
            prior = json.loads((args.output/"identity.json").read_text())
            if {k:v for k,v in prior.items() if k!="source_commit"}!={k:v for k,v in identity.items() if k!="source_commit"}:
                raise ValueError("training identity differs from prior run")
            identity = prior
        else:
            atomic_json(args.output/"identity.json",identity)
        if args.preflight:
            preflight(args,data,device,identity)
        else:
            train(args,data,device,identity)


if __name__=="__main__":
    main()
