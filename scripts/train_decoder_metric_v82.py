#!/usr/bin/env python3
"""Matched state-only decoder metric versus continued image-code regression."""
import argparse
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.clean_image_actor import tensor_digest
from ilm.visual_lm.codec_block_writer import CodecBlockWriter, load_qualified_writing_codec
from ilm.visual_lm.metric_image_continuation import MetricImageLanguageModel
from ilm.visual_lm.native_image_continuation import NativeImageLanguageModel
from ilm.visual_lm.semantic_writing_data import ExactEpochSampler
from scripts.audit_native_neural_transcripts_v80 import write
from scripts.cache_semantic_images_v72 import ROOT, CODEC, sha, committed_identity
from scripts.evaluate_trained_readout_v81 import incremental_reference_stream
from scripts.infer_semantic_ilm_v72 import save_image_answer
from scripts.probe_corpus_fit_v75 import metrics, save_raster
from scripts.probe_native_corpus_fit_v80 import cases_and_data, prompt, targets
from scripts.score_native_training_samples_v80 import check_answer, raster
from scripts.train_adaptive_readout_v81 import (
    SOURCES as OLD_SOURCES, build, condition_checks, cpu_tree, frozen_hash, pool_for,
    prefix_hash, save_checkpoint, tree_equal,
)
from scripts.train_clean_corpus_v75 import setup
from scripts.train_native_chinese_v80 import batch, new_optimizer, restore_values, same_identity
from scripts.verify_codec_language_v68 import assert_close

SEED = 20268210
UPDATES = 2000
WARM_SHA = "7da31d229f4c5cc5bb51a5f5776a2c6e61e444f171ba412de547165911cb532c"
PLAN = "references/decoder_metric_v82_protocol.md"
SOURCES = tuple(dict.fromkeys(("scripts/train_decoder_metric_v82.py", PLAN,
    "ilm/visual_lm/decoder_pullback.py", "ilm/visual_lm/metric_image_continuation.py",
    "scripts/evaluate_trained_readout_v81.py", *OLD_SOURCES)))
ARMS = dict(control=0., metric=.001)


def identity_for(previous):
    evaluation = ROOT/"artifacts/adaptive_readout_v81eval_20261107"
    v = json.loads((evaluation/"verification.json").read_text())
    if (v["report_sha256"] != sha(evaluation/"report.json") or v["training_fit_repair_pass"]
            or not v["all_64_outputs_independently_scored"]
            or v["identity"]["checkpoints_sha256"]["adapted"] != WARM_SHA
            or sha(previous/"adapted/checkpoint_latest.pt") != WARM_SHA):
        raise ValueError("the completed failed adapted checkpoint is required")
    for name,digest in v["identity"]["sources"].items():
        if sha(ROOT/name) != digest:raise ValueError("the earlier evaluation implementation changed")
    derivative = {}
    for device in ("cpu", "gpu"):
        path = ROOT/f"artifacts/decoder_pullback_v82_{device}_20261107/report.json"
        receipt = json.loads(path.read_text())
        if not receipt["passed"] or not receipt["no_optimization_or_new_generation"]:
            raise ValueError("both actual derivative preflights must pass")
        for name,digest in receipt["identity"]["sources"].items():
            if sha(ROOT/name) != digest:raise ValueError("the preflight implementation changed")
        if sha(path.parent/"derivatives.pt") != receipt["retained_derivatives_sha256"]:
            raise ValueError("retained derivative evidence changed")
        derivative[device] = sha(path)
    cases, _, _ = cases_and_data()
    return dict(**committed_identity(SOURCES), purpose="matched_state_only_decoder_metric",
        warm_checkpoint_sha256=WARM_SHA, previous_evaluation_verification_sha256=sha(evaluation/"verification.json"),
        derivative_preflight_sha256=derivative, arms=ARMS, seed=SEED, updates_per_arm=UPDATES,
        batch_size=4, training_examples=32, external_pretraining=True, training_only=True,
        language_quality_claim=False, general_ilm_goal_complete=False), cases


def bind(out, identity):
    path = out/"identity.json"
    if path.exists():
        stored = json.loads(path.read_text())
        if not same_identity(identity,stored):raise ValueError("preserve the source-bound run")
        return stored
    if out.exists() and any(p.name != "launch.log" for p in out.iterdir()):
        raise ValueError("preserve nonempty unbound output")
    write(path,identity);return identity


def select(actor, warm, strength):
    model = MetricImageLanguageModel(actor.writing.continuation.context, actor.writing.writer,
        input_scale=float(warm["input_scale"]), query_seed=20268103, metric_strength=strength).to(actor.device)
    model.load_trainable_state_dict(warm)
    actor.writing = model;actor.train()
    return model


def update(model, optimizer, pool, sampler, generator, step, *, schedule=True, gradients=False):
    chosen = sampler.take(4)
    prefix,codes,lengths = batch(pool,chosen,model.device)
    if schedule:
        factor = (step+1)/100 if step<100 else .1+.9*.5*(1+math.cos(math.pi*(step-100)/(UPDATES-100)))
        optimizer.param_groups[0]["lr"] = 1e-4*factor
    optimizer.zero_grad(set_to_none=True)
    loss,values = model.training_loss(prefix,codes,lengths,generator=generator)
    if not torch.isfinite(loss):raise FloatingPointError("nonfinite physical-metric objective")
    loss.backward()
    groups = {}
    if gradients:
        selections = {name:[p for n,p in model.named_parameters() if n.startswith(name)]
            for name in ("image_input.","flow.","stop.")}
        selections["start_query"] = [model.start_query]
        for ending in (".down",".up"):
            selections["adapter"+ending] = [p for n,p in model.named_parameters()
                if n.startswith("continuation.context.") and n.endswith(ending)]
        groups = {name:bool(parameters) and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters)
            and any(bool(p.grad.abs().max()>0) for p in parameters) for name,parameters in selections.items()}
        if not all(groups.values()):raise AssertionError("missing finite nonzero gradients: "+str(groups))
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    optimizer.step()
    return dict(update=step+1,indices=chosen,loss=float(loss.detach()),flow=float(values["flow"]),
        clean_mse=float(values["clean"]),decoder_tangent_mse=float(values["decoder_tangent"]),
        metric_strength=model.metric_strength,stop=float(values["stop"]),gradient_norm=float(norm),
        gradient_groups=groups,lr=optimizer.param_groups[0]["lr"])


def preflight(actor,warm,cases,data,indices,out,identity):
    if (out/"preflight/report.json").exists():raise FileExistsError("preserve completed preflight")
    selected = [0,2,3,17]
    pool = pool_for(actor,cases,data,indices,selected,out)
    prefixes = [prefix_hash(p) for p,_ in pool];reports = {}
    for arm,strength in ARMS.items():
        model = select(actor,warm,strength); frozen = frozen_hash(actor)
        if not tree_equal(model.trainable_state_dict(),warm):raise AssertionError("warm start differs")
        zero_exact = None
        if strength == 0:
            prefix,codes,lengths = batch(pool,list(range(4)),model.device)
            a = torch.Generator(device=model.device).manual_seed(SEED+7)
            b = torch.Generator(device=model.device).manual_seed(SEED+7)
            wanted,wvalues = NativeImageLanguageModel.training_loss(model,prefix,codes,lengths,generator=a)
            got,gvalues = model.training_loss(prefix,codes,lengths,generator=b)
            zero_exact = torch.equal(wanted,got) and all(torch.equal(wvalues[k],gvalues[k]) for k in ("flow","clean","stop"))
            if not zero_exact or not torch.equal(a.get_state(),b.get_state()):raise AssertionError("zero metric changes original objective")
            del wanted,got,wvalues,gvalues,prefix,codes,lengths
        optimizer = new_optimizer(model);sampler = ExactEpochSampler(4,SEED+1)
        generator = torch.Generator(device=model.device).manual_seed(SEED+2)
        rows = [update(model,optimizer,pool,sampler,generator,0,schedule=False)]
        rows.append(update(model,optimizer,pool,sampler,generator,1,schedule=False,gradients=True))
        snapshot = out/"preflight"/arm/"resume_check.pt"
        save_checkpoint(snapshot,model,optimizer,sampler,generator,2,identity,arm)
        expected = update(model,optimizer,pool,sampler,generator,2,schedule=False)
        expected_model = model.trainable_state_dict();expected_optimizer = cpu_tree(optimizer.state_dict())
        saved = torch.load(snapshot,map_location="cpu",weights_only=True)
        restore_values(saved,actor,optimizer,sampler,generator)
        actual = update(model,optimizer,pool,sampler,generator,2,schedule=False)
        if (actual != expected or not tree_equal(model.trainable_state_dict(),expected_model)
                or not tree_equal(cpu_tree(optimizer.state_dict()),expected_optimizer)):
            raise AssertionError("physical-metric update does not resume exactly")
        checks = condition_checks(model,pool)
        prefix,_ = actor.prefill(prompt(data,indices[cases[selected[0]]["source"]],cases[selected[0]]))
        if prefix_hash(prefix) != prefixes[0]:raise AssertionError("updated suffix changed prompt prefill")
        del prefix
        sample = model.generate(pool[0][0].to(model.device),generator=torch.Generator(device=model.device).manual_seed(SEED+20))
        check_answer(sample)
        if not len(sample["images"]):raise AssertionError("preflight must exercise emitted-image feedback")
        for pixels,digest in zip(sample["images"],sample["feedback_sha256"]):
            if tensor_digest(model.writer.encode_targets(pixels[None].to(model.device))) != digest:
                raise AssertionError("feedback is not the actual emitted pixels")
        save_image_answer(sample,out/"preflight"/arm/"output")
        if frozen_hash(actor) != frozen or [prefix_hash(p) for p,_ in pool] != prefixes:
            raise AssertionError("frozen parameters or prompt cache changed")
        reports[arm] = dict(checks,rows=rows+[actual],strength_zero_objective_exact=zero_exact,
            exact_model_optimizer_noise_resume=True,frozen_parameters_bit_exact=True,prompt_kv_bit_exact=True,
            physical_feedback_replayed=True,generated_blocks=len(sample["images"]),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            complete_actor_parameters=sum(p.numel() for p in actor.parameters()))
        write(out/"preflight"/arm/"checks.json",reports[arm])
        del optimizer,saved,expected_model,expected_optimizer
        gc.collect();torch.cuda.empty_cache()
    peak = torch.cuda.max_memory_allocated()/1024**3
    if peak>=24:raise RuntimeError("physical metric exceeds single-card allocated-memory limit")
    write(out/"preflight/report.json",dict(identity=identity,passed=True,arms=reports,peak_allocated_vram_gib=peak,
        training_fit_measured=False,language_quality_claim=False,general_ilm_goal_complete=False))
    write(out/"status.json",dict(stage="preflight_completed",pid=None))


@torch.no_grad()
def outputs(model,pool,cases,data,indices,out,*,assisted,binding):
    for i,(prefix,_) in enumerate(pool):
        path = out/f"{i:02}"
        expected = dict(position=i,source=cases[i]["source"],seed=20267420+cases[i]["pair"],checkpoint_binding=binding)
        if (path/"header.json").exists():
            header = json.loads((path/"header.json").read_text())
            if any(header[k]!=v for k,v in expected.items()):raise ValueError("retained output identity differs")
            for name,digest in header["files_sha256"].items():
                if sha(path/name)!=digest:raise ValueError("retained image evidence changed")
            continue
        if path.exists() and any(path.iterdir()):raise ValueError("preserve partial unbound output")
        free = model.generate(prefix.to(model.device),generator=torch.Generator(device=model.device).manual_seed(expected["seed"]))
        saved = save_image_answer(free,path)
        memory,codes,reference = targets(data,indices[cases[i]["source"]],cases[i])
        Image.fromarray(reference.numpy()).save(path/"reference.png")
        if assisted:
            images,trace = incremental_reference_stream(model,prefix.to(model.device),codes.to(model.device),expected["seed"])
            soft = model.writer.predict(codes.to(model.device))
            supplied = torch.where(soft<0,-torch.ones_like(soft),torch.ones_like(soft)).cpu()
            if (trace["stop_probabilities"][0]!=free["stop_probabilities"][0]
                    or (len(free["images"]) and not torch.equal(free["images"][:1],images[:1]))):
                raise AssertionError("common incremental first decision differs")
            torch.save(trace,path/"assisted_trace.pt")
            for name,value in (("assisted",images),("supplied",supplied)):
                torch.save(value,path/(name+".pt"));save_raster(path/(name+".png"),value)
        else:
            previous = ROOT/f"artifacts/adaptive_readout_v81eval_20261107/adapted/predictions/{i:02}/answer.pt"
            if not tree_equal(free,torch.load(previous,map_location="cpu",weights_only=True)):
                raise AssertionError("warm checkpoint does not replay original autonomous answer")
        write(path/"header.json",dict(**expected,output=saved,reader_memory_sha256=tensor_digest(memory[None]),
            files_sha256={p.name:sha(p) for p in sorted(path.iterdir())}))


def train(actor,warm,cases,data,indices,out,identity):
    pre = json.loads((out/"preflight/report.json").read_text())
    if not pre["passed"] or not same_identity(pre["identity"],identity):raise ValueError("matching actual preflight required")
    pool = pool_for(actor,cases,data,indices,list(range(32)),out)
    prefixes = [prefix_hash(p) for p,_ in pool]
    for arm,strength in ARMS.items():
        folder = out/arm
        if (folder/"report.json").exists():continue
        model = select(actor,warm,strength);optimizer = new_optimizer(model)
        sampler = ExactEpochSampler(32,SEED+1);generator = torch.Generator(device=model.device).manual_seed(SEED+2)
        checkpoint = folder/"checkpoint_latest.pt";step=0
        if checkpoint.exists():
            saved = torch.load(checkpoint,map_location="cpu",weights_only=True)
            if not same_identity(saved["identity"],identity) or saved["arm"]!=arm:raise ValueError("checkpoint identity changed")
            restore_values(saved,actor,optimizer,sampler,generator);step=saved["update"];del saved
        else:
            outputs(model,pool,cases,data,indices,folder/"before/predictions",assisted=False,binding=WARM_SHA)
        begun = time.perf_counter();folder.mkdir(parents=True,exist_ok=True)
        with (folder/"metrics.jsonl").open("a") as log:
            while step<UPDATES:
                row = update(model,optimizer,pool,sampler,generator,step,gradients=step==1);step+=1
                if step==1 or step%10==0:
                    log.write(json.dumps(row)+"\n");log.flush()
                    write(out/"status.json",dict(stage="training",pid=os.getpid(),arm=arm,**row))
                if step%250==0:save_checkpoint(checkpoint,model,optimizer,sampler,generator,step,identity,arm)
        if [prefix_hash(p) for p,_ in pool]!=prefixes:raise AssertionError("training changed prompt pool")
        model.eval()
        outputs(model,pool,cases,data,indices,folder/"after/predictions",assisted=True,binding=sha(checkpoint))
        references=[targets(data,indices[c["source"]],c)[2] for c in cases]
        rows,summary=metrics(folder/"after",cases,references)
        write(folder/"report.json",dict(identity=identity,arm=arm,metric_strength=strength,updates=step,draws=sampler.draws,
            checkpoint_sha256=sha(checkpoint),rows=rows,summary=summary,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            complete_actor_parameters=sum(p.numel() for p in actor.parameters()),elapsed_seconds=time.perf_counter()-begun,
            timing_scope="current training/final-output invocation, excluding shared setup/prefill and initial outputs",
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
            language_quality_claim=False,general_ilm_goal_complete=False))
        del optimizer;gc.collect();torch.cuda.empty_cache()
    write(out/"status.json",dict(stage="awaiting_cpu_verification",pid=None))


def verify(out,identity,cases):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized():raise ValueError("explicit CPU-only verifier required")
    torch.set_num_threads(4);_,data,indices=cases_and_data()
    writer=CodecBlockWriter(load_qualified_writing_codec(CODEC,torch.device("cpu")))
    summaries={};replays=0;checkpoint_states={}
    for arm in ARMS:
        folder=out/arm;report=json.loads((folder/"report.json").read_text())
        if not same_identity(report["identity"],identity) or report["checkpoint_sha256"]!=sha(folder/"checkpoint_latest.pt"):
            raise ValueError("final checkpoint/report changed")
        saved=torch.load(folder/"checkpoint_latest.pt",map_location="cpu",weights_only=True)
        if saved["update"]!=2000 or not saved["training_complete"] or saved["arm"]!=arm:raise ValueError("incomplete final checkpoint")
        checkpoint_states[arm]=(saved["pool_sampler"],saved["generator"]);del saved
        references=[]
        for i,case in enumerate(cases):
            if not torch.equal(prompt(data,indices[case["source"]],case),prompt(data,indices[case["source"]],case,independent=True)):
                raise ValueError("independent prompt reconstruction differs")
            memory,codes,reference=targets(data,indices[case["source"]],case);references.append(reference)
            for stage in ("before","after"):
                path=folder/stage/"predictions"/f"{i:02}";header=json.loads((path/"header.json").read_text())
                if (header["source"]!=case["source"] or header["position"]!=i
                        or header["seed"]!=20267420+case["pair"]
                        or header["reader_memory_sha256"]!=tensor_digest(memory[None])
                        or header["checkpoint_binding"]!=(WARM_SHA if stage=="before" else report["checkpoint_sha256"])):
                    raise ValueError("prediction source identity changed")
                for name,digest in header["files_sha256"].items():
                    if sha(path/name)!=digest:raise ValueError("retained physical output changed")
                free=torch.load(path/"answer.pt",map_location="cpu",weights_only=True);pixels=check_answer(free)
                if len(free["images"]) and not np.array_equal(pixels,np.asarray(Image.open(path/"answer.png"))):raise ValueError("free PNG differs")
                if not np.array_equal(reference.numpy(),np.asarray(Image.open(path/"reference.png"))):raise ValueError("reference PNG differs")
                if stage=="before":
                    previous=ROOT/f"artifacts/adaptive_readout_v81eval_20261107/adapted/predictions/{i:02}/answer.pt"
                    if not tree_equal(free,torch.load(previous,map_location="cpu",weights_only=True)):raise ValueError("initial autonomous replay differs")
                    replays+=1;continue
                trace=torch.load(path/"assisted_trace.pt",map_location="cpu",weights_only=True)
                if trace["reference_codes_sha256"]!=tensor_digest(codes) or len(trace["stop_probabilities"])!=len(codes)+1:
                    raise ValueError("assisted target/history binding differs")
                for name,coordinates in (("assisted",trace["coordinates"]),("supplied",codes)):
                    with torch.inference_mode():soft=writer.predict(coordinates)
                    expected=torch.where(soft<0,-torch.ones_like(soft),torch.ones_like(soft))
                    images=torch.load(path/(name+".pt"),map_location="cpu",weights_only=True)
                    if not torch.equal(images,expected) or not np.array_equal(raster(images),np.asarray(Image.open(path/(name+".png")))):
                        raise ValueError("independent control coordinate decode differs")
                assisted=torch.load(path/"assisted.pt",map_location="cpu",weights_only=True)
                if (trace["stop_probabilities"][0]!=free["stop_probabilities"][0]
                        or (len(free["images"]) and not torch.equal(free["images"][:1],assisted[:1]))):
                    raise ValueError("canonical first decision differs")
        rows,summary=metrics(folder/"after",cases,references,independent=True)
        assert_close(rows,report["rows"]);assert_close(summary,report["summary"]);summaries[arm]=summary
    if replays!=64 or not tree_equal(checkpoint_states["control"],checkpoint_states["metric"]):
        raise ValueError("matched initial replays, case sampler or explicit noise RNG differ")
    control,repaired=summaries["control"],summaries["metric"]
    gates=dict(free_exact_at_least_24=repaired["free_exact"]>=24,free_f1_at_least_098=repaired["free_f1"]>=.98,
        free_exact_exceeds_control=repaired["free_exact"]>control["free_exact"],free_f1_exceeds_control=repaired["free_f1"]>control["free_f1"])
    write(out/"verification.json",dict(identity=identity,summaries=summaries,gates=gates,training_fit_repair_pass=all(gates.values()),
        all_64_final_outputs_independently_scored=True,all_64_initial_generations_bit_exact=True,
        matched_case_sampler_and_noise_rng=True,assisted_coordinate_decodes=64,supplied_code_decodes=64,
        full_context_and_sampler_neural_forward_replayed=False,training_only=True,language_quality_claim=False,general_ilm_goal_complete=False))
    write(out/"status.json",dict(stage="completed_verified",pid=None))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"artifacts/decoder_metric_v82_20261107")
    parser.add_argument("--stage",choices=("preflight","train","verify"),required=True)
    args=parser.parse_args();out=args.out.resolve()
    previous=ROOT/"artifacts/adaptive_readout_v81r2_20261106"
    identity,cases=identity_for(previous);identity=bind(out,identity)
    try:
        if args.stage=="verify":verify(out,identity,cases);return
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise ValueError("explicit deterministic CUDA workspace required")
        with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);device=setup()
            write(out/"runtime_handoff.json",dict(pid=os.getpid(),stage=args.stage,gpu=0,noVNC_url=None,owned_gui_processes=[]))
            actor,provenance,old_image,old_adapters=build(ROOT/"artifacts/native_chinese_v80_20261106",device)
            del old_image,old_adapters
            saved=torch.load(previous/"adapted/checkpoint_latest.pt",map_location="cpu",weights_only=True)
            if saved["update"]!=2000 or not saved["training_complete"]:raise ValueError("completed warm checkpoint required")
            warm=saved["model"];del saved
            _,data,indices=cases_and_data();write(out/"provenance.json",provenance)
            dict(preflight=preflight,train=train)[args.stage](actor,warm,cases,data,indices,out,identity)
    except BaseException as exc:
        write(out/"failure.json",dict(stage=args.stage,error=repr(exc),language_quality_claim=False));raise


if __name__=="__main__":main()
