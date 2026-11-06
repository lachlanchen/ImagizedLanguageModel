#!/usr/bin/env python3
"""Train and audit a matched continuous-readout repair on actual Chinese images."""
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

from ilm.visual_lm.adaptive_image_continuation import (
    AdaptiveImageLanguageModel, adapter_parameters, install_suffix_adapters,
)
from ilm.visual_lm.clean_image_actor import tensor_digest
from ilm.visual_lm.codec_block_writer import CodecBlockWriter, load_qualified_writing_codec
from ilm.visual_lm.native_image_actor import load_trained_native_image_actor
from ilm.visual_lm.native_image_continuation import NativeImageLanguageModel, NativeImagePrefix
from ilm.visual_lm.semantic_writing_data import ExactEpochSampler
from scripts.audit_native_neural_transcripts_v80 import ready, write
from scripts.cache_semantic_images_v72 import ROOT, CODEC, CONTROLS, sha, committed_identity
from scripts.infer_semantic_ilm_v72 import save_image_answer
from scripts.probe_corpus_fit_v75 import metrics, save_raster
from scripts.probe_native_corpus_fit_v80 import (
    assisted_stream, cases_and_data, prepared, prompt, targets,
    SOURCES as DIAGNOSTIC_SOURCES,
)
from scripts.score_native_training_samples_v80 import check_answer, raster
from scripts.train_clean_corpus_v75 import setup
from scripts.train_native_chinese_v80 import (
    SOURCES as TRAINING_SOURCES, batch, new_optimizer, restore_values, same_identity,
)
from scripts.verify_codec_language_v68 import assert_close

SEED = 20268100
UPDATES = 2000
WARM_SHA = "2e267b023118552e2462971c5f8e0dc49b57f4e401ef7ba58082c05fc85fc0f5"
PLAN = "references/adaptive_readout_v81r2_protocol.md"
SOURCES = tuple(dict.fromkeys(("scripts/train_adaptive_readout_v81.py", PLAN,
    "references/adaptive_readout_v81_protocol.md",
    "ilm/visual_lm/adaptive_image_continuation.py", *TRAINING_SOURCES, *DIAGNOSTIC_SOURCES)))


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple,list)):
        return len(a) == len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return a == b


def experiment_identity(source):
    completed = ready(source)
    diagnostic = source/"corpus_fit_v80r2"
    verification = json.loads((diagnostic/"verification.json").read_text())
    if (completed["checkpoint_sha256"] != WARM_SHA
            or verification["report_sha256"] != sha(diagnostic/"report.json")
            or not verification["all_32_free_and_assisted_outputs_independently_scored"]):
        raise ValueError("completed authenticated native failure diagnosis required")
    _, cases = prepared(source, diagnostic)
    return dict(**committed_identity(SOURCES), purpose="matched_adaptive_image_readout",
        original_pipeline=completed, original_diagnosis_sha256=sha(diagnostic/"verification.json"),
        cohort_sha256=sha(diagnostic/"cases.json"), arms=["control","adapted"], seed=SEED,
        updates_per_arm=UPDATES, batch_size=4, training_examples=32, flow_steps=32,
        maximum_blocks=128, warm_checkpoint_sha256=WARM_SHA,
        query_and_transition_composite_intervention=True, training_fit_only=True,
        external_pretraining=True, tokenizer=False, teacher_service=False), cases


def bind(out, identity):
    path = out/"identity.json"
    if path.exists():
        value = json.loads(path.read_text())
        if not same_identity(value, identity):
            raise ValueError("repair source or configuration changed; preserve the run")
        return value
    if out.exists() and any(p.name != "launch.log" for p in out.iterdir()):
        raise ValueError("preserve the unbound output directory")
    write(path, identity)
    return identity


def build(source, device):
    local = ROOT.parent/"LocalLLM/.local"
    actor, provenance = load_trained_native_image_actor(checkpoint=source/"training/checkpoint_latest.pt",
        controls=CONTROLS, codec=CODEC, ollama_root=local/"models/ollama",
        gguf_source=local/"tools/llama.cpp/gguf-py", device=device)
    context = actor.writing.continuation.context
    install_suffix_adapters(context)
    image_state = actor.writing.trainable_state_dict()
    adapters = {n:p.detach().cpu().clone() for n,p in adapter_parameters(context)}
    if len(adapters) != 216 or any(bool(v.any()) for n,v in adapters.items() if n.endswith(".up")):
        raise ValueError("the original zero-output allocated adapters are required")
    return actor, provenance, image_state, adapters


def select_arm(actor, arm, image_state, adapters):
    device = actor.device
    context, writer = actor.writing.continuation.context, actor.writing.writer
    with torch.no_grad():
        for name, parameter in adapter_parameters(context):
            parameter.copy_(adapters[name].to(device))
    kwargs = dict(input_scale=float(image_state["input_scale"]))
    if arm == "adapted":
        model = AdaptiveImageLanguageModel(context, writer, query_seed=SEED+3, **kwargs).to(device)
        model.load_native_image_state(image_state)
    elif arm == "control":
        model = NativeImageLanguageModel(context, writer, **kwargs).to(device)
        model.load_trainable_state_dict(image_state)
    else:
        raise ValueError("unknown matched arm")
    actor.writing = model
    actor.train()
    return model


def pool_for(actor, cases, data, indices, selected, out):
    pool = []
    for position, i in enumerate(selected):
        case = cases[i]
        pixels = prompt(data, indices[case["source"]], case)
        prefix, metadata = actor.prefill(pixels)
        memory, codes, _ = targets(data, indices[case["source"]], case)
        if metadata["reader_memory_sha256"] != tensor_digest(memory[None]) or not torch.equal(prefix.initial_state.cpu()[0,0], memory[-1]):
            raise ValueError("base-only prompt prefill differs from authenticated source memory")
        pool.append((prefix.to("cpu"), codes))
        del prefix
        write(out/"status.json",dict(stage="prefill",pid=os.getpid(),completed=position+1,total=len(selected)))
    return pool


def prefix_hash(prefix):
    return dict(initial=tensor_digest(prefix.initial_state),rotary=tensor_digest(prefix.rotary_offsets),
        values=[[tensor_digest(v) for v in pair] for pair in prefix.key_values])


def frozen_hash(actor):
    return {name:tensor_digest(value) for name,value in actor.named_parameters() if not value.requires_grad}


def update(model, optimizer, pool, sampler, generator, step, *, schedule=True, gradients=False):
    chosen = sampler.take(4)
    prefix, codes, lengths = batch(pool, chosen, model.device)
    if schedule:
        factor = (step+1)/100 if step < 100 else .1+.9*.5*(1+math.cos(math.pi*(step-100)/(UPDATES-100)))
        optimizer.param_groups[0]["lr"] = 1e-4*factor
    optimizer.zero_grad(set_to_none=True)
    loss, values = model.training_loss(prefix, codes, lengths, generator=generator)
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite continuous readout loss")
    loss.backward()
    groups = {}
    if gradients:
        selections = {name:[p for n,p in model.named_parameters() if n.startswith(name)]
            for name in ("image_input.","flow.","stop.")}
        if isinstance(model, AdaptiveImageLanguageModel):
            selections["start_query"] = [model.start_query]
            for ending in (".down",".up"):
                selections["adapter"+ending] = [p for n,p in adapter_parameters(model.continuation.context) if n.endswith(ending)]
        for name, parameters in selections.items():
            groups[name] = bool(parameters) and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters) and any(bool(p.grad.abs().max()>0) for p in parameters)
        if not all(groups.values()):
            raise AssertionError("missing/nonfinite learned readout gradients: "+str(groups))
    parameters = [p for p in model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
    optimizer.step()
    return dict(update=step+1,indices=chosen,loss=float(loss.detach()),flow=float(values["flow"]),
        clean_mse=float(values["clean"]),stop=float(values["stop"]),gradient_norm=float(norm),
        lr=optimizer.param_groups[0]["lr"],gradient_groups=groups)


def save_checkpoint(path, model, optimizer, sampler, generator, step, identity, arm):
    value = dict(model=model.trainable_state_dict(),optimizer=cpu_tree(optimizer.state_dict()),
        pool_sampler=sampler.state_dict(),generator=generator.get_state(),cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(),update=step,identity=identity,arm=arm,
        training_complete=step==UPDATES,frozen_weights_saved=False)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(".partial")
    torch.save(value,temporary)
    if path.exists():
        path.replace(path.with_name("checkpoint_previous.pt"))
    temporary.replace(path)


@torch.no_grad()
def condition_checks(model, pool):
    prefix, codes, lengths = batch(pool, list(range(len(pool))), model.device)
    full = model.conditions(prefix,codes,checkpoint_layers=False)
    altered = codes.clone();altered[:,-1] = -altered[:,-1]
    future = model.conditions(prefix,altered,checkpoint_layers=False)
    causal = float((full[:,:-1]-future[:,:-1]).abs().max())
    padding, incremental, scaled_padding, scaled_incremental = 0.,0.,0.,0.
    def scaled_error(actual, reference):
        # A normwise comparison accommodates cancellation in individual
        # components. The absolute discrepancy is still reported separately.
        bound = torch.maximum(torch.full_like(reference[..., :1],1e-4),
            reference.abs().amax(-1,keepdim=True)*1e-5)
        return float(((actual-reference).abs()/bound).max())
    for i,(original,target) in enumerate(pool):
        own = original.to(model.device);target = target.to(model.device)
        states = model.conditions(own,target[None],checkpoint_layers=False)
        padding = max(padding,float((full[i,:len(states[0])]-states[0]).abs().max()))
        scaled_padding = max(scaled_padding,scaled_error(full[i,:len(states[0])],states[0]))
        if isinstance(model, AdaptiveImageLanguageModel):
            current = model.start_prefix(own)
        else:
            current = own
        observed = [current.initial_state[:,0].float()]
        for block in target:
            hidden, values = model.continuation(model.embed_images(block[None,None]),current,
                checkpoint_layers=False,collect_cache=True)
            current = NativeImagePrefix(values,hidden[:,-1:],current.rotary_offsets+2,current.blocked_before)
            observed.append(hidden[:,-1])
        got = torch.stack(observed,dim=1)
        incremental = max(incremental,float((got-states).abs().max()))
        scaled_incremental = max(scaled_incremental,scaled_error(got,states))
        extra = int(isinstance(model,AdaptiveImageLanguageModel))
        if current.length != own.length+extra+2*len(target) or not torch.equal(current.rotary_offsets,own.rotary_offsets+extra+2*len(target)):
            raise AssertionError("read-query or image-history position drift")
    if causal != 0 or scaled_padding > 1 or scaled_incremental > 1:
        raise AssertionError("native causal/normwise-padding/incremental mismatch: "+str((causal,scaled_padding,scaled_incremental)))
    return dict(future_max_abs_error=causal,padding_max_abs_error=padding,incremental_max_abs_error=incremental,
        padding_max_scaled_error=scaled_padding,incremental_max_scaled_error=scaled_incremental,
        comparison="per-state infinity norm: max(1e-4, 1e-5 * reference infinity norm)",
        original_absolute_1e4_check_pass=padding<=1e-4 and incremental<=1e-4)


def preflight(actor, image_state, adapters, cases, data, indices, out, identity):
    path = out/"preflight/report.json"
    if path.exists():
        raise FileExistsError("preserve completed preflight")
    selected = [0,2,3,17]
    pool = pool_for(actor,cases,data,indices,selected,out)
    original_prefixes = [prefix_hash(prefix) for prefix,_ in pool]
    reports = {}
    for arm in ("control","adapted"):
        model = select_arm(actor,arm,image_state,adapters)
        original_frozen = frozen_hash(actor)
        optimizer = new_optimizer(model)
        sampler = ExactEpochSampler(4,SEED+1)
        generator = torch.Generator(device=model.device).manual_seed(SEED+2)
        rows = [update(model,optimizer,pool,sampler,generator,0,schedule=False)]
        rows.append(update(model,optimizer,pool,sampler,generator,1,schedule=False,gradients=True))
        snapshot = out/"preflight"/arm/"resume_check.pt"
        save_checkpoint(snapshot,model,optimizer,sampler,generator,2,identity,arm)
        expected = update(model,optimizer,pool,sampler,generator,2,schedule=False)
        expected_model = model.trainable_state_dict();expected_optimizer = cpu_tree(optimizer.state_dict())
        value = torch.load(snapshot,map_location="cpu",weights_only=True)
        restore_values(value,actor,optimizer,sampler,generator)
        actual = update(model,optimizer,pool,sampler,generator,2,schedule=False)
        if actual != expected or not tree_equal(model.trainable_state_dict(),expected_model) or not tree_equal(cpu_tree(optimizer.state_dict()),expected_optimizer):
            raise AssertionError("readout model/Adam/sampler/noise resume differs")
        checks = condition_checks(model,pool)
        # Re-run actual base prefill after nonzero adapter updates. Every
        # original KV tensor, not only its final memory, must remain exact.
        prefix, metadata = actor.prefill(prompt(data,indices[cases[selected[0]]["source"]],cases[selected[0]]))
        if prefix_hash(prefix) != original_prefixes[0]:
            raise AssertionError("adapter updates changed base-only prompt KV")
        del prefix
        sample = model.generate(pool[0][0].to(model.device),
            generator=torch.Generator(device=model.device).manual_seed(SEED+20))
        check_answer(sample)
        if not len(sample["images"]):
            raise AssertionError("preflight did not exercise actual emitted-pixel feedback")
        for image,digest in zip(sample["images"],sample["feedback_sha256"]):
            actual_codes = model.writer.encode_targets(image[None].to(model.device))
            if tensor_digest(actual_codes) != digest:
                raise AssertionError("feedback is not the encoding of the actual emitted pixels")
        save_image_answer(sample,out/"preflight"/arm/"output")
        if frozen_hash(actor) != original_frozen or [prefix_hash(prefix) for prefix,_ in pool] != original_prefixes:
            raise AssertionError("frozen weights or immutable CPU prompt pool changed")
        reports[arm] = dict(checks,rows=rows+[actual],exact_optimizer_noise_resume=True,
            base_prefill_after_updates_bit_exact=True,frozen_parameters_bit_exact=True,
            actual_pixel_feedback_replayed=True,generated_blocks=len(sample["images"]),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            complete_actor_parameters=sum(p.numel() for p in actor.parameters()))
        write(out/"preflight"/arm/"checks.json",reports[arm])
        del optimizer,value,expected_model,expected_optimizer
        gc.collect();torch.cuda.empty_cache()
    peak = torch.cuda.max_memory_allocated()/1024**3
    if peak >= 24:
        raise RuntimeError("readout preflight exceeds single-card allocated-memory ceiling")
    write(path,dict(identity=identity,passed=True,arms=reports,source_cases=[cases[i]["source"] for i in selected],
        peak_allocated_vram_gib=peak,training_fit_measured=False,language_quality_claim=False,general_ilm_goal_complete=False))


@torch.no_grad()
def outputs(actor, pool, cases, data, indices, out, *, assisted, binding):
    for i,(prefix,_) in enumerate(pool):
        folder = out/f"{i:02}"
        seed = 20267420+cases[i]["pair"]
        expected = dict(position=i,source=cases[i]["source"],seed=seed,checkpoint_binding=binding)
        if (folder/"header.json").exists():
            header = json.loads((folder/"header.json").read_text())
            if any(header[k] != v for k,v in expected.items()):
                raise ValueError("retained answer identity differs")
            for name,digest in header["files_sha256"].items():
                if sha(folder/name) != digest:
                    raise ValueError("retained answer payload changed")
            continue
        if folder.exists() and any(folder.iterdir()):
            raise ValueError("preserve unbound partial answer output")
        result = actor.writing.generate(prefix.to(actor.device),
            generator=torch.Generator(device=actor.device).manual_seed(seed))
        saved = save_image_answer(result,folder)
        memory,codes,reference = targets(data,indices[cases[i]["source"]],cases[i])
        Image.fromarray(reference.numpy()).save(folder/"reference.png")
        if assisted:
            images, trace = assisted_stream(actor.writing,prefix.to(actor.device),codes.to(actor.device),seed)
            supplied = actor.writing.writer.predict(codes.to(actor.device))
            supplied = torch.where(supplied<0,-torch.ones_like(supplied),torch.ones_like(supplied)).cpu()
            torch.save(images,folder/"assisted.pt");torch.save(trace,folder/"assisted_trace.pt")
            torch.save(supplied,folder/"supplied.pt")
            save_raster(folder/"assisted.png",images);save_raster(folder/"supplied.png",supplied)
            if (trace["stop_probabilities"][0] != result["stop_probabilities"][0]
                    or (len(result["images"]) and not torch.equal(result["images"][:1],images[:1]))):
                raise AssertionError("first free and correct-history decisions differ")
        write(folder/"header.json",dict(**expected,output=saved,
            reader_memory_sha256=tensor_digest(memory[None]),files_sha256={p.name:sha(p) for p in folder.iterdir()}))


def train(actor,image_state,adapters,cases,data,indices,out,identity):
    pre = json.loads((out/"preflight/report.json").read_text())
    if not pre["passed"] or not same_identity(pre["identity"],identity):
        raise ValueError("matching real-model preflight required")
    pool = pool_for(actor,cases,data,indices,list(range(32)),out)
    initial_prefixes = [prefix_hash(p) for p,_ in pool]
    for arm in ("control","adapted"):
        folder = out/arm
        if (folder/"report.json").exists():
            continue
        model = select_arm(actor,arm,image_state,adapters)
        optimizer = new_optimizer(model);sampler = ExactEpochSampler(32,SEED+1)
        generator = torch.Generator(device=model.device).manual_seed(SEED+2)
        checkpoint = folder/"checkpoint_latest.pt"
        step = 0
        if checkpoint.exists():
            value = torch.load(checkpoint,map_location="cpu",weights_only=True)
            if not same_identity(value["identity"],identity) or value["arm"] != arm:
                raise ValueError("saved readout checkpoint identity differs")
            restore_values(value,actor,optimizer,sampler,generator);step = value["update"]
            del value
        else:
            outputs(actor,pool,cases,data,indices,folder/"before/predictions",assisted=False,
                binding="warm:"+WARM_SHA+":"+arm)
        begun = time.perf_counter()
        folder.mkdir(parents=True,exist_ok=True)
        with (folder/"metrics.jsonl").open("a") as log:
            while step < UPDATES:
                row = update(model,optimizer,pool,sampler,generator,step,gradients=step==1)
                step += 1
                if step == 1 or step % 10 == 0:
                    log.write(json.dumps(row)+"\n");log.flush()
                    write(out/"status.json",dict(stage="training",pid=os.getpid(),arm=arm,**row))
                if step % 250 == 0 or step == UPDATES:
                    save_checkpoint(checkpoint,model,optimizer,sampler,generator,step,identity,arm)
        if [prefix_hash(p) for p,_ in pool] != initial_prefixes:
            raise AssertionError("training changed fixed prompt caches")
        model.eval()
        outputs(actor,pool,cases,data,indices,folder/"after/predictions",assisted=True,binding=sha(checkpoint))
        references = [targets(data,indices[c["source"]],c)[2] for c in cases]
        rows,summary = metrics(folder/"after",cases,references)
        write(folder/"report.json",dict(identity=identity,arm=arm,updates=step,draws=sampler.draws,
            checkpoint_sha256=sha(checkpoint),rows=rows,summary=summary,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            complete_actor_parameters=sum(p.numel() for p in actor.parameters()),
            elapsed_seconds=time.perf_counter()-begun,
            timing_scope="current training and final-output invocation, excluding shared setup and initial outputs",
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
            files_sha256={str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*'))
                if p.is_file() and p.suffix in (".png",".json")},
            language_quality_claim=False,general_ilm_goal_complete=False))
        del optimizer
        gc.collect();torch.cuda.empty_cache()
    write(out/"status.json",dict(stage="awaiting_cpu_verification",pid=None))


def verify(source,out,identity,cases):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("explicit CPU-only verification required")
    torch.set_num_threads(4)
    _,data,indices = cases_and_data()
    writer = CodecBlockWriter(load_qualified_writing_codec(CODEC,torch.device("cpu")))
    summaries = {}
    original_free_matches = 0
    for arm in ("control","adapted"):
        folder = out/arm;report = json.loads((folder/"report.json").read_text())
        if not same_identity(report["identity"],identity) or report["checkpoint_sha256"] != sha(folder/"checkpoint_latest.pt"):
            raise ValueError("final repair checkpoint changed")
        for relative,digest in report["files_sha256"].items():
            if sha(folder/relative) != digest:
                raise ValueError("retained repair evidence changed")
        references = []
        for i,case in enumerate(cases):
            if not torch.equal(prompt(data,indices[case["source"]],case),prompt(data,indices[case["source"]],case,independent=True)):
                raise ValueError("independent prompt reconstruction differs")
            _,codes,reference = targets(data,indices[case["source"]],case);references.append(reference)
            for stage in ("before","after"):
                path = folder/stage/"predictions"/f"{i:02}"
                header = json.loads((path/"header.json").read_text())
                for name,digest in header["files_sha256"].items():
                    if sha(path/name) != digest:
                        raise ValueError("retained answer payload changed")
                result = torch.load(path/"answer.pt",map_location="cpu",weights_only=True)
                pixels = check_answer(result)
                if len(result["images"]) and not np.array_equal(pixels,np.asarray(Image.open(path/"answer.png"))):
                    raise ValueError("answer PNG differs from emitted pixels")
                if not np.array_equal(np.asarray(Image.open(path/"reference.png")),reference.numpy()):
                    raise ValueError("complete owning reference differs")
                if stage == "before":
                    if arm == "control":
                        previous = torch.load(source/f"corpus_fit_v80r2/predictions/{i:02}/answer.pt",map_location="cpu",weights_only=True)
                        if (not torch.equal(result["images"],previous["images"])
                                or result["stop_probabilities"] != previous["stop_probabilities"]
                                or result["feedback_sha256"] != previous["feedback_sha256"]):
                            raise ValueError("unchanged control failed to reproduce original native generation")
                        original_free_matches += 1
                    continue
                trace = torch.load(path/"assisted_trace.pt",map_location="cpu",weights_only=True)
                with torch.inference_mode():
                    supplied = writer.predict(codes)
                    decoded = writer.predict(trace["coordinates"])
                for name,soft in (("supplied",supplied),("assisted",decoded)):
                    images = torch.load(path/(name+".pt"),map_location="cpu",weights_only=True)
                    wanted = torch.where(soft<0,-torch.ones_like(soft),torch.ones_like(soft))
                    if not torch.equal(images,wanted) or not np.array_equal(raster(images),np.asarray(Image.open(path/(name+".png")))):
                        raise ValueError("independent control-coordinate decode differs")
                if (trace["reference_codes_sha256"] != tensor_digest(codes)
                        or len(trace["stop_probabilities"]) != len(codes)+1
                        or trace["stop_probabilities"][0] != result["stop_probabilities"][0]):
                    raise ValueError("correct-history binding or initial stop differs")
        rows,summary = metrics(folder/"after",cases,references,independent=True)
        assert_close(rows,report["rows"]);assert_close(summary,report["summary"])
        summaries[arm] = summary
    repaired = summaries["adapted"]
    passed = (repaired["free_exact"] >= 24 and repaired["free_f1"] >= .98
        and repaired["free_exact"] > summaries["control"]["free_exact"]
        and repaired["free_f1"] > summaries["control"]["free_f1"])
    write(out/"verification.json",dict(identity=identity,summaries=summaries,training_fit_repair_pass=passed,
        all_64_final_outputs_independently_scored=True,supplied_code_decodes=64,assisted_coordinate_decodes=64,
        all_32_original_control_generations_bit_exact=original_free_matches==32,
        full_context_and_sampler_neural_forward_replayed=False,training_only=True,
        language_quality_claim=False,general_ilm_goal_complete=False))
    write(out/"status.json",dict(stage="completed_verified",pid=None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT/"artifacts/native_chinese_v80_20261106")
    parser.add_argument("--run",type=Path,default=ROOT/"artifacts/adaptive_readout_v81r2_20261106")
    parser.add_argument("--stage",choices=("preflight","train","verify"),required=True)
    args = parser.parse_args();source = args.source.resolve();out = args.run.resolve()
    if source == out or out.is_relative_to(source):
        raise ValueError("preserve the original source run in a separate directory")
    identity,cases = experiment_identity(source);identity = bind(out,identity)
    try:
        if args.stage == "verify":
            verify(source,out,identity,cases)
            return
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python")
        with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            device = setup()
            write(out/"runtime_handoff.json",dict(pid=os.getpid(),stage=args.stage,gpu=0,
                noVNC_url=None,owned_gui_processes=[]))
            _,data,indices = cases_and_data()
            actor,provenance,image_state,adapters = build(source,device)
            write(out/"provenance.json",provenance)
            if args.stage == "preflight":
                preflight(actor,image_state,adapters,cases,data,indices,out,identity)
                write(out/"status.json",dict(stage="preflight_completed",pid=None))
            else:
                train(actor,image_state,adapters,cases,data,indices,out,identity)
    except BaseException as error:
        write(out/"failure.json",dict(stage=args.stage,error=repr(error),language_quality_claim=False))
        raise


if __name__ == "__main__":
    main()
