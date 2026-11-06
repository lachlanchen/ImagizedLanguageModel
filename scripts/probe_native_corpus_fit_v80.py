#!/usr/bin/env python3
"""Diagnose native free writing, correct-history prediction and reconstruction."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from ilm.visual_lm.clean_image_actor import tensor_digest
from ilm.visual_lm.codec_block_writer import CodecBlockWriter, load_qualified_writing_codec
from ilm.visual_lm.native_image_actor import load_trained_native_image_actor
from scripts.audit_native_neural_transcripts_v80 import PipelinePending, ready, write
from scripts.cache_semantic_images_v72 import ROOT, CODEC, CONTROLS, sha, pixels_sha, committed_identity
from scripts.infer_semantic_ilm_v72 import save_image_answer
from scripts.probe_corpus_fit_v75 import COHORT, TIMES, fixed_cases, latent_errors, metrics, save_raster
from scripts.score_native_training_samples_v80 import check_answer, raster
from scripts.train_clean_corpus_v75 import setup
from scripts.train_native_chinese_v80 import ChineseImages
from scripts.verify_codec_language_v68 import assert_close, independent_actor_prompt

PLAN = "references/native_corpus_fit_v80_protocol.md"
COHORT_SHA = "86809a7b29bc57c84ee4991d10d320e04031cf56da2ad6d49724e693aa20f884"
SEED = 20267420
SOURCES = (
    "scripts/probe_native_corpus_fit_v80.py", PLAN,
    "scripts/probe_corpus_fit_v75.py", "references/evidence/clean_learning_v74_result.json",
    "scripts/train_native_chinese_v80.py", "scripts/audit_native_chinese_v80.py",
    "scripts/audit_native_neural_transcripts_v80.py", "scripts/score_native_training_samples_v80.py",
    "scripts/cache_semantic_images_v72.py", "scripts/train_clean_corpus_v75.py",
    "scripts/infer_semantic_ilm_v72.py", "scripts/verify_codec_language_v68.py",
    "ilm/visual_lm/native_image_actor.py", "ilm/visual_lm/native_image_continuation.py",
    "ilm/visual_lm/codec_block_writer.py", "ilm/visual_lm/clean_writing_flow.py",
    "ilm/visual_lm/complete_writing_data.py", "ilm/visual_lm/semantic_writing_data.py",
)


def cpu_only():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("explicit CPU-only preparation or verification required")
    torch.set_num_threads(4)


def cases_and_data():
    if sha(COHORT) != COHORT_SHA:
        raise ValueError("the existing fixed training cohort changed")
    cases = fixed_cases()
    data = ChineseImages()
    indices = {name:i for i,(name,_) in enumerate(data.rows)}
    if len({c["source"] for c in cases}) != 32:
        raise ValueError("all 32 distinct original training questions required")
    for i, case in enumerate(cases):
        if (case["source"] not in indices
                or case["context_sha256"] != cases[i ^ 1]["context_sha256"]
                or case["target_sha256"] == cases[i ^ 1]["target_sha256"]):
            raise ValueError("original distinct-answer same-passage pair changed")
    return cases, data, indices


def prompt(data, index, case, *, independent=False):
    if sha(data.images.path / case["source"]) != case["source_sha256"]:
        raise ValueError("owning training image changed")
    if independent:
        raw = data.images.prompt(data.image_index[case["source"]])[0,0]
        raw = ((raw+1)*127.5).to(torch.uint8)
        value = independent_actor_prompt(dict(prompt=raw))
    else:
        value = data.prompt(index)
    if pixels_sha(value) != case["prompt_sha256"]:
        raise ValueError("actual prompt layout differs from original")
    return value


def targets(data, index, case):
    relative = data.codes.names[data.rows[index][1]]
    if relative != case["cache_record"] or sha(data.codes.path / relative) != case["cache_record_sha256"]:
        raise ValueError("owning image-code training record changed")
    memory, codes = data.target(index)
    image = data.images.target(data.image_index[case["source"]])[..., :-128]
    if pixels_sha(image) != case["target_sha256"] or image.shape[-1] != 64*len(codes) or len(codes) != case["blocks"]:
        raise ValueError("complete original writing target changed")
    return memory, codes, ((image[0,0]+1)*127.5).to(torch.uint8)


def prepare(run, out):
    cpu_only()
    identity = dict(**committed_identity(SOURCES),training_identity_sha256=sha(run/"identity.json"),
        cohort_sha256=COHORT_SHA,seed=SEED,first_block_noise_seed=20267597,times=TIMES,
        actual_model_calls=0,training_cases_only=True,primary_gates_unchanged=True)
    if (out/"identity.json").exists():
        previous = json.loads((out/"identity.json").read_text())
        if {k:v for k,v in identity.items() if k!="source_commit"} != {k:v for k,v in previous.items() if k!="source_commit"}:
            raise ValueError("preserve the earlier diagnostic identity")
        identity = previous
    elif out.exists() and any(out.iterdir()):
        raise ValueError("preserve the nonempty diagnostic directory")
    cases, data, indices = cases_and_data()
    rows = []
    for case in cases:
        index = indices[case["source"]]
        image = prompt(data,index,case)
        if not torch.equal(image,prompt(data,index,case,independent=True)):
            raise ValueError("independent prompt reconstruction differs")
        memory, codes, reference = targets(data,index,case)
        rows.append(dict(**case,memory_sha256=tensor_digest(memory[None]),
            codes_sha256=tensor_digest(codes),prompt_shape=list(image.shape),reference_shape=list(reference.shape)))
    for name,value in (("identity.json",identity),("cases.json",rows)):
        path = out/name
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError("prepared diagnostic source binding changed")
        write(path,value)
    write(out/"preparation.json",dict(identity_sha256=sha(out/"identity.json"),cases_sha256=sha(out/"cases.json"),
        cases=32,training_only=True,source_and_code_hashes_verified=True,
        independent_prompt_reconstructions=32,actual_model_calls=0,general_ilm_goal_complete=False))
    print((out/"preparation.json").read_text())


def prepared(run, out):
    identity = json.loads((out/"identity.json").read_text())
    receipt = json.loads((out/"preparation.json").read_text())
    if (identity["training_identity_sha256"] != sha(run/"identity.json")
            or identity["cohort_sha256"] != COHORT_SHA or sha(COHORT) != COHORT_SHA
            or receipt["identity_sha256"] != sha(out/"identity.json")
            or receipt["cases_sha256"] != sha(out/"cases.json") or receipt["cases"] != 32):
        raise ValueError("prepared diagnostic identity differs")
    for name,digest in identity["sources"].items():
        if sha(ROOT/name) != digest:
            raise ValueError("prepared diagnostic implementation changed")
    return identity, json.loads((out/"cases.json").read_text())


@torch.inference_mode()
def assisted_stream(model, prefix, codes, seed):
    # Full teacher history is causal; no predicted pixel enters this control.
    states = model.conditions(prefix,codes[None],checkpoint_layers=False)
    generator = torch.Generator(device=model.device).manual_seed(seed)
    images, coordinates, probabilities = [], [], []
    for position in range(len(codes)+1):
        condition = states[:,position].float()
        probabilities.append(model.stop(condition).float().sigmoid().item())
        if position == len(codes):
            break
        _, endpoint = model.flow.sample_endpoint(condition,generator=generator,steps=model.flow_steps)
        normalized = F.layer_norm(endpoint.reshape(-1,768),(768,),
            eps=model.writer.codec.config.layer_norm_epsilon).reshape(1,2,768)
        soft = model.writer.predict(normalized)
        images.append(torch.where(soft<0,-torch.ones_like(soft),torch.ones_like(soft)).cpu())
        coordinates.append(normalized.cpu())
    return torch.cat(images),dict(states=states.cpu(),coordinates=torch.cat(coordinates),
        stop_probabilities=probabilities,reference_codes_sha256=tensor_digest(codes),
        reference_history_supplied=True,reference_length_supplied=True)


@torch.inference_mode()
def evaluate(run, out):
    completed = ready(run)  # Guard before initialization, allocation or new output.
    if (out/"report.json").exists():
        raise FileExistsError("completed diagnosis is immutable; use verify")
    identity, cases = prepared(run,out)
    execution = dict(preparation_identity_sha256=sha(out/"identity.json"),completed_original_pipeline=completed)
    path = out/"execution_identity.json"
    if path.exists() and json.loads(path.read_text()) != execution:
        raise ValueError("completed checkpoint or original audit changed")
    _, data, indices = cases_and_data()
    checkpoint = run/"training/checkpoint_latest.pt"
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if ready(run) != completed:
            raise ValueError("original pipeline changed before GPU startup")
        device = setup();begun = time.perf_counter()
        write(path,execution)
        local = ROOT.parent/"LocalLLM/.local"
        actor, provenance = load_trained_native_image_actor(checkpoint=checkpoint,controls=CONTROLS,codec=CODEC,
            ollama_root=local/"models/ollama",gguf_source=local/"tools/llama.cpp/gguf-py",device=device)
        setup_seconds = time.perf_counter()-begun
        references, conditions, first_targets, inventory = [], [], [], {}
        for i, case in enumerate(cases):
            folder = out/"predictions"/f"{i:02}"
            source = prompt(data,indices[case["source"]],case)
            seed = SEED+case["pair"]
            expected = dict(position=i,source=case["source"],seed=seed,prompt_sha256=pixels_sha(source),
                execution_identity_sha256=sha(path))
            if not (folder/"header.json").exists():
                if folder.exists() and any(folder.iterdir()):
                    raise ValueError("retain the unbound partial diagnostic and investigate it")
                prefix, metadata = actor.prefill(source)
                free = actor.writing.generate(prefix,generator=torch.Generator(device=device).manual_seed(seed))
                free.update(image_input_image_output=True,reader_memory_sha256=metadata["reader_memory_sha256"])
                saved = save_image_answer(free,folder)
                # Targets become available to the controls after the autonomous call.
                memory, codes, reference = targets(data,indices[case["source"]],case)
                if (metadata["reader_memory_sha256"] != tensor_digest(memory[None])
                        or not torch.equal(prefix.initial_state.cpu()[0,0],memory[-1])):
                    raise ValueError("actual native reading differs from original prompt-only cache")
                assisted, trace = assisted_stream(actor.writing,prefix,codes.to(device),seed)
                soft = actor.writing.writer.predict(codes.to(device))
                supplied = torch.where(soft<0,-torch.ones_like(soft),torch.ones_like(soft)).cpu()
                if (trace["stop_probabilities"][0] != free["stop_probabilities"][0]
                        or (len(free["images"]) and not torch.equal(free["images"][:1],assisted[:1]))):
                    raise ValueError("common first writing decision differs")
                for name,value in (("assisted.pt",assisted),("supplied.pt",supplied),
                        ("assisted_trace.pt",trace),("initial_state.pt",prefix.initial_state.cpu())):
                    torch.save(value,folder/name)
                save_raster(folder/"assisted.png",assisted);save_raster(folder/"supplied.png",supplied)
                Image.fromarray(reference.numpy()).save(folder/"reference.png")
                files = {p.name:sha(p) for p in folder.iterdir()}
                write(folder/"header.json",dict(**expected,output=saved,files_sha256=files,
                    prompt_memory_sha256=metadata["reader_memory_sha256"],native_prefix_positions=prefix.length))
                del prefix
            header = json.loads((folder/"header.json").read_text())
            if any(header[k] != value for k,value in expected.items()):
                raise ValueError("retained diagnostic source differs")
            for name,digest in header["files_sha256"].items():
                if sha(folder/name) != digest:
                    raise ValueError("retained diagnostic output changed")
            memory, codes, reference = targets(data,indices[case["source"]],case)
            state = torch.load(folder/"initial_state.pt",map_location="cpu",weights_only=True)
            if not torch.equal(state[0,0],memory[-1]) or header["prompt_memory_sha256"] != case["memory_sha256"]:
                raise ValueError("retained native initial reading differs")
            conditions.append(state[0,0].float());first_targets.append(codes[0]);references.append(reference)
            for p in folder.iterdir():
                inventory[str(p.relative_to(out))] = sha(p)
            write(out/"status.json",dict(stage="native_training_diagnosis",pid=os.getpid(),completed=i+1,total=32))
        target = torch.stack(first_targets).to(device).flatten(1)
        own = torch.stack(conditions).to(device)
        noise = torch.randn(target.shape,device=device,generator=torch.Generator(device=device).manual_seed(20267597))
        values = dict(times=TIMES,target=target.cpu(),noise=noise.cpu(),own=[],paired=[])
        with torch.autocast("cuda",dtype=torch.bfloat16):
            for t in TIMES:
                noisy = (1-t)*noise+t*target;times = torch.full((32,),t,device=device)
                values["own"].append(actor.writing.flow(noisy,times,own).cpu())
                values["paired"].append(actor.writing.flow(noisy,times,own[torch.arange(32,device=device)^1]).cpu())
        values["own"] = torch.stack(values["own"]);values["paired"] = torch.stack(values["paired"])
        torch.save(values,out/"conditioning.pt")
        rows, summary = metrics(out,cases,references)
        write(out/"report.json",dict(identity=identity,execution_identity=execution,rows=rows,summary=summary,
            provenance=provenance,files_sha256=inventory,conditioning_sha256=sha(out/"conditioning.pt"),
            latent_errors=latent_errors(values),actual_image_input_calls=32,assisted_streams=32,
            reference_history_and_length_supplied=True,training_cases_only=True,primary_gates_unchanged=True,
            elapsed_seconds=time.perf_counter()-begun,setup_seconds=setup_seconds,
            timing_scope="current evaluation invocation; retained calls may include resumed cases",
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
            language_quality_claim=False,general_ilm_goal_complete=False))
        write(out/"status.json",dict(stage="awaiting_cpu_verification",pid=None))


def verify(run, out):
    cpu_only();identity, cases = prepared(run,out)
    report = json.loads((out/"report.json").read_text())
    execution = json.loads((out/"execution_identity.json").read_text())
    if identity != report["identity"] or execution != report["execution_identity"] or ready(run) != execution["completed_original_pipeline"]:
        raise ValueError("completed diagnostic provenance changed")
    for name,digest in report["files_sha256"].items():
        if sha(out/name) != digest:
            raise ValueError("raw diagnostic artifact changed")
    _, data, indices = cases_and_data()
    writer = CodecBlockWriter(load_qualified_writing_codec(CODEC,torch.device("cpu")))
    references, first_targets, stop_consistent = [], [], []
    for i, case in enumerate(cases):
        folder = out/"predictions"/f"{i:02}"
        header = json.loads((folder/"header.json").read_text())
        actual_prompt = prompt(data,indices[case["source"]],case,independent=True)
        if (header["position"] != i or header["source"] != case["source"] or header["seed"] != SEED+case["pair"]
                or header["prompt_sha256"] != pixels_sha(actual_prompt)
                or header["execution_identity_sha256"] != sha(out/"execution_identity.json")):
            raise ValueError("independently reconstructed prompt binding differs")
        free = torch.load(folder/"answer.pt",map_location="cpu",weights_only=True);strip = check_answer(free)
        if sha(folder/"answer.pt") != header["output"]["raw_sha256"]:
            raise ValueError("autonomous raw output binding differs")
        if len(free["images"]) and not np.array_equal(np.asarray(Image.open(folder/"answer.png")),strip):
            raise ValueError("autonomous preview differs from emitted pixels")
        if not len(free["images"]) and (folder/"answer.png").exists():
            raise ValueError("empty autonomous answer has a stale preview")
        memory, codes, reference = targets(data,indices[case["source"]],case)
        initial = torch.load(folder/"initial_state.pt",map_location="cpu",weights_only=True)
        assisted = torch.load(folder/"assisted.pt",map_location="cpu",weights_only=True)
        supplied = torch.load(folder/"supplied.pt",map_location="cpu",weights_only=True)
        trace = torch.load(folder/"assisted_trace.pt",map_location="cpu",weights_only=True)
        if (initial.shape != (1,1,4096) or initial.dtype != torch.bfloat16
                or not torch.equal(initial[0,0],memory[-1]) or tensor_digest(memory[None]) != header["prompt_memory_sha256"]
                or free["reader_memory_sha256"] != header["prompt_memory_sha256"]
                or trace["reference_codes_sha256"] != tensor_digest(codes)
                or trace["states"].shape != (1,len(codes)+1,4096) or trace["states"].dtype != torch.float32
                or not torch.isfinite(trace["states"]).all()
                or trace["coordinates"].shape != (len(codes),2,768) or trace["coordinates"].dtype != torch.float32
                or not torch.isfinite(trace["coordinates"]).all()
                or not trace["reference_history_supplied"] or not trace["reference_length_supplied"]
                or not torch.equal(trace["states"][:,:1],initial.float())
                or trace["stop_probabilities"][0] != free["stop_probabilities"][0]
                or (len(free["images"]) and not torch.equal(free["images"][:1],assisted[:1]))):
            raise ValueError("owning reading, reference history or first writing decision differs")
        for name,images in (("assisted",assisted),("supplied",supplied)):
            if len(images) != len(codes) or not np.array_equal(np.asarray(Image.open(folder/(name+".png"))),raster(images)):
                raise ValueError("full assisted/reconstruction image differs")
        for coordinates,images in ((codes,supplied),(trace["coordinates"],assisted)):
            decoded = writer.predict(coordinates)
            if not torch.equal(torch.where(decoded<0,-torch.ones_like(decoded),torch.ones_like(decoded)),images):
                raise ValueError("CPU supplied/retained-coordinate decoder replay differs")
        if not np.array_equal(np.asarray(Image.open(folder/"reference.png")),reference.numpy()):
            raise ValueError("complete original reference differs")
        probabilities = trace["stop_probabilities"]
        if len(probabilities) != len(codes)+1 or not all(np.isfinite(p) and 0<=p<=1 for p in probabilities):
            raise ValueError("invalid reference-history stop trace")
        stop_consistent.append(all(p<.5 for p in probabilities[:-1]) and probabilities[-1]>=.5)
        references.append(reference);first_targets.append(codes[0])
    rows, summary = metrics(out,cases,references,independent=True)
    assert_close(rows,report["rows"]);assert_close(summary,report["summary"])
    values = torch.load(out/"conditioning.pt",map_location="cpu",weights_only=True)
    if (sha(out/"conditioning.pt") != report["conditioning_sha256"] or values["times"] != TIMES
            or values["own"].shape != (len(TIMES),32,1536) or values["paired"].shape != values["own"].shape
            or not torch.equal(values["target"],torch.stack(first_targets).flatten(1))
            or not torch.isfinite(values["own"]).all() or not torch.isfinite(values["paired"]).all()):
        raise ValueError("first-block conditional probe differs")
    errors = latent_errors(values,independent=True);assert_close(errors,report["latent_errors"])
    write(out/"verification.json",dict(report_sha256=sha(out/"report.json"),summary=summary,latent_errors=errors,
        all_32_free_and_assisted_outputs_independently_scored=True,independent_prompt_reconstructions=32,
        independent_supplied_code_replays=32,independent_assisted_coordinate_decodes=32,
        reference_history_stop_consistent=sum(stop_consistent),reference_history_stop_cases=32,
        full_neural_forward_replayed=False,training_cases_only=True,language_quality_claim=False,general_ilm_goal_complete=False))
    write(out/"status.json",dict(stage="completed_verified_training_diagnosis",pid=None))
    print((out/"verification.json").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True);parser.add_argument("--out",type=Path)
    parser.add_argument("--stage",choices=("prepare","evaluate","verify"),required=True)
    args = parser.parse_args();run = args.run.resolve();out = (args.out or run/"corpus_fit_v80").resolve()
    if not out.is_relative_to(run) or out == run:
        raise ValueError("separate diagnostic output inside the original run required")
    try:
        dict(prepare=prepare,evaluate=evaluate,verify=verify)[args.stage](run,out)
    except PipelinePending:
        raise
    except BaseException as error:
        if out.exists():write(out/"failure.json",dict(stage=args.stage,error=repr(error),completed_score=False))
        raise
