#!/usr/bin/env python3
"""Compare free, exact-history and supplied-code writing on fixed training cases."""

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

from ilm.visual_lm.clean_image_actor import load_clean_image_actor, tensor_digest
from ilm.visual_lm.codec_block_writer import CodecBlockWriter, load_qualified_writing_codec
from ilm.visual_lm.codec_language_audit import reference_scores
from ilm.visual_lm.complete_writing_data import CompleteWritingImageData
from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt
from ilm.visual_lm.semantic_writing_data import SemanticWritingData
from scripts.audit_semantic_ilm_v72 import strip_from_images
from scripts.cache_semantic_images_v72 import ROOT, TRAIN, TRAIN_SHA, CODEC, CONTROLS, sha, pixels_sha, atomic_json, committed_identity
from scripts.infer_semantic_ilm_v72 import save_image_answer
from scripts.train_semantic_ilm_v72 import setup
from scripts.verify_codec_language_v68 import assert_close, independent_actor_prompt, physical_scores

COHORT=ROOT/"references/evidence/clean_learning_v74_result.json"
TIMES=[0.,.1,.3,.5,.9,.96875,1.]
SOURCES=("scripts/probe_corpus_fit_v75.py","references/corpus_fit_v75_protocol.md",
    "references/evidence/clean_learning_v74_result.json","ilm/visual_lm/clean_image_actor.py",
    "ilm/visual_lm/clean_writing_flow.py","ilm/visual_lm/full_rank_writing_flow.py",
    "ilm/visual_lm/semantic_writing_model.py","ilm/visual_lm/semantic_writing_data.py",
    "ilm/visual_lm/complete_writing_data.py","ilm/visual_lm/codec_block_writer.py",
    "ilm/visual_lm/semantic_image_memory.py","scripts/verify_codec_language_v68.py")


def fixed_cases():
    value=json.loads(COHORT.read_text())
    cases=value["cases"]
    if len(cases)!=32 or any(c["position"]!=i or c["pair"]!=i//2 for i,c in enumerate(cases)):
        raise ValueError("reuse the exact existing training cohort")
    return cases


def raster(images):
    return strip_from_images(images)


def save_raster(path,images):
    Image.fromarray(raster(images).numpy()).save(path)


def metrics(out,cases,references,*,independent=False):
    rows=[]
    for i,case in enumerate(cases):
        folder=out/"predictions"/f"{i:02}"
        free=torch.load(folder/"answer.pt",map_location="cpu",weights_only=True)
        assisted=torch.load(folder/"assisted.pt",map_location="cpu",weights_only=True)
        supplied=torch.load(folder/"supplied.pt",map_location="cpu",weights_only=True)
        def measure(images,stopped):
            strip=raster(images)
            return (physical_scores(strip.numpy()==0,[references[i]],stopped) if independent
                else reference_scores(strip,[references[i]],stopped=stopped))
        def pixels_only(images):
            score=measure(images,True)
            return dict(foreground_f1=score["foreground_f1"],exact_pixel_stream=score["exact"])
        first_free=free["images"][:1]
        first_target=references[i][:,:64]
        first=(physical_scores(raster(first_free).numpy()==0,[first_target],True) if independent
            else reference_scores(raster(first_free),[first_target],stopped=True))
        rows.append(dict(free=measure(free["images"],free["stopped"]),assisted=pixels_only(assisted),
            supplied=pixels_only(supplied),first_block_f1=first["foreground_f1"],
            first_block_pixels_equal=bool(torch.equal(first_free,assisted[:1])) if len(first_free) else None))
    summary=dict(cases=32,free_exact=sum(r["free"]["exact"] for r in rows),
        free_f1=float(np.mean([r["free"]["foreground_f1"] for r in rows])),
        free_stopped=sum(r["free"]["stopped"] for r in rows),
        first_block_f1=float(np.mean([r["first_block_f1"] for r in rows])),
        assisted_exact_pixels=sum(r["assisted"]["exact_pixel_stream"] for r in rows),
        assisted_f1=float(np.mean([r["assisted"]["foreground_f1"] for r in rows])),
        supplied_exact_pixels=sum(r["supplied"]["exact_pixel_stream"] for r in rows),
        supplied_f1=float(np.mean([r["supplied"]["foreground_f1"] for r in rows])),
        first_blocks_compared=sum(r["first_block_pixels_equal"] is not None for r in rows),
        first_blocks_equal=sum(r["first_block_pixels_equal"] is True for r in rows))
    return rows,summary


def latent_errors(value,*,independent=False):
    rows=[]
    for i,t in enumerate(value["times"]):
        own=value["own"][i].double().numpy()
        paired=value["paired"][i].double().numpy()
        target=value["target"].double().numpy()
        if independent:
            correct=np.sum(np.square(own-target),axis=1)/target.shape[1]
            wrong=np.sum(np.square(paired-target),axis=1)/target.shape[1]
        else:
            correct=np.mean((own-target)**2,axis=1)
            wrong=np.mean((paired-target)**2,axis=1)
        rows.append(dict(time=t,own_mse=float(correct.mean()),paired_mse=float(wrong.mean()),
            paired_minus_own_mse=float((wrong-correct).mean())))
    return rows


@torch.inference_mode()
def assisted_stream(actor,memory,codes,seed):
    model=actor.writing
    generator=torch.Generator(device=actor.device).manual_seed(seed)
    images=[]
    with torch.autocast("cuda",dtype=torch.bfloat16):
        initial_stop=None
        for t in range(len(codes)):
            condition=model.conditions(memory,codes[None,:t])[:,-1]
            if t==0:
                initial_stop=model.stop(condition).float().sigmoid().item()
            _,endpoint=model.flow.sample_endpoint(condition,generator=generator,steps=model.config.flow_steps)
            coordinates=F.layer_norm(endpoint.reshape(-1,768),(768,),
                eps=actor.writer.codec.config.layer_norm_epsilon).reshape(1,2,768)
            soft=actor.writer.predict(coordinates)
            images.append(torch.where(soft<0,-torch.ones_like(soft),torch.ones_like(soft)).cpu())
    return torch.cat(images),initial_stop


def evaluate(args,cases):
    if (args.out/"report.json").exists():
        raise ValueError("completed diagnosis is immutable; use verification")
    original=json.loads((args.run/"verification.json").read_text())
    copy=json.loads((args.copy/"verification.json").read_text())
    if (not original["offline_checks_pass"] or not copy["all_12_outputs_independently_scored"]
            or original["report_sha256"]!=sha(args.run/"development/report.json")
            or copy["report_sha256"]!=sha(args.copy/"report.json")):
        raise ValueError("finish the original and queued copy audits first")
    checkpoint=args.run/"training/checkpoint_latest.pt"
    training=json.loads((args.run/"training/training_report.json").read_text())
    if training["updates"]!=40000 or sha(checkpoint)!=training["checkpoint_sha256"]:
        raise ValueError("completed full-corpus state required")
    identity=dict(**committed_identity(SOURCES),checkpoint_sha256=sha(checkpoint),
        original_verification_sha256=sha(args.run/"verification.json"),
        copy_verification_sha256=sha(args.copy/"verification.json"),cohort_sha256=sha(COHORT),seed=20267420)
    args.out.mkdir(parents=True,exist_ok=True)
    path=args.out/"identity.json"
    if path.exists():
        old=json.loads(path.read_text())
        if {k:v for k,v in old.items() if k!="source_commit"}!={k:v for k,v in identity.items() if k!="source_commit"}:
            raise ValueError("diagnostic resume identity changed")
        identity=old
    else:
        atomic_json(path,identity)
    data=CompleteWritingImageData(TRAIN,purpose="train",expected_report_sha256=TRAIN_SHA)
    index={name:i for i,name in enumerate(data.names)}
    cache=SemanticWritingData(args.run/"cache")
    cache_index={name:i for i,name in enumerate(cache.names)}
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        device=setup()
        begun=time.perf_counter()
        local=ROOT.parent/"LocalLLM/.local"
        actor,provenance=load_clean_image_actor(checkpoint=checkpoint,controls=CONTROLS,codec=CODEC,
            ollama_root=local/"models/ollama",gguf_source=local/"tools/llama.cpp/gguf-py",device=device)
        setup_seconds=time.perf_counter()-begun
        references,memories,targets,replays=[],[],[],[]
        inventory={}
        for i,case in enumerate(cases):
            folder=args.out/"predictions"/f"{i:02}"
            prompt=fold_writing_prompt(data.prompt(index[case["source"]])).pixels
            expected=dict(position=i,source=case["source"],seed=identity["seed"]+case["pair"],
                prompt_sha256=pixels_sha(prompt),checkpoint_sha256=identity["checkpoint_sha256"])
            if not (folder/"header.json").exists():
                if folder.exists() and any(folder.iterdir()):
                    raise ValueError("retain partial unbound output and use a fresh directory")
                free,memory=actor.generate_with_memory(prompt,seed=expected["seed"])
                saved=save_image_answer(free,folder)
                # Gold code/history and lengths are accessed only after the free call.
                cached,codes=cache[cache_index[case["cache_record"]]]
                reference=((data.target(index[case["source"]])[0,0,:,:-128]+1)*127.5).to(torch.uint8)
                assisted,initial_stop=assisted_stream(actor,memory.to(device),codes.to(device),expected["seed"])
                with torch.inference_mode():
                    soft=actor.writer.predict(codes.to(device)).cpu()
                    supplied=torch.where(soft<0,-torch.ones_like(soft),torch.ones_like(soft))
                torch.save(memory,folder/"memory.pt")
                torch.save(assisted,folder/"assisted.pt")
                torch.save(supplied,folder/"supplied.pt")
                save_raster(folder/"assisted.png",assisted)
                save_raster(folder/"supplied.png",supplied)
                Image.fromarray(reference.numpy()).save(folder/"reference.png")
                file_hashes={p.name:sha(p) for p in folder.iterdir()}
                atomic_json(folder/"header.json",dict(**expected,output=saved,files_sha256=file_hashes,
                    memory_sha256=tensor_digest(memory),assisted_initial_stop=initial_stop,
                    reference_history_supplied=True,reference_length_supplied=True))
            header=json.loads((folder/"header.json").read_text())
            if any(header[k]!=v for k,v in expected.items()):
                raise ValueError("source generation identity changed")
            for name,digest in header["files_sha256"].items():
                if sha(folder/name)!=digest:
                    raise ValueError("retained diagnostic changed")
            memory=torch.load(folder/"memory.pt",map_location="cpu",weights_only=True)
            cached,codes=cache[cache_index[case["cache_record"]]]
            references.append(((data.target(index[case["source"]])[0,0,:,:-128]+1)*127.5).to(torch.uint8))
            memories.append(memory)
            targets.append(codes[0])
            replays.append(dict(exact=torch.equal(memory[0],cached),
                maximum_abs_difference=float((memory[0].float()-cached.float()).abs().max())))
            for p in folder.iterdir():
                inventory[str(p.relative_to(args.out))]=sha(p)
            atomic_json(args.out/"status.json",dict(stage="training_fit_diagnosis",pid=os.getpid(),completed=i+1,total=32))
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            conditions=torch.cat([actor.writing.conditions(m.to(device),torch.empty(1,0,2,768,device=device))[:,-1]
                for m in memories])
            target=torch.stack(targets).to(device).flatten(1)
            noise=torch.randn(target.shape,device=device,generator=torch.Generator(device=device).manual_seed(20267597))
            partner=torch.arange(32,device=device)^1
            values=dict(times=TIMES,target=target.cpu(),noise=noise.cpu(),own=[],paired=[])
            for t in values["times"]:
                time_value=torch.full((32,),t,device=device)
                noisy=(1-t)*noise+t*target
                values["own"].append(actor.writing.flow(noisy,time_value,conditions).cpu())
                values["paired"].append(actor.writing.flow(noisy,time_value,conditions[partner]).cpu())
            values["own"],values["paired"]=torch.stack(values["own"]),torch.stack(values["paired"])
        torch.save(values,args.out/"conditioning.pt")
        rows,summary=metrics(args.out,cases,references)
        atomic_json(args.out/"report.json",dict(identity=identity,provenance=provenance,rows=rows,summary=summary,
            files_sha256=inventory,conditioning_sha256=sha(args.out/"conditioning.pt"),
            latent_errors=latent_errors(values),reader_cache_replay=replays,actual_image_input_calls=32,
            assisted_streams=32,assisted_reference_history_and_length=True,training_cases_only=True,
            elapsed_seconds=time.perf_counter()-begun,setup_seconds=setup_seconds,
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,language_quality_claim=False))
        atomic_json(args.out/"status.json",dict(stage="completed_awaiting_cpu_verification",pid=None))


def verify(args,cases):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized():
        raise ValueError("CPU-only verification required")
    report=json.loads((args.out/"report.json").read_text())
    identity=json.loads((args.out/"identity.json").read_text())
    if (identity!=report["identity"] or identity["cohort_sha256"]!=sha(COHORT)
            or identity["checkpoint_sha256"]!=sha(args.run/"training/checkpoint_latest.pt")
            or identity["original_verification_sha256"]!=sha(args.run/"verification.json")
            or identity["copy_verification_sha256"]!=sha(args.copy/"verification.json")):
        raise ValueError("completed diagnostic provenance changed")
    for name,digest in identity["sources"].items():
        if sha(ROOT/name)!=digest:
            raise ValueError("diagnostic implementation changed")
    for name,digest in report["files_sha256"].items():
        if sha(args.out/name)!=digest:
            raise ValueError("raw diagnostic file changed")
    data=CompleteWritingImageData(TRAIN,purpose="train",expected_report_sha256=TRAIN_SHA)
    index={name:i for i,name in enumerate(data.names)}
    cache=SemanticWritingData(args.run/"cache")
    cache_index={name:i for i,name in enumerate(cache.names)}
    writer=CodecBlockWriter(load_qualified_writing_codec(CODEC,torch.device("cpu")))
    references,replays,targets=[],[],[]
    for i,case in enumerate(cases):
        folder=args.out/"predictions"/f"{i:02}"
        header=json.loads((folder/"header.json").read_text())
        normal=((data.prompt(index[case["source"]])[0,0]+1)*127.5).to(torch.uint8)
        prompt=independent_actor_prompt(dict(prompt=normal))
        if (header["position"]!=i or header["source"]!=case["source"] or header["seed"]!=20267420+case["pair"]
                or header["prompt_sha256"]!=pixels_sha(prompt) or header["checkpoint_sha256"]!=identity["checkpoint_sha256"]):
            raise ValueError("independent source input differs")
        free=torch.load(folder/"answer.pt",map_location="cpu",weights_only=True)
        pixels=raster(free["images"])
        probabilities=free["stop_probabilities"]
        if (type(free["stopped"])is not bool or type(free["capped"])is not bool
                or len(probabilities)!=len(free["images"])+1 or free["stopped"]==free["capped"]
                or any(not math.isfinite(p) or not 0<=p<=1 for p in probabilities)
                or any(p>=.5 for p in probabilities[:-1]) or (probabilities[-1]>=.5)!=free["stopped"]
                or (free["capped"] and len(free["images"])!=128)
                or not free["image_input_image_output"] or not free["feedback_from_emitted_pixels"]
                or len(free["feedback_sha256"])!=len(free["images"])):
            raise ValueError("invalid autonomous stopping/feedback trace")
        expected_output=dict(raw_sha256=sha(folder/"answer.pt"),blocks=len(free["images"]),
            stopped=free["stopped"],capped=free["capped"],feedback_from_emitted_pixels=True,
            answer_png_sha256=sha(folder/"answer.png") if len(free["images"]) else None)
        if header["output"]!=expected_output:
            raise ValueError("saved free output metadata differs")
        if len(free["images"]):
            if not np.array_equal(np.array(Image.open(folder/"answer.png")),pixels.numpy()):
                raise ValueError("free PNG differs")
        elif (folder/"answer.png").exists():
            raise ValueError("empty free output has a placeholder PNG")
        memory=torch.load(folder/"memory.pt",map_location="cpu",weights_only=True)
        cached,codes=cache[cache_index[case["cache_record"]]]
        if (tensor_digest(memory)!=header["memory_sha256"] or free["reader_memory_sha256"]!=tensor_digest(memory)
                or header["assisted_initial_stop"]!=probabilities[0]):
            raise ValueError("reader or common initial state differs")
        reference=((data.target(index[case["source"]])[0,0,:,:-128]+1)*127.5).to(torch.uint8)
        if not np.array_equal(np.array(Image.open(folder/"reference.png")),reference.numpy()):
            raise ValueError("original reference differs")
        for name in ("assisted","supplied"):
            images=torch.load(folder/(name+".pt"),map_location="cpu",weights_only=True)
            if len(images)!=len(codes) or not np.array_equal(np.array(Image.open(folder/(name+".png"))),raster(images).numpy()):
                raise ValueError("assisted/supplied full image differs")
        decoded=writer.predict(codes).detach()
        supplied=torch.where(decoded<0,-torch.ones_like(decoded),torch.ones_like(decoded))
        if not torch.equal(supplied,torch.load(folder/"supplied.pt",map_location="cpu",weights_only=True)):
            raise ValueError("independent fixed-codec replay differs")
        references.append(reference)
        targets.append(codes[0])
        replays.append(dict(exact=torch.equal(memory[0],cached),
            maximum_abs_difference=float((memory[0].float()-cached.float()).abs().max())))
    rows,summary=metrics(args.out,cases,references,independent=True)
    assert_close(rows,report["rows"])
    assert_close(summary,report["summary"])
    if replays!=report["reader_cache_replay"] or sha(args.out/"conditioning.pt")!=report["conditioning_sha256"]:
        raise ValueError("reader/conditioning evidence differs")
    values=torch.load(args.out/"conditioning.pt",map_location="cpu",weights_only=True)
    if (values["times"]!=TIMES or values["own"].shape!=(len(TIMES),32,1536)
            or values["paired"].shape!=(len(TIMES),32,1536) or not torch.equal(values["target"],torch.stack(targets).flatten(1))
            or not torch.isfinite(values["own"]).all() or not torch.isfinite(values["paired"]).all()):
        raise ValueError("invalid retained latent-prediction probe")
    errors=latent_errors(values,independent=True)
    assert_close(errors,report["latent_errors"])
    atomic_json(args.out/"verification.json",dict(report_sha256=sha(args.out/"report.json"),summary=summary,
        latent_errors=errors,all_32_free_and_assisted_outputs_independently_scored=True,
        independent_supplied_code_replays=32,reader_cache_exact_count=sum(r["exact"] for r in replays),
        full_neural_forward_replayed=False,training_cases_only=True,assisted_history_and_length_supplied=True,
        language_quality_claim=False,general_ilm_goal_complete=False))
    print(json.dumps(dict(summary=summary,latent_errors=errors),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=ROOT/"artifacts/clean_corpus_v75_20261106")
    parser.add_argument("--copy",type=Path,default=ROOT/"artifacts/copy_transfer_v75_20261106")
    parser.add_argument("--out",type=Path,default=ROOT/"artifacts/corpus_fit_v75_20261106")
    parser.add_argument("--stage",choices=("evaluate","verify"),required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    cases=fixed_cases()
    if args.stage=="evaluate":
        evaluate(args,cases)
    else:
        verify(args,cases)
