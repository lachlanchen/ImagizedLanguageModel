#!/usr/bin/env python3
"""Evaluate and independently score new literal-writing image requests."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.clean_image_actor import load_clean_image_actor
from ilm.visual_lm.codec_language_audit import reference_scores
from scripts.audit_semantic_ilm_v72 import strip_from_images
from scripts.cache_semantic_images_v72 import ROOT, CODEC, CONTROLS, sha, pixels_sha, atomic_json, committed_identity
from scripts.infer_semantic_ilm_v72 import save_image_answer
from scripts.train_semantic_ilm_v72 import setup
from scripts.verify_codec_language_v68 import assert_close, independent_actor_prompt, physical_scores

SOURCES = ("scripts/audit_copy_transfer_v75.py", "scripts/audit_semantic_ilm_v72.py",
    "scripts/infer_semantic_ilm_v72.py", "scripts/verify_codec_language_v68.py",
    "ilm/visual_lm/clean_image_actor.py", "ilm/visual_lm/semantic_image_actor.py",
    "ilm/visual_lm/clean_writing_flow.py", "ilm/visual_lm/full_rank_writing_flow.py",
    "ilm/visual_lm/semantic_writing_model.py", "ilm/visual_lm/semantic_image_memory.py",
    "ilm/visual_lm/codec_block_writer.py", "ilm/visual_lm/codec_language_audit.py",
    "references/copy_transfer_v75_protocol.md")


def source(args):
    value = json.loads((args.data/"report.json").read_text())
    if (not value["evaluation_only"] or value["model_calls"]!=0 or value["gpu_allocations"]!=0
            or value["exact_prompt_overlaps"] or value["training_images_checked"]!=12976
            or len(value["cases"])!=12 or value["sealed_examples_used"]!=0):
        raise ValueError("complete fixed evaluation-only copy requests required")
    if any(case["position"]!=i or case["pair"]!=i//2
            or case["requested_script"]!=("zh" if i<4 else "en" if i<8 else "numeric")
            for i,case in enumerate(value["cases"])):
        raise ValueError("fixed case order or script groups changed")
    for name, digest in value["identity"]["sources"].items():
        if sha(ROOT/name)!=digest:
            raise ValueError("prepared copy source implementation changed")
    return value


def inputs(data, case):
    def allowed(name):
        path=Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("unsafe data path")
        return path
    return (data/allowed(case["input_file"]), data/allowed(case["source_strip_file"]),
        data/allowed(case["reference_file"]))


def summarize(cases, rows):
    groups={}
    for key in ("all", "zh", "en", "numeric"):
        selected=[rows[i] for i,c in enumerate(cases) if key=="all" or c["requested_script"]==key]
        groups[key]=dict(cases=len(selected), exact=sum(r["normal"]["exact"] for r in selected),
            foreground_f1=float(np.mean([r["normal"]["foreground_f1"] for r in selected])),
            learned_stopping=sum(r["normal"]["stopped"] for r in selected),
            own_minus_partner_f1=float(np.mean([r["normal"]["foreground_f1"]-
                r["cross"]["foreground_f1"] for r in selected])))
    return dict(groups=groups, both_exact_pairs=sum(all(rows[i]["normal"]["exact"]
        for i in (a,a+1)) for a in range(0,12,2)), pairs=6)


def score(args, prepared, *, independent=False):
    references=[]
    for case in prepared["cases"]:
        _, _, path=inputs(args.data,case)
        if sha(path)!=prepared["files_sha256"][case["reference_file"]]:
            raise ValueError("reference raster changed")
        array=np.array(Image.open(path),copy=True)
        if array.ndim!=2 or array.shape[0]!=32 or array.shape[1]%64 or not np.isin(array,[0,255]).all():
            raise ValueError("complete binary reference required")
        references.append(torch.from_numpy(array))
    if any(torch.equal(references[i],references[i+1]) for i in range(0,12,2)):
        raise ValueError("pair references must differ")
    rows=[]
    for i, case in enumerate(prepared["cases"]):
        folder=args.out/"predictions"/f"{i:02}"
        result=torch.load(folder/"answer.pt",map_location="cpu",weights_only=True)
        raster=strip_from_images(result["images"])
        def measure(target):
            return (physical_scores(raster.numpy()==0,[target],result["stopped"]) if independent
                else reference_scores(raster,[target],stopped=result["stopped"]))
        rows.append(dict(normal=measure(references[i]),cross=measure(references[i^1])))
    return rows,summarize(prepared["cases"],rows)


def evaluate(args, prepared):
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/"report.json").exists():
        raise ValueError("retain completed answers and run verification instead")
    verification=json.loads((args.run/"verification.json").read_text())
    original=json.loads((args.run/"development/report.json").read_text())
    training=json.loads((args.run/"training/training_report.json").read_text())
    checkpoint=args.run/"training/checkpoint_latest.pt"
    if (not verification["offline_checks_pass"] or not verification["all_384_outputs_independently_scored"]
            or verification["report_sha256"]!=sha(args.run/"development/report.json")
            or original["identity"]["checkpoint_sha256"]!=sha(checkpoint)
            or training["updates"]!=40000 or training["identity"]["purpose"]!="full_corpus_clean"):
        raise ValueError("complete original audit and CPU verification required first")
    identity=dict(**committed_identity(SOURCES), checkpoint_sha256=sha(checkpoint),
        original_verification_sha256=sha(args.run/"verification.json"),
        data_sha256=sha(args.data/"report.json"),seed=20267580,maximum_blocks=128)
    if (args.out/"identity.json").exists():
        previous=json.loads((args.out/"identity.json").read_text())
        if {k:v for k,v in previous.items() if k!="source_commit"}!={k:v for k,v in identity.items() if k!="source_commit"}:
            raise ValueError("copy diagnostic resume identity changed")
        identity=previous
    else:
        atomic_json(args.out/"identity.json",identity)
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        device=setup()
        begun=time.perf_counter()
        local=ROOT.parent/"LocalLLM/.local"
        actor,provenance=load_clean_image_actor(checkpoint=checkpoint,controls=CONTROLS,codec=CODEC,
            ollama_root=local/"models/ollama",gguf_source=local/"tools/llama.cpp/gguf-py",device=device)
        setup_seconds=time.perf_counter()-begun
        files={}
        for i,case in enumerate(prepared["cases"]):
            path,_,_=inputs(args.data,case)
            if sha(path)!=prepared["files_sha256"][case["input_file"]]:
                raise ValueError("actor input changed")
            prompt=torch.load(path,map_location="cpu",weights_only=True)
            folder=args.out/"predictions"/f"{i:02}"
            header=dict(identity_sha256=sha(args.out/"identity.json"),position=i,pair=case["pair"],
                seed=identity["seed"]+case["pair"],prompt_sha256=pixels_sha(prompt))
            if not (folder/"header.json").exists():
                if folder.exists() and any(folder.iterdir()):
                    raise ValueError("retain incomplete unbound output and use a fresh diagnostic directory")
                result=actor.generate(prompt,seed=header["seed"])
                saved=save_image_answer(result,folder)
                atomic_json(folder/"header.json",dict(**header,output=saved))
            retained=json.loads((folder/"header.json").read_text())
            if ({k:v for k,v in retained.items() if k!="output"}!=header
                    or sha(folder/"answer.pt")!=retained["output"]["raw_sha256"]):
                raise ValueError("retained actor call changed")
            for path in folder.iterdir():
                files[str(path.relative_to(args.out))]=sha(path)
            atomic_json(args.out/"status.json",dict(stage="generating",pid=os.getpid(),completed=i+1,total=12))
        # All reference pixel loading occurs after all autonomous calls finish.
        rows,summary=score(args,prepared)
        atomic_json(args.out/"report.json",dict(identity=identity,provenance=provenance,rows=rows,
            summary=summary,files_sha256=files,elapsed_seconds=time.perf_counter()-begun,
            setup_seconds=setup_seconds,peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
            model_calls=12,descriptive_diagnostic_only=True,language_quality_claim=False))
        atomic_json(args.out/"status.json",dict(stage="completed_awaiting_cpu_verification",pid=None))


def verify(args,prepared):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized():
        raise ValueError("verification must hide CUDA")
    report=json.loads((args.out/"report.json").read_text())
    identity=json.loads((args.out/"identity.json").read_text())
    if (report["identity"]!=identity or identity["data_sha256"]!=sha(args.data/"report.json")
            or identity["checkpoint_sha256"]!=sha(args.run/"training/checkpoint_latest.pt")
            or identity["original_verification_sha256"]!=sha(args.run/"verification.json")
            or identity["seed"]!=20267580 or identity["maximum_blocks"]!=128 or report["model_calls"]!=12):
        raise ValueError("copy diagnostic identity changed")
    for name,digest in identity["sources"].items():
        if sha(ROOT/name)!=digest:
            raise ValueError("generation implementation changed")
    for name,digest in prepared["files_sha256"].items():
        if sha(args.data/name)!=digest:
            raise ValueError("prepared input/reference changed")
    files={}
    for i,case in enumerate(prepared["cases"]):
        path,raw_path,_=inputs(args.data,case)
        raw=torch.from_numpy(np.array(Image.open(raw_path),copy=True))
        expected=independent_actor_prompt(dict(prompt=raw))
        if not torch.equal(expected,torch.load(path,map_location="cpu",weights_only=True)):
            raise ValueError("independent prompt placement differs")
        folder=args.out/"predictions"/f"{i:02}"
        header=json.loads((folder/"header.json").read_text())
        expected_header=dict(identity_sha256=sha(args.out/"identity.json"),position=i,pair=case["pair"],
            seed=identity["seed"]+case["pair"],prompt_sha256=pixels_sha(expected))
        if {k:v for k,v in header.items() if k!="output"}!=expected_header:
            raise ValueError("source actor call differs")
        result=torch.load(folder/"answer.pt",map_location="cpu",weights_only=True)
        raster=strip_from_images(result["images"])
        count=len(result["images"])
        probabilities=result["stop_probabilities"]
        if (type(result["stopped"])is not bool or type(result["capped"])is not bool
                or result["stopped"]==result["capped"] or (result["capped"] and count!=128)
                or len(probabilities)!=count+1 or not all(math.isfinite(p) and 0<=p<=1 for p in probabilities)
                or any(p>=.5 for p in probabilities[:-1]) or (probabilities[-1]>=.5)!=result["stopped"]
                or not result["feedback_from_emitted_pixels"] or len(result["feedback_sha256"])!=count
                or not result["image_input_image_output"]):
            raise ValueError("invalid autonomous image stream")
        png=folder/"answer.png"
        if count and not np.array_equal(np.array(Image.open(png)),raster.numpy()):
            raise ValueError("raw output and PNG differ")
        if not count and png.exists():
            raise ValueError("empty answer has a placeholder raster")
        output=dict(raw_sha256=sha(folder/"answer.pt"),blocks=count,stopped=result["stopped"],
            capped=result["capped"],answer_png_sha256=sha(png) if count else None,feedback_from_emitted_pixels=True)
        if output!=header["output"]:
            raise ValueError("saved output receipt differs")
        for path in folder.iterdir():
            files[str(path.relative_to(args.out))]=sha(path)
    if files!=report["files_sha256"]:
        raise ValueError("generation inventory differs")
    rows,summary=score(args,prepared,independent=True)
    assert_close(rows,report["rows"])
    assert_close(summary,report["summary"])
    atomic_json(args.out/"verification.json",dict(report_sha256=sha(args.out/"report.json"),
        summary=summary,all_12_outputs_independently_scored=True,source_input_pixels_checked=12,
        full_neural_forward_replayed=False,physical_feedback_hashes_numerically_replayed=False,
        descriptive_diagnostic_only=True,language_quality_claim=False,general_ilm_goal_complete=False))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=ROOT/"artifacts/clean_corpus_v75_20261106")
    parser.add_argument("--data",type=Path,default=ROOT/"data/general_instructions/copy_transfer_v75_20261106")
    parser.add_argument("--out",type=Path,default=ROOT/"artifacts/copy_transfer_v75_20261106")
    parser.add_argument("--stage",choices=("evaluate","verify"),required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    prepared=source(args)
    if args.stage=="evaluate":
        evaluate(args,prepared)
    else:
        verify(args,prepared)
