#!/usr/bin/env python3
"""Unselected full-answer image audit on all 192 fixed questions and controls."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import torch

from ilm.visual_lm.codec_language_audit import AUDIT_DATA_SHA, CodecLanguageAuditData, reference_scores, summarize
from ilm.visual_lm.semantic_image_actor import load_semantic_image_actor
from scripts.cache_semantic_images_v72 import ROOT, CODEC, CONTROLS, atomic_json, committed_identity, pixels_sha, sha
from scripts.infer_semantic_ilm_v72 import save_image_answer
from scripts.train_semantic_ilm_v72 import setup

SOURCES = ("scripts/audit_semantic_ilm_v72.py","scripts/verify_semantic_ilm_v72.py",
    "scripts/infer_semantic_ilm_v72.py","ilm/visual_lm/semantic_image_actor.py",
    "ilm/visual_lm/semantic_image_memory.py","ilm/visual_lm/image_context_interface.py",
    "ilm/visual_lm/semantic_writing_model.py","ilm/visual_lm/codec_language_audit.py",
    "ilm/visual_lm/codec_block_writer.py")
DATA = ROOT/"data/general_instructions/codec_language_audit_v68_20261105"


def strip_from_images(images):
    if images.ndim!=4 or images.shape[1:]!=(1,32,64) or len(images)>128:
        raise ValueError("invalid generated physical image stream")
    if images.device.type!="cpu" or images.dtype!=torch.float32 or not bool(((images==-1)|(images==1)).all()):
        raise ValueError("retain actual thresholded physical pixels without repair")
    if not len(images):
        return torch.empty((32,0),dtype=torch.uint8)
    return ((torch.cat(list(images),dim=-1)[0]+1)*127.5).to(torch.uint8)


def run(args):
    out = args.run/"development"
    out.mkdir(parents=True,exist_ok=True)
    training = json.loads((args.run/"training/training_report.json").read_text())
    checkpoint = args.run/"training/checkpoint_latest.pt"
    if training["updates"]!=40000 or sha(checkpoint)!=training["checkpoint_sha256"]:
        raise ValueError("the registered production checkpoint must finish before audit")
    implementation = committed_identity(SOURCES)
    identity = dict(**implementation,checkpoint_sha256=sha(checkpoint),
        training_report_sha256=sha(args.run/"training/training_report.json"),
        prepared_data_sha256=AUDIT_DATA_SHA,maximum_blocks=128,flow_steps=32,
        seed=20267204,normal_control_common_random_numbers=True)
    if (out/"identity.json").exists():
        old = json.loads((out/"identity.json").read_text())
        if {k:v for k,v in old.items() if k!="source_commit"}!={k:v for k,v in identity.items() if k!="source_commit"}:
            raise ValueError("audit resume identity changed")
        identity = old
    else:
        atomic_json(out/"identity.json",identity)
    data = CodecLanguageAuditData(DATA)
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        device = setup()
        begun = time.perf_counter()
        local = ROOT.parent/"LocalLLM/.local"
        actor,provenance = load_semantic_image_actor(checkpoint=checkpoint,controls=CONTROLS,
            codec=CODEC,ollama_root=local/"models/ollama",gguf_source=local/"tools/llama.cpp/gguf-py",device=device)
        setup_seconds = time.perf_counter()-begun
        scores,files = {},{}
        for index,row in enumerate(data.cases):
            key = row["source"]+"/"+row["id"]
            scores[key] = {}
            for route in ("normal","blank"):
                folder = out/"predictions"/f"{index:03}_{route}"
                seed = identity["seed"]+index
                prompt = data.prompt(index,blank=route=="blank")
                header = dict(identity_sha256=sha(out/"identity.json"),position=index,route=route,
                    source=row["source"],id=row["id"],seed=seed,prompt_sha256=pixels_sha(prompt))
                if not (folder/"header.json").exists():
                    if folder.exists() and any(folder.iterdir()):
                        raise ValueError("incomplete unbound prediction; preserve it and use a fresh audit directory")
                    result = actor.generate(prompt,seed=seed)
                    saved = save_image_answer(result,folder)
                    atomic_json(folder/"header.json",dict(**header,output=saved))
                retained = json.loads((folder/"header.json").read_text())
                if {k:v for k,v in retained.items() if k!="output"}!=header:
                    raise ValueError("generation header/source changed")
                if sha(folder/"answer.pt")!=retained["output"]["raw_sha256"]:
                    raise ValueError("retained generation payload changed")
                result = torch.load(folder/"answer.pt",map_location="cpu",weights_only=True)
                strip = strip_from_images(result["images"])
                # Reference access is downstream of the autonomous model call.
                scores[key][route] = reference_scores(strip,data.references(index),stopped=result["stopped"])
                for path in sorted(folder.iterdir()):
                    files[str(path.relative_to(out))] = sha(path)
                atomic_json(out/"status.json",dict(stage="autonomous_audit",pid=os.getpid(),
                    completed=sum(len(v) for v in scores.values()),total=384,last_position=index,last_route=route))
                print(json.dumps(dict(position=index,route=route,**scores[key][route])),flush=True)
        summary = summarize(data,scores)
        atomic_json(out/"scores.json",scores)
        atomic_json(out/"report.json",dict(identity=identity,provenance=provenance,
            generations=384,scores_sha256=sha(out/"scores.json"),files_sha256=files,
            elapsed_seconds=time.perf_counter()-begun,setup_seconds=setup_seconds,
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,**summary))
        atomic_json(out/"status.json",dict(stage="audit_complete_awaiting_verification",pid=None))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    run(parser.parse_args())
