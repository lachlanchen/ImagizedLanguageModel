#!/usr/bin/env python3
"""Offline reading check of the imported body; symbolic decoding is helper-only."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import PIL
from PIL import Image
import torch
from torch.nn import functional as F

from ilm.visual_lm.blind_vision_transcription import check_host_memory, gpu_memory
from ilm.visual_lm.gguf_language_trunk import load_gguf_reader_source
from ilm.visual_lm.gguf_visual_language import QwenVisualLanguageArchive
from ilm.visual_lm.grounded_image_weights import MODEL_SHA, load_grounded_image_actor
from ilm.visual_lm.image_context_interface import ImageContextInterface
from scripts.native_visual_baseline_v70 import TASK, score_response

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT/"artifacts/native_visual_baseline_v70_20261105/attempt01"
PREVIOUS_SHA="71926b423d97dab6f5ee99ff7b94633ad2919f18a4fbfddd806b27a057a0a35e"
SOURCES=("scripts/check_native_visual_transfer_v71.py","scripts/verify_native_visual_transfer_v71.py",
    "references/native_visual_transfer_v71_plan.md","ilm/visual_lm/image_context_interface.py",
    "ilm/visual_lm/grounded_image_actor.py","ilm/visual_lm/grounded_image_weights.py",
    "ilm/visual_lm/gguf_visual_language.py","scripts/native_visual_baseline_v70.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,value):
    temporary=path.with_suffix(".partial")
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n");temporary.replace(path)


def identity():
    revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    sources={name:sha(ROOT/name) for name in SOURCES}
    for name,digest in sources.items():
        raw=subprocess.check_output(["git","show",revision+":"+name],cwd=ROOT)
        if hashlib.sha256(raw).hexdigest()!=digest:
            raise ValueError("commit the fixed implementation before execution")
    return revision,sources


def archive_and_tokenizer():
    local=ROOT.parent/"LocalLLM"
    blob=local/".local/models/ollama/blobs"/("sha256-"+MODEL_SHA)
    if sha(blob)!=MODEL_SHA:
        raise ValueError("authenticated shared archive changed")
    archive=QwenVisualLanguageArchive(blob,load_gguf_reader_source(local/".local/tools/llama.cpp/gguf-py"))
    from transformers.integrations.ggml import convert_gguf_tokenizer
    table=archive.field("tokenizer.ggml.tokens")
    tokenizer,_=convert_gguf_tokenizer("qwen2",dict(tokens=table,
        merges=archive.field("tokenizer.ggml.merges"),eos_token_id=archive.field("tokenizer.ggml.eos_token_id")))
    markers={}
    for name in ("im_start","im_end","vision_start","vision_end"):
        literal="<|"+name+"|>"
        if table.count(literal)!=1:
            raise ValueError("source structural marker is not unique")
        markers[name]=table.index(literal)
    return archive,tokenizer,markers


def embed_rows(archive,ids):
    source=archive.omitted["token_embd.weight"]
    values=archive.gguf.dequantize(np.array(source.data[ids],copy=True),source.tensor_type)
    value=torch.from_numpy(np.array(values,dtype=np.float32,copy=True))
    if value.shape!=(len(ids),4096) or not torch.isfinite(value).all():
        raise ValueError("source embedding rows differ")
    return value


def compile_controls(archive,tokenizer,markers):
    encode=lambda text:tokenizer.encode(text,add_special_tokens=False).ids
    before=[markers["im_start"],*encode("user\n"),markers["vision_start"]]
    after=[markers["vision_end"],*encode(TASK),markers["im_end"],*encode("\n"),
           markers["im_start"],*encode("assistant\n")]
    if (tokenizer.decode(before,skip_special_tokens=False)!="<|im_start|>user\n<|vision_start|>"
            or tokenizer.decode(after,skip_special_tokens=False)!="<|vision_end|>"+TASK+"<|im_end|>\n<|im_start|>assistant\n"):
        raise ValueError("source structural-token round trip differs")
    return dict(before=embed_rows(archive,before),after=embed_rows(archive,after)),dict(before_ids=before,after_ids=after,
        source_model_sha256=MODEL_SHA,eog_ids=[int(x) for x in archive.field("tokenizer.ggml.eos_token_ids")],
        compiler_only_symbolic_data=True,actor_structural_state_contains_only_float_vectors=True)


def cpu_only():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized():
        raise ValueError("CPU stage requires CUDA hidden")
    torch.set_num_threads(4)


def prepare(out):
    cpu_only();revision,sources=identity()
    if sha(PREVIOUS/"verification.json")!=PREVIOUS_SHA:
        raise ValueError("fixed source visual diagnostic changed")
    old=json.loads((PREVIOUS/"preparation.json").read_text())
    geometry=json.loads((PREVIOUS/"geometry_audit.json").read_text())
    if geometry["source_verification_sha256"]!=PREVIOUS_SHA:
        raise ValueError("native sampling geometry changed")
    archive,tokenizer,markers=archive_and_tokenizer();controls,metadata=compile_controls(archive,tokenizer,markers)
    out.mkdir(parents=True,exist_ok=False)
    torch.save(controls,out/"structural_vectors.pt");atomic_json(out/"compiler.json",metadata)
    cases=[]
    for row in old["cases"][:8]:
        item={k:row[k] for k in ("position","source","id","article","source_prompt","references")};item["images"]={}
        for route in ("normal","blank"):
            source=row["images"][route];path=PREVIOUS/source["file"]
            if sha(path)!=source["sha256"]:
                raise ValueError("source image changed")
            original=Image.open(path).convert("RGB")
            for scale in ("raw","native_budget"):
                dimensions=geometry["records"][row["position"]]["inferred_native_resized_shape"]
                image=original if scale=="raw" else original.resize((dimensions[1],dimensions[0]),Image.Resampling.BILINEAR)
                name=f"case_{row['position']:02}_{scale}_{route}.png";image.save(out/name)
                item["images"][scale+"/"+route]=dict(file=name,sha256=sha(out/name),shape=[image.height,image.width],
                    source_png_sha256=source["sha256"])
        cases.append(item)
    atomic_json(out/"preparation.json",dict(source_commit=revision,sources=sources,cases=cases,
        previous_verification_sha256=PREVIOUS_SHA,previous_preparation_sha256=sha(PREVIOUS/"preparation.json"),
        geometry_sha256=sha(PREVIOUS/"geometry_audit.json"),structural_vectors_sha256=sha(out/"structural_vectors.pt"),
        compiler_sha256=sha(out/"compiler.json"),pillow_version=PIL.__version__,resize="Pillow RGB bilinear",
        exact_native_pixel_equivalence=False,model_calls=0,gpu_allocations=0,sealed_examples=0))


def image_tensor(path):
    array=np.asarray(Image.open(path).convert("L")).copy()
    return torch.from_numpy(array).float().div(127.5).sub(1)[None,None].clone()


@torch.no_grad()
def generate(out):
    revision,sources=identity();prep=json.loads((out/"preparation.json").read_text())
    if prep["sources"]!=sources or sha(out/"structural_vectors.pt")!=prep["structural_vectors_sha256"]:
        raise ValueError("prepared structural interface changed")
    if (out/"generation_report.json").exists():
        raise FileExistsError("completed generations are immutable")
    if (os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or os.environ.get("CUDA_DEVICE_ORDER")!="PCI_BUS_ID"
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8"):
        raise ValueError("deterministic physical GPU 0 environment required")
    check_host_memory()
    if gpu_memory()>256:
        raise RuntimeError("GPU 0 is occupied; preserve other work")
    lock=(out/"operation.lock").open("a+");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.manual_seed(20267100);torch.cuda.manual_seed_all(20267100)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    begun=time.perf_counter();local=ROOT.parent/"LocalLLM";device=torch.device("cuda:0")
    actor,provenance=load_grounded_image_actor(ollama_root=local/".local/models/ollama",
        gguf_source=local/".local/tools/llama.cpp/gguf-py",device=device)
    actor.writer=torch.nn.Identity();actor.requires_grad_(False).eval()
    if any(bool(p.any()) for n,p in actor.named_parameters() if n.endswith(".up")):
        raise ValueError("source contextual adapters must have zero output")
    controls=torch.load(out/"structural_vectors.pt",map_location="cpu",weights_only=True)
    interface=ImageContextInterface(actor,**controls).requires_grad_(False).eval()
    archive,tokenizer,markers=archive_and_tokenizer()
    compiled,compiler=compile_controls(archive,tokenizer,markers)
    if compiler!=json.loads((out/"compiler.json").read_text()) or any(not torch.equal(compiled[k],controls[k]) for k in controls):
        raise ValueError("numeric source controls differ from their compiler")
    # Local evaluator variable only: never register or attach this vocabulary
    # projection or token-row lookup to the independent image interface.
    source_head=archive.omitted["output.weight"]
    head=torch.from_numpy(np.array(archive.gguf.dequantize(source_head.data,source_head.tensor_type),copy=True)).to(device,torch.bfloat16)
    eog=set(compiler["eog_ids"]);setup_seconds=time.perf_counter()-begun

    def prefill_text(text):
        encode=lambda s:tokenizer.encode(s,add_special_tokens=False).ids
        ids=[markers["im_start"],*encode("user\n"),*encode(TASK+"\n\n"+text),markers["im_end"],
            *encode("\n"),markers["im_start"],*encode("assistant\n")]
        embedding=embed_rows(archive,ids).to(device)[None]
        positions=torch.arange(len(ids),device=device)[None,None].expand(4,1,-1)
        output=actor.context(inputs_embeds=embedding,position_ids=positions,use_cache=True)
        return output.last_hidden_state[:,-1:],output.past_key_values,dict(sequence_length=len(ids),rotary_offset=len(ids),visual_positions=0)

    def decode(state,past,metadata):
        tokens,states,selected_logits=[],[],[];offset=metadata["rotary_offset"];stopped=False
        for step in range(64):
            states.append(state[0,-1].float().cpu())
            logits=F.linear(state[:,-1].float(),head.float())
            if not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite source projection")
            token=int(logits.argmax(-1));tokens.append(token);selected_logits.append(float(logits[0,token]));del logits
            if token in eog:
                stopped=True;break
            if step+1<64:
                embedding=embed_rows(archive,[token]).to(device)[None]
                sequence=past.get_seq_length()
                positions=torch.tensor([sequence,offset,offset,offset],device=device).reshape(4,1,1)
                output=actor.context(inputs_embeds=embedding,position_ids=positions,past_key_values=past,use_cache=True)
                state=output.last_hidden_state[:,-1:];past=output.past_key_values;offset+=1
        text=tokenizer.decode(tokens[:-1] if stopped else tokens,skip_special_tokens=False)
        return dict(selected_ids=tokens,selected_logits=selected_logits,hidden_states=torch.stack(states),text=text,stopped=stopped,capped=not stopped,
            metadata=metadata,helper_only_symbolic_decoding=True)

    files={};completed=0
    for row in prep["cases"]:
        for route in ("raw/normal","raw/blank","native_budget/normal","native_budget/blank","text"):
            name=f"case_{row['position']:02}_"+route.replace("/","_")+".pt";path=out/name
            if not path.exists():
                if route=="text":state,past,metadata=prefill_text(row["source_prompt"])
                else:
                    entry=row["images"][route]
                    if sha(out/entry["file"])!=entry["sha256"]:
                        raise ValueError("prepared input image changed")
                    state,past,metadata=interface(image_tensor(out/entry["file"]))
                value=decode(state,past,metadata);del state,past
                temporary=path.with_suffix(".partial");torch.save(dict(position=row["position"],route=route,
                    structural_vectors_sha256=prep["structural_vectors_sha256"],result=value),temporary);temporary.replace(path)
            saved=torch.load(path,map_location="cpu",weights_only=True)
            if saved["position"]!=row["position"] or saved["route"]!=route or saved["structural_vectors_sha256"]!=prep["structural_vectors_sha256"]:
                raise ValueError("partial generation identity changed")
            files[name]=sha(path);completed+=1
            atomic_json(out/"status.json",dict(stage="reading_imported_images",pid=os.getpid(),completed=completed,total=40))
            print(json.dumps(dict(completed=completed,route=route,text=saved["result"]["text"]),ensure_ascii=False),flush=True)
    if actor.context.embed_tokens is not None or any(n.startswith("head") for n,_ in interface.named_parameters()):
        raise ValueError("helper vocabulary entered the image interface")
    atomic_json(out/"generation_report.json",dict(source_commit=revision,sources=sources,
        preparation_sha256=sha(out/"preparation.json"),raw_files_sha256=files,provenance=provenance,
        elapsed_seconds=time.perf_counter()-begun,setup_seconds=setup_seconds,
        peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,generations=40,
        vocabulary_module_absent=True,source_output_head_helper_only=True,optimizer_updates=0,
        image_actor_generated_images=0,language_quality_claim=False,general_ilm_goal_complete=False))
    atomic_json(out/"status.json",dict(stage="generation_complete",pid=None))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True);parser.add_argument("--stage",choices=("prepare","generate"),required=True)
    args=parser.parse_args();{"prepare":prepare,"generate":generate}[args.stage](args.out)
