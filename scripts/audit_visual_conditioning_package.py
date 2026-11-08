#!/usr/bin/env python3
"""Inspect every pinned component header; never download tensor payloads.

Hub summary counts can omit nonstandard components. Stored tensor elements
also include buffers, so they are not automatically runtime parameter counts.
Optional meta construction checks that distinction without loading weights.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
REPO="deepgenteam/DeepGen-1.0-diffusers"
REVISION="85c2ed257b9406d8531965c68e140dc7fdb81382"
BASE_REPO="Qwen/Qwen2.5-VL-3B-Instruct"
BASE_REVISION="66285546d2b821cf421d4f5eb2576359d3770cd3"
DTYPE_BYTES={"BF16":2,"F16":2,"F32":4,"F64":8,"I64":8,"I32":4,"I16":2,"I8":1,"U8":1,"BOOL":1}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_url(url,limit):
    with urllib.request.urlopen(url,timeout=30) as response:body=response.read(limit+1)
    if len(body)>limit:raise ValueError("bounded metadata payload exceeded")
    return body


def metadata(repo,revision,path):
    if path.exists():value=json.loads(path.read_text())
    else:
        value=json.loads(read_url(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true",2**20))
        path.write_text(json.dumps(value,indent=2)+"\n")
    if value["sha"]!=revision or value.get("gated"):raise ValueError("pinned public revision required")
    return value


def small_file(row,output):
    name=row["rfilename"];target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
    body=target.read_bytes() if target.exists() else read_url(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}",2**20)
    if len(body)!=row["size"] or hashlib.sha1(f"blob {len(body)}\0".encode()+body).hexdigest()!=row["blobId"]:
        raise ValueError("small source file differs from pinned Git blob: "+name)
    if not target.exists():target.write_bytes(body)
    return sha(target)


def tensor_header(row,output):
    name=row["rfilename"];path=output/"headers"/(name.replace("/","_")+".prefix")
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        url=f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
        def prefix(count):
            request=urllib.request.Request(url,headers={"Range":f"bytes=0-{count-1}","Accept-Encoding":"identity"})
            with urllib.request.urlopen(request,timeout=45) as response:
                content_range=response.headers.get("Content-Range")
                if content_range and not content_range.startswith("bytes 0-"):
                    raise ValueError("unexpected header range start")
                # Even if a server ignores Range, stop after this bounded prefix.
                body=response.read(count)
            if len(body)!=count:raise ValueError("truncated safetensors prefix")
            return body
        size=struct.unpack("<Q",prefix(8))[0]
        if not 2<=size<=4*2**20:raise ValueError("unexpected safetensors header size")
        body=prefix(8+size)
        if struct.unpack("<Q",body[:8])[0]!=size:raise ValueError("safetensors prefix changed between requests")
        path.write_bytes(body)
    body=path.read_bytes();size=struct.unpack("<Q",body[:8])[0]
    if len(body)!=size+8:raise ValueError("stored prefix must contain only header bytes")
    header=json.loads(body[8:]);tensors={k:v for k,v in header.items() if k!="__metadata__"}
    counts={};intervals=[]
    for name_,value in tensors.items():
        count=math.prod(value["shape"]);dtype=value["dtype"];lo,hi=value["data_offsets"]
        if hi-lo!=count*DTYPE_BYTES[dtype]:raise ValueError("tensor shape/offset/dtype mismatch")
        counts[dtype]=counts.get(dtype,0)+count;intervals.append((lo,hi,name_))
    intervals.sort();end=0
    for lo,hi,_ in intervals:
        if lo!=end:raise ValueError("noncontiguous tensor offsets")
        end=hi
    if end+8+size!=row["size"]:raise ValueError("header offsets do not describe the pinned file size")
    return dict(file=name,file_bytes=row["size"],expected_full_file_sha256=row["lfs"]["sha256"],
        full_file_checksum_verified=False,header_prefix_sha256=sha(path),header_prefix_bytes=len(body),
        stored_tensors=len(tensors),stored_elements=sum(counts.values()),elements_by_dtype=counts,
        tensor_payload_bytes=end,tensor_payload_downloaded=False,
        named_elements={k:math.prod(v["shape"]) for k,v in tensors.items()})


def meta_counts(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("hide CUDA for meta structure inspection")
    import torch
    from diffusers import AutoencoderKL
    from transformers import Qwen2_5_VLConfig,Qwen2_5_VLForConditionalGeneration
    compatibility=ROOT/"artifacts/visual_conditioning_compatibility_v1"
    prepared=json.loads((compatibility/"inputs.json").read_text());wheel=compatibility/prepared["einops_wheel"]
    if sha(wheel)!=prepared["files_sha256"][wheel.name]:raise ValueError("isolated dependency changed")
    sys.path.insert(0,str(wheel))
    source=ROOT/"artifacts/external/visual_conditioning_source_audit"/("deepgen_diffusers_"+REVISION)/"deepgen_pipeline.py"
    if sha(source)!="41e0f263d9ad55940426a4b2092688a74711395499c12a71ee69dc378ec9d3a9":raise ValueError("audited source changed")
    spec=importlib.util.spec_from_file_location("ilm_packaging_deepgen_source",source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    results={}
    for component in ("transformer","vae","connector","vlm"):
        config=json.loads((output/component/"config.json").read_text())
        with torch.device("meta"):
            if component=="transformer":model=module.SD3Transformer2DModel.from_config(config)
            elif component=="vae":model=AutoencoderKL.from_config(config)
            elif component=="vlm":model=Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLConfig.from_dict(config))
            else:
                connector=config["connector"].copy();connector["_attn_implementation"]="sdpa"
                keys=("num_queries","llm_hidden_size","projector_1_in","projector_1_out","projector_2_in","projector_2_out","projector_3_in","projector_3_out")
                model=module.DeepGenConnector(connector_config=connector,**{k:config[k] for k in keys})
        results[component]=dict(parameters=sum(p.numel() for p in model.parameters()),
            buffers=sum(p.numel() for p in model.buffers()),
            all_parameters_meta=all(p.is_meta for p in model.parameters()),
            all_buffers_meta=all(p.is_meta for p in model.buffers()),
            buffer_elements={k:v.numel() for k,v in model.named_buffers()})
        state_keys=set(model.state_dict())
        results[component]["persistent_buffer_elements"]=sum(v.numel() for k,v in model.named_buffers() if k in state_keys)
        if component=="vlm":
            results[component]["output_head_tied_to_input"]=model.get_output_embeddings().weight is model.get_input_embeddings().weight
            model.tie_weights()
            results[component]["parameters_after_native_tie"]=sum(p.numel() for p in model.parameters())
            results[component]["output_head_tied_after_native_tie"]=model.get_output_embeddings().weight is model.get_input_embeddings().weight
        del model
    if torch.cuda.is_initialized():raise AssertionError("metadata probe initialized CUDA")
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--meta",action="store_true")
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/"result.json").exists():raise ValueError("preserve completed audit")
    hub=metadata(REPO,REVISION,output/"hub_metadata.json")
    base=metadata(BASE_REPO,BASE_REVISION,output/"qwen_base_metadata.json")
    configs={}
    for row in hub["siblings"]:
        if row["rfilename"] in ("connector/config.json","transformer/config.json","vae/config.json","vlm/config.json","vlm/model.safetensors.index.json","prompt_template.json"):
            configs[row["rfilename"]]=small_file(row,output)
    records=[]
    for row in hub["siblings"]:
        if row["rfilename"].endswith(".safetensors"):
            records.append(tensor_header(row,output));print(row["rfilename"],records[-1]["stored_elements"],flush=True)
    components={name:sum(r["stored_elements"] for r in records if r["file"].startswith(name+"/")) for name in ("connector","transformer","vae","vlm")}
    if sum(components.values())!=sum(r["stored_elements"] for r in records):raise ValueError("unaccounted weight component")
    base_hashes={r["rfilename"]:r.get("lfs",{}).get("sha256") for r in base["siblings"] if r["rfilename"].endswith(".safetensors")}
    packaged_hashes={Path(r["file"]).name:r["expected_full_file_sha256"] for r in records if r["file"].startswith("vlm/")}
    result=dict(repo=REPO,revision=REVISION,files=records,configs_sha256=configs,
        component_stored_elements=components,total_stored_elements=sum(components.values()),
        hub_summary_elements=hub.get("safetensors",{}).get("total"),packaged_vlm_present=bool(packaged_hashes),
        packaged_vlm_files_identical_to_base_files=packaged_hashes==base_hashes,
        differing_file_hashes_do_not_alone_prove_tensor_value_changes=True,
        base_source_repo=BASE_REPO,base_source_revision=BASE_REVISION,base_file_sha256=base_hashes,
        total_weight_file_bytes=sum(r["file_bytes"] for r in records),tensor_payload_downloaded=False,
        weights_loaded=False,neural_forward_performed=False,implementation_sha256=sha(__file__),
        runtime_parameter_counts_from_loaded_weights_established=False,
        source_candidate_quality_established=False)
    if args.meta:
        result["meta_structure"]=meta_counts(output)
        result["meta_architecture_parameter_total"]=sum(v["parameters"] for v in result["meta_structure"].values())
        result["meta_architecture_buffer_total"]=sum(v["buffers"] for v in result["meta_structure"].values())
        result["meta_parameter_total_after_native_tie"]=sum(v.get("parameters_after_native_tie",v["parameters"]) for v in result["meta_structure"].values())
        result["meta_unique_parameters_plus_persistent_buffers_after_native_tie"]=result["meta_parameter_total_after_native_tie"]+sum(v["persistent_buffer_elements"] for v in result["meta_structure"].values())
    (output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("component_stored_elements","total_stored_elements","hub_summary_elements","total_weight_file_bytes","packaged_vlm_present","packaged_vlm_files_identical_to_base_files")},indent=2))


if __name__=="__main__":main()
