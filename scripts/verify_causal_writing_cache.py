#!/usr/bin/env python3
"""Verify incremental states against actual trained full-history computation."""

import argparse
import json
import os
from pathlib import Path
import time

import torch

from ilm.visual_lm.causal_writing_cache import CausalWritingCache
from ilm.visual_lm.clean_writing_flow import CleanSemanticWritingModel, CleanWritingConfig
from ilm.visual_lm.semantic_writing_data import SemanticWritingData
from scripts.cache_semantic_images_v72 import ROOT, TRAIN, sha, committed_identity, atomic_json

SOURCES=("scripts/verify_causal_writing_cache.py","ilm/visual_lm/causal_writing_cache.py",
    "ilm/visual_lm/semantic_writing_model.py","ilm/visual_lm/semantic_writing_data.py",
    "ilm/visual_lm/clean_writing_flow.py","references/causal_writing_cache_v76_protocol.md")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized():
        raise ValueError("CPU-only numerical checking required")
    if args.out.exists():
        raise ValueError("retain completed evidence; use a fresh report path")
    torch.set_num_threads(4)
    identity=committed_identity(SOURCES)
    receipt=json.loads((ROOT/"references/evidence/clean_learning_v74_result.json").read_text())
    path=ROOT/receipt["source_run"]/"training/checkpoint_latest.pt"
    if sha(path)!=receipt["checkpoint_sha256"]:
        raise ValueError("completed paired-learning model changed")
    saved=torch.load(path,map_location="cpu",weights_only=True)
    if saved["update"]!=4000 or saved["identity"]["purpose"]!="paired_learning_control":
        raise ValueError("use the completed known checkpoint, not live V75 training")
    model=CleanSemanticWritingModel(CleanWritingConfig(**saved["model_config"]))
    model.load_state_dict(saved["model"],strict=True)
    model.eval()
    del saved
    data=SemanticWritingData(ROOT/"artifacts/semantic_ilm_v72_20261105/cache")
    source_index={name:i for i,name in enumerate(data.recipe["names"])}
    first_chinese=next(i for i,name in enumerate(data.recipe["names"]) if name.startswith("cmrc/"))
    manifest=TRAIN/"dolly_accepted.jsonl"
    source_report=json.loads((TRAIN/"report.json").read_text())
    if sha(manifest)!=source_report["summaries"]["dolly"]["accepted_manifest_sha256"]:
        raise ValueError("complete-answer manifest changed")
    long_source=min(json.loads(line)["file"] for line in manifest.read_text().splitlines()
        if json.loads(line)["answer"]["blocks"]==128)
    selected=[first_chinese,source_index[long_source]]
    flags=next(line for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("flags"))
    native_bf16=bool({"avx512_bf16","amx_bf16"}.intersection(flags.split()))
    precisions=("float32","bfloat16") if native_bf16 else ("float32",)
    results=[]
    for index in selected:
        memory,codes=data[index]
        memory=memory[None]
        n=len(codes)
        for precision in precisions:
            try:
                with torch.inference_mode(), torch.autocast("cpu",dtype=torch.bfloat16,enabled=precision=="bfloat16"):
                    begun=time.perf_counter()
                    original=torch.stack([model.conditions(memory,codes[None,:i])[:, -1]
                        for i in range(n+1)])
                    full_seconds=time.perf_counter()-begun
                    begun=time.perf_counter()
                    cache=CausalWritingCache(model)
                    states=[cache.start(memory)]
                    for block in codes:
                        states.append(cache.append(block[None]))
                    cached=torch.stack(states)
                    cached_seconds=time.perf_counter()-begun
                    difference=(original.float()-cached.float())
                    stop_original=model.stop(original).float().sigmoid()
                    stop_cached=model.stop(cached).float().sigmoid()
                    close=torch.allclose(original,cached,rtol=1e-5,atol=2e-5)
                    stopping=torch.equal(stop_original>=.5,stop_cached>=.5)
                    if precision=="float32":
                        torch.testing.assert_close(original,cached,rtol=1e-5,atol=2e-5)
                        if not stopping:
                            raise AssertionError("FP32 stopping decisions differ")
                    if cache.sequence_length!=memory.shape[1]+n+1:
                        raise AssertionError("cache sequence growth differs")
                    if n==128:
                        try:
                            cache.append(codes[-1:])
                        except ValueError:
                            pass
                        else:
                            raise AssertionError("history capacity was not enforced")
                    prefix=memory.shape[1]+1
                    results.append(dict(source=data.recipe["names"][index],cache_file=data.names[index],
                        cache_payload_sha256=data.report["files_sha256"][data.names[index]],
                        blocks=n,conditions=n+1,precision=precision,
                        max_abs_error=float(difference.abs().max()),rms_error=float(difference.square().mean().sqrt()),
                        fp32_tolerance_satisfied=bool(close),stop_thresholds_equal=stopping,
                        max_stop_probability_difference=float((stop_original-stop_cached).abs().max()),
                        full_history_seconds=full_seconds,cached_seconds=cached_seconds,
                        final_cache_bytes=cache.storage_bytes,
                        full_projection_rows_per_layer=sum(prefix+i for i in range(n+1)),
                        cached_projection_rows_per_layer=prefix+n,
                        full_attention_scores_per_head=sum((prefix+i)**2 for i in range(n+1)),
                        cached_attention_scores_per_head=prefix**2+sum(prefix+i for i in range(1,n+1))))
            except RuntimeError as error:
                if precision!="bfloat16":
                    raise
                results.append(dict(source=data.recipe["names"][index],precision=precision,
                    cpu_backend_unsupported=str(error)))
    with torch.inference_mode():
        cache=CausalWritingCache(model)
        cache.start(memory)
        model.start.add_(.125)
        try:
            cache.append(codes[:1])
        except ValueError:
            stale_guard=True
        else:
            raise AssertionError("parameter mutation did not invalidate the cache")
    report=dict(identity=identity,checkpoint_sha256=receipt["checkpoint_sha256"],
        cache_report_sha256=data.report_sha256,torch_version=str(torch.__version__),results=results,
        cpu_bf16_skipped_without_native_support=not native_bf16,
        stale_parameter_guard_pass=stale_guard,image_reader_codec_sampler_calls=0,
        autonomous_image_generation_calls=0,gpu_allocations=0,
        assisted_reference_history=True,actual_gpu_generation_replayed=False,
        used_by_running_v75=False,language_quality_claim=False,general_ilm_goal_complete=False)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(args.out,report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
