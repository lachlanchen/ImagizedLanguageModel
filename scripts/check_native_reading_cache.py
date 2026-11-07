#!/usr/bin/env python3
"""Replay every original native readout decision using incremental KV state."""
import argparse
import json
from pathlib import Path
import time
import torch

from ilm.native_unified.showo2_source import load, reading_prefix, sha
from ilm.native_unified.reading import cached_decisions
from probe_native_unified_source import gpu_slot, save, ANSWER_INSTRUCTION, TRANSCRIBE_INSTRUCTION


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets",type=Path,default=Path("artifacts/external/native_unified_source"))
    p.add_argument("--original",type=Path,default=Path("artifacts/native_unified_probe/reading_v1"))
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    original=json.loads((args.original/"outputs.json").read_text())
    inputs=json.loads(Path("artifacts/native_unified_probe/prepare_v1/inputs.json").read_text())
    if len(original)!=20 or len(inputs)!=12:raise ValueError("complete fixed source observation required")
    positions={x["id"]:i for i,x in enumerate(inputs)}
    start=time.monotonic();results=[];torch.set_num_threads(8)
    save(args.output/"status.json",dict(state="running"))
    with gpu_slot(),torch.inference_mode():
        bundle=load(args.assets,dtype=torch.float32);torch.cuda.reset_peak_memory_stats()
        for row in original:
            folder=args.original/f'{positions[row["id"]]:02}'
            latent_path=folder/"latent.pt"
            if sha(latent_path)!=row["latent_sha256"]:raise ValueError("original input latent changed")
            z=torch.load(latent_path,map_location=bundle.device,weights_only=True)
            instruction=ANSWER_INSTRUCTION if row["kind"]=="answer" else TRANSCRIBE_INSTRUCTION
            prefix,mask=reading_prefix(bundle,z,instruction)
            t=time.monotonic();ids=cached_decisions(bundle,prefix,mask,seed=row["readout_seed"],cap=row["cap"])
            result=dict(id=row["id"],kind=row["kind"],all_decisions_equal=ids==row["ids"],
                original_ids=row["ids"],cached_ids=ids,original_seconds=row["elapsed_seconds"],
                cached_seconds=time.monotonic()-t)
            results.append(result);save(args.output/"results.json",results)
            print(row["id"],row["kind"],result["all_decisions_equal"],flush=True)
        report=dict(calls=20,equal_calls=sum(x["all_decisions_equal"] for x in results),
            original_decisions=sum(len(x["original_ids"]) for x in results),
            original_decode_seconds=sum(x["original_seconds"] for x in results),cached_decode_seconds=sum(x["cached_seconds"] for x in results),
            elapsed_seconds=time.monotonic()-start,peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
            model_counts=bundle.counts,implementation_sha256=sha(__file__),
            adapter_source_sha256=sha(Path(__file__).parents[1]/"ilm/native_unified/reading.py"),
            optimizer_updates=0,quality_qualification=False,general_ilm_goal_complete=False)
        save(args.output/"report.json",report)
    save(args.output/"status.json",dict(state="completed",equal_calls=report["equal_calls"]))
    if report["equal_calls"]!=20:raise ValueError("cache decisions differ; preserve and reject equivalence claim")


if __name__=="__main__":main()
