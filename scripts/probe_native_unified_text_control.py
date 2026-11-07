#!/usr/bin/env python3
"""Compare the same native body on original rendered text, never a student."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import torch

from ilm.image_edit.paired_data import copy_cases, qa_candidates
from ilm.native_unified.showo2_source import load, sha
from probe_native_unified_source import gpu_slot, save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets",type=Path,default=Path("artifacts/external/native_unified_source"))
    p.add_argument("--images",type=Path,default=Path("artifacts/native_unified_probe/prepare_v1/inputs.json"))
    p.add_argument("--manifest",type=Path,default=Path("artifacts/image_edit_learning/pages_v2/pairs.jsonl"))
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/"status.json",dict(state="running"))
    try:
        rows=json.loads(args.images.read_text())
        if len(rows)!=12:raise ValueError("complete twelve-case control required")
        pages={x["id"]:x for x in [json.loads(s) for s in args.manifest.read_text().splitlines()]}
        source={x["id"]:x["prompt"] for x in copy_cases()}
        source.update({x["id"]:x["prompt"] for pair in qa_candidates(Path("data/visual_qa/cmrc2018_short/train.jsonl")) for x in pair})
        prompts=[]
        for row in rows:
            text=source[row["id"]]
            digest=hashlib.sha256(text.encode()).hexdigest()
            if digest!=pages[row["id"]]["input_geometry"]["text_sha256"]:
                raise ValueError("text does not match the actual rendered page")
            prompts.append(dict(id=row["id"],text=text,sha256=digest))
        save(args.output/"source_prompts.json",prompts)
        start=time.monotonic();results=[];torch.set_num_threads(8)
        with gpu_slot(),torch.inference_mode():
            b=load(args.assets,dtype=torch.float32);torch.cuda.reset_peak_memory_stats()
            for i,row in enumerate(prompts):
                text="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"+row["text"]+"\nFollow the request. Output only the requested answer, with no explanation.\n<|im_start|>assistant\n"
                ids=torch.tensor([b.tokenizer(text,add_special_tokens=False).input_ids],device=b.device)
                prefix=b.model.showo.model.embed_tokens(ids)
                mask=b.native.masks.causal_attn_mask_naive(1,prefix.shape[1],b.device).to(b.dtype)
                torch.manual_seed(20268821+i);torch.cuda.manual_seed_all(20268821+i)
                generated=b.model.mmu_generate(input_embeds=prefix,attention_mask=mask,max_new_tokens=96,
                    top_k=1,temperature=1.0,eos_token=b.tokenizer.eos_token_id)
                generated=[int(x) for x in generated]
                result=dict(id=row["id"],source_text_sha256=row["sha256"],seed=20268821+i,
                    ids=generated,text=b.tokenizer.decode(generated,skip_special_tokens=True),
                    stopped=bool(generated and generated[-1]==b.tokenizer.eos_token_id),
                    native_text_input=True,student_answer=False)
                results.append(result);save(args.output/"outputs.json",results)
                print(row["id"],repr(result["text"]),flush=True)
            save(args.output/"report.json",dict(complete_calls=len(results),counts=b.counts,
                elapsed_seconds=time.monotonic()-start,peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
                implementation_sha256=sha(__file__),source_adapter_sha256=sha(Path(__file__).parents[1]/"ilm/native_unified/showo2_source.py"),
                optimizer_updates=0,student_language_test=False,language_quality_qualified=False,general_ilm_goal_complete=False))
        save(args.output/"status.json",dict(state="completed"))
    except Exception as e:
        save(args.output/"status.json",dict(state="failed",error=repr(e),traceback=traceback.format_exc()));raise


if __name__=="__main__":main()
