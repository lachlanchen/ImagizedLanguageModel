#!/usr/bin/env python3
"""Prepare complete Chinese/English writing images on CPU, training split only."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.full_text_renderer import CompleteLineRenderer

SOURCES={
    "cmrc":("data/visual_qa/cmrc2018_short/train.jsonl",3211,"742a473db53d1a55e319f0b52befc8da5fa3d884df395b37e66581e882d1496c"),
    "dolly":("data/general_instructions/dolly15k/train.jsonl",12070,"86cb87ccd8a6742177f2fd6907bd83ee263f4303a4c737c3eaf7c5f296155b3a"),
}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",default="data/general_instructions/complete_lines_train_20261105")
    args=parser.parse_args();out=Path(args.out)
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("explicit CPU-only CUDA_VISIBLE_DEVICES='' required")
    if (out/"report.json").exists():raise FileExistsError("completed candidate cache is immutable")
    torch.set_num_threads(2);renderer=CompleteLineRenderer()
    provenance={name:digest(name) for name in [__file__,"ilm/visual_lm/full_text_renderer.py",
        "references/full_text_renderer_plan.md"]}
    identity=dict(source_sha256={name:expected for name,(_,_,expected) in SOURCES.items()},
        renderer=renderer.provenance,source_code_sha256=provenance,
        maximum_prompt_blocks=240,maximum_answer_blocks=128,termination_blocks=2)
    out.mkdir(parents=True,exist_ok=True)
    if (out/"identity.json").exists() and json.loads((out/"identity.json").read_text())!=identity:
        raise ValueError("partial candidate cache identity changed")
    (out/"identity.json").write_text(json.dumps(identity,indent=2)+"\n")
    begun=time.perf_counter();summaries={};files={};first=[]
    for source,(path,count,expected) in SOURCES.items():
        if digest(path)!=expected:raise ValueError("pinned training source changed")
        records=[json.loads(line) for line in Path(path).read_text().splitlines()]
        if len(records)!=count:raise ValueError("training source count differs")
        folder=out/source;folder.mkdir(exist_ok=True);counts=Counter();categories=Counter()
        with (out/f"{source}_accepted.jsonl").open("w") as accepted,(out/f"{source}_rejected.jsonl").open("w") as rejected:
            for index,record in enumerate(records):
                counts["source_records"]+=1
                prompt=record["prompt_flat"] if source=="dolly" else record["prompt"]
                answer=record["answer_flat"] if source=="dolly" else record["references"][0]["text"]
                try:
                    p,pm=renderer.render(prompt,maximum_blocks=240)
                    a,am=renderer.render(answer,maximum_blocks=128)
                except (ValueError,OverflowError) as error:
                    counts["rejected"]+=1;counts[str(error)]+=1
                    rejected.write(json.dumps(dict(id=record["id"],reason=str(error)))+"\n")
                    if index<4:first.append(dict(source=source,id=record["id"],accepted=False,reason=str(error)))
                    continue
                category=record.get("category","chinese_reading")
                counts["accepted"]+=1;categories[category]+=1
                pixels=np.concatenate((p,a,np.full((32,128),255,dtype=np.uint8)),axis=1)
                bits=np.packbits((pixels.reshape(-1)>=128).astype(np.uint8))
                name=folder/(record["id"]+".pt")
                temporary=name.with_suffix(".partial")
                torch.save(dict(bits=torch.from_numpy(bits),width=pixels.shape[1],prompt_blocks=pm["blocks"],
                    id=record["id"],source=source),temporary);temporary.replace(name)
                files[str(name.relative_to(out))]=digest(name)
                accepted.write(json.dumps(dict(id=record["id"],file=str(name.relative_to(out)),category=category,
                    prompt=pm,answer=am),ensure_ascii=False)+"\n")
                # Lossless pack/unpack is checked against the actual complete
                # prompt, answer and termination image for every retained row.
                restored=np.unpackbits(bits,count=pixels.size).reshape(pixels.shape)*255
                if not np.array_equal(restored,pixels):raise AssertionError("image bits changed")
                if index<4:
                    Image.fromarray(p,mode="L").save(out/f"{source}_{index}_prompt.png")
                    Image.fromarray(a,mode="L").save(out/f"{source}_{index}_answer.png")
                    first.append(dict(source=source,id=record["id"],accepted=True,prompt=pm,answer=am))
                if index%1000==0:print(json.dumps(dict(source=source,processed=index+1,accepted=counts["accepted"])),flush=True)
        summaries[source]=dict(counts=counts,categories=categories,
            accepted_manifest_sha256=digest(out/f"{source}_accepted.jsonl"),
            rejected_manifest_sha256=digest(out/f"{source}_rejected.jsonl"))
    report=dict(**identity,summaries=summaries,files_sha256=files,first_fixed_samples=first,
        elapsed_seconds=time.perf_counter()-begun,training_only=True,model_calls=0,gpu_allocations=0,
        all_retained_rows_lossless=True,original_sources_modified=False,V62_data_modified=False,
        language_quality_claim=False,candidate_only=True,
        licenses=dict(cmrc="CC-BY-SA-4.0",dolly="CC-BY-SA-3.0"),
        limitations="New proportional layout and size differ from both old cell grids and released native font. Training/evaluation protocol still required.")
    (out/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=="__main__":main()
