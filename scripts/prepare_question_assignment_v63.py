#!/usr/bin/env python3
"""Pin training-only same-paragraph pairs with distinct physical answers."""

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path

import torch

from ilm.visual_lm.visual_qa_data import VisualQARenderer,read_questions
from scripts.prepare_native_pixel_qa_v61 import sha256
from scripts.train_canonical_glyph_binding_v43 import _atomic_json
from scripts.train_native_pixel_instruction_v61 import DEFAULT_DATA,ImageData

PROTOCOL="references/question_assignment_v63_protocol.md"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",default="artifacts/question_assignment_v63_20261105/data")
    args=parser.parse_args();out=Path(args.out)
    if (out/"report.json").exists():raise FileExistsError("completed pair manifest is immutable")
    source=Path("data/visual_qa/cmrc2018_short/train.jsonl")
    if sha256(source)!="742a473db53d1a55e319f0b52befc8da5fa3d884df395b37e66581e882d1496c":
        raise ValueError("training source differs")
    cache=ImageData(DEFAULT_DATA);renderer=VisualQARenderer()
    records=sorted(read_questions(source),key=lambda r:r["id"]);groups=defaultdict(list)
    physical={};maximum={};first_answers={}
    for row in records:
        groups[row["context_hash"]].append(row)
        physical[row["id"]]={hashlib.sha256(renderer(ref["text"]).numpy().tobytes()).hexdigest() for ref in row["references"]}
        pixels,boundary=cache.get("train/"+row["id"]+".pt")
        maximum[row["id"]]=(boundary,pixels.shape[-1]//64-boundary)
        first_answers[row["id"]]=hashlib.sha256(pixels[...,boundary*64:].numpy().tobytes()).hexdigest()
        assert bool((pixels[...,-128:]==1).all())
    pairs=[];rejected=defaultdict(int)
    for context,rows in sorted(groups.items()):
        for a,b in itertools.combinations(rows,2):
            if not {r["text"] for r in a["references"]}.isdisjoint(r["text"] for r in b["references"]):
                rejected["shared_reference_text"]+=1;continue
            if not physical[a["id"]].isdisjoint(physical[b["id"]]) or first_answers[a["id"]]==first_answers[b["id"]]:
                rejected["shared_reference_pixels"]+=1;continue
            ids=[a["id"],b["id"]]
            largest=max(maximum[i][0]+maximum[j][1] for i in ids for j in ids)
            pairs.append(dict(context_hash=context,record_ids=ids,maximum_joined_blocks=largest))
    contexts=sorted({p["context_hash"] for p in pairs})
    out.mkdir(parents=True,exist_ok=True)
    manifest=out/"pairs.jsonl";manifest.write_text("".join(json.dumps(p)+"\n" for p in pairs))
    longest=max(pairs,key=lambda p:p["maximum_joined_blocks"])
    report=dict(training_only=True,development_examples=0,model_calls=0,records=len(records),
        paired_records=len({i for p in pairs for i in p["record_ids"]}),pairs=len(pairs),contexts=len(contexts),
        rejected=dict(rejected),longest_pair=longest,manifest_sha256=sha256(manifest),source_sha256=sha256(source),
        image_cache=str(Path(DEFAULT_DATA)),image_report_sha256=sha256(Path(DEFAULT_DATA)/"report.json"),
        provenance={p:sha256(p) for p in [__file__,PROTOCOL,"ilm/visual_lm/visual_qa_data.py"]},
        selection="sorted complete pairs; disjoint retained text and physical references; uniform context then pair")
    _atomic_json(report,out/"report.json");print(json.dumps(report,indent=2))


if __name__=="__main__":main()
