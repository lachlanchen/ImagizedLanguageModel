#!/usr/bin/env python3
"""Prepare blinded training-only references and negative image controls."""

import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageOps
import torch

from ilm.visual_lm.visual_cell_data import pack_visual_cells
from ilm.visual_lm.visual_qa_data import VisualQARenderer,read_questions


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",default="artifacts/v62_transcript_calibration_20261105")
    args=parser.parse_args();out=Path(args.out)
    if (out/"report.json").exists():raise FileExistsError("completed calibration images are immutable")
    source=Path("data/visual_qa/cmrc2018_short/train.jsonl")
    if digest(source)!="742a473db53d1a55e319f0b52befc8da5fa3d884df395b37e66581e882d1496c":
        raise ValueError("training source changed")
    groups=defaultdict(list)
    for row in sorted(read_questions(source),key=lambda r:r["id"]):groups[row["article_title"]].append(row)
    titles=sorted(groups);rng=np.random.default_rng(20266370)
    if len(titles)<128:raise ValueError("insufficient distinct training articles")
    selected=[]
    for index in rng.choice(len(titles),128,replace=False):
        candidates=groups[titles[index]];selected.append(candidates[int(rng.integers(len(candidates)))])
    renderer=VisualQARenderer();out.mkdir(parents=True,exist_ok=True);images=out/"images";images.mkdir(exist_ok=True)
    records=[]

    def save(cells,kind,expected,identifier=None):
        raw=pack_visual_cells(cells,columns=len(cells),gutter=0)
        enlarged=raw.resize((raw.width*4,raw.height*4),Image.Resampling.NEAREST)
        view=ImageOps.expand(enlarged,border=16,fill=255)
        buffer=io.BytesIO();view.save(buffer,format="PNG");data=buffer.getvalue()
        sha=hashlib.sha256(data).hexdigest();path=images/(sha+".png")
        if path.exists() and path.read_bytes()!=data:raise ValueError("image hash collision")
        path.write_bytes(data)
        records.append(dict(image=str(path.relative_to(out)),image_sha256=sha,kind=kind,
                            expected_text=expected,source_id=identifier))

    for row in selected:save(renderer(row["references"][0]["text"]),"reference",row["references"][0]["text"],row["id"])
    noise=np.random.default_rng(20266372)
    for length in range(1,17):
        save(torch.zeros(length,1,32,32),"blank","")
        save(torch.from_numpy((noise.random((length,1,32,32))<.15).astype(np.float32)),"random_ink","")
    manifest=out/"private_expected.jsonl"
    manifest.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in records))
    report=dict(records=len(records),references=128,reference_articles=128,blank=16,random_ink=16,
        unique_images=len(set(row["image_sha256"] for row in records)),source_sha256=digest(source),
        manifest_sha256=digest(manifest),source_code_sha256=digest(__file__),
        protocol_sha256=digest("references/v62_blind_vision_transcription_plan.md"),
        image_transform="nearest neighbour 4x, 16 white pixels each side; no repair",
        training_only=True,development_examples=0,model_calls=0,
        warning="Expected labels and source IDs are private scoring metadata and must never enter recognition requests")
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
