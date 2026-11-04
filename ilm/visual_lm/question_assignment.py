"""Paired question/answer-image likelihood, outside the deployed actor."""

from collections import defaultdict
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.prepare_native_pixel_qa_v61 import sha256
from scripts.train_native_pixel_instruction_v61 import ImageData


def assignment_objective(energies,weight,temperature=.01):
    if energies.shape!=(2,2) or weight not in (0.,1.):
        raise ValueError("expected a two-question assignment and a registered weight")
    margin=energies[0,1]+energies[1,0]-energies[0,0]-energies[1,1]
    positive=(energies[0,0]+energies[1,1])*.5
    contrast=temperature*F.softplus(-margin/temperature)
    return positive+weight*contrast,dict(positive_nll=positive,assignment_loss=contrast,assignment_margin=margin)


class QuestionImagePairs:
    def __init__(self,path):
        self.path=Path(path);self.report=json.loads((self.path/"report.json").read_text())
        if not self.report["training_only"] or self.report["development_examples"]!=0:
            raise ValueError("only training pairs allowed")
        if sha256(self.path/"pairs.jsonl")!=self.report["manifest_sha256"]:
            raise ValueError("pair manifest changed")
        if any(sha256(p)!=h for p,h in self.report["provenance"].items()):
            raise ValueError("pair preparation sources changed")
        self.cache=ImageData(self.report["image_cache"])
        if sha256(self.cache.path/"report.json")!=self.report["image_report_sha256"]:
            raise ValueError("image cache identity changed")
        self.pairs=[json.loads(s) for s in (self.path/"pairs.jsonl").read_text().splitlines()]
        if len(self.pairs)!=self.report["pairs"]:raise ValueError("pair count differs")
        groups=defaultdict(list)
        for index,row in enumerate(self.pairs):groups[row["context_hash"]].append(index)
        self.contexts=[groups[k] for k in sorted(groups)]

    def sample(self,generator):
        group=int(torch.randint(len(self.contexts),(1,),generator=generator))
        candidates=self.contexts[group]
        return candidates[int(torch.randint(len(candidates),(1,),generator=generator))]

    def streams(self,index):
        row=self.pairs[index];prefixes=[];answers=[]
        for identifier in row["record_ids"]:
            pixels,boundary=self.cache.get("train/"+identifier+".pt")
            prefixes.append(pixels[...,:boundary*64]);answers.append(pixels[...,boundary*64:])
        streams=[(torch.cat((p,y),-1),p.shape[-1]//64) for p in prefixes for y in answers]
        if max(p.shape[-1]//64 for p,_ in streams)!=row["maximum_joined_blocks"]:
            raise ValueError("physical assembly differs from pair manifest")
        return streams


def image_assignment_energies(model,streams,device):
    if len(streams)!=4:raise ValueError("four physical streams required")
    return torch.stack([model.answer_nll(pixels.to(device),boundary) for pixels,boundary in streams]).reshape(2,2)
