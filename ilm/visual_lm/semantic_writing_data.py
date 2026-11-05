"""Authenticated continuous training cache; never part of image inference."""

from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path

import torch


class SemanticWritingData:
    def __init__(self,path,*,preflight=False):
        self.path = Path(path)
        self.report_bytes = (self.path/"report.json").read_bytes()
        self.report_sha256 = hashlib.sha256(self.report_bytes).hexdigest()
        self.report = json.loads(self.report_bytes)
        self.recipe_bytes = (self.path/"recipe.json").read_bytes()
        self.recipe_sha256 = hashlib.sha256(self.recipe_bytes).hexdigest()
        self.recipe = json.loads(self.recipe_bytes)
        if self.recipe_sha256 != self.report["recipe_sha256"]:
            raise ValueError("cache recipe digest changed")
        if (self.report["total"] != 12976 or self.report["completed"] != len(self.report["files_sha256"])
                or (not preflight and not self.report["complete"])):
            raise ValueError("production requires the entire admitted corpus")
        if not self.recipe["prompt_only_memory"] or self.recipe["symbolic_helper_loaded"]:
            raise ValueError("cache used incompatible semantic inputs")
        self.names = sorted(self.report["files_sha256"])
        if not self.names:
            raise ValueError("empty cache")
        for name in self.names:
            p = Path(name)
            if p.is_absolute() or ".." in p.parts or p.suffix != ".pt":
                raise ValueError("unsafe cache payload path")

    def __len__(self):
        return len(self.names)

    @lru_cache(maxsize=256)
    def __getitem__(self,index):
        name = self.names[index]
        raw = (self.path/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.report["files_sha256"][name]:
            raise ValueError("cached training payload changed")
        row = torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
        if row["recipe_sha256"] != self.recipe_sha256:
            raise ValueError("payload recipe changed")
        if row["name"] != self.recipe["names"][int(Path(name).stem)]:
            raise ValueError("cache order/source differs")
        memory,codes = row["memory"],row["codes"]
        if (memory.dtype != torch.bfloat16 or memory.ndim != 2 or memory.shape[1] != 4096
                or codes.dtype != torch.float32 or codes.ndim != 3 or codes.shape[1:] != (2,768)
                or not 1 <= len(codes) <= 128 or not torch.isfinite(memory).all()
                or not torch.isfinite(codes).all()):
            raise ValueError("invalid image-semantic training tensors")
        return memory,codes

    def batch(self,indices,device):
        rows = [self[i] for i in indices]
        memory = torch.stack([r[0] for r in rows]).to(device)
        lengths = torch.tensor([len(r[1]) for r in rows],device=device,dtype=torch.long)
        codes = torch.zeros((len(rows),int(lengths.max()),2,768),device=device)
        for i,(_,value) in enumerate(rows):
            codes[i,:len(value)] = value.to(device)
        return memory,codes,lengths


class ExactEpochSampler:
    """Fresh full-corpus permutations with serialized generator and cursor."""

    def __init__(self,size,seed=20267202):
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size,generator=self.generator)
        self.cursor = 0
        self.epoch = 0
        self.draws = 0

    def take(self,count):
        result = []
        while len(result)<count:
            if self.cursor==self.size:
                self.order = torch.randperm(self.size,generator=self.generator)
                self.cursor = 0
                self.epoch += 1
            selected = self.order[self.cursor:self.cursor+count-len(result)].tolist()
            result.extend(selected)
            self.cursor += len(selected)
        self.draws += count
        return result

    def state_dict(self):
        return dict(size=self.size,order=self.order.clone(),cursor=self.cursor,
            epoch=self.epoch,draws=self.draws,generator=self.generator.get_state())

    def load_state_dict(self,state):
        if state["size"] != self.size or not torch.equal(state["order"].sort().values,torch.arange(self.size)):
            raise ValueError("invalid sampler state")
        self.order = state["order"].clone()
        self.cursor,self.epoch,self.draws = state["cursor"],state["epoch"],state["draws"]
        self.generator.set_state(state["generator"])
