"""Predict an entire contextual source field and its attention measure directly from ink."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.ordered_translation import Block


class OpticalStateReader(nn.Module):
    def __init__(self, output_size=512, width=256):
        super().__init__()
        self.ink = nn.Sequential(nn.Linear(256,width), nn.LayerNorm(width), nn.GELU(), nn.Linear(width,width))
        self.boundaries = nn.Parameter(torch.randn(2,width)*.02)
        self.local = nn.Conv1d(width,width,5,padding=2)
        self.layers = nn.ModuleList([Block(width) for _ in range(4)])
        self.norm = nn.LayerNorm(width)
        self.field = nn.Linear(width,output_size)
        self.measure = nn.Linear(width,1)
        phase = torch.arange(132)[:,None] * torch.exp(-math.log(10000)*torch.arange(0,width,2)/width)
        self.register_buffer("position",torch.stack((phase.sin(),phase.cos()),-1).flatten(1),persistent=False)
        self.register_buffer("center",torch.zeros(output_size))
        self.register_buffer("spread",torch.ones(output_size))

    def forward(self, features, source):
        lengths = source.ne(0).sum(1)
        raw = self.ink(features)[source].masked_fill(source.eq(0).unsqueeze(-1),0)
        x = F.pad(raw,(0,0,1,1))
        x[:,0] = self.boundaries[0].to(x.dtype)
        x[torch.arange(len(x),device=x.device),lengths+1] = self.boundaries[1].to(x.dtype)
        valid = torch.arange(x.size(1),device=x.device)[None,:] < lengths[:,None]+2
        x = x + self.local(x.transpose(1,2)).transpose(1,2) + self.position[:x.size(1)]
        for block in self.layers:
            x, _ = block(x,valid[:,None,None,:])
        x = self.norm(x)
        return self.field(x), self.measure(x).squeeze(-1), valid

    def states(self, features, source):
        field, mass, valid = self(features,source)
        return field.float()*self.spread+self.center, valid.long(), mass.float()
