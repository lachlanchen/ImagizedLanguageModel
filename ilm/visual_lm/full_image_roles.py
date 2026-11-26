"""Condition an optical sentence writer on semantic fields without removing source ink."""
import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.ordered_translation import Block, OrderedTranslator


class EventMarker(nn.Module):
    """Three overlapping ink-cell masks estimated without a marked verb at inference."""
    def __init__(self, width=80):
        super().__init__()
        self.eye = nn.Sequential(nn.Linear(256,width),nn.LayerNorm(width),nn.GELU(),nn.Linear(width,width))
        self.layers = nn.ModuleList(nn.Sequential(nn.LayerNorm(width),nn.Linear(width,2*width)) for _ in range(3))
        self.convs = nn.ModuleList(nn.Conv1d(width,width,5,padding=2*d,dilation=d) for d in (1,2,4))
        self.context = Block(width)
        self.out = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,3))

    def forward(self, features, cells):
        valid = cells.ne(0)
        x = self.eye(features)[cells]
        for gate,conv in zip(self.layers,self.convs,strict=True):
            content,weight = gate(x).chunk(2,-1)
            local = conv((content*valid.unsqueeze(-1)).transpose(1,2)).transpose(1,2)
            x = x + F.gelu(local)*weight.sigmoid()
        x,_ = self.context(x,valid[:,None,None,:])
        return self.out(x)


class FullImageWriter(OrderedTranslator):
    def __init__(self):
        super().__init__(1)
        self.role_field = nn.Linear(3,self.pos.size(1),bias=False)
        nn.init.zeros_(self.role_field.weight)

    def encode(self, bank, src):
        ids = src[:,:,0].long()
        valid = ids.ne(0)
        cells = bank[ids]*valid.unsqueeze(-1)
        windows = F.pad(cells,(0,0,0,3)).unfold(1,4,1).transpose(2,3).flatten(2)
        x = self.window(windows)+self.pos[:ids.size(1)]
        x = x+self.role_field(src[:,:,1:].to(x.dtype)/255)
        mask = valid[:,None,None,:]
        for layer in self.enc:
            x,_ = layer(x,mask)
        return [layer.cross.keys(x) for layer in self.dec],mask


def pictured_input(cells, fields):
    """Column zero addresses a measured physical bitmap; the other columns carry probabilities."""
    return torch.cat((cells.unsqueeze(-1).to(torch.int32),fields.to(torch.int32)),dim=-1)
