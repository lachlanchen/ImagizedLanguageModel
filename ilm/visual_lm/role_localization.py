"""Locate two disjoint answer spans in a sequence of glyph images."""
import numpy as np
import torch
from torch import nn

from ilm.visual_lm.natural_roles import NaturalRoleReader
from ilm.visual_lm.ordered_translation import Block


class RoleLocator(nn.Module):
    def __init__(self,width=80):
        super().__init__()
        trunk=NaturalRoleReader(width)
        self.eye=trunk.eye
        self.local=trunk.local
        self.norm=trunk.norm
        self.geometry=nn.Linear(4,width)
        self.attention=nn.ModuleList(Block(width) for _ in range(2))
        self.output=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,4))

    def read(self,features,verb,valid,blind=False):
        loc=torch.arange(verb.size(1),device=verb.device).float()[None]
        delta=loc-(loc*verb).sum(1,keepdim=True)/verb.sum(1,keepdim=True)
        coords=torch.stack((delta.sign(),(delta/32).clamp(-4,4),verb.float(),
                           (loc/valid.sum(1,keepdim=True)).clamp_max(1)),-1)
        x=self.geometry(coords)
        if not blind:x=x+features
        for norm,conv in zip(self.norm,self.local,strict=True):
            x=x+conv((norm(x)*valid.unsqueeze(-1)).transpose(1,2)).transpose(1,2)
        for layer in self.attention:
            x,_=layer(x,valid[:,None,None,:])
        return self.output(x).transpose(1,2).masked_fill(~(valid & ~verb)[:,None],-1e4)


def best_pair(scores,legal):
    """Exact maximum over both possible span orders. Endpoints are inclusive internally."""
    n=len(legal)
    cells=np.arange(n)
    invalid=np.r_[0,np.cumsum(~np.asarray(legal,dtype=bool))]
    feasible=(cells[:,None]<=cells[None,:]) & (invalid[1:][None,:]==invalid[:-1,None])
    mats=[np.where(feasible,scores[i,:,None]+scores[i+1,None,:],-np.inf) for i in (0,2)]
    best=-np.inf;answer=None
    for first,second in ((0,1),(1,0)):
        # Score the best first span ending before each possible start of the second.
        ends=mats[first].max(0)
        first_start=mats[first].argmax(0)
        second_end=mats[second].argmax(1)
        prefix=np.maximum.accumulate(ends)
        totals=prefix[:-1]+mats[second][cells[1:],second_end[1:]]
        if not len(totals):continue
        b=int(totals.argmax())+1
        if totals[b-1]>best:
            best=totals[b-1]
            e=int(ends[:b].argmax())
            a=(int(first_start[e]),e+1);c=(b,int(second_end[b])+1)
            answer=(a,c) if first==0 else (c,a)
    if answer is None:raise ValueError("The pictured source does not contain two legal answer spans")
    return answer
