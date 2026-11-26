"""Bounded, source-dependent glyph decoding with a shared causal answer history."""

import math
import torch
from torch.nn import functional as F

BOUND = math.log(4.0)


def evidence(logits):
    """First channel is the actual source. Others are equiprobable backgrounds."""
    logp = logits.float().log_softmax(-1)
    full = logp[:,0]
    background = torch.logsumexp(logp[:,1:],dim=1) - math.log(logp.size(1)-1)
    return full, (full-background).clamp(-BOUND,BOUND)


def backgrounds(src, table):
    lengths = src.count_nonzero(1)
    if torch.any((lengths < 1) | (lengths > 96)):
        raise ValueError("Invalid source length")
    others = table[lengths]
    width = max(src.size(1), int(others.count_nonzero(-1).max()))
    original = F.pad(src,(0,width-src.size(1)))
    return torch.cat((original[:,None],others[:,:,:width]),dim=1)


@torch.inference_mode()
def write(net, features, src, inv, direction, table, gain, limit=128):
    if gain==0:
        return net.generate(features,src,inv,direction,limit)
    sources=backgrounds(src,table)
    n,k=sources.shape[:2]
    bank=net.optical(features)
    ctx,valid=net.encode(bank,sources.flatten(0,1))
    prototypes=net.prototypes(bank,inv)
    x=net.begin[direction].expand(n*k,1,-1)
    done=torch.zeros(n,dtype=torch.bool,device=src.device)
    results=[]
    past=None
    for step in range(limit+1):
        h,past=net.decode(x,ctx,valid,past,step)
        logits=(h[:,0] @ prototypes.T)/math.sqrt(h.size(-1))
        full,ratio=evidence(logits.reshape(n,k,-1))
        ids=inv[(full+gain*ratio).argmax(-1)]
        results.append(ids.masked_fill(done,0))
        done |= ids==1
        if done.all():
            break
        x=bank[ids].repeat_interleave(k,dim=0)[:,None]
    return torch.stack(results,1)
