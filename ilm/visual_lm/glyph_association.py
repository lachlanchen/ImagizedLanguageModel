"""Source-restricted nearest-neighbor recall from observed optical decoder states."""

import math
import torch
from torch.nn import functional as F


def source_code(ctx, valid):
    keys = ctx[-1][0].transpose(1,2).flatten(2)
    mask = valid[:,0,0,:,None]
    return F.normalize((keys*mask).sum(1)/mask.sum(1), dim=-1)


class GlyphMemory:
    def __init__(self, archive, device):
        names=("sources","states","labels","shuffled","offsets")
        self.sources,self.states,self.labels,self.shuffled,self.offsets = (
            torch.as_tensor(archive[name],device=device) for name in names
        )

    def passages(self, source, count=32):
        choices=(source.float() @ self.sources.float().T).topk(count,dim=1).indices
        length=self.offsets[choices+1]-self.offsets[choices]
        pos=torch.arange(int(length.max()),device=choices.device)
        idx=self.offsets[choices].unsqueeze(-1) + pos
        valid=(pos < length.unsqueeze(-1)).flatten(1)
        idx=idx.flatten(1).clamp_max(len(self.states)-1)
        return self.states[idx],self.labels[idx],self.shuffled[idx],valid

    @staticmethod
    def neighbors(hidden, passages):
        keys, labels, shuffled, valid=passages
        query=F.normalize(hidden.float(),dim=-1)
        score=(query.to(keys.dtype) @ keys.transpose(1,2)).float()
        score=score.masked_fill(~valid[:,None],-float("inf"))
        values,ix=score.topk(16,dim=-1)
        def select(label):
            return torch.gather(label.long()[:,None].expand(-1,ix.size(1),-1),2,ix)
        return values,select(labels),select(shuffled)


@torch.inference_mode()
def generate(net, features, src, inv, direction, memory, temperature, mixing, shuffled=False, limit=128):
    if mixing == 0:
        return net.generate(features,src,inv,direction,limit)
    bank=net.optical(features)
    ctx,valid=net.encode(bank,src)
    nearby=memory.passages(source_code(ctx,valid))
    prototypes=net.prototypes(bank,inv)
    x=net.begin[direction].expand(len(src),1,-1)
    done=torch.zeros(len(src),dtype=torch.bool,device=src.device)
    out=[]
    past=None
    for step in range(limit+1):
        h,past=net.decode(x,ctx,valid,past,step)
        logits=h@prototypes.T/math.sqrt(h.size(-1))
        prob=logits.float().softmax(-1)*(1-mixing)
        sim,labels,control=memory.neighbors(h,nearby)
        prob.scatter_add_(-1,control if shuffled else labels,(sim/temperature).softmax(-1)*mixing)
        ids=inv[prob[:,0].argmax(-1)]
        out.append(ids.masked_fill(done,0))
        done |= ids==1
        if done.all():
            break
        x=bank[ids][:,None]
    return torch.stack(out,dim=1)
