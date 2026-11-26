"""Independent visual pointer over labelled dictionary rows. Values remain raw source pixels."""
import torch
from torch import nn
from torch.nn import functional as F


class OriginValueReader(nn.Module):
    def __init__(self, visual_match=True):
        super().__init__()
        self.visual_match=visual_match
        self.ink=nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Linear(64,48))
        self.query=nn.GRU(48,64,batch_first=True)
        self.label=nn.GRU(48,64,batch_first=True)
        self.step=nn.GRUCell(64,64)
        self.begin=nn.Parameter(torch.zeros(64))
        self.score=nn.Sequential(nn.Linear(193,96),nn.GELU(),nn.Linear(96,1))
        self.stop=nn.Linear(64,1)

    def line(self,features,handles,reader):
        b=handles.shape[:-1]
        ids=handles.reshape(-1,handles.shape[-1])
        seq,_=reader(features[ids])
        last=(ids.ne(0).sum(1)-1).clamp_min(0)
        return seq[torch.arange(len(ids),device=ids.device),last].reshape(*b,-1)

    def forward(self, unique_ink, question, labels, targets=None):
        # Handles are local addresses of exact input bitmaps shared across lines.
        # The encoder receives grayscale ink counts; the optional equality channel
        # compares physical bitmaps. Neither path embeds a handle or a code point.
        small=F.avg_pool2d(unique_ink[:,None].float(),4).flatten(1)
        features=self.ink(small)
        context=self.line(features,question,self.query)
        rows=self.line(features,labels,self.label)
        valid=labels[:,:,0].ne(0)
        match=(labels[:,:,0,None]==question[:,None,:]).any(-1)&valid
        if not self.visual_match: match=torch.zeros_like(match)
        h=context
        previous=self.begin.expand_as(h)
        logits=[]
        stop=rows.shape[1]
        options=F.pad(rows,(0,0,0,1))
        for i in range(3 if targets is not None else 5):
            h=self.step(previous,h)
            repeated=h[:,None].expand_as(rows)
            local=self.score(torch.cat([rows,repeated,rows*repeated,match[:,:,None].float()],-1)).squeeze(-1)
            scores=torch.cat([local.masked_fill(~valid,-1e4),self.stop(h)],1)
            logits.append(scores)
            choice=targets[:,i] if targets is not None else scores.argmax(1)
            previous=options[torch.arange(len(h),device=h.device),choice]
        return torch.stack(logits,1)


def emit_values(page, sequence):
    output=[]
    for idx in sequence:
        if idx>=len(page): return output,True
        if len(output)==4: return output,False
        output.append(page[idx]["value"])
    return output,False
