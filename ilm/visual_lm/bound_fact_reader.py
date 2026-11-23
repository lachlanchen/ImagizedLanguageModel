"""Predict two physical spans and a factual relation from a pictured passage and question."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from safetensors.torch import load_file

from ilm.visual_lm.font_invariant_reader import FontInvariantReader
from ilm.visual_lm.ordered_page_features import contextualize


class FactHead(nn.Module):
    def __init__(self, relations):
        super().__init__()
        self.summary = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 128), nn.GELU())
        self.local = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 128))
        self.aux = nn.Linear(5, 128)
        self.roles = nn.Linear(128, relations)
        self.bounds = nn.Linear(128, 4)

    def forward(self, summary, page, mask):
        context = self.summary(summary)
        hidden = F.gelu(self.local(page[:,:,:768]) + self.aux(page[:,:,768:]) + context[:,None])
        ends = self.bounds(hidden)
        # The original trained QA reader provides a prior for the requested object.
        ends = torch.cat((ends[:,:,:2], ends[:,:,2:] + page[:,:,768:770]), dim=-1)
        return self.roles(context), ends.masked_fill(~mask[:,:,None], -1e4)


def features_batch(features, device):
    cls, local = zip(*features)
    cls = torch.stack(cls).to(device, dtype=torch.float32)
    page = pad_sequence(local,batch_first=True).to(device,dtype=torch.float32)
    lengths = torch.tensor([len(v) for v in local],device=device)
    mask = torch.arange(page.shape[1],device=device)[None] < lengths[:,None]
    return cls, page, mask


def best_bounds(logits, width=48):
    start,end = logits.unbind(-1)
    width = min(width, start.size(1))
    windows = F.pad(end,(0,width-1),value=-1e4).unfold(1,width,1)
    best = (start[:,:,None] + windows).flatten(1).argmax(1)
    first = best // width
    return torch.stack((first,first + best % width + 1),dim=-1)


def render_fact(frame, subject, obj):
    opening, link, ending = frame
    return opening + subject + link + obj + ending


class BoundFactReader:
    """Outputs bitmap strings; callers may also request the exact selected object."""
    def __init__(self,reader,folder,device="cpu"):
        folder=Path(folder)
        self.reader=FontInvariantReader.restore(reader,device)
        self.frames=[[bytes.fromhex(piece) for piece in parts] for parts in json.loads((folder/"frames.json").read_text())]
        self.head=FactHead(len(self.frames)).to(device)
        self.head.load_state_dict(load_file(str(folder/"head.safetensors")))
        self.head.eval()
        self.device=device

    @torch.inference_mode()
    def fields_many(self,pairs):
        if not pairs:return []
        if any(not p or not q or len(p)%128 or len(q)%128 for p,q in pairs):
            raise ValueError("Each input must consist of complete, nonempty 32 × 32 cells")
        output=[]
        for start in range(0,len(pairs),32):
            items=pairs[start:start+32]
            values=contextualize(self.reader,items)
            roles,ends=self.head(*features_batch(values,self.device))
            subjects=best_bounds(ends[:,:,:2]).tolist()
            objects=best_bounds(ends[:,:,2:]).tolist()
            for (p,_),r,(a,b),(c,d) in zip(items,roles.argmax(1).tolist(),subjects,objects,strict=True):
                obj=p[c*128:d*128]
                output.append((p[a*128:b*128],r,obj))
        return output

    def answer_many(self,pairs,direct=False):
        return [obj if direct else render_fact(self.frames[role],subject,obj)
                for subject,role,obj in self.fields_many(pairs)]
