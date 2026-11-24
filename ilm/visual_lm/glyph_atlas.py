"""Read an aligned question, locate its headword in an image index and copy fields."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class AtlasReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape = nn.Sequential(nn.Linear(256,96),nn.GELU(),nn.Linear(96,96))
        self.layers = nn.ModuleList(nn.Sequential(nn.Conv1d(96,96,5,dilation=d,padding=2*d),
                                    nn.GELU(),nn.Conv1d(96,96,1)) for d in (1,2,4,8))
        self.norms = nn.ModuleList(nn.LayerNorm(96) for _ in self.layers)
        self.head = nn.Linear(96,1)
        self.fields = nn.Linear(96,8)

    def forward(self,x,valid):
        hidden = self.shape(x)
        for conv,norm in zip(self.layers,self.norms):
            state = norm(hidden).masked_fill(~valid[:,:,None],0).transpose(1,2)
            hidden = hidden + conv(state).transpose(1,2)
        return self.head(hidden).squeeze(-1).float().masked_fill(~valid,-torch.inf),self.fields(hidden).float()

    def key(self,x):
        return F.normalize(self.shape(x).float(),dim=-1)


def mentions(labels):
    """Emit one action per requested field in the order of its first mention."""
    return tuple(dict.fromkeys(int(k)-1 for k in labels if k))


def requested(actions,spans):
    fields = []
    for f in actions:
        if f == 6:
            fields.extend(i for i in (4,0,1,2,3) if spans[i,1])
        else:
            fields.append(f)
    return tuple(dict.fromkeys(fields))


def payload(book,entry,fields):
    if entry < 0:
        return []
    result = []
    for f in fields:
        offset,n = book["spans"][entry,f]
        result.append(book["stream"][offset:offset+n])
    return result


@torch.inference_mode()
def address(net,questions,valid,head_bank,*,approximate=False):
    head_score,field_score = net(questions,valid)
    rows = torch.arange(len(questions),device=questions.device)
    focus = head_score.argmax(1)
    cell = questions[rows,focus]
    # Cosine matching of canonical observations also supplies a raw-image control.
    raw = F.normalize(cell) @ F.normalize(head_bank).T
    literal = raw.argmax(1)
    exact = (cell == head_bank[literal]).all(1)
    learned = (net.key(cell) @ net.key(head_bank).T).argmax(1)
    fallback = learned if approximate else torch.full_like(learned,-1)
    entry = torch.where(exact,literal,fallback)
    labels = field_score.argmax(-1).masked_fill(~valid,0)
    return (entry.cpu().numpy(),focus.cpu().numpy(),labels.cpu().numpy(),
            literal.cpu().numpy(),exact.cpu().numpy())
