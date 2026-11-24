"""A conditional masked image writer for aligned five- and seven-cell lines."""
import torch
from torch import nn
from torch.nn import functional as F


class ReadingBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3*width)
        self.projection = nn.Linear(width,width)
        self.feed = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,3*width),
                                  nn.GELU(),nn.Linear(3*width,width))

    def forward(self, state):
        batch, length, width = state.shape
        qkv = self.qkv(self.norm(state)).reshape(batch,length,3,4,width//4)
        q,k,v = qkv.permute(2,0,3,1,4).unbind(0)
        out = F.scaled_dot_product_attention(q,k,v)
        state = state+self.projection(out.transpose(1,2).reshape_as(state))
        return state+self.feed(state)


class VerseCanvas(nn.Module):
    def __init__(self, vocab, width=192):
        super().__init__()
        self.source = nn.Sequential(nn.Linear(256,width),nn.GELU(),nn.Linear(width,width))
        self.target = nn.Sequential(nn.Linear(256,width),nn.GELU(),nn.Linear(width,width))
        self.blank = nn.Parameter(torch.randn(width)*.03)
        self.position = nn.Parameter(torch.randn(7,width)*.03)
        self.metre = nn.Parameter(torch.randn(2,width)*.03)
        self.body = nn.Sequential(*(ReadingBlock(width) for _ in range(4)),nn.LayerNorm(width))
        self.ink = nn.Linear(width,vocab)

    def encode(self, source, target, visible):
        length = source.shape[1]
        if length not in (5,7):
            raise ValueError('Expected five or seven aligned glyphs')
        state = self.source(source)+self.position[:length]+self.metre[int(length==7)]
        state = state+torch.where(visible[:,:,None],self.target(target),self.blank)
        return self.body(state)

    def forward(self, source, target, visible):
        return self.ink(self.encode(source,target,visible)).float()

    @torch.inference_mode()
    def fill(self, source, target, visible, palette, order='confidence'):
        """Preserve supplied ink; commit one unknown cell per row and iteration."""
        if order not in ('confidence','left'):
            raise ValueError(order)
        visible = visible.clone()
        target = target.clone()
        output = torch.zeros_like(visible,dtype=torch.long)
        rows = torch.arange(len(source),device=source.device)
        for _ in range(source.shape[1]):
            hidden = ~visible
            active = hidden.any(1)
            if not active.any():
                break
            logits = self(source,target,visible)
            value,index = logits.max(-1)
            if order=='confidence':
                priority = value-torch.logsumexp(logits,-1)
                pos = priority.masked_fill(visible,-torch.inf).argmax(1)
            else:
                pos = hidden.long().argmax(1)
            r,c = rows[active],pos[active]
            chosen = index[r,c]+1
            output[r,c] = chosen
            target[r,c] = palette[chosen]
            visible[r,c] = True
        return output
