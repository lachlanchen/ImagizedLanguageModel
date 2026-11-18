"""Residual global attention over a pictured question, optionally aware of reading distance."""
import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.grounded_glyph_decoder import WIDTH, WindowEncoder


class ScopedQuestion(WindowEncoder):
    def __init__(self, ordered=True):
        super().__init__()
        self.ordered = ordered
        self.heads = 4
        self.context_qkv = nn.Linear(WIDTH, WIDTH * 3)
        self.context_out = nn.Linear(WIDTH, WIDTH)
        nn.init.zeros_(self.context_out.weight)
        nn.init.zeros_(self.context_out.bias)

    def forward(self, x, valid):
        x = super().forward(x, valid)
        batch, length, width = x.shape
        q, k, v = self.context_qkv(x).reshape(batch, length, 3, self.heads, width//self.heads).permute(2,0,3,1,4)
        # Two heads look left and two right, each at a short or long length scale.
        pos = torch.arange(length,device=x.device)
        offset = pos[None,:] - pos[:,None]
        directions = x.new_tensor([-1,-1,1,1])[:,None,None]
        scales = x.new_tensor([12,48,12,48])[:,None,None]
        bias = -offset.abs()/scales - 8*(directions*offset < 0)
        if not self.ordered:
            bias = torch.zeros_like(bias)
        bias = bias[None].masked_fill(~valid[:,None,None,:], -1e4)
        contextual = F.scaled_dot_product_attention(q,k,v,attn_mask=bias.to(q.dtype))
        contextual = contextual.transpose(1,2).reshape(batch,length,width)
        return (x + self.context_out(contextual))*valid[...,None]
