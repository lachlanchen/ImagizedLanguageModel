"""Translation from ordered visual windows with a causal glyph-image decoder."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.q = nn.Linear(width, width, bias=False)
        self.kv = nn.Linear(width, width * 2, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def heads(self, x):
        return x.reshape(x.size(0), x.size(1), 4, -1).transpose(1, 2)

    def keys(self, x):
        return tuple(self.heads(z) for z in self.kv(x).chunk(2, -1))

    def forward(self, x, kv, mask=None, causal=False):
        v = F.scaled_dot_product_attention(
            self.heads(self.q(x)), *kv, attn_mask=mask, is_causal=causal
        )
        return self.out(v.transpose(1, 2).reshape_as(x))


class Block(nn.Module):
    def __init__(self, width, decoder=False):
        super().__init__()
        self.selfatt = Attention(width)
        self.norm = nn.LayerNorm(width)
        self.ffnorm = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, 3 * width), nn.GELU(), nn.Linear(3 * width, width)
        )
        if decoder:
            self.cross = Attention(width)
            self.cnorm = nn.LayerNorm(width)

    def forward(self, x, valid, source=None, past=None):
        z = self.norm(x)
        k, v = self.selfatt.keys(z)
        if past is not None:
            k, v = torch.cat((past[0], k), 2), torch.cat((past[1], v), 2)
        mask = None if source is not None else valid
        x = x + self.selfatt(z, (k, v), mask, source is not None and past is None)
        if source is not None:
            x = x + self.cross(self.cnorm(x), source, valid)
        return x + self.ff(self.ffnorm(x)), (k, v)


class OrderedTranslator(nn.Module):
    def __init__(self, stride, width=160):
        super().__init__()
        if stride not in (1, 4):
            raise ValueError("Stride must be 1 or 4")
        self.stride = stride
        self.optical = nn.Sequential(
            nn.Linear(256, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.window = nn.Linear(4 * width, width)
        self.enc = nn.ModuleList(Block(width) for _ in range(3))
        self.dec = nn.ModuleList(Block(width, True) for _ in range(2))
        self.final = nn.LayerNorm(width)
        self.target = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
        self.begin = nn.Parameter(torch.randn(2, width) * 0.03)
        self.end = nn.Parameter(torch.randn(width) * 0.03)
        pos = torch.arange(132)[:, None] * torch.exp(
            -math.log(10000) * torch.arange(0, width, 2) / width
        )
        self.register_buffer(
            "pos", torch.stack((pos.sin(), pos.cos()), -1).flatten(1), persistent=False
        )

    def encode(self, bank, src):
        cells = bank[src].masked_fill((src == 0)[:, :, None], 0)
        windows = (
            F.pad(cells, (0, 0, 0, 3))
            .unfold(1, 4, self.stride)
            .transpose(2, 3)
            .flatten(2)
        )
        x = self.window(windows) + self.pos[: windows.size(1)]
        valid = (src[:, :: self.stride] != 0)[:, None, None, :]
        for layer in self.enc:
            x, _ = layer(x, valid)
        return [layer.cross.keys(x) for layer in self.dec], valid

    def decode(self, x, source, valid, past=None, offset=0):
        x = x + self.pos[offset : offset + x.size(1)]
        cache = []
        for i, layer in enumerate(self.dec):
            x, state = layer(x, valid, source[i], None if past is None else past[i])
            cache.append(state)
        return self.final(x), cache

    def prototypes(self, bank, inventory):
        return torch.cat((self.end[None], self.target(bank[inventory[1:]])))

    def forward(self, features, src, tgt, inventory, direction):
        bank = self.optical(features)
        context, valid = self.encode(bank, src)
        x = torch.cat(
            (self.begin[direction].expand(len(src), 1, -1), bank[tgt[:, :-1]]), 1
        )
        h, _ = self.decode(x, context, valid)
        return h @ self.prototypes(bank, inventory).T / math.sqrt(h.size(-1))

    @torch.inference_mode()
    def generate(self, features, src, inventory, direction, limit=128):
        bank = self.optical(features)
        ctx, valid = self.encode(bank, src)
        keys = self.prototypes(bank, inventory)
        x = self.begin[direction].expand(len(src), 1, -1)
        done = torch.zeros(len(src), dtype=torch.bool, device=src.device)
        result = []
        past = None
        for i in range(limit + 1):
            h, past = self.decode(x, ctx, valid, past, i)
            ids = inventory[(h[:, 0] @ keys.T).argmax(-1)]
            result.append(ids.masked_fill(done, 0))
            done = done | (ids == 1)
            if done.all():
                break
            x = bank[ids][:, None]
        return torch.stack(result, 1)
