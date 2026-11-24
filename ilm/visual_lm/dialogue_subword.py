"""Small conditional Transformer operating entirely on subword IDs."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, 2 * width)
        self.out = nn.Linear(width, width)

    def split(self, x):
        return x.unflatten(-1, (self.heads, -1)).transpose(1, 2)

    def keys(self, x):
        k, v = self.kv(x).chunk(2, dim=-1)
        return self.split(k), self.split(v)

    def attend(self, x, keys, mask=None, causal=False):
        h = F.scaled_dot_product_attention(
            self.split(self.q(x)), *keys, attn_mask=mask, is_causal=causal
        )
        return self.out(h.transpose(1, 2).flatten(2))


class Block(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.self_att = Attention(96, 4)
        self.norm = nn.LayerNorm(96)
        self.ff_norm = nn.LayerNorm(96)
        self.ff = nn.Sequential(nn.Linear(96, 256), nn.GELU(), nn.Linear(256, 96))
        if decoder:
            self.cross = Attention(96, 4)
            self.cross_norm = nn.LayerNorm(96)

    def forward(self, x, mask=None, cross=None, incremental=False, past=None):
        h = self.norm(x)
        keys = self.self_att.keys(h)
        if past is not None:
            keys = tuple(torch.cat((a, b), dim=2) for a, b in zip(past, keys))
        x = x + self.self_att.attend(
            h,
            keys,
            mask if cross is None else None,
            causal=cross is not None and not incremental,
        )
        if cross is not None:
            x = x + self.cross.attend(self.cross_norm(x), cross, mask)
        return x + self.ff(self.ff_norm(x)), keys


class DialogueSubword(nn.Module):
    def __init__(self, vocab=4096):
        super().__init__()
        self.embedding = nn.Embedding(vocab, 96, padding_idx=0)
        nn.init.normal_(self.embedding.weight, std=0.04)
        self.segment = nn.Embedding(3, 96)
        nn.init.normal_(self.segment.weight, std=0.04)
        self.encoder = nn.ModuleList(Block(False) for _ in range(2))
        self.decoder = nn.ModuleList(Block(True) for _ in range(2))
        self.enc_norm = nn.LayerNorm(96)
        self.dec_norm = nn.LayerNorm(96)
        self.evidence = nn.Linear(96, 2)
        scale = torch.exp(torch.arange(0, 96, 2) * (-math.log(10000.0) / 96))
        self.register_buffer("frequency", scale)

    def embed(self, ids, offset=0):
        phase = (torch.arange(ids.size(1), device=ids.device) + offset)[
            :, None
        ] * self.frequency
        position = torch.stack((phase.sin(), phase.cos()), -1).flatten(1)
        return self.embedding(ids) * 4 + position

    def encode(self, ids, segments):
        mask = (ids != 0)[:, None, None]
        h = self.embed(ids) + self.segment(segments)
        for block in self.encoder:
            h, _ = block(h, mask)
        h = self.enc_norm(h)
        return [b.cross.keys(h) for b in self.decoder], mask, self.evidence(h)

    def decode(self, ids, source, mask, step=None, cache=None):
        h = self.embed(ids, step or 0)
        fresh = []
        for j, block in enumerate(self.decoder):
            h, k = block(
                h,
                mask,
                source[j],
                step is not None,
                None if cache is None else cache[j],
            )
            fresh.append(k)
        return F.linear(self.dec_norm(h), self.embedding.weight), fresh

    @torch.inference_mode()
    def generate(self, ids, segments, maximum=96):
        source, mask, _ = self.encode(ids, segments)
        prev = ids.new_full((len(ids), 1), 1)
        out = []
        done = torch.zeros(len(ids), device=ids.device, dtype=torch.bool)
        cache = None
        for step in range(maximum + 1):
            logits, cache = self.decode(prev, source, mask, step, cache)
            logits[:, :, 0:2] = -torch.inf
            logits[:, :, 3:5] = -torch.inf
            prev = logits.argmax(-1).masked_fill(done[:, None], 2)
            done |= prev[:, 0] == 2
            out.append(prev)
            if bool(done.all()):
                break
        return torch.cat(out, 1), done
