"""A causal reader of cell observations with prototype, copy and suffix distributions."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def observations(data, mode):
    if mode == "ink":
        bits = np.unpackbits(data["tiles"], axis=1).reshape(-1, 16, 2, 16, 2)
        return np.ascontiguousarray(bits.mean(axis=(2, 4), dtype=np.float32).reshape(-1, 256))
    if mode == "byte":
        rng = np.random.default_rng(194671)
        signs = rng.choice(np.array([-0.5, 0.5], np.float32), (256, 64))
        signs[0] = 0
        return signs[data["bytes"]].reshape(-1, 256)
    raise ValueError(mode)


class CausalBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.project = nn.Linear(width, width)
        self.feed = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2),
                                  nn.GELU(), nn.Linear(width * 2, width))

    def forward(self, state):
        batch, length, width = state.shape
        packed = self.qkv(self.attention_norm(state)).reshape(batch, length, 3, 4, width // 4)
        q, k, v = packed.permute(2, 0, 3, 1, 4).unbind()
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        state = state + self.project(out.transpose(1, 2).reshape_as(state))
        return state + self.feed(state)


class BookWriter(nn.Module):
    def __init__(self, vocab, hybrid, width=128):
        super().__init__()
        self.hybrid = hybrid
        self.input = nn.Linear(256, width)
        self.position = nn.Parameter(torch.randn(64, width) * 0.02)
        self.body = nn.Sequential(CausalBlock(width), CausalBlock(width), nn.LayerNorm(width))
        self.output = nn.Linear(width, vocab)
        self.pointer = nn.Linear(width, 128)
        self.router = nn.Linear(width + (3 if hybrid else 0), 3 if hybrid else 2)

    def forward(self, features, confidence, output_from=0):
        length = features.size(1)
        state = self.body(self.input(features) + self.position[:length])
        query, key = self.pointer(state).chunk(2, -1)
        score = (query[:, output_from:] @ key.transpose(1, 2)).float() / 8
        mask = torch.ones((length, length), device=score.device, dtype=torch.bool).triu(1)
        copy = score.masked_fill(mask[output_from:], -torch.inf).softmax(-1)
        state = state[:, output_from:]
        route_input = torch.cat((state, confidence[:, output_from:]), -1) if self.hybrid else state
        return self.output(state).float(), copy, self.router(route_input).float().softmax(-1)

    def loss(self, features, ids, labels, confidence, count_prob, first=5):
        logits, copy, route = self(features, confidence, first)
        truth = labels[:, first:]
        head = logits.log_softmax(-1).gather(-1, (truth - 1).unsqueeze(-1)).squeeze(-1)
        copy_mass = (copy * (truth.unsqueeze(-1) == ids.unsqueeze(1))).sum(-1)
        mass = head.exp() * route[:, :, 0] + copy_mass * route[:, :, 1]
        if self.hybrid:
            mass = mass + route[:, :, 2] * count_prob[:, first:]
        return -mass.clamp_min(1e-35).log().mean()
