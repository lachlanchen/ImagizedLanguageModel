"""Widen a trained optical translator without changing its mathematical logits."""

import math
import torch
from torch import nn
from ilm.visual_lm.ordered_translation import OrderedTranslator, Attention


class WiderAttention(Attention):
    def __init__(self, width, heads):
        super().__init__(width)
        self.head_count = heads

    def heads(self, x):
        return x.reshape(x.size(0), x.size(1), self.head_count, -1).transpose(1, 2)


class WideTranslator(OrderedTranslator):
    def __init__(self, replicas=3, base_width=160):
        if replicas < 2:
            raise ValueError("Widening requires at least two copies")
        super().__init__(1, replicas * base_width)
        self.replicas = replicas
        for block in [*self.enc, *self.dec]:
            block.selfatt = WiderAttention(replicas * base_width, 4 * replicas)
            if hasattr(block, "cross"):
                block.cross = WiderAttention(replicas * base_width, 4 * replicas)
        # Keep positional frequencies of the starting network in each replica.
        base_pos = torch.arange(132)[:, None] * torch.exp(
            -math.log(10000) * torch.arange(0, base_width, 2) / base_width
        )
        self.pos = (
            torch.stack((base_pos.sin(), base_pos.cos()), -1)
            .flatten(1)
            .repeat(1, replicas)
        )


def index_map(size, groups, factor):
    return torch.arange(size).reshape(groups, -1).repeat(1, factor).flatten()


@torch.no_grad()
def widen(source, replicas=3, seed=77135):
    """CPU initialization. Noise sums to zero within each group of copied input coordinates."""
    if source.stride != 1 or next(source.parameters()).device.type != "cpu":
        raise ValueError("Expected a stride-one CPU source")
    width = source.begin.size(-1)
    model = WideTranslator(replicas, width)
    rng = torch.Generator().manual_seed(seed)
    for name, old in source.named_modules():
        new = model.get_submodule(name)
        if isinstance(old, nn.Linear):
            in_rep = 1 if name == "optical.0" else replicas
            in_groups = 4 if name == "window" else 1
            out_groups = 2 if name.endswith(".kv") else 1
            rows = index_map(old.out_features, out_groups, replicas)
            cols = index_map(old.in_features, in_groups, in_rep)
            weight = old.weight[rows][:, cols] / in_rep
            if in_rep > 1:
                shape = (len(rows), in_groups, in_rep, old.in_features // in_groups)
                noise = torch.randn(shape, generator=rng)
                noise -= noise.mean(dim=2, keepdim=True)
                weight += noise.reshape_as(weight) * (0.1 * old.weight.std() / in_rep)
            scale = math.sqrt(replicas) if name == "target.1" else 1
            new.weight.copy_(weight / scale)
            if old.bias is not None:
                new.bias.copy_(old.bias[rows] / scale)
        elif isinstance(old, nn.LayerNorm):
            new.weight.copy_(old.weight.repeat(replicas))
            new.bias.copy_(old.bias.repeat(replicas))
    model.begin.copy_(source.begin.repeat(1, replicas))
    model.end.copy_(source.end.repeat(replicas) / math.sqrt(replicas))
    return model
