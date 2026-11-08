"""Actual native image conditions and low-rank parameters for flow learning."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class NativeLowRankLinear(nn.Module):
    """Frozen W plus B A, with FP32 masters and native-dtype multiplication.

    Unlike an autocast-dependent wrapper, this also works inside the source's
    plain no-grad sampler. Zero B preserves the native output before learning.
    """
    def __init__(self, base, rank):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.down = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.up = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))

    @property
    def weight(self): return self.base.weight

    @property
    def bias(self): return self.base.bias

    def forward(self, value):
        original = self.base(value)
        delta = F.linear(F.linear(value.to(original.dtype), self.down.to(original.dtype)), self.up.to(original.dtype))
        return original + delta


def add_adapters(pipe, rank=16):
    if not 1 <= rank <= 128: raise ValueError("invalid adapter rank")
    selected = {}
    for component in ("transformer", "connector_module"):
        model = getattr(pipe, component); model.requires_grad_(False)
        names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and
            (component == "connector_module" or n.startswith("transformer_blocks.") or n in ("context_embedder", "proj_out"))]
        if not names: raise ValueError("native component has no selected projections")
        for name in names:
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, NativeLowRankLinear(getattr(parent, child), rank))
        selected[component] = names
    return selected


def trainable(pipe):
    return {prefix + "." + name: p for prefix in ("transformer", "connector_module")
        for name, p in getattr(pipe, prefix).named_parameters() if p.requires_grad}


def adapter_state(pipe):
    return {name: p.detach().cpu().clone() for name, p in trainable(pipe).items()}


def pixels(pipe, path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as source:
        image = source.convert("RGB")
        if image.size != (1024, 1024): raise ValueError("complete native 1024-square page required")
        value = torch.from_numpy(np.array(image)).float() / 255.0
    return (2 * value - 1).permute(2, 0, 1).to(device=pipe._gpu_device, dtype=pipe.transformer.dtype)


@torch.no_grad()
def frozen_input_features(pipe, path):
    """Same image path as native inference, before the trainable connector.

    Takes an input image only. No target, transcript, accepted answer or
    target-dependent crop enters this feature computation.
    """
    image = pixels(pipe, path)
    features, grid = pipe.get_semantic_features_dynamic([image])
    reference = pipe.pixels_to_latents(image[None])[0]
    queries = pipe.connector_module.meta_queries[None].expand(2, pipe.num_queries, -1)
    inputs = pipe._ilm_condition(queries, image_embeds=torch.cat(features * 2), image_grid_thw=grid.repeat(2, 1))
    output = pipe.llm(**inputs, return_dict=True, output_hidden_states=True)
    count = len(output.hidden_states) - 1
    selected = list(range(count - 1, 0, -6))
    merged = torch.cat([output.hidden_states[i] for i in selected], dim=-1)
    if not torch.isfinite(merged).all() or not torch.isfinite(reference).all():
        raise FloatingPointError("nonfinite native input features")
    return merged.detach(), reference.detach(), dict(selected_reader_layers=selected,
        sequence_length=merged.shape[1], hidden_width=merged.shape[2], input_only=True)


@torch.no_grad()
def encode_target(pipe, path):
    return pipe.pixels_to_latents(pixels(pipe, path)[None]).detach()
