"""Native-scale semantic windows over a complete high-resolution latent image.

This changes source feature extraction, adds no parameters, and retains every
visual position in the language context. It is an experimental reading path,
not a demonstrated image-answer ILM.
"""
import math

import torch
from torch.nn import functional as F


def origins(size, window=27):
    if size < window:
        raise ValueError("input grid smaller than native window")
    count = math.ceil(size / window)
    return [0] if count == 1 else [round(i * (size - window) / (count - 1)) for i in range(count)]


def semantic_windows(bundle, raw, height, width):
    model = bundle.model
    if raw.shape[1] != height * width:
        raise ValueError("projected image order differs from spatial grid")
    rows, columns = origins(height), origins(width)
    positions = model.position_embedding(torch.arange(729, device=raw.device)[None])
    if height == width == 27:
        # No averaging or extra rounding in the source-equivalence condition.
        result = model.und_trans(raw + positions)["last_hidden_state"]
    else:
        grid = raw.reshape(raw.shape[0], height, width, raw.shape[-1])
        total = None
        coverage = torch.zeros((1, 1, height, width), device=raw.device, dtype=torch.float32)
        for row in rows:
            for column in columns:
                window = grid[:, row:row + 27, column:column + 27].reshape(raw.shape[0], 729, raw.shape[-1])
                encoded = model.und_trans(window + positions)["last_hidden_state"]
                encoded = encoded.transpose(1, 2).reshape(raw.shape[0], raw.shape[-1], 27, 27)
                padding = (column, width - column - 27, row, height - row - 27)
                contribution = F.pad(encoded.float(), padding)
                total = contribution if total is None else total + contribution
                coverage += F.pad(torch.ones((1, 1, 27, 27), device=raw.device), padding)
        if not bool((coverage > 0).all()):
            raise ValueError("window assembly omitted an image patch")
        result = (total / coverage).to(raw.dtype).flatten(2).transpose(1, 2)
    return result, dict(grid=[height, width], window=[27, 27], rows=rows, columns=columns,
        semantic_windows=len(rows) * len(columns), output_visual_positions=height * width,
        patches_dropped=0, averaging="uniform FP32 sum and coverage, then cast to source dtype")


@torch.inference_mode()
def windowed_prefix(bundle, latents, instruction):
    """Preserve the source chat/mask layout after local semantic extraction."""
    model, native, tok, ids = bundle.model, bundle.native, bundle.tokenizer, bundle.ids
    z = latents.to(bundle.dtype)
    p = model.config.patch_size
    height, width = z.shape[-2] // p, z.shape[-1] // p
    raw = model.image_embedder_und(z)
    generative = model.image_embedder_gen(z)
    semantic, geometry = semantic_windows(bundle, raw, height, width)
    visual = model.fusion_proj(torch.cat([semantic, generative], dim=-1))
    encode = lambda words: tok(words, add_special_tokens=False).input_ids
    a = [ids["bos_id"]] + encode("system\nYou are a helpful assistant.<|im_end|>") + encode("\n<|im_start|>user\n")
    b = [ids["boi_id"], ids["eoi_id"]] + encode(instruction) + encode("\n<|im_start|>assistant\n")
    ta, tb = [torch.tensor([x], device=z.device) for x in (a, b)]
    ea, eb = [model.showo.model.embed_tokens(x) for x in (ta, tb)]
    t = model.time_embed(torch.ones((1, 1), device=z.device), ea.dtype)
    if hasattr(model, "time_embed_proj"):
        t = model.time_embed_proj(t)
    embeddings = torch.cat([ea, eb[:, :1], t, visual, eb[:, 1:]], dim=1).to(bundle.dtype)
    positions = torch.tensor([[[len(a) + 2, visual.shape[1] + 1]]], device=z.device)
    mask = native.masks.omni_attn_mask_naive(1, embeddings.shape[1], positions, z.device, inverted=True).to(bundle.dtype)
    return embeddings, mask, geometry
