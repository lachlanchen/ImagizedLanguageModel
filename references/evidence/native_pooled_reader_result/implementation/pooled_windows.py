"""Test native-length context after full-resolution windowed visual encoding."""
import torch
from torch.nn import functional as F

from .windowed import windowed_prefix


@torch.inference_mode()
def pooled_windowed_prefix(bundle, latents, instruction):
    prefix, _, geometry = windowed_prefix(bundle, latents, instruction)
    height, width = geometry["grid"]
    if height == width == 27:
        # Preserve the exact native single-window path, including its mask.
        from .showo2_source import reading_prefix
        direct, mask = reading_prefix(bundle, latents, instruction)
        if not torch.equal(prefix, direct):
            raise ValueError("native window and original prefix differ")
        return prefix, mask, dict(geometry, context_resampled=False)
    tok, ids = bundle.tokenizer, bundle.ids
    encode = lambda words: tok(words, add_special_tokens=False).input_ids
    a = [ids["bos_id"]] + encode("system\nYou are a helpful assistant.<|im_end|>") + encode("\n<|im_start|>user\n")
    start = len(a) + 2
    count = height * width
    features = prefix[:, start:start + count]
    grid = features.reshape(features.shape[0], height, width, features.shape[-1]).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(grid.float(), (27, 27)).to(prefix.dtype).flatten(2).transpose(1, 2)
    compact = torch.cat([prefix[:, :start], pooled, prefix[:, start + count:]], dim=1)
    positions = torch.tensor([[[start, 730]]], device=prefix.device)
    mask = bundle.native.masks.omni_attn_mask_naive(1, compact.shape[1], positions, prefix.device, inverted=True).to(prefix.dtype)
    return compact, mask, dict(geometry, context_resampled=True,
        original_visual_positions=count, output_visual_positions=729,
        context_pool="adaptive average of fused features in FP32; no source-image resize")
