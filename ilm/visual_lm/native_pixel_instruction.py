"""Answer-only native pixel likelihood and unrestricted image-block writing."""

from __future__ import annotations

import torch
from torch.nn import functional as F

BLOCK = 64
REVISION = "c5d5bcfbad2f290296fd83810558f396c6a6333a"
WEIGHT_SHA256 = "be5706028246c147998a141110fcc29c9ce2026623b31bd85d20ed0916659af8"


def complete_pairs(images):
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 32, 32):
        raise ValueError("expected binary image cells [L,1,32,32]")
    if not bool(((images == 0) | (images == 1)).all()):
        raise ValueError("binary ink pixels required")
    return (torch.cat((images, torch.zeros_like(images[:1])))
            if len(images) % 2 else images)


def native_strip(images):
    if len(images) % 2:
        raise ValueError("native blocks require complete cell pairs")
    return (1-2*images).permute(1, 2, 0, 3).reshape(1, 1, 32, len(images)*32)


def training_strip(prompt, answer):
    if not 1 <= len(prompt) <= 480 or not 1 <= len(answer) <= 16:
        raise ValueError("source or answer outside registered bounds")
    prompt, answer = complete_pairs(prompt), complete_pairs(answer)
    terminal = torch.zeros_like(prompt[:1]).expand(4, -1, -1, -1)
    return native_strip(torch.cat((prompt, answer, terminal))), len(prompt)//2


def answer_only_bce(logits, pixels, prompt_blocks):
    if logits.shape != pixels.shape or pixels.shape[:3] != (1, 1, 32):
        raise ValueError("native strip/logit shape differs")
    if pixels.shape[-1] % BLOCK or not 1 <= prompt_blocks < pixels.shape[-1]//BLOCK:
        raise ValueError("invalid prompt boundary")
    # State at native block t predicts t+1. Prompt labels never contribute.
    predicted = logits[..., (prompt_blocks-1)*BLOCK:-BLOCK].float()
    targets = pixels[..., prompt_blocks*BLOCK:].float().add(1).mul(.5)
    if predicted.shape != targets.shape:
        raise AssertionError("causal answer shift differs")
    return F.binary_cross_entropy_with_logits(predicted, targets)


def image_cells(strip):
    if strip.shape[:3] != (1, 1, 32) or strip.shape[-1] % 32:
        raise ValueError("invalid output strip")
    count = strip.shape[-1]//32
    return (strip[0] <= 0).float().reshape(1, 32, count, 32).permute(2, 0, 1, 3)


@torch.no_grad()
def generate(model, prompt, maximum_blocks=10):
    if maximum_blocks != 10:
        raise ValueError("registered output maximum is ten native blocks")
    current = native_strip(complete_pairs(prompt))
    mask = torch.ones(1, current.shape[-1]//BLOCK, device=current.device)
    outputs, events, cache, stopped = [], [], None, False
    for step in range(maximum_blocks):
        with torch.autocast(device_type=current.device.type, dtype=torch.bfloat16,
                            enabled=current.device.type == "cuda"):
            result = model(inputs_embeds=current if cache is None else current[..., -BLOCK:],
                           attention_mask=mask, past_key_values=cache, use_cache=True)
        cache = result.past_key_values
        nxt = (result.logits[..., -BLOCK:] > 0).float().mul(2).sub(1)
        stopped = bool((nxt == 1).all())
        events.append(dict(block=step, route="native_free_pixels", blank_termination=stopped))
        if stopped:
            break
        outputs.append(nxt)
        current = torch.cat((current, nxt), -1)
        mask = torch.cat((mask, mask.new_ones(1, 1)), -1)
    if outputs:
        images = image_cells(torch.cat(outputs, -1))
        # Padding convention only: never alter a nonblank generated cell.
        while len(images) and not bool(images[-1].any()):
            images = images[:-1]
    else:
        images = prompt[:0].clone()
    return dict(images=images, stopped=stopped, capped=not stopped, events=events)
