"""Independent learned whole-image density loading and autoregressive writing."""

from __future__ import annotations

from pathlib import Path

import torch

from .correlated_native_density import CorrelatedNativeLanguage, expert_mode_candidates
from .native_pixel_instruction import complete_pairs, image_cells, native_strip, REVISION


def load_correlated(checkpoint, snapshot, device, *, allow_smoke=False):
    from transformers import AutoConfig, AutoModelForCausalLM

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["architecture"] != "correlated-native-v62" or payload["external_revision"] != REVISION:
        raise ValueError("unexpected whole-image model")
    if payload["smoke_only"] and not allow_smoke:
        raise ValueError("smoke weights are not a language result")
    config = AutoConfig.from_pretrained(str(snapshot), trust_remote_code=True, local_files_only=True)
    for name in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                 "num_channel", "patch_len", "pixels_per_patch"):
        if getattr(config, name) != payload["config"][name]:
            raise ValueError("native architecture changed")
    config._attn_implementation = "eager"
    # Architecture only: no original foundation weights are loaded here.
    author = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model = CorrelatedNativeLanguage.from_author(author, payload["components"], noise=0)
    model.load_state_dict(payload["ema_model"], strict=True)
    return model.to(device).requires_grad_(False).eval()


@torch.no_grad()
def generate_correlated(model, prompt, maximum_blocks=10):
    if maximum_blocks != 10 or not 1 <= len(prompt) <= 480:
        raise ValueError("registered prompt/output bounds differ")
    current = native_strip(complete_pairs(prompt))
    mask = torch.ones(1, current.shape[-1]//64, device=current.device)
    outputs, events, cache, stopped = [], [], None, False
    for step in range(maximum_blocks):
        with torch.autocast(device_type=current.device.type, dtype=torch.bfloat16,
                            enabled=current.device.type == "cuda"):
            logits, gate, cache = model(current if cache is None else current[..., -64:],
                mask, use_cache=True, past_key_values=cache)
        logits, gate = logits[:, -1], gate[:, -1]
        bitmap, logp = expert_mode_candidates(logits, gate)
        modes = (logits > 0).float()
        matches = (modes == bitmap[:, None]).all(-1)
        # Equivalent expert modes share an image; choose the first solely for
        # descriptive usage accounting. Image selection uses the full density.
        expert = int(matches[0].float().argmax())
        probabilities = gate.float().softmax(-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
        nxt = bitmap.reshape(1, 1, 32, 64).mul(2).sub(1)
        stopped = bool((nxt == 1).all())
        events.append(dict(block=step, route="conditional_whole_image", expert=expert,
            gate_entropy=float(entropy), image_log_probability=float(logp), blank_termination=stopped))
        if stopped:
            break
        outputs.append(nxt)
        current = torch.cat((current, nxt), -1)
        mask = torch.cat((mask, mask.new_ones(1, 1)), -1)
    if outputs:
        images = image_cells(torch.cat(outputs, -1))
        while len(images) and not bool(images[-1].any()):
            images = images[:-1]
    else:
        images = prompt[:0].clone()
    return dict(images=images, stopped=stopped, capped=not stopped, events=events)
