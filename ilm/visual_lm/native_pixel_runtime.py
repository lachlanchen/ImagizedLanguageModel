"""Load learned pixel weights without loading an external model's weights."""

from __future__ import annotations

from pathlib import Path

import torch

from .native_pixel_instruction import REVISION


def load_trained(checkpoint, snapshot, device, *, allow_smoke=False):
    from transformers import AutoConfig, AutoModelForCausalLM
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        # A save_pretrained bundle carries its own learned weights and custom code.
        return AutoModelForCausalLM.from_pretrained(str(checkpoint), local_files_only=True,
            trust_remote_code=True, torch_dtype=torch.float32,
            attn_implementation="eager").to(device).requires_grad_(False).eval()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["architecture"] != "native-pixel-instruction-v61" or payload["external_revision"] != REVISION:
        raise ValueError("unexpected native trained model")
    if payload["smoke_only"] and not allow_smoke:
        raise ValueError("smoke weights are not a trained language result")
    config = AutoConfig.from_pretrained(str(snapshot), trust_remote_code=True, local_files_only=True)
    for name in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                 "num_channel", "patch_len", "pixels_per_patch"):
        if getattr(config, name) != payload["config"][name]:
            raise ValueError("native architecture changed")
    config._attn_implementation = "eager"
    # Construct from configuration only: no source-release weights are loaded.
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.load_state_dict(payload["ema_model"], strict=True)
    return model.to(device).requires_grad_(False).eval()
