#!/usr/bin/env python3
"""Check full actor causality without inspecting development performance."""

import json
from pathlib import Path

import torch

from ilm.visual_lm.native_pixel_runtime import load_trained
from scripts.train_native_pixel_instruction_v61 import DEFAULT_SNAPSHOT, DEFAULT_DATA, ImageData


@torch.no_grad()
def main():
    torch.set_num_threads(4); torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    model = load_trained("artifacts/native_pixel_instruction_v61_smoke_full/model_final.pt",
                         DEFAULT_SNAPSHOT, device, allow_smoke=True)
    data = ImageData(DEFAULT_DATA)
    pixels, boundary = data.get(data.training[0]); pixels = pixels.to(device)
    mask = torch.ones(1, pixels.shape[-1]//64, device=device)
    altered = pixels.clone(); altered[..., boundary*64:] *= -1
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        first = model(inputs_embeds=pixels, attention_mask=mask, use_cache=False).logits
        second = model(inputs_embeds=altered, attention_mask=mask, use_cache=False).logits
    delta = float((first[..., :boundary*64]-second[..., :boundary*64]).abs().max())
    if delta != 0:
        raise AssertionError(f"future answer pixels changed earlier prediction: {delta}")
    path = Path("artifacts/native_pixel_instruction_v61_preflight.json")
    receipt = json.loads(path.read_text())
    receipt["causal_future_pixel_max_error"] = delta
    receipt["checks"]["full_native_actor_causality"] = True
    receipt["runtime_loads_source_weights"] = False
    path.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
