#!/usr/bin/env python3
"""Explore a pinned external pixel-language checkpoint without symbolic inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

REPOSITORY = "TYTTYTTYT/llama-pixar-477m-p32-stage2"
REVISION = "c5d5bcfbad2f290296fd83810558f396c6a6333a"
WEIGHT_SHA256 = "be5706028246c147998a141110fcc29c9ce2026623b31bd85d20ed0916659af8"
PROMPTS = [
    "The capital of France is",
    "Water freezes at",
    "The word alphabet comes from",
    "中国的首都是",
    "汉字由",
    "汉字“言”的本义是",
]


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument(
        "--out", default="artifacts/external/mixar477_baseline_20261105"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--system-bindings", default="/usr/lib/python3/dist-packages")
    parser.add_argument(
        "--foreign-bindings",
        default="artifacts/external/native_bindings/usr/lib/python3/dist-packages/gi",
    )
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        raise RuntimeError("author custom code requires Python 3.11 or later")
    out = Path(args.out)
    if (out / "receipt.json").exists():
        raise FileExistsError("completed baseline output is immutable")
    snapshot = Path(args.snapshot)
    if (
        snapshot.name != REVISION
        or digest(snapshot / "model.safetensors") != WEIGHT_SHA256
    ):
        raise ValueError("external baseline weights/revision differ")
    # Explicit native imports precede pixar_render. Its missing-cairo fallback
    # launches a Conda environment updater; never invoke that fallback here.
    if Path(args.system_bindings).is_dir():
        sys.path.append(args.system_bindings)
    try:
        import cairo
        import gi
        import manimpango

        if Path(args.foreign_bindings).is_dir():
            gi.__path__.append(str(Path(args.foreign_bindings).resolve()))
        gi.require_foreign("cairo")
    except ImportError as exc:
        raise RuntimeError(
            "native Cairo/GI/Pango dependencies must be prepared explicitly; no automatic installer is permitted"
        ) from exc
    import numpy as np
    from PIL import Image
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    from pixar_render import PixarProcessor

    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.set_float32_matmul_precision("high")
    model = (
        AutoModelForCausalLM.from_pretrained(
            str(snapshot),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        .to(device)
        .eval()
    )
    processor = PixarProcessor.load(str(snapshot / "render_config"))
    block_width = processor._block_width
    if block_width != 64:
        raise ValueError("baseline expects two 32px patches per block")
    encoded = processor.render(
        PROMPTS, padding_side="left", add_eos=False, truncate=True
    )
    encoded = processor.align_text_to_right_edge(encoded, 2)
    grey = encoded.pixel_values.to(device).float().mean(dim=1, keepdim=True)
    current = (grey >= 128).float().mul(2).sub(1)
    attention = encoded.attention_mask.to(device)
    prompt = current.clone()
    outputs = []
    cache = None
    stopped = [False] * len(PROMPTS)
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(32):
            prediction = model(
                inputs_embeds=current if cache is None else current[..., -block_width:],
                attention_mask=attention,
                past_key_values=cache,
                use_cache=True,
            )
            cache = prediction.past_key_values
            nxt = (prediction.logits[..., -block_width:] > 0).float().mul(2).sub(1)
            for i in range(len(PROMPTS)):
                if float(nxt[i].mean()) > 0.9:
                    stopped[i] = True
                if stopped[i]:
                    nxt[i] = 1
            outputs.append(nxt)
            current = torch.cat((current, nxt), dim=-1)
            attention = torch.cat(
                (attention, attention.new_ones(len(PROMPTS), 1)), dim=-1
            )
            if all(stopped):
                break
    generated = torch.cat(outputs, dim=-1).cpu()
    elapsed = time.perf_counter() - started

    def write_image(tensor, path):
        pixels = ((tensor[0] > 0).to(torch.uint8) * 255).numpy()
        Image.fromarray(pixels, mode="L").save(path)

    def write_grid(tensor, path):
        pixels = ((tensor[0] > 0).to(torch.uint8) * 255).numpy()
        width = 1024
        rows = (pixels.shape[1] + width - 1) // width
        page = np.full((rows * 32, width), 255, dtype=np.uint8)
        for row in range(rows):
            chunk = pixels[:, row * width : (row + 1) * width]
            page[row * 32 : (row + 1) * 32, : chunk.shape[1]] = chunk
        Image.fromarray(page, mode="L").save(path)

    for i in range(len(PROMPTS)):
        write_image(prompt[i].cpu(), out / f"case_{i:02}_prompt.png")
        write_image(generated[i], out / f"case_{i:02}_generated.png")
        write_grid(generated[i], out / f"case_{i:02}_generated_grid.png")
    torch.save(
        {
            "prompt_pixels": prompt.cpu(),
            "generated_pixels": generated,
            "attention_mask": encoded.attention_mask,
            "prompts_used_only_by_host_renderer": PROMPTS,
        },
        out / "raw_pixels.pt",
    )
    receipt = {
        "external_pretrained_baseline": True,
        "qualification": "exploratory external baseline; not a local ILM qualification",
        "repository": REPOSITORY,
        "revision": REVISION,
        "weights_sha256": WEIGHT_SHA256,
        "license": "Apache-2.0 per author release",
        "parameters": sum(p.numel() for p in model.parameters()),
        "dtype": "float32",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": sys.version.split()[0],
        "cairo_version": cairo.cairo_version_string(),
        "gi_version": gi.__version__,
        "manimpango_version": getattr(manimpango, "__version__", None),
        "model_calls_receive_images_only": True,
        "tokenizer_in_runtime": False,
        "ocr_in_runtime": False,
        "external_network_model_call": False,
        "selection_or_repair": False,
        "native_dependency_preflight": True,
        "prompt_strings_used_by_host_renderer_only": PROMPTS,
        "generated_blocks": len(outputs),
        "maximum_new_blocks": 32,
        "blank_eos_flags": stopped,
        "elapsed_generation_seconds": elapsed,
        "peak_allocated_vram_gib": torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0,
        "source_sha256": digest(__file__),
        "author_source_sha256": {p.name: digest(p) for p in snapshot.glob("*.py")},
        "render_config_sha256": digest(
            snapshot / "render_config/pixar_processor_conf.json"
        ),
    }
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
