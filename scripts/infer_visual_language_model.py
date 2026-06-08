#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from ilm.visual_lm import ImageToImageUNet, RenderConfig
from ilm.visual_lm.dataset import pil_to_tensor, tensor_to_pil
from ilm.visual_lm.rendering import render_prompt_page


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run ILM-V image-first inference.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--char", default="言", help="Character to render into a visual prompt if --prompt-image is omitted.")
    ap.add_argument("--prompt-image", default=None, help="Optional existing prompt image.")
    ap.add_argument("--out", default="artifacts/visual_lm_infer")
    ap.add_argument("--device", default="auto")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    train_args = ckpt.get("args", {})
    model_cfg = ckpt.get("model_config", {"base_ch": train_args.get("base_ch", 32), "depth": train_args.get("depth", 3)})
    image_size = int(train_args.get("image_size", 384))
    model = ImageToImageUNet(base_ch=model_cfg["base_ch"], depth=model_cfg["depth"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if args.prompt_image:
        prompt = Image.open(args.prompt_image).convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
        prompt_source = args.prompt_image
    else:
        prompt = render_prompt_page(args.char, RenderConfig(image_size=image_size))
        prompt_source = "rendered-text-to-image prompt"
    x = pil_to_tensor(prompt).unsqueeze(0).to(device)
    with torch.no_grad():
        y = model(x)[0]
    answer = tensor_to_pil(y)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt.save(out_dir / "prompt.png")
    answer.save(out_dir / "answer.png")
    metadata = {
        "checkpoint": str(args.checkpoint),
        "char": args.char,
        "prompt_source": prompt_source,
        "output_contract": "answer.png is the primary image output; OCR/transcoding is optional after generation.",
        "codec_text_hint": args.char if len(args.char) == 1 else None,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(json.dumps({"prompt": str(out_dir / "prompt.png"), "answer": str(out_dir / "answer.png"), "metadata": str(out_dir / "metadata.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
