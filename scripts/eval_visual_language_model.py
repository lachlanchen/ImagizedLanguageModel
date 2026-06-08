#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ilm.visual_lm import GlyphCorpus, ImageToImageUNet, RenderConfig, VisualLanguageDataset
from ilm.visual_lm.dataset import tensor_to_pil
from ilm.visual_lm.model import psnr_from_l1
from ilm.visual_lm.rendering import make_triptych


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate an ILM-V image-to-image checkpoint.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--glyph-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="auto")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    train_args = ckpt.get("args", {})
    model_cfg = ckpt.get("model_config", {"base_ch": train_args.get("base_ch", 32), "depth": train_args.get("depth", 3)})
    model = ImageToImageUNet(base_ch=model_cfg["base_ch"], depth=model_cfg["depth"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    characters = ckpt.get("characters") or train_args.get("characters", "言,中").split(",")
    image_size = int(train_args.get("image_size", 384))
    corpus = GlyphCorpus(args.glyph_root or train_args.get("glyph_root"), characters=characters)
    ds = VisualLanguageDataset(corpus, render_config=RenderConfig(image_size=image_size), length=args.samples, seed=args.seed, characters=characters)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    out_dir = Path(args.out) if args.out else Path(args.checkpoint).with_suffix("").parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    total_l1 = 0.0
    total_psnr = 0.0
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(dl):
            prompt = batch["prompt"].to(device)
            target = batch["target"].to(device)
            pred = model(prompt)
            total_l1 += float(F.l1_loss(pred, target).cpu()) * prompt.size(0)
            total_psnr += psnr_from_l1(pred, target) * prompt.size(0)
            n += prompt.size(0)
            if i < 3:
                make_triptych(tensor_to_pil(prompt[0]), tensor_to_pil(pred[0]), tensor_to_pil(target[0])).save(out_dir / f"eval_{i:02d}.png")
    metrics = {"samples": n, "l1": total_l1 / max(1, n), "psnr": total_psnr / max(1, n)}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
