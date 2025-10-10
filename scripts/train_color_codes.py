#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# Ensure repo root on sys.path for `ilm` imports when running as script
try:
    from ilm.data.loader import make_dataloader
    from ilm.encoders.glyph_cnn import GlyphCNN
    from ilm.code.product import ProductCode
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from ilm.data.loader import make_dataloader
    from ilm.encoders.glyph_cnn import GlyphCNN
    from ilm.code.product import ProductCode


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def cosine_anneal(start: float, end: float, t: float) -> float:
    import math
    return end + (start - end) * (1 + math.cos(math.pi * t)) / 2


def main():
    ap = argparse.ArgumentParser(description="Train 3x32 product color codes with InfoNCE alignment")
    ap.add_argument("--config", default="configs/color.yaml")
    ap.add_argument("--smoke-test", action="store_true", help="Run a quick gradient check with random data")
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs for quick tests")
    ap.add_argument("--batch-size", type=int, default=None, help="Override batch size for quick tests")
    ap.add_argument("--auto-generate-missing", action="store_true", help="Generate glyph images on the fly if missing")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))

    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    if args.smoke_test:
        # Create synthetic loader
        from torch.utils.data import DataLoader, TensorDataset
        B = 1024
        H = W = cfg["data"].get("image_size", 128)
        X = torch.rand(B, 3, H, W)
        ds = TensorDataset(X)
        bs = args.batch_size or cfg["data"].get("batch_size", 256)
        loader = DataLoader(ds, batch_size=bs, shuffle=True)
    else:
        idx_path = cfg["data"]["index_path"]
        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"index.tsv not found: {idx_path}. Run image builders in scripts/.")
        loader = make_dataloader(
            index_path=idx_path,
            batch_size=args.batch_size or cfg["data"].get("batch_size", 256),
            shuffle=True,
            num_workers=cfg["data"].get("num_workers", 4),
            image_size=cfg["data"].get("image_size", None),
            auto_generate_missing=args.auto_generate_missing,
        )

    # Model
    d_glyph = cfg["model"]["d_glyph"]
    d_code = cfg["model"]["d_code"]
    K = cfg["model"]["K"]
    C = cfg["model"]["C"]
    tau_start = cfg["model"]["tau_start"]
    tau_end = cfg["model"]["tau_end"]
    tau_epochs = cfg["model"]["tau_anneal_epochs"]
    temp_info = cfg["model"]["temperature"]
    straight = bool(cfg["model"].get("straight_through", True))

    glyph_cnn = GlyphCNN(d=d_glyph, in_channels=3).to(device)
    code = ProductCode(d_in=d_glyph, d=d_code, K=K, C=C, tau=tau_start, straight_through=straight).to(device)

    params = list(glyph_cnn.parameters()) + list(code.parameters())
    opt = optim.AdamW(params, lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["wd"])
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    w_info = cfg["loss_weights"]["info"]
    w_usage = cfg["loss_weights"]["usage"]
    w_indep = cfg["loss_weights"]["indep"]

    log_every = cfg["log"]["log_every"]
    ckpt_dir = Path(cfg["log"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_every = cfg["log"].get("ckpt_every", 1)

    if args.epochs is not None:
        cfg["optim"]["epochs"] = int(args.epochs)
    epochs = int(cfg["optim"]["epochs"])
    step = 0

    for epoch in range(1, epochs + 1):
        glyph_cnn.train()
        code.train()

        # anneal tau
        tfrac = min(1.0, max(0.0, (epoch - 1) / max(1, tau_epochs)))
        tau = cosine_anneal(tau_start, tau_end, tfrac)

        for batch in loader:
            if args.smoke_test:
                images = batch[0]
            else:
                images = batch["image"]
            images = images.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                g = glyph_cnn(images)               # (B×d_glyph)
                out = code(g, tau=tau)              # embed, logits, y
                losses, total = code.compute_losses(
                    glyph=g,
                    out=out,
                    temperature=temp_info,
                    w_info=w_info,
                    w_usage=w_usage,
                    w_indep=w_indep,
                )
            scaler.scale(total).backward()
            if cfg["optim"].get("grad_clip") is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, cfg["optim"]["grad_clip"])
            scaler.step(opt)
            scaler.update()

            if step % log_every == 0:
                print({
                    "epoch": epoch,
                    "step": step,
                    "loss_total": float(total.detach().cpu()),
                    "loss_info": float(losses.info_nce.detach().cpu()),
                    "loss_usage": float(losses.usage_kl.detach().cpu()),
                    "loss_indep": float(losses.indep.detach().cpu()),
                    "tau": float(tau),
                })
            step += 1

        if (epoch % ckpt_every) == 0:
            ckpt_path = ckpt_dir / f"color_codes_e{epoch}.pt"
            torch.save({
                "glyph_cnn": glyph_cnn.state_dict(),
                "code": code.state_dict(),
                "epoch": epoch,
                "cfg": cfg,
            }, ckpt_path)
            print({"saved": str(ckpt_path)})

    print("Training complete.")


if __name__ == "__main__":
    main()
