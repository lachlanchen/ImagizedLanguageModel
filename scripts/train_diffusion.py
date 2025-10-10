#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from ilm.frames.pack import SentenceFrameDataset
    from ilm.encoders.glyph_cnn import GlyphCNN
    from ilm.utils.glyphs import make_rgb_token_image
    from ilm.diffusion.unet2d import UNet2D
    from ilm.diffusion.discrete import sample_mask, masked_mse, corruption
except ModuleNotFoundError:
    import sys
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from ilm.frames.pack import SentenceFrameDataset
    from ilm.encoders.glyph_cnn import GlyphCNN
    from ilm.utils.glyphs import make_rgb_token_image
    from ilm.diffusion.unet2d import UNet2D
    from ilm.diffusion.discrete import sample_mask, masked_mse, corruption


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def collate_tokens(batch: List[dict], glyph_size: int):
    # Flatten tokens across batch and remember mapping to grid positions
    tokens: List[str] = []
    langs: List[str] = []
    grids: List[tuple[int, int, int, int]] = []  # (start, count, H, W)
    masks: List[torch.Tensor] = []
    for ex in batch:
        toks: List[str] = ex["tokens"]
        mask_flat: List[int] = ex["mask"]
        H, W = ex["H"], ex["W"]
        start = len(tokens)
        tokens.extend(toks)
        langs.extend([ex["lang"]] * len(toks))
        grids.append((start, len(toks), H, W))
        masks.append(torch.tensor(mask_flat, dtype=torch.float32))
    # Render glyph images
    Ximgs = []
    for lang, tok in zip(langs, tokens):
        rgb = make_rgb_token_image(lang, tok, size=glyph_size)
        t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        Ximgs.append(t)
    X = torch.stack(Ximgs, dim=0)  # (N,C,h,w)
    return {"X": X, "grids": grids, "masks": masks}


def tokens_to_frames(G: torch.Tensor, grids: List[tuple[int, int, int, int]], masks: List[torch.Tensor], d: int):
    # G: (N_tokens, d)
    frames = []
    valid_masks = []
    for (start, count, H, W), mflat in zip(grids, masks):
        feats = G[start:start+count]  # (H*W, d)
        feats = feats.view(H, W, d).permute(2, 0, 1).contiguous()  # (d,H,W)
        frames.append(feats)
        vm = mflat.view(H, W)[None, :, :]  # (1,H,W)
        valid_masks.append(vm)
    Fstack = torch.stack(frames, dim=0)
    Vstack = torch.stack(valid_masks, dim=0)
    return Fstack, Vstack


def main():
    ap = argparse.ArgumentParser(description="Train small 2D UNet on sentence frames with masked infilling")
    ap.add_argument("--config", default="configs/diffusion.yaml")
    ap.add_argument("--resume-glyph", default=None, help="optional color_codes checkpoint for glyph CNN init")
    # Overrides for serious training
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grid-h", type=int, default=None)
    ap.add_argument("--grid-w", type=int, default=None)
    ap.add_argument("--mask-ratio-min", type=float, default=None)
    ap.add_argument("--mask-ratio-max", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--accum-steps", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--save-every-epochs", type=int, default=1, help="checkpoint frequency (epochs)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))
    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset/loader
    jsonl = cfg["data"]["jsonl"]
    H = args.grid_h or cfg["data"]["grid_h"]
    W = args.grid_w or cfg["data"]["grid_w"]
    glyph_size = cfg["data"]["glyph_size"]
    ds = SentenceFrameDataset(jsonl, H=H, W=W)
    loader = DataLoader(ds, batch_size=(args.batch_size or cfg["data"]["batch_size"]), shuffle=True, num_workers=cfg["data"].get("num_workers", 0), collate_fn=lambda b: collate_tokens(b, glyph_size))

    # Models
    d_glyph = cfg["model"]["d_glyph"]
    glyph_cnn = GlyphCNN(d=d_glyph, in_channels=3).to(device)
    if args.resume_glyph and Path(args.resume_glyph).exists():
        ck = torch.load(args.resume_glyph, map_location=device)
        if "glyph_cnn" in ck:
            glyph_cnn.load_state_dict(ck["glyph_cnn"])  # type: ignore[index]
    glyph_cnn.eval()
    for p in glyph_cnn.parameters():
        p.requires_grad = False

    in_ch = cfg["model"]["in_channels"]
    base_ch = cfg["model"]["base_ch"]
    depth = cfg["model"]["depth"]
    out_ch = cfg["model"]["out_channels"]
    unet = UNet2D(in_ch=in_ch, base_ch=base_ch, depth=depth, out_ch=out_ch).to(device)

    if args.lr is not None:
        cfg["optim"]["lr"] = args.lr
    if args.wd is not None:
        cfg["optim"]["wd"] = args.wd
    opt = optim.AdamW(unet.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["wd"])
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    log_every = cfg["log"]["log_every"]
    ckpt_dir = Path(cfg["log"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = cfg["log"]["ckpt_name"]

    if args.epochs is not None:
        cfg["optim"]["epochs"] = int(args.epochs)
    epochs = int(cfg["optim"]["epochs"])
    step = 0
    for epoch in range(1, epochs + 1):
        unet.train()
        for batch in loader:
            X = batch["X"].to(device)
            grids = batch["grids"]
            masks = [m.to(device) for m in batch["masks"]]

            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                G = glyph_cnn(X)  # (N_tokens, d)
            Ftrue, Vmask = tokens_to_frames(G, grids, masks, d=d_glyph)
            # Create random mask per sample
            B = Ftrue.size(0)
            Hf, Wf = Ftrue.shape[-2:]
            ratio = torch.rand(B, device=device) * (cfg["train"]["mask_ratio_max"] - cfg["train"]["mask_ratio_min"]) + cfg["train"]["mask_ratio_min"]
            M = torch.stack([sample_mask((1, Hf, Wf), float(r.item()), device=device) for r in ratio], dim=0)
            # Only mask valid positions (avoid masking pads)
            M = (M * Vmask).clamp(0, 1)
            # Build model input: corrupted features + mask channel
            Fcorr = corruption(Ftrue, M)
            Xin = torch.cat([Fcorr, M], dim=1)

            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                pred = unet(Xin, t_scalar=ratio)
                loss = masked_mse(pred, Ftrue, M) / max(1, args.accum_steps)
            scaler.scale(loss).backward()

            if (step + 1) % max(1, args.accum_steps) == 0:
                if cfg["optim"].get("grad_clip") is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), cfg["optim"]["grad_clip"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            if step % log_every == 0:
                print({"epoch": epoch, "step": step, "loss": float(loss.detach().cpu()), "mask_ratio_mean": float(ratio.mean().item())})
            step += 1

        # Save checkpoint per epoch
        if (epoch % max(1, args.save_every_epochs)) == 0:
            ckpt_path = ckpt_dir / (ckpt_name if args.save_every_epochs == 0 else ckpt_name.replace('.pt', f"_e{epoch}.pt"))
            torch.save({
                "unet": unet.state_dict(),
                "cfg": cfg,
            }, ckpt_path)
            print(json.dumps({"saved": str(ckpt_path)}))

    # Final checkpoint
    final_path = ckpt_dir / ckpt_name
    torch.save({"unet": unet.state_dict(), "cfg": cfg}, final_path)
    print(json.dumps({"saved_final": str(final_path)}))


if __name__ == "__main__":
    main()
