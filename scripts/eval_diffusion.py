#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
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


def collate_tokens(batch: List[dict], glyph_size: int, glyph_db_path: str | None = None):
    from ilm.db.glyph_db import GlyphDB
    db = GlyphDB(glyph_db_path) if glyph_db_path else None
    tokens: List[str] = []
    langs: List[str] = []
    grids: List[tuple[int, int, int, int]] = []
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
    Ximgs = []
    for lang, tok in zip(langs, tokens):
        if db is not None:
            path = db.ensure_glyph(lang, tok, size=glyph_size)
            import PIL.Image as Image
            t = torch.from_numpy(np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
        else:
            rgb = make_rgb_token_image(lang, tok, size=glyph_size)
            t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        Ximgs.append(t)
    X = torch.stack(Ximgs, dim=0)
    return {"X": X, "grids": grids, "masks": masks}


def tokens_to_frames(G: torch.Tensor, grids: List[tuple[int, int, int, int]], masks: List[torch.Tensor], d: int):
    frames = []
    valid_masks = []
    for (start, count, H, W), mflat in zip(grids, masks):
        feats = G[start:start+count]
        feats = feats.view(H, W, d).permute(2, 0, 1).contiguous()
        frames.append(feats)
        vm = mflat.view(H, W)[None, :, :]
        valid_masks.append(vm)
    Fstack = torch.stack(frames, dim=0)
    Vstack = torch.stack(valid_masks, dim=0)
    return Fstack, Vstack


def main():
    ap = argparse.ArgumentParser(description="Evaluate masked infilling accuracy of diffusion UNet on sentence frames")
    ap.add_argument("--jsonl", default="data/processed/test_100.jsonl")
    ap.add_argument("--checkpoint", default="artifacts/diffusion_unet.pt")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--glyph-size", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--glyph-db", default=None, help="Optional SQLite glyph DB to cache glyphs for frames")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    ds = SentenceFrameDataset(args.jsonl, H=8, W=8)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=lambda b: collate_tokens(b, glyph_size=args.glyph_size, glyph_db_path=args.glyph_db))

    glyph_cnn = GlyphCNN(d=128, in_channels=3).to(device)
    glyph_cnn.eval()
    for p in glyph_cnn.parameters():
        p.requires_grad = False

    ck = torch.load(args.checkpoint, map_location=device)
    cfg = ck.get("cfg", {})
    in_ch = cfg.get("model", {}).get("in_channels", 129)
    base_ch = cfg.get("model", {}).get("base_ch", 64)
    depth = cfg.get("model", {}).get("depth", 2)
    out_ch = cfg.get("model", {}).get("out_channels", 128)
    unet = UNet2D(in_ch=in_ch, base_ch=base_ch, depth=depth, out_ch=out_ch).to(device)
    unet.load_state_dict(ck["unet"])  # type: ignore[index]
    unet.eval()

    mses = []
    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            grids = batch["grids"]
            masks = [m.to(device) for m in batch["masks"]]
            G = glyph_cnn(X)
            Ftrue, Vmask = tokens_to_frames(G, grids, masks, d=128)
            B, d, Hf, Wf = Ftrue.shape
            ratio = torch.full((B,), 0.5, device=device)
            M = torch.stack([sample_mask((1, Hf, Wf), 0.5, device=device) for _ in range(B)], dim=0)
            M = (M * Vmask).clamp(0, 1)
            Xin = torch.cat([corruption(Ftrue, M), M], dim=1)
            pred = unet(Xin, t_scalar=ratio)
            mse = float(masked_mse(pred, Ftrue, M).detach().cpu())
            mses.append(mse)

    out_dir = Path("artifacts/diffusion_metrics") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"masked_mse": float(np.mean(mses)), "N_batches": len(mses)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"metrics_dir": str(out_dir), **summary}))


if __name__ == "__main__":
    main()
