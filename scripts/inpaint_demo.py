#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ilm.datasets.alpaca_glyph_dataset import tokenize_en, tokenize_zh
from ilm.models.product_codebook import ProductCodebook, ProductCodebookConfig
from ilm.diffusion.inpaint_unet2d import InpaintNet


def layout_ids(ids: List[int], grid: int) -> np.ndarray:
    T = min(len(ids), grid * grid)
    frame = np.full((grid, grid), fill_value=-1, dtype=np.int64)
    for i in range(T):
        r = i // grid
        c = i % grid
        frame[r, c] = ids[i]
    return frame


def build_frame_image(hard_codes: np.ndarray, K: int, grid: int = 16, cell: int = 8) -> Image.Image:
    T = grid * grid
    H = grid * cell
    W = grid * cell
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(T):
        r = i // grid
        c = i % grid
        # color by code tuple
        if i < hard_codes.shape[0]:
            code = hard_codes[i]
        else:
            code = np.zeros((3,), dtype=np.int64)
        vals = (code.astype(np.float32) / max(1, K - 1)).tolist()
        rsum = sum(vals[0::3]) / max(1, len(vals[0::3]))
        gsum = sum(vals[1::3]) / max(1, len(vals[1::3]))
        bsum = sum(vals[2::3]) / max(1, len(vals[2::3]))
        color = np.array([rsum, gsum, bsum]) * 255.0
        y0 = r * cell
        x0 = c * cell
        img[y0 : y0 + cell, x0 : x0 + cell, :] = np.clip(color, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Inpaint a sentence frame with trained UNet")
    ap.add_argument("--ckpt-code", required=True, help="Codebook checkpoint")
    ap.add_argument("--ckpt-inpaint", required=True, help="Inpaint UNet checkpoint")
    ap.add_argument("--lang", required=True, choices=["en", "zh"])
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="artifacts/inpaint_demo")
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--cell", type=int, default=8)
    ap.add_argument("--mask-ratio", type=float, default=0.3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Load codebook
    ckpt = torch.load(args.ckpt_code, map_location="cpu")
    vocab = ckpt["vocab"]
    cfg = ckpt.get("config", {})
    d_model = cfg.get("d_model", 96)
    n_channels = cfg.get("n_channels", 3)
    n_codes = cfg.get("n_codes", 32)
    code_cfg = ProductCodebookConfig(d_model=d_model, n_channels=n_channels, n_codes=n_codes)
    codebook = ProductCodebook(vocab_size=len(vocab), cfg=code_cfg)
    codebook.load_state_dict(ckpt["codebook"])  # type: ignore
    codebook.eval().to(args.device)

    # Tokens and ids
    toks = tokenize_en(args.text) if args.lang == "en" else tokenize_zh(args.text)
    ids = [vocab.get(f"{args.lang}::{t}", 0) for t in toks]
    frame_ids = layout_ids(ids, args.grid)
    H = W = args.grid
    flat_ids = torch.tensor([i if i >= 0 else 0 for i in frame_ids.reshape(-1)], dtype=torch.long, device=args.device)
    with torch.no_grad():
        emb = codebook.token_embedding(flat_ids)  # (G*G,d)
        y = emb.reshape(H, W, d_model).permute(2, 0, 1).contiguous().unsqueeze(0)  # (1,d,H,W)

    # Load inpaint UNet
    ip_ckpt = torch.load(args.ckpt_inpaint, map_location="cpu")
    net = InpaintNet(d_model=d_model, r=ip_ckpt.get("r_channels", 16))
    net.load_state_dict(ip_ckpt["net"])  # type: ignore
    net.eval().to(args.device)

    # Build random mask on valid tokens
    valid = (torch.tensor(frame_ids, device=args.device) >= 0)
    vpos = valid.nonzero(as_tuple=False)
    m = torch.zeros((H, W), dtype=torch.float32, device=args.device)
    if vpos.numel() > 0:
        k = max(1, int(vpos.size(0) * args.mask_ratio))
        idx = torch.randperm(vpos.size(0))[:k]
        sel = vpos[idx]
        m[sel[:, 0], sel[:, 1]] = 1.0
    mask = m.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    with torch.no_grad():
        y_hat, _, _ = net(y.to(args.device), mask)
        # Merge predictions at masked locations only
        y_pred = y * (1.0 - mask) + y_hat * mask

        # For visualization: get hard codes via nearest neighbor to codebook embeddings
        V = len(vocab)
        all_ids = torch.arange(V, dtype=torch.long, device=args.device)
        all_emb = F.normalize(codebook.token_embedding(all_ids), dim=-1)  # (V,d)
        y_seq = y_pred.squeeze(0).permute(1, 2, 0).reshape(-1, d_model)
        y_seq = F.normalize(y_seq, dim=-1)
        sims = y_seq @ all_emb.T  # (G*G, V)
        top = sims.argmax(dim=1).cpu().numpy()

    # Build hard code tuples for first T cells
    with torch.no_grad():
        codes = codebook.token_codes_hard(torch.tensor(top, dtype=torch.long))  # (G*G,C)
        codes = codes.cpu().numpy()
    frame_img = build_frame_image(codes, K=n_codes, grid=args.grid, cell=args.cell)
    frame_img.save(out / "predicted_frame.png")
    print(f"Saved {out / 'predicted_frame.png'}")


if __name__ == "__main__":
    main()

