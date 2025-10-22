#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from ilm.datasets.alpaca_glyph_dataset import tokenize_en, tokenize_zh
from ilm.db.glyph_db import GlyphDB
from ilm.models.product_codebook import ProductCodebook, ProductCodebookConfig


def load_ckpt(path: str) -> dict:
    return torch.load(path, map_location="cpu")


def make_glyph_contact_sheet(db: GlyphDB, lang: str, tokens: List[str], size: int = 128, cols: int = 8) -> Image.Image:
    N = len(tokens)
    cols = max(1, cols)
    rows = (N + cols - 1) // cols
    sheet = Image.new("RGB", (cols * size, rows * size), color=(0, 0, 0))
    for i, tok in enumerate(tokens):
        r = i // cols
        c = i % cols
        p = db.ensure_glyph(lang, tok, size)
        im = Image.open(p).convert("RGB")
        sheet.paste(im, (c * size, r * size))
    return sheet


def build_code_image_hstripe(hard_codes: np.ndarray, K: int, cell: int = 4) -> Image.Image:
    # hard_codes: (T, C) int indices; render C horizontal stripes over time axis
    T, C = hard_codes.shape
    W = T * cell
    H = C * cell
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # simple colormap per channel: vary RGB weights cyclically
    base_colors = np.array([
        [255, 64, 64],
        [64, 255, 64],
        [64, 64, 255],
        [255, 192, 64],
        [64, 192, 255],
        [192, 64, 255],
        [160, 160, 64],
        [64, 160, 160],
    ], dtype=np.float32)
    for t in range(T):
        for c in range(C):
            idx = hard_codes[t, c]
            frac = 0.0 if K <= 1 else float(idx) / float(K - 1)
            color = base_colors[c % len(base_colors)] * (0.2 + 0.8 * frac)
            y0 = c * cell
            x0 = t * cell
            img[y0 : y0 + cell, x0 : x0 + cell, :] = np.clip(color, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def frame_layout_tokens(tokens: List[str], grid: int = 16) -> Tuple[List[Tuple[int, int]], int, int]:
    # Map token index to (row,col) in grid x grid frame; truncate to grid^2
    T = min(len(tokens), grid * grid)
    coords = []
    for i in range(T):
        r = i // grid
        c = i % grid
        coords.append((r, c))
    return coords, grid, T


def build_code_frame(hard_codes: np.ndarray, K: int, grid: int = 16, cell: int = 8) -> Image.Image:
    # Render tokens into a 2D frame (grid x grid), each token cell is cell x cell colored by average over channels
    T, C = hard_codes.shape
    coords, grid, Tused = frame_layout_tokens(list(range(T)), grid=grid)
    H = grid * cell
    W = grid * cell
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(Tused):
        r, c = coords[i]
        code = hard_codes[i]
        # map C codes -> RGB by 3-channel average of normalized codes
        vals = (code.astype(np.float32) / max(1, K - 1)).tolist()
        # simple projection: pack channels into RGB sums
        rsum = sum(vals[0::3]) / max(1, len(vals[0::3]))
        gsum = sum(vals[1::3]) / max(1, len(vals[1::3]))
        bsum = sum(vals[2::3]) / max(1, len(vals[2::3]))
        color = np.array([rsum, gsum, bsum]) * 255.0
        y0 = r * cell
        x0 = c * cell
        img[y0 : y0 + cell, x0 : x0 + cell, :] = np.clip(color, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def simple_mask_and_infill(tokens: List[str], emb: torch.Tensor, vocab_emb: torch.Tensor, mask_ratio: float = 0.3, window: int = 2) -> Tuple[List[int], List[int]]:
    # emb: (T,d) token embeddings for sequence; vocab_emb: (V,d) all token embeddings
    T, d = emb.shape
    device = emb.device
    masked = np.zeros(T, dtype=bool)
    # choose mask positions
    mcount = max(1, int(T * mask_ratio))
    idxs = np.random.choice(T, size=mcount, replace=False)
    masked[idxs] = True
    # fill masked by nearest neighbor to average of neighbors within window
    pred_ids = []
    sim = None
    for t in range(T):
        if not masked[t]:
            pred_ids.append(-1)
            continue
        # neighbor indices
        nbrs = []
        for o in range(-window, window + 1):
            j = t + o
            if j < 0 or j >= T or masked[j] or o == 0:
                continue
            nbrs.append(j)
        if not nbrs:
            nbrs = [i for i in range(T) if not masked[i]] or [0]
        targ = emb[nbrs].mean(dim=0, keepdim=True)  # (1,d)
        targ = F.normalize(targ, dim=-1)
        Vn = F.normalize(vocab_emb, dim=-1)
        sims = (targ @ Vn.T).squeeze(0)  # (V,)
        pid = int(sims.argmax().item())
        pred_ids.append(pid)
    return idxs.tolist(), pred_ids


def decode_ids_to_text(ids: List[int], inv_vocab: List[str], lang: str) -> str:
    toks = [inv_vocab[i].split("::", 1)[1] for i in ids]
    if lang == "en":
        return " ".join(toks)
    else:
        return "".join(toks)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Demo ILM pipeline: glyphs, code images, frame, simple infilling")
    ap.add_argument("--ckpt", required=True, help="Checkpoint path")
    ap.add_argument("--lang", required=True, choices=["en", "zh"], help="Language of input text")
    ap.add_argument("--text", required=True, help="Input text")
    ap.add_argument("--out", default="artifacts/ilm_demo", help="Output directory")
    ap.add_argument("--glyph-db", default="data/glyphdb/glyphs.sqlite3")
    ap.add_argument("--grid", type=int, default=16, help="Frame grid size (tokens per side)")
    ap.add_argument("--cell", type=int, default=8, help="Frame cell size in pixels")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Tokenize
    tokens = tokenize_en(args.text) if args.lang == "en" else tokenize_zh(args.text)
    # Glyph contact sheet
    db = GlyphDB(args.glyph_db)
    sheet = make_glyph_contact_sheet(db, args.lang, tokens, size=128, cols=min(8, max(1, len(tokens))))
    sheet.save(out / "01_input_glyphs.png")

    # Load codebook
    ckpt = load_ckpt(args.ckpt)
    vocab = ckpt["vocab"]
    inv_vocab = [None] * len(vocab)
    for k, i in vocab.items():
        inv_vocab[i] = k
    d_model = ckpt.get("config", {}).get("d_model", 128)
    n_channels = ckpt.get("config", {}).get("n_channels", 4)
    n_codes = ckpt.get("config", {}).get("n_codes", 64)
    cfg = ProductCodebookConfig(d_model=d_model, n_channels=n_channels, n_codes=n_codes)
    codebook = ProductCodebook(vocab_size=len(vocab), cfg=cfg)
    codebook.load_state_dict(ckpt["codebook"])  # type: ignore
    codebook.eval()

    # Token ids
    ids = []
    for t in tokens:
        key = f"{args.lang}::{t}"
        if key in vocab:
            ids.append(vocab[key])
        else:
            # unknown token fallback: choose closest by string or 0
            ids.append(0)
    ids_t = torch.tensor(ids, dtype=torch.long)

    # Hard codes for visualization
    with torch.no_grad():
        hard = codebook.token_codes_hard(ids_t).cpu().numpy()  # (T,C)
        emb_seq = F.normalize(codebook.token_embedding(ids_t), dim=-1)  # (T,d)
        emb_all = F.normalize(codebook.token_embedding(torch.arange(len(vocab))), dim=-1)  # (V,d)

    # Horizontal stripe code image
    hstripe = build_code_image_hstripe(hard, K=n_codes, cell=6)
    hstripe.save(out / "02_code_hstripes.png")

    # 2D frame image of codes
    frame_img = build_code_frame(hard, K=n_codes, grid=args.grid, cell=args.cell)
    frame_img.save(out / "03_code_frame.png")

    # Simple mask + infill using nearest neighbor in codebook space
    mask_idx, pred_ids = simple_mask_and_infill(tokens, emb_seq, emb_all, mask_ratio=0.3)
    # Build predicted tokens list: copy original ids, replace masked with pred ids
    pred_seq_ids = ids.copy()
    for t, pid in enumerate(pred_ids):
        if pid >= 0:
            pred_seq_ids[t] = pid
    pred_ids_t = torch.tensor(pred_seq_ids, dtype=torch.long)
    with torch.no_grad():
        pred_hard = codebook.token_codes_hard(pred_ids_t).cpu().numpy()

    pred_frame = build_code_frame(pred_hard, K=n_codes, grid=args.grid, cell=args.cell)
    pred_frame.save(out / "04_code_frame_pred.png")

    # Decode original and predicted text from ids
    orig_text = decode_ids_to_text(ids, inv_vocab, args.lang)
    pred_text = decode_ids_to_text(pred_seq_ids, inv_vocab, args.lang)
    (out / "05_text.txt").write_text(f"ORIG ({args.lang}):\n{orig_text}\n\nMASKED POS: {mask_idx}\nPRED:\n{pred_text}\n", encoding="utf-8")

    print(f"Saved demo artifacts to {out}")


if __name__ == "__main__":
    main()

