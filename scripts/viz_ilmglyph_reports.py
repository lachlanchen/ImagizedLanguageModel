#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

from ilm.models.product_codebook import ProductCodebook, ProductCodebookConfig
from ilm.encoders.glyph_cnn import GlyphCNN
from ilm.db.glyph_db import GlyphDB


def load_ckpt(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    return ckpt


def pca_2d(X: np.ndarray, k: int = 2) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T


def nearest_neighbors(emb: np.ndarray, langs: List[str], vocab_inv: List[str], topk: int = 5) -> List[Tuple[str, List[Tuple[str, float]]]]:
    # Cosine sim
    X = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sims = X @ X.T
    results = []
    idx_en = [i for i, l in enumerate(langs) if l == "en"]
    idx_zh = [i for i, l in enumerate(langs) if l == "zh"]
    for i in idx_en:
        s = sims[i, idx_zh]
        order = np.argsort(-s)[:topk]
        # IMPORTANT: use ordered zh indices for both token and similarity
        nbrs = [(vocab_inv[idx_zh[order[j]]], float(s[order[j]])) for j in range(len(order))]
        results.append((vocab_inv[i], nbrs))
    return results


def plot_scatter(save_path: Path, Z: np.ndarray, langs: List[str], c0: np.ndarray | None = None) -> None:
    if not HAS_PLT:
        return
    colors = ["tab:blue" if l == "en" else "tab:red" for l in langs]
    plt.figure(figsize=(6, 5))
    if c0 is not None:
        # jitter by channel 0 if provided
        plt.scatter(Z[:, 0], Z[:, 1], c=c0, cmap="viridis", s=8, alpha=0.8)
        cb = plt.colorbar()
        cb.set_label("channel-0 index")
    else:
        plt.scatter(Z[:, 0], Z[:, 1], c=colors, s=8, alpha=0.8)
    plt.title("Token embeddings (PCA-2D)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize ILM glyph/code training results")
    ap.add_argument("--ckpt", required=True, help="Checkpoint path (ckpt_epochX.pt)")
    ap.add_argument("--out", default="artifacts/ilm_viz", help="Output directory")
    ap.add_argument("--glyph-db", default="data/glyphdb/glyphs.sqlite3")
    ap.add_argument("--hard", action="store_true", help="Use hard codes for embedding (argmax); default uses soft")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_ckpt(args.ckpt)
    vocab: Dict[str, int] = ckpt["vocab"]
    inv = [None] * len(vocab)
    for k, i in vocab.items():
        inv[i] = k
    langs = [k.split("::", 1)[0] for k in inv]

    d_model = ckpt.get("config", {}).get("d_model", 128)
    n_channels = ckpt.get("config", {}).get("n_channels", 3)
    n_codes = ckpt.get("config", {}).get("n_codes", 32)
    cfg = ProductCodebookConfig(d_model=d_model, n_channels=n_channels, n_codes=n_codes)
    codebook = ProductCodebook(vocab_size=len(vocab), cfg=cfg)
    codebook.load_state_dict(ckpt["codebook"])
    codebook.eval()

    with torch.no_grad():
        ids = torch.arange(len(vocab), dtype=torch.long)
        if args.hard:
            # use hard codes then sum code vectors
            codes = codebook.token_codes_hard(ids)
            d_per = d_model // n_channels
            emb_parts = []
            for c in range(n_channels):
                idx = codes[:, c]
                part = codebook.codebooks[c, idx]  # (V, d_per)
                emb_parts.append(part)
            E = torch.cat(emb_parts, dim=1)
        else:
            E = codebook.token_embedding(ids)
        E = F.normalize(E, dim=-1).cpu().numpy()

    Z = pca_2d(E, k=2)
    plot_scatter(out_dir / "pca_tokens.png", Z, langs)

    # Channel-0 index color scatter
    with torch.no_grad():
        probs = F.softmax(codebook.assign_logits, dim=-1).cpu().numpy()  # (V,C,K)
        c0_idx = probs[:, 0, :].argmax(axis=1)
    plot_scatter(out_dir / "pca_tokens_c0.png", Z, langs, c0_idx)

    # Nearest neighbors EN->ZH
    nn_results = nearest_neighbors(E, langs, inv, topk=5)
    with open(out_dir / "nn_en_to_zh.tsv", "w", encoding="utf-8") as f:
        f.write("en_token\tzh_neighbor\tsim\n")
        for en_tok, nbrs in nn_results:
            for zh_tok, sim in nbrs:
                f.write(f"{en_tok}\t{zh_tok}\t{sim:.4f}\n")

    print(f"Saved visualizations to {out_dir}")


if __name__ == "__main__":
    main()
