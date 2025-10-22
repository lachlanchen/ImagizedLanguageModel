#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ilm.datasets.alpaca_glyph_dataset import (
    QAPair,
    build_vocab_and_cache_glyphs,
    iter_alpaca_qa,
)
from ilm.db.glyph_db import GlyphDB
from ilm.encoders.glyph_cnn import GlyphCNN
from ilm.models.product_codebook import ProductCodebook, ProductCodebookConfig


def load_img_rgb(path: str) -> np.ndarray:
    arr = np.array(Image.open(path).convert("RGB"))
    return arr


def gather_token_batch(
    qa_batch: List[QAPair],
    vocab: dict,
    db: GlyphDB,
    glyph_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[str, str]]]:
    """Collect unique tokens across a QA mini-batch into tensors.

    Returns:
    - images: float tensor (U, 3, H, W)
    - token_ids: long tensor (U,)
    - keys: list of (lang, token) in the same order
    """
    keys: List[Tuple[str, str]] = []
    seen = set()
    for pair in qa_batch:
        for token in pair.q_tokens + pair.a_tokens:
            key = (pair.lang, token)
            if key not in seen:
                seen.add(key)
                keys.append(key)
    imgs = []
    ids = []
    for lang, token in keys:
        rec_path = db.ensure_glyph(lang, token, glyph_size)
        imgs.append(load_img_rgb(rec_path))
        ids.append(vocab[f"{lang}::{token}"])
    images = torch.from_numpy(np.stack(imgs, axis=0)).float().permute(0, 3, 1, 2) / 255.0
    token_ids = torch.tensor(ids, dtype=torch.long)
    return images, token_ids, keys


def encode_sentence(token_list: List[str], lang: str, vocab: dict, codebook: ProductCodebook) -> torch.Tensor:
    ids = [vocab[f"{lang}::{t}"] for t in token_list if f"{lang}::{t}" in vocab]
    if not ids:
        return torch.zeros(codebook.cfg.d_model, device=codebook.codebooks.device)
    ids_t = torch.tensor(ids, dtype=torch.long, device=codebook.codebooks.device)
    emb = codebook.token_embedding(ids_t)  # (L, d)
    return emb.mean(dim=0)


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    # x: (L, d), normalized
    return x @ x.T


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train ILM glyph-based codebook on Alpaca QA")
    ap.add_argument("--en", required=False, help="Path to Alpaca-style JSON for English")
    ap.add_argument("--zh", required=False, help="Path to Alpaca-style JSON for Chinese")
    ap.add_argument("--glyph-db", default="data/glyphdb/glyphs.sqlite3", help="SQLite DB for glyphs")
    ap.add_argument("--out", default="artifacts/ilm_glyph_train", help="Output directory")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8, help="QA pairs per step")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-channels", type=int, default=4)
    ap.add_argument("--n-codes", type=int, default=64)
    ap.add_argument("--glyph-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load QA data
    pairs: List[QAPair] = []
    if args.en:
        pairs.extend(list(iter_alpaca_qa(args.en, "en")))
    if args.zh:
        pairs.extend(list(iter_alpaca_qa(args.zh, "zh")))
    if not pairs:
        raise SystemExit("No QA data provided. Pass --en and/or --zh Alpaca JSON paths.")

    # Init glyph DB
    db = GlyphDB(args.glyph_db)

    # Build vocab and render glyphs
    vocab, pairs = build_vocab_and_cache_glyphs(pairs, db, glyph_size=args.glyph_size)
    vocab_size = len(vocab)
    (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    # Models
    glyph_cnn = GlyphCNN(d=args.d_model, in_channels=3).to(args.device)
    code_cfg = ProductCodebookConfig(d_model=args.d_model, n_channels=args.n_channels, n_codes=args.n_codes)
    codebook = ProductCodebook(vocab_size=vocab_size, cfg=code_cfg).to(args.device)

    params = list(glyph_cnn.parameters()) + list(codebook.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)

    # Training loop
    rng = random.Random(0)
    for epoch in range(args.epochs):
        rng.shuffle(pairs)
        for step in range(args.steps_per_epoch):
            # Sample QA mini-batch
            batch = pairs[(step * args.batch_size) % len(pairs) : ((step + 1) * args.batch_size) % len(pairs)]
            if len(batch) < 1:
                batch = pairs[: args.batch_size]

            images, token_ids_cpu, keys = gather_token_batch(batch, vocab, db, args.glyph_size)
            images = images.to(args.device)
            token_ids = token_ids_cpu.to(args.device)

            # Stage 1: glyph ↔ code InfoNCE
            g = glyph_cnn(images)  # (U,d)
            e = codebook.token_embedding(token_ids)  # (U,d)
            g = F.normalize(g, dim=-1)
            e = F.normalize(e, dim=-1)
            logits = g @ e.T  # (U,U)
            targets = torch.arange(logits.size(0), device=logits.device)
            loss_img = F.cross_entropy(logits / 0.07, targets)

            # Stage 2: QA InfoNCE + Gram
            q_embs = []
            a_embs = []
            gram_losses = []
            for pair in batch:
                q_ids = [vocab.get(f"{pair.lang}::{t}") for t in pair.q_tokens]
                a_ids = [vocab.get(f"{pair.lang}::{t}") for t in pair.a_tokens]
                q_ids = [i for i in q_ids if i is not None]
                a_ids = [i for i in a_ids if i is not None]
                if not q_ids or not a_ids:
                    continue
                q_e = codebook.token_embedding(torch.tensor(q_ids, device=args.device))
                a_e = codebook.token_embedding(torch.tensor(a_ids, device=args.device))
                q_embs.append(q_e.mean(dim=0))
                a_embs.append(a_e.mean(dim=0))

                # Gram alignment on normalized token embeddings
                qn = F.normalize(q_e, dim=-1)
                an = F.normalize(a_e, dim=-1)
                Gq = gram_matrix(qn)
                Ga = gram_matrix(an)
                m = min(Gq.size(0), Ga.size(0))
                gram_losses.append(F.mse_loss(Gq[:m, :m], Ga[:m, :m]))

            if not q_embs:
                # fallback to only image-code loss
                q_embs = [e.mean(dim=0)]
                a_embs = [g.mean(dim=0)]
                gram_losses.append(torch.zeros(1, device=args.device).mean())

            Q = F.normalize(torch.stack(q_embs, dim=0), dim=-1)
            A = F.normalize(torch.stack(a_embs, dim=0), dim=-1)
            qa_logits = Q @ A.T
            qa_targets = torch.arange(qa_logits.size(0), device=qa_logits.device)
            loss_qa = F.cross_entropy(qa_logits / 0.07, qa_targets)
            loss_gram = torch.stack(gram_losses).mean()

            # Regularization: encourage assignment entropy (diversity)
            loss_reg = -0.01 * codebook.usage_entropy()

            loss = loss_img + loss_qa + 0.1 * loss_gram + loss_reg

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            if (step + 1) % 20 == 0:
                print(
                    f"epoch {epoch+1} step {step+1}: loss={loss.item():.4f} img={loss_img.item():.4f} qa={loss_qa.item():.4f} gram={loss_gram.item():.4f}"
                )

        # Save checkpoint each epoch
        ckpt = {
            "glyph_cnn": glyph_cnn.state_dict(),
            "codebook": codebook.state_dict(),
            "vocab": vocab,
        }
        torch.save(ckpt, out_dir / f"ckpt_epoch{epoch+1}.pt")

    # Export token code indices (hard) for memory table mapping
    inv_vocab = [None] * len(vocab)
    for k, idx in vocab.items():
        inv_vocab[idx] = k
    all_ids = torch.arange(len(vocab), dtype=torch.long, device=args.device)
    hard_codes = codebook.token_codes_hard(all_ids).cpu().numpy()  # (V,C)
    with open(out_dir / "token_codes.tsv", "w", encoding="utf-8") as f:
        f.write("token\t" + "\t".join([f"c{i}" for i in range(codebook.cfg.n_channels)]) + "\n")
        for i, key in enumerate(inv_vocab):
            f.write(f"{key}\t" + "\t".join(map(str, hard_codes[i].tolist())) + "\n")

    print(f"Training complete. Artifacts saved in {out_dir}")


if __name__ == "__main__":
    main()
