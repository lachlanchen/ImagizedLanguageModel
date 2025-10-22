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
    id_to_lang_from_vocab,
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
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--n-channels", type=int, default=3)
    ap.add_argument("--n-codes", type=int, default=32)
    ap.add_argument("--glyph-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-every", type=int, default=50, help="Print logs every N steps with accuracy metrics")
    # Optional regularizers
    ap.add_argument("--lambda-uniq", type=float, default=0.0, help="Uniqueness regularizer weight (L_uniq)")
    ap.add_argument("--lambda-adv", type=float, default=0.0, help="Adversarial language invariance weight for early channels")
    ap.add_argument("--adv-early-channels", type=int, default=-1, help="How many early channels to use for adversary; -1 means C-1")
    ap.add_argument("--strict-codes", action="store_true", help="Abort if vocab size exceeds K^C capacity for unique hard codes")
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
    vocab, pairs, id_to_lang = build_vocab_and_cache_glyphs(pairs, db, glyph_size=args.glyph_size)
    vocab_size = len(vocab)
    (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    # Capacity check: unique hard-code capacity is K^C
    capacity = (args.n_codes ** args.n_channels)
    if vocab_size > capacity:
        msg = (
            f"WARNING: vocab size {vocab_size} exceeds hard-code capacity K^C = {args.n_codes}^{args.n_channels} = {capacity}.\n"
            "Hard code export (token_codes.tsv) will collide; training can proceed (soft assignments), but uniqueness is not guaranteed.\n"
            "Use more channels or codes, or enable subword/character tokenization to reduce vocab."
        )
        print(msg)
        if args.strict_codes:
            raise SystemExit("Aborting due to --strict-codes.")

    # Models
    # Ensure d_model is divisible by n_channels; round up if needed to avoid assertion.
    if args.d_model % args.n_channels != 0:
        old_dm = args.d_model
        args.d_model = ((args.d_model + args.n_channels - 1) // args.n_channels) * args.n_channels
        print(f"Adjusted d_model from {old_dm} to {args.d_model} to be divisible by n_channels={args.n_channels}")

    glyph_cnn = GlyphCNN(d=args.d_model, in_channels=3).to(args.device)
    code_cfg = ProductCodebookConfig(d_model=args.d_model, n_channels=args.n_channels, n_codes=args.n_codes)
    codebook = ProductCodebook(vocab_size=vocab_size, cfg=code_cfg).to(args.device)

    params_main = list(glyph_cnn.parameters()) + list(codebook.parameters())
    opt = torch.optim.AdamW(params_main, lr=args.lr)

    # Optional adversarial language classifier over early channels
    lang2id = {"en": 0, "zh": 1}
    d_per = code_cfg.d_model // code_cfg.n_channels
    early_channels = (code_cfg.n_channels - 1) if args.adv_early_channels < 0 else max(0, min(code_cfg.n_channels, args.adv_early_channels))
    if args.lambda_adv > 0 and early_channels > 0:
        adv_in = d_per * early_channels
        lang_clf = nn.Sequential(
            nn.Linear(adv_in, max(32, adv_in // 2)),
            nn.SiLU(),
            nn.Linear(max(32, adv_in // 2), len(lang2id)),
        ).to(args.device)
        opt_lang = torch.optim.AdamW(lang_clf.parameters(), lr=args.lr)
    else:
        lang_clf = None
        opt_lang = None

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
            # Retrieval accuracy (image -> code embedding)
            with torch.no_grad():
                pred_ic = logits.argmax(dim=1)
                acc_ic = (pred_ic == targets).float().mean().item()

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
            with torch.no_grad():
                pred_qa = qa_logits.argmax(dim=1)
                acc_qa = (pred_qa == qa_targets).float().mean().item() if qa_logits.size(0) > 0 else 0.0
            loss_gram = torch.stack(gram_losses).mean()

            # Regularization: encourage assignment entropy (diversity)
            loss_reg = -0.01 * codebook.usage_entropy()

            # Optional uniqueness regularizer (L_uniq) over current unique tokens
            loss_uniq = torch.zeros((), device=args.device)
            if args.lambda_uniq > 0 and token_ids.numel() > 1:
                with torch.no_grad():
                    uniq_ids, inv = torch.unique(token_ids, sorted=False, return_inverse=True)
                # Soft assignments for unique ids: (Uu,C,K)
                probs = F.softmax(codebook.assign_logits[uniq_ids], dim=-1)
                # Similarity per pair: product over channels of inner products over K
                # To avoid underflow, use log-domain
                Uu = probs.size(0)
                sim_sum = torch.zeros((), device=args.device)
                pair_count = 0
                for i in range(Uu):
                    pi = probs[i]  # (C,K)
                    for j in range(i + 1, Uu):
                        pj = probs[j]
                        ip = torch.einsum("ck,ck->c", pi, pj).clamp_min(1e-8)
                        log_prod = torch.log(ip).sum()
                        sim = torch.exp(log_prod)
                        sim_sum = sim_sum + sim
                        pair_count += 1
                if pair_count > 0:
                    loss_uniq = args.lambda_uniq * (sim_sum / pair_count)

            # Optional adversarial language invariance over early channels
            loss_lang_main = torch.zeros((), device=args.device)
            if lang_clf is not None:
                # Build a token-level sample with language labels for classifier
                with torch.no_grad():
                    uniq_ids, inv = torch.unique(token_ids, sorted=False, return_inverse=True)
                    langs = [k[0] for k in keys]  # keys are (lang, token) per occurrence order
                    # Map to per-uniq id labels by scanning keys order
                    # Build lang label per uniq id: pick first occurrence
                    id_to_label = {}
                    for (lang, tok), tid in zip(keys, token_ids_cpu.tolist()):
                        if tid not in id_to_label:
                            id_to_label[tid] = lang2id.get(lang, 0)
                    labels = torch.tensor([id_to_label.get(i, 0) for i in uniq_ids.tolist()], device=args.device, dtype=torch.long)
                # Early embedding slice
                emb_full = codebook.token_embedding(uniq_ids.to(args.device))  # (Uu,d)
                if early_channels > 0:
                    emb_early = emb_full[:, : d_per * early_channels]
                else:
                    emb_early = emb_full[:, :0]
                # Update language classifier to minimize CE
                opt_lang.zero_grad()
                logits_lang = lang_clf(emb_early.detach())
                loss_lang_clf = F.cross_entropy(logits_lang, labels)
                loss_lang_clf.backward()
                opt_lang.step()
                # Main model adversarial: maximize CE -> minimize negative CE
                logits_lang_main = lang_clf(emb_early)
                loss_lang_main = -args.lambda_adv * F.cross_entropy(logits_lang_main, labels)

            loss = loss_img + loss_qa + 0.1 * loss_gram + loss_reg + loss_uniq + loss_lang_main

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_main, 1.0)
            opt.step()

            if (step + 1) % args.log_every == 0:
                ent = codebook.usage_entropy().item()
                print(
                    f"epoch {epoch+1} step {step+1}: loss={loss.item():.4f} img={loss_img.item():.4f} qa={loss_qa.item():.4f} gram={loss_gram.item():.4f} uniq={loss_uniq.item():.4f} langAdv={loss_lang_main.item():.4f} acc_ic@1={acc_ic:.3f} acc_qa@1={acc_qa:.3f} entropy={ent:.2f}"
                )

        # Save checkpoint each epoch
        ckpt = {
            "glyph_cnn": glyph_cnn.state_dict(),
            "codebook": codebook.state_dict(),
            "vocab": vocab,
            "lang_clf": (lang_clf.state_dict() if lang_clf is not None else None),
            "config": {
                "d_model": args.d_model,
                "n_channels": args.n_channels,
                "n_codes": args.n_codes,
                "lambda_uniq": args.lambda_uniq,
                "lambda_adv": args.lambda_adv,
                "adv_early_channels": early_channels,
            },
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
