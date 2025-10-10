#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from ilm.data.alpaca_pairs import AlpacaPairs, QAPair
    from ilm.encoders.glyph_cnn import GlyphCNN
    from ilm.code.product import ProductCode
    from ilm.utils.glyphs import make_rgb_token_image
except ModuleNotFoundError:
    import sys
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from ilm.data.alpaca_pairs import AlpacaPairs, QAPair
    from ilm.encoders.glyph_cnn import GlyphCNN
    from ilm.code.product import ProductCode
    from ilm.utils.glyphs import make_rgb_token_image


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


def collate_render(batch: List[QAPair], image_size: int = 128) -> Dict:
    # Gather unique tokens across QA in the batch
    uniq: Dict[Tuple[str, str], int] = {}
    all_pairs: List[Tuple[List[int], List[int], str]] = []
    for s in batch:
        q_idx: List[int] = []
        a_idx: List[int] = []
        for t in s.q_tokens:
            key = (s.lang, t)
            if key not in uniq:
                uniq[key] = len(uniq)
            q_idx.append(uniq[key])
        for t in s.a_tokens:
            key = (s.lang, t)
            if key not in uniq:
                uniq[key] = len(uniq)
            a_idx.append(uniq[key])
        all_pairs.append((q_idx, a_idx, s.lang))

    # Render images for uniq tokens
    keys = list(uniq.keys())
    imgs = []
    for lang, tok in keys:
        rgb = make_rgb_token_image(lang, tok, size=image_size)
        x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        imgs.append(x)
    X = torch.stack(imgs, dim=0)  # (U,C,H,W)
    return {"X": X, "keys": keys, "pairs": all_pairs}


def sentence_embed(indexes: List[int], token_feats: torch.Tensor, token_codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # token_feats: (U,d_glyph), token_codes: (U,d_code)
    idx = torch.as_tensor(indexes, dtype=torch.long, device=token_feats.device)
    z_g = token_feats.index_select(0, idx)  # (n,d_glyph)
    z_e = token_codes.index_select(0, idx)  # (n,d_code)
    # mean pool
    g = z_g.mean(dim=0)
    e = z_e.mean(dim=0)
    return g, e


def gram_matrix(Z: torch.Tensor) -> torch.Tensor:
    # Z: (n, d) normalized
    if Z.ndim != 2:
        Z = Z.view(Z.size(0), -1)
    Z = F.normalize(Z, dim=-1)
    C = Z.t() @ Z / (Z.size(0) + 1e-6)  # (d,d)
    return C


def main():
    ap = argparse.ArgumentParser(description="Train color codes with QA (Alpaca) supervision and autocorrelation alignment")
    ap.add_argument("--en-json", default="data/raw/alpaca_en.json")
    ap.add_argument("--zh-json", default="data/raw/alpaca_zh.json")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.5, help="Gumbel-Softmax temp for codes during QA training")
    ap.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature")
    ap.add_argument("--w-info", type=float, default=1.0)
    ap.add_argument("--w-auto", type=float, default=0.1, help="autocorrelation (Gram) alignment weight")
    ap.add_argument("--w-usage", type=float, default=0.05)
    ap.add_argument("--w-indep", type=float, default=0.05)
    ap.add_argument("--resume", default=None, help="optional color_codes checkpoint to resume from")
    args = ap.parse_args()

    set_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    # Data
    en_ds = AlpacaPairs(args.en_json, default_lang="en", max_len=args.max_len)
    zh_ds = AlpacaPairs(args.zh_json, default_lang="zh", max_len=args.max_len)
    # mix datasets
    full = list(en_ds.pairs) + list(zh_ds.pairs)
    ds = type("_DS", (), {"__len__": lambda self: len(full), "__getitem__": lambda self, i: full[i]})()
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=lambda b: collate_render(b, image_size=args.image_size))

    # Model
    d_glyph = 128
    d_code = 128
    K = 32
    C = 3
    glyph_cnn = GlyphCNN(d=d_glyph, in_channels=3).to(device)
    code = ProductCode(d_in=d_glyph, d=d_code, K=K, C=C, tau=args.tau, straight_through=True).to(device)
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        glyph_cnn.load_state_dict(ckpt["glyph_cnn"])  # type: ignore[index]
        code.load_state_dict(ckpt["code"])            # type: ignore[index]

    opt = optim.AdamW(list(glyph_cnn.parameters()) + list(code.parameters()), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    step = 0
    for epoch in range(1, args.epochs + 1):
        glyph_cnn.train(); code.train()
        for batch in loader:
            X = batch["X"].to(device)
            keys = batch["keys"]
            pairs = batch["pairs"]

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                G = glyph_cnn(X)                # (U,d_glyph)
                out = code(G, tau=args.tau)
                E = out["embed"]               # (U,d_code)
                Y = out["y"]                   # (U,C,K)

                # Build sentence embeddings and losses
                Qe: List[torch.Tensor] = []
                Ae: List[torch.Tensor] = []
                auto_losses: List[torch.Tensor] = []
                for q_idx, a_idx, lang in pairs:
                    gq, eq = sentence_embed(q_idx, G, E)
                    ga, ea = sentence_embed(a_idx, G, E)
                    Qe.append(eq)
                    Ae.append(ea)
                    # autocorrelation (Gram) alignment between token-level code embeddings
                    Zq = F.normalize(E.index_select(0, torch.as_tensor(q_idx, device=E.device)), dim=-1)
                    Za = F.normalize(E.index_select(0, torch.as_tensor(a_idx, device=E.device)), dim=-1)
                    Cq = gram_matrix(Zq)
                    Ca = gram_matrix(Za)
                    auto_losses.append(F.mse_loss(Cq, Ca))

                Q = F.normalize(torch.stack(Qe, dim=0), dim=-1)
                A = F.normalize(torch.stack(Ae, dim=0), dim=-1)
                logits = (Q @ A.t()) / args.temperature
                targets = torch.arange(Q.size(0), device=Q.device)
                L_info = F.cross_entropy(logits, targets)
                L_auto = torch.stack(auto_losses, dim=0).mean() if auto_losses else torch.tensor(0.0, device=Q.device)

                # usage/indep over all tokens in batch
                # reshape Y from (U,C,K) but usage_kl expects (B,C,K); use as batch
                L_usage = code.usage_kl(Y)
                L_indep = code.independence(Y)

                total = args.w_info * L_info + args.w_auto * L_auto + args.w_usage * L_usage + args.w_indep * L_indep

            scaler.scale(total).backward()
            scaler.step(opt)
            scaler.update()

            if step % 20 == 0:
                print({
                    "epoch": epoch,
                    "step": step,
                    "loss_total": float(total.detach().cpu()),
                    "L_info": float(L_info.detach().cpu()),
                    "L_auto": float(L_auto.detach().cpu()),
                    "L_usage": float(L_usage.detach().cpu()),
                    "L_indep": float(L_indep.detach().cpu()),
                    "U_tokens": int(X.size(0)),
                    "pairs": len(pairs),
                })
            step += 1

    # Save checkpoint
    out = Path("artifacts")
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "color_codes_qa.pt"
    torch.save({
        "glyph_cnn": glyph_cnn.state_dict(),
        "code": code.state_dict(),
        "cfg": {"qa": True, "epochs": args.epochs, "tau": args.tau, "temperature": args.temperature},
    }, ckpt_path)
    print(json.dumps({"saved": str(ckpt_path)}))


if __name__ == "__main__":
    main()

