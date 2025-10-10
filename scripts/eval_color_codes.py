#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from ilm.data.loader import ImageIndexDataset
    from ilm.encoders.glyph_cnn import GlyphCNN
    from ilm.code.product import ProductCode
except ModuleNotFoundError:
    import sys
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from ilm.data.loader import ImageIndexDataset
    from ilm.encoders.glyph_cnn import GlyphCNN
    from ilm.code.product import ProductCode


def load_checkpoint(path: str, device: str) -> Tuple[Dict, Dict]:
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("cfg", {})
    return ckpt, cfg


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def eval_metrics(index_path: str, ckpt_path: str, out_root: str,
                 batch_size: int = 256, image_size: int | None = 128,
                 device: str = "cuda", auto_generate_missing: bool = False,
                 pairs_path: str | None = None):
    # Output directory with datetime suffix
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_root) / ts
    ensure_dir(out_dir)

    # Load checkpoint and config
    ckpt, cfg = load_checkpoint(ckpt_path, device)
    d_glyph = cfg.get("model", {}).get("d_glyph", 128)
    d_code = cfg.get("model", {}).get("d_code", 128)
    K = cfg.get("model", {}).get("K", 32)
    C = cfg.get("model", {}).get("C", 3)

    # Build dataset/loader
    ds = ImageIndexDataset(index_path=index_path, image_size=image_size,
                           auto_generate_missing=auto_generate_missing)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Instantiate models
    glyph_cnn = GlyphCNN(d=d_glyph, in_channels=3).to(device)
    code = ProductCode(d_in=d_glyph, d=d_code, K=K, C=C, tau=0.2, straight_through=False).to(device)
    glyph_cnn.load_state_dict(ckpt["glyph_cnn"])  # type: ignore[index]
    code.load_state_dict(ckpt["code"])            # type: ignore[index]
    glyph_cnn.eval()
    code.eval()

    # Accumulators
    N = 0
    sum_y = [torch.zeros(K, device=device) for _ in range(C)]
    sum_outer = [[torch.zeros(K, K, device=device) for _ in range(C)] for __ in range(C)]
    embeds = []        # code embeddings
    glyph_feats = []   # glyph cnn outputs
    langs: List[str] = []
    tokens: List[str] = []
    paths: List[str] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            g = glyph_cnn(x)              # (B×d_glyph)
            out = code(g, tau=0.2)
            y = out["y"]                  # (B×C×K)
            e = out["embed"]             # (B×d_code)

            B = x.size(0)
            N += B

            # Sums per channel
            for c in range(C):
                yc = y[:, c, :]  # (B×K)
                sum_y[c] += yc.sum(dim=0)
            # Cross outer products for independence
            for a in range(C):
                Ya = y[:, a, :]
                for b in range(a + 1, C):
                    Yb = y[:, b, :]
                    sum_outer[a][b] += Ya.t() @ Yb

            embeds.append(e.detach().cpu())
            glyph_feats.append(g.detach().cpu())
            langs.extend(batch["lang"])  # type: ignore[arg-type]
            tokens.extend(batch["token"])  # type: ignore[arg-type]
            paths.extend(batch["path"])  # type: ignore[arg-type]

    embeds = torch.cat(embeds, dim=0).numpy().astype(np.float16)
    glyph_feats = torch.cat(glyph_feats, dim=0).numpy().astype(np.float16)

    # Usage and entropy
    usage = []
    entropies = []
    for c in range(C):
        p = (sum_y[c] / N).clamp_min(1e-9)
        usage.append(p.detach().cpu().numpy())
        H = - (p * p.log()).sum().item()
        entropies.append(H)

    # Independence metric
    indep_vals = []
    means = [sum_y[c] / N for c in range(C)]
    for a in range(C):
        for b in range(a + 1, C):
            S = sum_outer[a][b] / N
            cov = S - torch.ger(means[a], means[b])
            indep_vals.append(float((cov.pow(2).sum()).item()))
    indep_mean = float(np.mean(indep_vals)) if indep_vals else 0.0

    # Save arrays and ids
    np.save(out_dir / "codes.npy", embeds)
    np.save(out_dir / "glyphs.npy", glyph_feats)
    with open(out_dir / "ids.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["idx", "lang", "token", "path"])  # header
        for i, (la, tok, pa) in enumerate(zip(langs, tokens, paths)):
            w.writerow([i, la, tok, pa])

    # Optional: cross-lingual retrieval metrics
    retrieval = None
    if pairs_path and os.path.exists(pairs_path):
        retrieval = compute_retrieval(out_dir, pairs_path)

    # Summarize
    summary = {
        "dataset_size": N,
        "channels": C,
        "K": K,
        "entropy": entropies,
        "independence_mean": indep_mean,
        "retrieval": retrieval,
        "checkpoint": ckpt_path,
        "index": index_path,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save usage CSV
    with open(out_dir / "usage.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "bin", "prob"])
        for c in range(C):
            for k in range(K):
                w.writerow([c, k, float(usage[c][k])])

    # Save a few NN examples
    save_nn_examples(out_dir)

    print(json.dumps({"metrics_dir": str(out_dir)}))


def compute_retrieval(out_dir: Path, pairs_path: str) -> Dict[str, float]:
    codes = np.load(out_dir / "codes.npy").astype(np.float32)
    ids = []
    with open(out_dir / "ids.tsv", "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            ids.append(row)
    # Indices by token/lang
    idx_by_tok = {}
    for row in ids:
        idx_by_tok.setdefault((row["lang"], row["token"]), int(row["idx"]))
    # Read pairs
    pairs = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        pr = csv.DictReader(f, delimiter="\t")
        for row in pr:
            if "en" in row and "zh" in row:
                pairs.append((row["en"], row["zh"]))
    # Build matrices
    en_idx = [idx_by_tok.get(("en", t)) for t, _ in pairs]
    zh_idx = [idx_by_tok.get(("zh", t)) for _, t in pairs]
    # Filter missing
    keep = [i for i, (e, z) in enumerate(zip(en_idx, zh_idx)) if e is not None and z is not None]
    en_idx = [en_idx[i] for i in keep]
    zh_idx = [zh_idx[i] for i in keep]
    if len(en_idx) == 0:
        return {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "pairs_used": 0}
    E = codes[en_idx]
    Z = codes
    # restrict Z to zh only
    zh_rows = [int(row["idx"]) for row in ids if row["lang"] == "zh"]
    Z = codes[zh_rows]
    # normalize
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    sims = E @ Z.T
    # compute ranks
    recall = {1: 0, 5: 0, 10: 0}
    for qi, eidx in enumerate(en_idx):
        # the correct zh index among zh_rows
        true_zh_global = zh_idx[qi]
        try:
            true_pos = zh_rows.index(true_zh_global)
        except ValueError:
            continue
        order = np.argsort(-sims[qi])
        rank = int(np.where(order == true_pos)[0]) + 1
        for k in recall.keys():
            if rank <= k:
                recall[k] += 1
    total = len(en_idx)
    res = {f"R@{k}": float(recall[k] / total) for k in recall}
    res["pairs_used"] = int(total)
    return res


def save_nn_examples(out_dir: Path, n_samples: int = 10, k: int = 10):
    codes = np.load(out_dir / "codes.npy").astype(np.float32)
    ids = []
    with open(out_dir / "ids.tsv", "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            ids.append(row)
    # normalize
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-9)
    n = len(ids)
    rng = np.random.default_rng(0)
    samples = rng.choice(n, size=min(n_samples, n), replace=False)
    with open(out_dir / "nn_examples.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["query_idx", "query_lang", "query_token", "nn_rank", "nn_idx", "nn_lang", "nn_token", "sim"])
        for qi in samples:
            sims = codes[qi:qi+1] @ codes.T
            order = np.argsort(-sims[0])
            for r, j in enumerate(order[:k]):
                w.writerow([
                    qi, ids[qi]["lang"], ids[qi]["token"],
                    r + 1, j, ids[j]["lang"], ids[j]["token"], float(sims[0, j])
                ])


def main():
    ap = argparse.ArgumentParser(description="Evaluate product color codes and generate metrics with timestamped folder")
    ap.add_argument("--index", default="data/processed/images_common_freq/index.tsv")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-root", default="artifacts/metrics")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--auto-generate-missing", action="store_true")
    ap.add_argument("--pairs", default=None, help="Optional bilingual pairs TSV with columns: en\tzh")
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    eval_metrics(
        index_path=args.index,
        ckpt_path=args.checkpoint,
        out_root=args.out_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=device,
        auto_generate_missing=args.auto_generate_missing,
        pairs_path=args.pairs,
    )


if __name__ == "__main__":
    main()

