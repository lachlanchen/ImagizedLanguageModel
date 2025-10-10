#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from ilm.data.alpaca_pairs import AlpacaPairs, QAPair
    from ilm.encoders.glyph_cnn import GlyphCNN
    from ilm.code.product import ProductCode
    from scripts.train_codes_from_qa import collate_render, sentence_embed
except ModuleNotFoundError:
    import sys
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from ilm.data.alpaca_pairs import AlpacaPairs, QAPair
    from ilm.encoders.glyph_cnn import GlyphCNN
    from ilm.code.product import ProductCode
    from scripts.train_codes_from_qa import collate_render, sentence_embed


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def encode_side(ds: AlpacaPairs, glyph_cnn: GlyphCNN, code: ProductCode, side: str, batch_size: int, image_size: int, device: str) -> Tuple[np.ndarray, List[str]]:
    assert side in ("q", "a")
    # Convert to thin dataset returning only the selected side
    pairs = list(ds.pairs)
    subset = type("_subset", (), {"__len__": lambda self: len(pairs), "__getitem__": lambda self, i: pairs[i]})()
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=lambda b: collate_render(b, image_size=image_size))
    embs: List[np.ndarray] = []
    texts: List[str] = []
    with torch.no_grad():
        glyph_cnn.eval(); code.eval()
        for batch in loader:
            X = batch["X"].to(device)
            pairs_b = batch["pairs"]
            G = glyph_cnn(X)
            E = code(G, tau=0.2)["embed"]
            S: List[torch.Tensor] = []
            for q_idx, a_idx, lang in pairs_b:
                idx = q_idx if side == "q" else a_idx
                _, e = sentence_embed(idx, G, E)
                S.append(F.normalize(e, dim=-1))
            S = torch.stack(S, dim=0)
            embs.append(S.detach().cpu().numpy().astype(np.float32))
            # store joined text just for ids
            for q_idx, a_idx, lang in pairs_b:
                toks = pairs[pairs_b.index((q_idx, a_idx, lang))].q_tokens if side == "q" else pairs[pairs_b.index((q_idx, a_idx, lang))].a_tokens
                texts.append(" ".join(toks[:16]))
    Eall = np.concatenate(embs, axis=0)
    return Eall, texts


def main():
    ap = argparse.ArgumentParser(description="Evaluate QA retrieval (Q->A) with the trained color-code encoder")
    ap.add_argument("--en-json", default="data/raw/alpaca_en.json")
    ap.add_argument("--zh-json", default="data/raw/alpaca_zh.json")
    ap.add_argument("--checkpoint", default="artifacts/color_codes_qa.pt")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-root", default="artifacts/qa_metrics")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    en_ds = AlpacaPairs(args.en_json, default_lang="en", max_len=64)
    zh_ds = AlpacaPairs(args.zh_json, default_lang="zh", max_len=64)
    # combine
    class _C: pass
    C = _C()
    C.pairs = en_ds.pairs + zh_ds.pairs

    glyph_cnn = GlyphCNN(d=128, in_channels=3).to(device)
    code = ProductCode(d_in=128, d=128, K=32, C=3, tau=0.2, straight_through=False).to(device)
    if Path(args.checkpoint).exists():
        ck = torch.load(args.checkpoint, map_location=device)
        glyph_cnn.load_state_dict(ck["glyph_cnn"])  # type: ignore[index]
        code.load_state_dict(ck["code"])            # type: ignore[index]

    # Encode Q and A
    Q, q_texts = encode_side(C, glyph_cnn, code, side="q", batch_size=args.batch_size, image_size=args.image_size, device=device)
    A, a_texts = encode_side(C, glyph_cnn, code, side="a", batch_size=args.batch_size, image_size=args.image_size, device=device)

    # Retrieval
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    sims = Q @ A.T
    order = np.argsort(-sims, axis=1)
    ranks = np.empty(Q.shape[0], dtype=np.int32)
    for i in range(Q.shape[0]):
        # ground truth answer is same index
        ranks[i] = int(np.where(order[i] == i)[0][0]) + 1
    def recall_at(k):
        return float(np.mean(ranks <= k))

    metrics = {"R@1": recall_at(1), "R@5": recall_at(5), "R@10": recall_at(10), "N": int(Q.shape[0])}

    out_dir = Path(args.out_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"metrics_dir": str(out_dir), **metrics}))


if __name__ == "__main__":
    main()

