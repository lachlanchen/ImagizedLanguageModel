#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ilm.datasets.alpaca_glyph_dataset import iter_alpaca_qa, special_token_ids
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


class AnswerFrames(Dataset):
    def __init__(self, en_path: str | None, zh_path: str | None, codebook: ProductCodebook, grid: int = 16):
        self.items: List[tuple[str, List[str]]] = []
        if en_path:
            for qa in iter_alpaca_qa(en_path, "en"):
                self.items.append((qa.lang, qa.a_tokens))
        if zh_path:
            for qa in iter_alpaca_qa(zh_path, "zh"):
                self.items.append((qa.lang, qa.a_tokens))
        self.vocab = None
        self.codebook = codebook
        self.grid = grid

    def set_vocab(self, vocab: dict):
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        lang, toks = self.items[idx]
        # Compose [BOS] + tokens + [EOS], pad rest with [PAD]
        bos, eos, pad = special_token_ids(self.vocab, lang)
        seq: list[int] = []
        if bos is not None:
            seq.append(bos)
        for t in toks:
            key = f"{lang}::{t}"
            if key in self.vocab:
                seq.append(self.vocab[key])
        if eos is not None:
            seq.append(eos)
        # Map into frame and pad remainder with PAD (or 0 if missing)
        frame = np.full((self.grid, self.grid), fill_value=-1, dtype=np.int64)
        T = min(len(seq), self.grid * self.grid)
        for i in range(T):
            r = i // self.grid
            c = i % self.grid
            frame[r, c] = seq[i]
        # Replace -1 with PAD id if available
        pad_id = pad if pad is not None else 0
        frame[frame < 0] = pad_id
        frame_ids = frame
        valid = (frame_ids != pad_id)
        H = W = self.grid
        with torch.no_grad():
            flat = frame_ids.reshape(-1)
            flat_ids = torch.tensor([i if i >= 0 else 0 for i in flat], dtype=torch.long)
            emb = self.codebook.token_embedding(flat_ids)  # (G*G,d)
            d = emb.size(1)
            y = emb.reshape(H, W, d).permute(2, 0, 1).contiguous()  # (d,H,W)
        return y, torch.tensor(valid.reshape(H, W), dtype=torch.bool), torch.tensor(frame_ids.reshape(H, W), dtype=torch.long)


def random_mask(valid: torch.Tensor, min_ratio: float = 0.2, max_ratio: float = 0.5) -> torch.Tensor:
    H, W = valid.shape
    mask = torch.zeros_like(valid, dtype=torch.float32)
    vpos = valid.nonzero(as_tuple=False)
    if vpos.numel() == 0:
        return mask.unsqueeze(0)
    ratio = random.uniform(min_ratio, max_ratio)
    k = max(1, int(vpos.size(0) * ratio))
    idx = torch.randperm(vpos.size(0))[:k]
    sel = vpos[idx]
    mask[sel[:, 0], sel[:, 1]] = 1.0
    return mask.unsqueeze(0)  # (1,H,W)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train 2D UNet masked inpainting on code frames")
    ap.add_argument("--ckpt-code", required=True, help="Trained codebook checkpoint (ckpt_epochX.pt)")
    ap.add_argument("--en", default="data/raw/alpaca_en.json")
    ap.add_argument("--zh", default="data/raw/alpaca_zh.json")
    ap.add_argument("--out", default="artifacts/inpaint")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--r-channels", type=int, default=16)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Resume
    ap.add_argument("--resume-from", default=None, help="Path to inpaint ckpt to resume from")
    ap.add_argument("--resume-auto", action="store_true", help="Auto-resume from latest ckpt in --out directory")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    ds = AnswerFrames(args.en, args.zh, codebook=codebook, grid=args.grid)
    ds.set_vocab(vocab)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    net = InpaintNet(d_model=d_model, r=args.r_channels).to(args.device)
    # Optional resume
    start_epoch = 0
    def latest_ckpt(root: Path) -> Path | None:
        best = None
        best_n = -1
        for p in root.glob("ckpt_epoch*.pt"):
            try:
                n = int(p.stem.split("epoch")[-1])
            except Exception:
                continue
            if n > best_n:
                best_n = n
                best = p
        return best
    if args.resume_auto:
        last = latest_ckpt(out_dir)
        if last is not None:
            args.resume_from = str(last)
    if args.resume_from:
        ip_ckpt = torch.load(args.resume_from, map_location="cpu")
        r_prev = ip_ckpt.get("r_channels", args.r_channels)
        if r_prev != args.r_channels:
            args.r_channels = r_prev
            net = InpaintNet(d_model=d_model, r=r_prev).to(args.device)
        net.load_state_dict(ip_ckpt["net"])  # type: ignore
        try:
            start_epoch = int(Path(args.resume_from).stem.split("epoch")[-1])
        except Exception:
            start_epoch = 0
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)

    # Precompute normalized vocab embeddings once per epoch
    V = len(vocab)
    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        for step, (y, valid, ids_frame) in enumerate(dl, 1):
            y = y.to(args.device)  # (B,d,H,W)
            ids_frame = ids_frame.to(args.device)  # (B,H,W)
            mask = torch.stack([random_mask(v) for v in valid], dim=0).to(args.device)  # (B,1,H,W)
            y_hat, y_r_hat, y_r = net(y, mask)
            # L2 on masked positions only
            m = mask
            loss = ((y_hat - y) ** 2 * m).sum() / (m.sum() + 1e-6)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            # Periodic masked top-1 token accuracy
            acc_msg = ""
            if step % args.log_every == 0:
                with torch.no_grad():
                    y_pred = y * (1.0 - m) + y_hat * m
                    # Nearest-neighbor to vocab
                    all_ids = torch.arange(V, dtype=torch.long, device=args.device)
                    all_emb = F.normalize(codebook.token_embedding(all_ids), dim=-1)  # (V,d)
                    Bh, d, H, W = y_pred.shape
                    y_seq = y_pred.permute(0, 2, 3, 1).reshape(Bh * H * W, d)
                    y_seq = F.normalize(y_seq, dim=-1)
                    sims = y_seq @ all_emb.T
                    top = sims.argmax(dim=1).reshape(Bh, H, W)
                    masked_pos = (m.squeeze(1) > 0.5)
                    correct = (top == ids_frame) & masked_pos
                    denom = masked_pos.sum().clamp_min(1)
                    acc = correct.sum().float() / denom.float()
                    acc_msg = f" acc_masked_top1={acc.item():.3f}"
                print(f"epoch {epoch} step {step}: loss={loss.item():.4f}{acc_msg}")
        torch.save({
            "net": net.state_dict(),
            "d_model": d_model,
            "r_channels": args.r_channels,
            "grid": args.grid,
        }, out_dir / f"ckpt_epoch{epoch}.pt")
    print(f"Training complete. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
