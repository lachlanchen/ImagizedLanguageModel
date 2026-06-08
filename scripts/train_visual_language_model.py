#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ilm.visual_lm import GlyphCorpus, ImageToImageUNet, RenderConfig, VisualLanguageDataset
from ilm.visual_lm.dataset import tensor_to_pil
from ilm.visual_lm.model import image_gradient_loss, psnr_from_l1
from ilm.visual_lm.rendering import make_triptych


DEFAULT_CHARS = "言,中,水,日,月,人,山,火,木,口,学"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train ILM-V image-to-image page model.")
    ap.add_argument("--glyph-root", default=None, help="Historic glyph root; defaults to local incoder data if present.")
    ap.add_argument("--characters", default=DEFAULT_CHARS, help="Comma-separated characters, or 'auto'.")
    ap.add_argument("--max-characters", type=int, default=128, help="Used when --characters auto.")
    ap.add_argument("--out", default="artifacts/visual_lm", help="Output directory for checkpoints and samples.")
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--samples-per-epoch", type=int, default=512)
    ap.add_argument("--val-samples", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-weight", type=float, default=0.25)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--limit-steps", type=int, default=None, help="Optional cap for quick validation runs.")
    return ap.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_characters(args: argparse.Namespace, corpus: GlyphCorpus) -> list[str]:
    if args.characters.strip().lower() == "auto":
        return corpus.discover_characters(max_chars=args.max_characters)
    chars = [c.strip() for c in args.characters.split(",") if c.strip()]
    return [c for c in chars if (corpus.root / c).exists()] or corpus.characters[: args.max_characters]


@torch.no_grad()
def evaluate(
    model: ImageToImageUNet,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    epoch: int,
) -> dict[str, float]:
    model.eval()
    total_l1 = 0.0
    total_psnr = 0.0
    n = 0
    saved = False
    for batch in loader:
        prompt = batch["prompt"].to(device)
        target = batch["target"].to(device)
        pred = model(prompt)
        total_l1 += float(F.l1_loss(pred, target).detach().cpu()) * prompt.size(0)
        total_psnr += psnr_from_l1(pred, target) * prompt.size(0)
        n += prompt.size(0)
        if not saved:
            sample_dir = out_dir / "samples"
            sample_dir.mkdir(parents=True, exist_ok=True)
            trip = make_triptych(tensor_to_pil(prompt[0]), tensor_to_pil(pred[0]), tensor_to_pil(target[0]))
            trip.save(sample_dir / f"epoch{epoch:04d}.png")
            saved = True
    return {"val_l1": total_l1 / max(1, n), "val_psnr": total_psnr / max(1, n)}


def save_checkpoint(
    path: Path,
    model: ImageToImageUNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    characters: list[str],
    metrics: dict[str, Any],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "characters": characters,
            "args": vars(args),
            "metrics": metrics,
            "model_config": {
                "in_ch": 3,
                "out_ch": 3,
                "base_ch": args.base_ch,
                "depth": args.depth,
            },
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_corpus = GlyphCorpus(args.glyph_root)
    characters = load_characters(args, seed_corpus)
    corpus = GlyphCorpus(args.glyph_root, characters=characters)
    args.glyph_root = str(corpus.root)
    cfg = RenderConfig(image_size=args.image_size)
    train_ds = VisualLanguageDataset(corpus, render_config=cfg, length=args.samples_per_epoch, seed=args.seed, characters=characters)
    val_ds = VisualLanguageDataset(corpus, render_config=cfg, length=args.val_samples, seed=args.seed + 10_000, characters=characters)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ImageToImageUNet(base_ch=args.base_ch, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0))

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "characters": characters, "glyph_root": str(corpus.root)}, f, ensure_ascii=False, indent=2)

    print(json.dumps({"device": str(device), "characters": characters, "glyph_root": str(corpus.root)}, ensure_ascii=False))
    global_step = 0
    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        model.train()
        losses: list[float] = []
        for step, batch in enumerate(train_dl, 1):
            prompt = batch["prompt"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                pred = model(prompt)
                loss_l1 = F.l1_loss(pred, target)
                loss_grad = image_gradient_loss(pred, target)
                loss = loss_l1 + args.grad_weight * loss_grad
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            if step == 1 or step % 25 == 0:
                print(json.dumps({"epoch": epoch, "step": step, "loss": losses[-1], "l1": float(loss_l1.detach().cpu())}))
            global_step += 1
            if args.limit_steps is not None and step >= args.limit_steps:
                break
        metrics = evaluate(model, val_dl, device, out_dir, epoch)
        metrics["train_loss"] = float(np.mean(losses)) if losses else 0.0
        print(json.dumps({"epoch": epoch, **metrics}))
        if epoch % max(1, args.save_every) == 0:
            save_checkpoint(out_dir / f"ckpt_epoch{epoch}.pt", model, optimizer, epoch, args, characters, metrics)
        save_checkpoint(out_dir / "ckpt_latest.pt", model, optimizer, epoch, args, characters, metrics)
    print(json.dumps({"saved": str(out_dir / "ckpt_latest.pt"), "steps": global_step}))


if __name__ == "__main__":
    main()
