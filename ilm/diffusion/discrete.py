from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_mask(shape, mask_ratio: float, device: str | torch.device = "cpu") -> torch.Tensor:
    """Bernoulli mask with probability=mask_ratio for masking (1 means masked)."""
    return (torch.rand(shape, device=device) < mask_ratio).to(torch.float32)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE only on masked positions. pred/target: (B,C,H,W), mask: (B,1,H,W) with 1=masked."""
    diff = (pred - target) ** 2
    if mask.shape[1] != 1:
        mask = mask[:, :1]
    w = mask
    denom = w.sum().clamp_min(1.0)
    return (diff * w).sum() / denom


def corruption(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply mask by zeroing out masked positions in x."""
    if mask.shape[1] != 1:
        mask = mask[:, :1]
    return x * (1.0 - mask)

