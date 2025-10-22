from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProductCodebookConfig:
    d_model: int = 128  # embedding dimension
    n_channels: int = 4  # number of codebook channels
    n_codes: int = 64  # codes per channel


class ProductCodebook(nn.Module):
    """Multi-channel codebook with per-token soft assignments (Gumbel-Softmax).

    - Code embeddings: C channels × K codes × (d_model/C) each channel summed across channels.
    - Token assignments: per-token logits [V × C × K].
    """

    def __init__(self, vocab_size: int, cfg: ProductCodebookConfig):
        super().__init__()
        self.vocab_size = vocab_size
        self.cfg = cfg
        d_per = cfg.d_model // cfg.n_channels
        assert cfg.d_model % cfg.n_channels == 0, "d_model must be divisible by n_channels"

        # Codebook embeddings per channel
        self.codebooks = nn.Parameter(
            torch.randn(cfg.n_channels, cfg.n_codes, d_per) * (1.0 / math.sqrt(d_per))
        )
        # Token logits per channel/code
        self.assign_logits = nn.Parameter(torch.zeros(vocab_size, cfg.n_channels, cfg.n_codes))

    def token_embedding(
        self, token_ids: torch.Tensor, *, tau: float = 1.0, hard: bool = False
    ) -> torch.Tensor:
        """Compute token embeddings via soft (or hard) code selection.

        token_ids: (B,) longs
        returns: (B, d_model)
        """
        logits = self.assign_logits[token_ids]  # (B, C, K)
        if hard:
            probs = F.one_hot(logits.argmax(dim=-1), num_classes=self.cfg.n_codes).float()
        else:
            probs = F.softmax(logits / tau, dim=-1)
        # (B,C,K) @ (C,K,d) -> (B,C,d)
        # expand codebooks to (1,C,K,d) and bmm per channel
        cb = self.codebooks  # (C,K,d)
        # einsum for clarity: b,c,k ; c,k,d -> b,c,d
        emb_c = torch.einsum("bck,ckd->bcd", probs, cb)
        # sum across channels
        emb = emb_c.reshape(emb_c.size(0), -1)  # (B, C*d_per) == (B, d_model)
        return emb

    def token_codes_hard(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return hard argmax codes per channel for each token: (B, C) longs."""
        logits = self.assign_logits[token_ids]  # (B,C,K)
        return logits.argmax(dim=-1)

    def usage_entropy(self) -> torch.Tensor:
        """Entropy of assignments across vocab (encourage diverse code use)."""
        probs = F.softmax(self.assign_logits, dim=-1)  # (V,C,K)
        ent = -torch.sum(probs * torch.log(probs.clamp_min(1e-8)), dim=-1)  # (V,C)
        return ent.mean()

