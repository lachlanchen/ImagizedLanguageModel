from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProductCodeLosses:
    info_nce: torch.Tensor
    usage_kl: torch.Tensor
    indep: torch.Tensor

    def total(self, w_info: float = 1.0, w_usage: float = 0.1, w_indep: float = 0.1) -> torch.Tensor:
        return w_info * self.info_nce + w_usage * self.usage_kl + w_indep * self.indep


class ProductCode(nn.Module):
    """
    Differentiable 3×K product code with Gumbel-Softmax and compositional embedding.

    Given input features g ∈ R^{B×d_in}, produce:
      - logits per channel (B×K)
      - soft one-hot selections y_c (B×K)
      - code embedding e ∈ R^{B×d}

    Embedding is E = sum_c y_c @ E_c (each E_c ∈ R^{K×d}).
    """

    def __init__(self, d_in: int, d: int = 128, K: int = 32, C: int = 3, tau: float = 1.0, straight_through: bool = True):
        super().__init__()
        self.C = C
        self.K = K
        self.d = d
        self.tau = tau
        self.straight_through = straight_through

        self.proj = nn.ModuleList([nn.Linear(d_in, K) for _ in range(C)])
        self.codebooks = nn.ParameterList([nn.Parameter(torch.randn(K, d) * 0.02) for _ in range(C)])
        # Optional per-channel scale
        self.scales = nn.Parameter(torch.ones(C))

    def forward(self, g: torch.Tensor, tau: float | None = None) -> Dict[str, torch.Tensor]:
        if tau is None:
            tau = self.tau
        B = g.size(0)
        logits_list = [p(g) for p in self.proj]
        y_list = [F.gumbel_softmax(logits, tau=tau, hard=self.straight_through, dim=-1) for logits in logits_list]
        # Compose embedding
        parts = []
        for c in range(self.C):
            # y (B×K) × E (K×d) → (B×d)
            parts.append(y_list[c] @ self.codebooks[c])
        e = torch.stack(parts, dim=1)  # (B×C×d)
        e = (self.scales.view(1, self.C, 1) * e).sum(dim=1)  # (B×d)
        out = {
            "logits": torch.stack(logits_list, dim=1),  # (B×C×K)
            "y": torch.stack(y_list, dim=1),            # (B×C×K)
            "embed": e,                                  # (B×d)
        }
        return out

    @staticmethod
    def info_nce(g: torch.Tensor, e: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        # Normalize
        g_n = F.normalize(g, dim=-1)
        e_n = F.normalize(e, dim=-1)
        logits = g_n @ e_n.t()  # (B×B)
        logits = logits / temperature
        target = torch.arange(g.size(0), device=g.device)
        loss = F.cross_entropy(logits, target)
        return loss

    def usage_kl(self, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """KL divergence of average assignment to uniform, summed over channels.
        y: (B×C×K)
        """
        B, C, K = y.shape
        p = y.mean(dim=0)  # (C×K)
        u = torch.full_like(p, 1.0 / K)
        kl = (p * (p.add(eps).log() - u.add(eps).log())).sum(dim=-1).mean()
        return kl

    def independence(self, y: torch.Tensor) -> torch.Tensor:
        """Measure dependence between channels by cross-covariance of assignments.
        Returns Frobenius norm of centered cross-covariance across all channel pairs.
        y: (B×C×K)
        """
        B, C, K = y.shape
        yc = y - y.mean(dim=0, keepdim=True)  # center per channel
        cov_sum = 0.0
        pairs = 0
        for a in range(C):
            Ya = yc[:, a, :]  # (B×K)
            for b in range(a + 1, C):
                Yb = yc[:, b, :]
                cov = (Ya.t() @ Yb) / (B - 1)  # (K×K)
                cov_sum = cov_sum + (cov.pow(2).sum())
                pairs += 1
        return cov_sum / max(pairs, 1)

    def compute_losses(self, glyph: torch.Tensor, out: Dict[str, torch.Tensor], temperature: float = 0.07,
                       w_info: float = 1.0, w_usage: float = 0.1, w_indep: float = 0.1) -> Tuple[ProductCodeLosses, torch.Tensor]:
        e = out["embed"]
        y = out["y"]
        l_info = self.info_nce(glyph, e, temperature=temperature)
        l_usage = self.usage_kl(y)
        l_indep = self.independence(y)
        total = w_info * l_info + w_usage * l_usage + w_indep * l_indep
        return ProductCodeLosses(l_info, l_usage, l_indep), total

