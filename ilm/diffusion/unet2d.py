from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t in [0,1], shape (B,)
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), half, device=t.device)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            pad = torch.zeros((emb.size(0), self.dim - emb.shape[-1]), device=t.device)
            emb = torch.cat([emb, pad], dim=-1)
        return self.proj(emb)


class ResBlock(nn.Module):
    def __init__(self, ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.time = nn.Linear(time_dim, ch * 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        scale, shift = self.time(t).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = h * (1 + scale) + shift
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return x + h


class UNet2D(nn.Module):
    def __init__(self, in_ch: int, base_ch: int = 64, depth: int = 2, out_ch: int | None = None, time_dim: int = 128):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbed(time_dim)
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.downs = nn.ModuleList()
        ch = base_ch
        for i in range(depth):
            self.downs.append(
                nn.ModuleDict({
                    "res": ResBlock(ch, time_dim),
                    "down": nn.Conv2d(ch, ch * 2, 4, stride=2, padding=1),
                })
            )
            ch *= 2
        self.mid = ResBlock(ch, time_dim)
        self.ups = nn.ModuleList()
        for i in range(depth):
            self.ups.append(
                nn.ModuleDict({
                    "up": nn.ConvTranspose2d(ch, ch // 2, 4, stride=2, padding=1),
                    "res": ResBlock(ch // 2, time_dim),
                })
            )
            ch //= 2
        self.out_conv = nn.Conv2d(ch, out_ch if out_ch is not None else in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W); t_scalar: (B,) in [0,1]
        t = self.time_embed(t_scalar)
        h = self.in_conv(x)
        skips = []
        for blk in self.downs:
            h = blk["res"](h, t)
            skips.append(h)
            h = blk["down"](h)
        h = self.mid(h, t)
        for blk in self.ups:
            h = blk["up"](h)
            # match skip size if needed
            if skips:
                s = skips.pop()
                if s.shape[-2:] != h.shape[-2:]:
                    # interpolate if odd sizes
                    s = F.interpolate(s, size=h.shape[-2:], mode="nearest")
                h = h + s
            h = blk["res"](h, t)
        out = self.out_conv(h)
        return out
