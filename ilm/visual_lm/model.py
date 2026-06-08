from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        groups = 8 if out_ch % 8 == 0 else 4 if out_ch % 4 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageToImageUNet(nn.Module):
    """Conditional page-image generator.

    The model maps an RGB prompt canvas to an RGB answer canvas. It is deliberately
    image-native: there is no text tokenizer, vocabulary, or Unicode decoder in
    the forward path.
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 3, base_ch: int = 32, depth: int = 3):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.base_ch = base_ch
        self.depth = depth

        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        ch = in_ch
        widths: list[int] = []
        for i in range(depth):
            out = base_ch * (2**i)
            self.down_blocks.append(ConvBlock(ch, out))
            self.downsample.append(nn.Conv2d(out, out, 4, stride=2, padding=1))
            widths.append(out)
            ch = out

        self.mid = ConvBlock(ch, ch * 2)
        ch = ch * 2

        self.up_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for out in reversed(widths):
            self.upsample.append(nn.ConvTranspose2d(ch, out, 4, stride=2, padding=1))
            self.up_blocks.append(ConvBlock(out * 2, out))
            ch = out

        self.out = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, out_ch, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[-2:]
        skips: list[torch.Tensor] = []
        h = x
        for block, down in zip(self.down_blocks, self.downsample):
            h = block(h)
            skips.append(h)
            h = down(h)
        h = self.mid(h)
        for up, block in zip(self.upsample, self.up_blocks):
            h = up(h)
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h)
        if h.shape[-2:] != orig_size:
            h = F.interpolate(h, size=orig_size, mode="bilinear", align_corners=False)
        return self.out(h)


def image_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    px = pred[..., :, 1:] - pred[..., :, :-1]
    tx = target[..., :, 1:] - target[..., :, :-1]
    py = pred[..., 1:, :] - pred[..., :-1, :]
    ty = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(px, tx) + F.l1_loss(py, ty)


def psnr_from_l1(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss((pred + 1) * 0.5, (target + 1) * 0.5).detach().clamp_min(1e-12)
    return float((-10.0 * torch.log10(mse)).cpu())
