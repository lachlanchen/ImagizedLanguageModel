from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=p)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.bn(self.conv(x)))


class UNet2D(nn.Module):
    """Small 2D U-Net for masked inpainting.

    Input: (B, Cin, H, W) where Cin = r_channels + 1 (mask channel)
    Output: (B, r_channels, H, W)
    """

    def __init__(self, in_ch: int, base: int = 64, out_ch: int | None = None):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch - 1
        self.enc1 = nn.Sequential(ConvBlock(in_ch, base), ConvBlock(base, base))
        self.down1 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.enc2 = nn.Sequential(ConvBlock(base * 2, base * 2), ConvBlock(base * 2, base * 2))
        self.down2 = nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1)
        self.bott = nn.Sequential(ConvBlock(base * 4, base * 4), ConvBlock(base * 4, base * 4))
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(ConvBlock(base * 4, base * 2), ConvBlock(base * 2, base * 2))
        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(ConvBlock(base * 2, base), ConvBlock(base, base))
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        b = self.bott(self.down2(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


class InpaintNet(nn.Module):
    """Wrapper with learnable compression/decompression around UNet.

    - Compress d_model channels to r via 1x1 conv
    - UNet predicts r channels for masked inpainting
    - Decompress r back to d_model via 1x1 conv
    """

    def __init__(self, d_model: int, r: int = 16):
        super().__init__()
        self.compress = nn.Conv2d(d_model, r, 1)
        self.unet = UNet2D(in_ch=r + 1, base=64, out_ch=r)
        self.decompress = nn.Conv2d(r, d_model, 1)

    def forward(self, y: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # y: (B,d,H,W) full target; mask: (B,1,H,W) 1=masked positions to predict
        y_r = self.compress(y)
        y_in = y_r * (1.0 - mask)  # zero masked positions
        unet_in = torch.cat([y_in, mask], dim=1)
        y_r_hat = self.unet(unet_in)
        y_hat = self.decompress(y_r_hat)
        return y_hat, y_r_hat, y_r

