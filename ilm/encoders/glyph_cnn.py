from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlyphCNN(nn.Module):
    """
    Lightweight CNN to encode 64x64 or 128x128 grayscale/RGB glyph images into a d-dim vector.

    Expects input shape: (B, C, H, W) with C=1 or 3. Outputs (B, d).
    """

    def __init__(self, d: int = 128, in_channels: int = 3):
        super().__init__()
        ch = [in_channels, 32, 64, 128, 128]
        self.conv1 = nn.Conv2d(ch[0], ch[1], 5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(ch[1])
        self.conv2 = nn.Conv2d(ch[1], ch[2], 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(ch[2])
        self.conv3 = nn.Conv2d(ch[2], ch[3], 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(ch[3])
        self.conv4 = nn.Conv2d(ch[3], ch[4], 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(ch[4])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch[4], d),
            nn.LayerNorm(d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.bn1(self.conv1(x)))
        x = F.silu(self.bn2(self.conv2(x)))
        x = F.silu(self.bn3(self.conv3(x)))
        x = F.silu(self.bn4(self.conv4(x)))
        x = self.head(x)
        return x

