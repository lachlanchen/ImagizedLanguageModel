"""An optical input adapter for an aligned, finite-repertoire translation writer."""

from pathlib import Path

import numpy as np
import torch
from torch import nn


class TypefaceReader(nn.Module):
    def __init__(self, n):
        super().__init__()
        layers = []
        previous = 1
        for width in (32, 64, 96):
            layers.extend(
                [
                    nn.Conv2d(previous, width, 3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.GELU(),
                    nn.MaxPool2d(2),
                ]
            )
            previous = width
        self.features = nn.Sequential(
            *layers,
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.head = nn.Linear(256, n)

    def forward(self, x):
        return self.head(self.features(x))

    @torch.inference_mode()
    def transport(self, cells, palette):
        shapes, inverse = np.unique(cells, axis=0, return_inverse=True)
        known = set(map(bytes, palette))
        missing = [i for i, v in enumerate(shapes) if bytes(v) not in known]
        for j in range(0, len(missing), 256):
            idx = np.asarray(missing[j : j + 256])
            x = np.unpackbits(shapes[idx], axis=1).reshape(-1, 1, 32, 32)
            scores = self(
                torch.from_numpy(x)
                .to(self.head.weight)
                .contiguous(memory_format=torch.channels_last)
            )
            confidence, ids = scores.float().softmax(-1).max(-1)
            accept = confidence.cpu().numpy() >= 0.5
            shapes[idx[accept]] = palette[ids.cpu().numpy()[accept]]
        return shapes[inverse]


def load_typeface_reader(folder, palette, device):
    import hashlib
    import json
    from safetensors.torch import load_file

    folder = Path(folder)
    fit = json.loads((folder / "fit.json").read_text())
    weights = folder / "weights.safetensors"
    if (
        hashlib.sha256(weights.read_bytes()).hexdigest() != fit["weights"]
        or hashlib.sha256(palette.tobytes()).hexdigest() != fit["palette"]
    ):
        raise ValueError("Optical adapter weights or reference shapes have changed")
    net = TypefaceReader(len(palette))
    net.load_state_dict(load_file(str(weights)))
    return net.eval().to(device, memory_format=torch.channels_last)
