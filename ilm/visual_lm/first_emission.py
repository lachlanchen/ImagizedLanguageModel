"""Adapt only the first physical drawing state of a visual phrase writer."""
import torch
from safetensors import safe_open
from torch import nn


class FirstEmission(nn.Module):
    def __init__(self, width=128, hidden=256):
        super().__init__()
        self.update = nn.Sequential(nn.Linear(3 * width, hidden), nn.GELU(), nn.Linear(hidden, width))
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)
        self.register_buffer("scale", torch.ones(()))

    def forward(self, boundary, initial, history):
        correction = self.update(torch.cat((boundary, initial, history), -1))
        return (boundary + history + self.scale * correction).tanh()

    @classmethod
    def restore(cls, path, interaction_digest, device):
        with safe_open(str(path), framework="pt") as file:
            meta = file.metadata() or {}
            if meta.get("format") != "ilm-first-emission-v1" or meta.get("interaction_sha256") != interaction_digest:
                raise ValueError("First-emission weights require the matching physical writer")
            net = cls()
            net.load_state_dict({key:file.get_tensor(key) for key in file.keys()})
        return net.to(device).eval()
