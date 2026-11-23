"""Recognize alternate printed forms while retaining original ink for writing."""

import hashlib

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.font_invariant_reader import GlyphMetric


def unpack(packed):
    shifts = torch.arange(7, -1, -1, device=packed.device)
    return ((packed[..., None] >> shifts) & 1).reshape(-1, 32, 32)


class TypefaceMap(nn.Module):
    def __init__(self, ids, prototypes):
        super().__init__()
        self.metric = GlyphMetric()
        self.register_buffer("ids", ids)
        self.register_buffer("prototypes", prototypes)

    def identify(self, packed):
        scores = self.metric(unpack(packed)) @ F.normalize(self.prototypes, dim=1).T
        best, labels = scores.topk(2, dim=1)
        accept = (best[:, 0] >= 0.65) & (best[:, 0] - best[:, 1] >= 0.06)
        return self.ids[labels[:, 0]], accept

    @classmethod
    def restore(cls, path, packed):
        with safe_open(path, framework="pt") as source:
            expected = source.metadata()["alphabet_sha256"]
        actual = hashlib.sha256(packed.detach().cpu().numpy().tobytes()).hexdigest()
        if actual != expected:
            raise ValueError("Typeface map belongs to another bitmap inventory")
        weights = load_file(str(path))
        ids = weights["ids"]
        if (
            len(ids) < 2
            or ids.min() < 1
            or ids.max() > len(packed)
            or len(ids.unique()) != len(ids)
        ):
            raise ValueError("Invalid bitmap indices in typeface map")
        model = cls(ids, weights["prototypes"])
        model.load_state_dict(weights)
        return model.to(packed.device).eval().requires_grad_(False)
