"""A function-preserving extension of the compact reader's feedback map."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class NonlinearVisualFeedback(nn.Module):
    """Keep the original map and add a zero-initialized nonlinear branch.

The old LayerNorm/linear pair restricts every prediction to one low-dimensional
affine family. A nonlinear branch removes that particular restriction; it does
not guarantee native-vector alignment or useful language. Source mean/scale
buffers remain outside this module, unchanged in the original visual encoder.
"""
    def __init__(self, original, *, hidden_width=4096, seed=202611184):
        super().__init__()
        if (not isinstance(original, nn.Sequential) or len(original) != 2
                or not isinstance(original[0], nn.LayerNorm)
                or not isinstance(original[1], nn.Linear)
                or original[0].normalized_shape != (original[1].in_features,)
                or type(hidden_width) is not int or hidden_width < original[1].out_features):
            raise ValueError("Require the original normalized affine map and sufficient branch width")
        self.original = original
        self.hidden_width = hidden_width
        # Initialize on CPU while preserving the caller's random streams.
        with torch.random.fork_rng(devices=[]), torch.device('cpu'):
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
            self.expand = nn.Linear(original[1].in_features, hidden_width)
            self.project = nn.Linear(hidden_width, original[1].out_features, bias=False)
            nn.init.zeros_(self.project.weight)
        self.expand.to(device=original[1].weight.device, dtype=original[1].weight.dtype)
        self.project.to(device=original[1].weight.device, dtype=original[1].weight.dtype)
        self.train(original.training)

    def forward(self, query):
        normalized = self.original[0](query)
        return self.original[1](normalized) + self.project(F.gelu(self.expand(normalized)))

    @property
    def added_parameters(self):
        return sum(p.numel() for module in (self.expand, self.project) for p in module.parameters())


def relative_feedback_error(prediction, target):
    """The original per-update native-vector objective, in FP32."""
    if prediction.shape != target.shape or prediction.ndim != 2 or not len(prediction):
        raise ValueError("Require complete nonempty aligned native-vector rows")
    a, b = prediction.float(), target.float()
    return (a-b).square().sum(-1) / b.square().sum(-1).clamp_min(1e-8)
