"""Context-dependent transitions between adjacent four-pixel measurements."""
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from ilm.visual_lm.idiom_ink import GROUPS


class ConditionalInk(nn.Module):
    """A shared visual encoder and eight pairwise potentials per transition."""

    def __init__(self):
        super().__init__()
        rank = 8
        self.encoder = nn.Sequential(nn.Linear(4 * 49 + 1, 96), nn.GELU(),
                                     nn.Linear(96, (GROUPS - 1) * rank))
        self.basis = nn.Parameter(torch.randn(GROUPS - 1, rank, 16, 16) * 0.05)
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)
        self.register_buffer('scale', torch.ones(()))

    def coefficients(self, signatures, onset):
        shifts = torch.arange(49, device=signatures.device)
        features = ((signatures[..., None] >> shifts) & 1).flatten(1).float()
        inputs = torch.cat((features, onset.float().reshape(-1, 1)), dim=1)
        return self.encoder(inputs).reshape(len(inputs), GROUPS - 1, -1) * self.scale

    def centered_basis(self):
        centered = self.basis - self.basis.mean(-2, keepdim=True)
        return centered - centered.mean(-1, keepdim=True)

    def teacher(self, signatures, onset, codes):
        # Pick the preceding observed group independently for each transition.
        groups = torch.arange(GROUPS - 1, device=codes.device)
        basis = self.centered_basis().permute(0, 2, 1, 3)
        selected = basis[groups[None], codes[:, :-1]]
        return (self.coefficients(signatures, onset)[..., None] * selected).sum(2)

    def transitions(self, signatures, onset):
        weights = self.coefficients(signatures, onset)
        return torch.einsum('bgr,grij->bgij', weights, self.centered_basis())

    @classmethod
    def restore(cls, path: Path, manifest_hash: str, device):
        with safe_open(str(path), framework='pt') as archive:
            meta = archive.metadata() or {}
            if meta.get('format') != 'ilm-conditional-ink-v1' or meta.get('counts_manifest') != manifest_hash:
                raise ValueError('Conditional ink checkpoint requires its matching local probability tables')
            model = cls()
            model.load_state_dict({key: archive.get_tensor(key) for key in archive.keys()}, strict=True)
        return model.to(device).eval()
