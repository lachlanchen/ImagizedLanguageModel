"""Matched continuous features and a small observed-image-block readout."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .canonical_glyph_language import orthonormal_dct_matrix
from .visual_answer_blocks import VisualAnswerBlocks
from .visual_copy_language import raster_signatures


def semi_orthogonal(inputs, outputs, seed):
    if inputs < outputs:
        raise ValueError("projection must not increase dimension")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q, _ = torch.linalg.qr(
        torch.randn(inputs, outputs, generator=generator), mode="reduced"
    )
    return q


def make_projectors():
    return {
        "semantic": semi_orthogonal(896, 384, 20265900),
        "fine": semi_orthogonal(1024, 128, 20265901),
        "basis": orthonormal_dct_matrix(32),
    }


def rms_features(values):
    return values.float() * torch.rsqrt(
        values.float().square().mean(-1, keepdim=True) + 1e-6
    )


def even_cells(images):
    if (
        images.ndim != 4
        or tuple(images.shape[1:]) != (1, 32, 32)
        or not 1 <= len(images) <= 480
    ):
        raise ValueError("prompt must be [1..480,1,32,32]")
    if not bool(((images == 0) | (images == 1)).all()):
        raise ValueError("binary prompt required")
    return (
        torch.cat((images, torch.zeros_like(images[:1])), 0)
        if len(images) % 2
        else images
    )


def native_strip(images):
    """Map ink cells to the author model's white=+1/black=-1 strip."""
    cells = even_cells(images)
    return (1 - 2 * cells).permute(1, 2, 0, 3).reshape(1, 1, 32, len(cells) * 32)


@torch.no_grad()
def extract_features(student, mixar, images, projectors):
    """Both encoders see prompt pixels alone; no reference answer argument."""
    count = len(images)
    cells = even_cells(images)
    device = images.device
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        student_pair = student.reader.language(cells[None])["hidden_states"][
            :, 1::2
        ].float()
        author_pair = mixar.model(
            inputs_embeds=native_strip(images),
            attention_mask=torch.ones(1, len(cells) // 2, device=device),
            use_cache=False,
        ).last_hidden_state.float()
    if student_pair.shape[1] != author_pair.shape[1] or author_pair.shape[-1] != 896:
        raise ValueError("pair alignment or author feature dimension differs")
    # Compute fixed coordinate projections in FP32 before the common FP16 cache.
    with torch.autocast(device_type=device.type, enabled=False):
        basis = projectors["basis"].to(device)
        signed = cells[:, 0].float().mul(2).sub(1)
        coordinates = (basis @ signed @ basis.T).flatten(1)
        fine = F.normalize(coordinates, dim=-1) @ projectors["fine"].to(device)
        fine = rms_features(fine)[:count]
        phase = torch.eye(2, device=device)[torch.arange(count, device=device) % 2]
        source = {
            "student": rms_features(student_pair[0]).repeat_interleave(2, 0)[:count],
            "mixar": rms_features(
                author_pair[0] @ projectors["semantic"].to(device)
            ).repeat_interleave(2, 0)[:count],
        }
        return {
            name: torch.cat((semantic, fine, phase), -1).half().cpu()
            for name, semantic in source.items()
        }


class FrozenFeatureBlockHead(nn.Module):
    maximum_answer_cells = 16

    def __init__(self, feature_dim=514, width=384):
        super().__init__()
        self.feature_dim = feature_dim
        self.width = width
        self.start_query = nn.Linear(feature_dim, width, bias=False)
        self.start_key = nn.Linear(feature_dim, width, bias=False)
        self.end_query = nn.Linear(feature_dim, width, bias=False)
        self.end_key = nn.Linear(feature_dim, width, bias=False)
        self.length_bias = nn.Parameter(torch.zeros(16))

    def forward(self, features, lengths):
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError("invalid continuous feature tensor")
        if lengths.shape != (len(features),) or not bool(
            ((lengths > 0) & (lengths <= features.shape[1])).all()
        ):
            raise ValueError("invalid packing lengths")
        final = features[
            torch.arange(len(features), device=features.device), lengths - 1
        ]
        start = torch.einsum(
            "bd,bld->bl", self.start_query(final), self.start_key(features)
        ).float() / math.sqrt(self.width)
        end = torch.einsum(
            "bd,bld->bl", self.end_query(final), self.end_key(features)
        ).float() / math.sqrt(self.width)
        source = torch.arange(features.shape[1], device=features.device)[:, None]
        last = source + torch.arange(16, device=features.device)[None]
        logits = (
            start[:, :, None]
            + end[:, last.clamp(max=features.shape[1] - 1)]
            + self.length_bias
        )
        valid = last[None] < lengths[:, None, None]
        return (
            logits.masked_fill(~valid, -torch.inf)
            .flatten(1)
            .log_softmax(-1)
            .reshape_as(logits)
        )

    def block_nll(self, logp, images, lengths, targets, target_lengths):
        # Reuse the frozen V58 exact image-sequence marginalization unchanged.
        return VisualAnswerBlocks.block_nll(
            self, logp, images, lengths, targets, target_lengths
        )

    @torch.no_grad()
    def decode(self, features, images):
        if len(features) != len(images):
            raise ValueError("feature/source mismatch")
        logp = self(
            features[None].float(), torch.tensor([len(images)], device=features.device)
        )[0]
        probability = logp.exp().cpu().double()
        keys = [tuple(row) for row in raster_signatures(images).cpu().tolist()]
        mass = {}
        representative = {}
        for start in range(len(images)):
            for length in range(1, min(16, len(images) - start) + 1):
                key = tuple(keys[start : start + length])
                if key not in mass:
                    mass[key] = 0.0
                    representative[key] = (start, length)
                mass[key] += float(probability[start, length - 1])
        winner = max(mass, key=mass.get)
        start, length = representative[winner]
        return {
            "images": images[start : start + length].clone(),
            "stopped": True,
            "capped": False,
            "events": [
                {
                    "route": "observed_block",
                    "source_start": start,
                    "source_end": start + length,
                    "marginal_block_probability": mass[winner],
                }
            ],
        }
