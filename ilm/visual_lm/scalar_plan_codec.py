"""Four-level spatial image measurements with an explicit 256-bit budget.

The finite scalar quantizer follows Mentzer et al., ICLR 2024, Appendix A.1.
The arithmetic alphabet is fixed; it contains no glyph or Unicode inventory.
"""
import math

import torch

from .binary_plan_glyph_density import binary_images, pack_plan
from .spatial_plan_codec import SpatialPlanCodec


class ScalarPlanCodec(SpatialPlanCodec):
    """Eight four-level features at each of sixteen spatial sites.

    Quantized decoder inputs are {-1, -.5, 0, .5}, serialized as two bits per
    scalar, least significant bit first. Every possible 256-bit plan is valid.
    Straight-through rounding supplies a biased training gradient, not a
    derivative of the discontinuous deployed quantizer.
    """
    levels = 4
    scalar_bits = 2
    epsilon = .001

    def __init__(self, channels=(32, 64, 96, 128), groups=8):
        super().__init__(8, channels=channels, groups=groups)
        self.scalar_count = 16 * self.planes
        self.code_bits = self.scalar_count * self.scalar_bits

    def bound(self, raw):
        half = (self.levels - 1) * (1 - self.epsilon) / 2
        # The published Appendix A.1 uses tan for this fixed even-level shift.
        return (raw + math.tan(.5 / half)).tanh() * half - .5

    def _draw(self, values):
        if values.ndim != 2 or values.shape[1] != self.scalar_count:
            raise ValueError('Complete 128-scalar spatial plan required')
        sites = values.reshape(-1, 4, 4, self.planes).permute(0, 3, 1, 2).contiguous()
        return self.decoder(self.expand(sites)).float()

    def indices_to_bits(self, indices):
        if (indices.ndim != 2 or indices.shape[1] != self.scalar_count or not len(indices)
                or not bool(((indices >= 0) & (indices < 4) & (indices == indices.long())).all())):
            raise ValueError('Complete four-level integer indices required')
        shifts = torch.arange(2, device=indices.device)
        return ((indices.long()[..., None] >> shifts) & 1).reshape(len(indices), self.code_bits).to(torch.uint8)

    def bits_to_values(self, bits):
        pack_plan(bits)
        if bits.shape[1] != self.code_bits:
            raise ValueError('Complete 256-bit scalar plan required')
        pairs = bits.reshape(len(bits), self.scalar_count, 2).long()
        indices = pairs[..., 0] + 2 * pairs[..., 1]
        return (indices.float() - 2) / 2

    def forward(self, ink):
        ink = binary_images(ink)
        bounded = self.bound(self.encoder(ink.float() * 2 - 1))
        rounded = bounded.round()
        # Exactly hard values in the forward pass, even in finite precision.
        values = (rounded.detach() + (bounded - bounded.detach())) / 2
        bits = self.indices_to_bits(rounded.detach().long() + 2)
        return dict(plan_bits=bits, plan_values=values, image_logits=self._draw(values))

    @torch.no_grad()
    def encode_plan(self, ink):
        ink = binary_images(ink)
        indices = self.bound(self.encoder(ink.float() * 2 - 1)).round().long() + 2
        return self.indices_to_bits(indices)

    def decode_logits(self, bits):
        return self._draw(self.bits_to_values(bits))


def scalar_codec_forward(codec, ink, mode):
    """The paired control uses the same bound and tensors without rounding."""
    if mode == 'binary':
        return codec(ink)
    if mode != 'continuous':
        raise ValueError('Fixed continuous/quantized comparison only')
    ink = binary_images(ink)
    values = codec.bound(codec.encoder(ink.float() * 2 - 1)) / 2
    return dict(plan_values=values, image_logits=codec._draw(values))
