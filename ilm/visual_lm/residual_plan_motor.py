"""A frozen visual-plan decoder followed by a normalized learned pixel repair.

This component is a physical writing motor, not a contextual language model.
Only generated plan bits are needed at its public generation boundary.
"""
import torch
from torch import nn

from .binary_plan_glyph_density import BinaryPlanCodec, binary_images, pack_plan, xor_correction
from .occupancy_glyph_density import OccupancyGlyphDensity, encode_occupancy


class ResidualPlanMotor(nn.Module):
    """Model P(Y|B) = P_R(Y XOR D(B)|B,D(B)), with D frozen.

    All images have positive mathematical support through the normalized
    physical tree. Identity initialization concerns greedy decoding only:
    root logits [0,-2] choose the empty correction for every possible plan.
    Their nonzero probabilities still admit learning nonempty corrections.
    """
    def __init__(self, codec, *, width=128, layers=2, heads=4):
        super().__init__()
        if not isinstance(codec, BinaryPlanCodec):
            raise TypeError('A fixed binary-plan neural codec is required')
        self.codec = codec.requires_grad_(False).eval()
        self.refinement = OccupancyGlyphDensity(codec.code_bits + 1024, mode='tree',
                                               width=width, layers=layers, heads=heads,
                                               maximum_span=1)
        with torch.no_grad():
            self.refinement.root_output.weight.zero_()
            self.refinement.root_output.bias.copy_(torch.tensor([0., -2.]))

    def train(self, mode=True):
        super().train(mode)
        self.codec.eval()
        return self

    def features(self, bits, base):
        """Form conditions from plan bits and their frozen decoder image.

        Training may reuse authenticated saved D(B) arrays. The public generate
        method computes D(B) itself; no caller-supplied answer image enters it.
        """
        pack_plan(bits)
        base = binary_images(base)
        if bits.shape != (len(base), self.codec.code_bits) or bits.device != base.device:
            raise ValueError('Complete matching plan and decoder image required')
        return torch.cat((bits.float() * 2 - 1, base.flatten(1).float() * 2 - 1), -1)

    def cached_image_log_prob(self, bits, base, target):
        """Training/assessment likelihood with authenticated fixed decoder arrays.

        This is exact conditional image likelihood for the supplied B; it is
        not the marginal language likelihood after an untrained plan prior.
        """
        residual = xor_correction(base, target)
        return self.refinement.log_prob(self.features(bits, base), encode_occupancy(residual, 'tree'))

    @torch.no_grad()
    def posterior_plan(self, target):
        """Supplied-image codec supervision only; never used by generate()."""
        bits = self.codec.encode_plan(target)
        return bits, self.codec.decode_plan(bits)

    @torch.no_grad()
    def generate_cached(self, bits, base, *, greedy=True, generator=None):
        """Internal evaluation with authenticated D(B); accepts no target."""
        residual, trace = self.refinement.generate(self.features(bits, base),
                                                  greedy=greedy, generator=generator)
        return xor_correction(base, residual), dict(base=base, residual=residual,
                                                  correction_trace=trace)

    @torch.no_grad()
    def generate(self, bits, *, greedy=True, generator=None):
        """Draw from generated bits; no target encoder, font or image lookup."""
        base = self.codec.decode_plan(bits)
        return self.generate_cached(bits, base, greedy=greedy, generator=generator)
