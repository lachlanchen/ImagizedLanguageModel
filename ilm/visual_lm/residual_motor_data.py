"""Decode serialized plan bytes without uint8 scalar-comparison overflow."""
import torch

from .binary_plan_glyph_density import pack_plan, unpack_plan


def decode_motor_patterns(patterns):
    if patterns.dtype != torch.uint8 or patterns.ndim != 2 or patterns.shape[1] != 32 or not len(patterns):
        raise ValueError('Complete serialized 256-bit plans required')
    # The shared validator compares with 256, outside uint8's scalar range.
    # Widen the serialized bytes before validation; never alter their values.
    bits = unpack_plan(patterns.to(torch.int64))
    if not torch.equal(pack_plan(bits).to(torch.uint8), patterns):
        raise ValueError('Serialized plan bytes failed exact recovery')
    return bits
