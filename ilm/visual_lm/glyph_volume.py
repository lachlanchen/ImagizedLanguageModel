"""Ordered temporal Haar packing of glyph images, without a symbol vocabulary.

This invertible reference operator shortens the time axis by moving information
into channels. It does not reduce the number of scalar coefficients, learn
language, or provide an entropy model. Only completed blocks are causal context.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GlyphVolume:
    coefficients: np.ndarray  # [ceil(T/group), group*C, H, W]
    length: int
    channels: int
    group: int


def pack_glyph_volume(ink: np.ndarray, group: int = 4) -> GlyphVolume:
    """Pack [T,C,H,W] ink in [0,1]; retain coarse AND temporal detail planes.

    Binary inputs have exact dyadic coefficients in float32. General float32
    inputs round-trip to floating-point precision. End padding is zero ink.
    """
    x = np.asarray(ink)
    if x.ndim != 4 or any(d < 1 for d in x.shape):
        raise ValueError("Expected a nonempty [T,C,H,W] glyph stream")
    if isinstance(group, bool) or not isinstance(group, int) or group < 1 or group & (group - 1):
        raise ValueError("group must be a positive power of two")
    if x.dtype not in (np.dtype('bool'), np.dtype('uint8'), np.dtype('float32')):
        raise TypeError("Use binary bool/uint8 or float32 ink in [0,1]")
    if not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1):
        raise ValueError("Ink must be finite and in [0,1]")
    length, channels, height, width = x.shape
    blocks = (length + group - 1) // group
    padded = np.zeros((blocks * group, channels, height, width), dtype=np.float32)
    padded[:length] = x
    low = padded.reshape(blocks, group, channels, height, width)
    details = []
    while low.shape[1] > 1:
        a, b = low[:, 0::2], low[:, 1::2]
        details.append((a - b) * np.float32(0.5))
        low = (a + b) * np.float32(0.5)
    coefficients = np.concatenate([low, *reversed(details)], axis=1)
    return GlyphVolume(coefficients.reshape(blocks, group * channels, height, width),
                       length, channels, group)


def unpack_glyph_volume(volume: GlyphVolume, *, retain_details: bool = True) -> np.ndarray:
    """Recover ordered images; dropping detail planes is deliberately lossy."""
    y = np.asarray(volume.coefficients)
    if y.ndim != 4 or volume.channels < 1 or volume.length < 1:
        raise ValueError("Invalid volume metadata")
    group = volume.group
    if isinstance(group, bool) or not isinstance(group, int) or group < 1 or group & (group - 1):
        raise ValueError("Invalid temporal group")
    if y.dtype != np.float32 or not np.isfinite(y).all():
        raise ValueError("Expected finite float32 coefficients")
    if y.shape[0] != (volume.length + group - 1) // group or y.shape[1] != group * volume.channels:
        raise ValueError("Coefficient shape disagrees with original stream length")
    planes = y.reshape(len(y), group, volume.channels, *y.shape[2:])
    low = planes[:, :1].copy()
    offset = 1
    while low.shape[1] < group:
        n = low.shape[1]
        detail = planes[:, offset:offset + n] if retain_details else np.zeros_like(low)
        expanded = np.empty((len(y), 2 * n, volume.channels, *y.shape[2:]), dtype=np.float32)
        expanded[:, 0::2], expanded[:, 1::2] = low + detail, low - detail
        low = expanded
        offset += n
    return low.reshape(-1, volume.channels, *y.shape[2:])[:volume.length]
