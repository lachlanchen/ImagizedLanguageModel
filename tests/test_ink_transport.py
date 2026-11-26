"""Boundary and integration checks for the optical input adapter."""

import numpy as np
import pytest
from ilm.visual_lm.ink_transport import InkTransport


def tile(*on):
    bits = np.zeros(1024, dtype=np.uint8)
    bits[list(on)] = 1
    return np.packbits(bits)


def test_open_radius_and_abstention():
    left, right = tile(), tile(0, 1, 2, 3)
    model = InkTransport(np.stack([left, right]))
    observed = np.stack([tile(0), tile(0, 1), right, tile(0, 1, 2)])
    expected = np.stack([left, tile(0, 1), right, right])
    assert np.array_equal(model.apply(observed), expected)
    assert np.array_equal(model.apply(expected), expected)
    with pytest.raises(ValueError, match="distinct"):
        InkTransport(np.stack([left, left]))


def test_adapter_preserves_canonical_cells_in_original_order():
    from ilm.visual_lm.glyph_typeface_reader import TypefaceReader

    bank = np.stack([tile(), tile(0, 1, 2, 3)])
    shapes = bank[[1, 0, 1, 1]]
    reader = TypefaceReader(len(bank)).eval()

    # Exact canonical shapes never need a neural call.
    def unexpected(module, inputs):
        raise AssertionError("Unnecessary optical inference")

    reader.register_forward_pre_hook(unexpected)
    assert np.array_equal(reader.transport(shapes, bank), shapes)
