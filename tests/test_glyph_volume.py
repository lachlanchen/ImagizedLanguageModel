"""Information preservation and block causality, not language-model scores."""
import numpy as np
import pytest

from ilm.visual_lm.glyph_volume import GlyphVolume, pack_glyph_volume, unpack_glyph_volume


@pytest.mark.parametrize('length,group', [(1, 1), (1, 4), (7, 4), (16, 8), (513, 16)])
def test_binary_roundtrip_preserves_shape_order_and_padding(length, group):
    x = np.random.default_rng(13).integers(0, 2, (length, 1, 32, 32), dtype=np.uint8)
    untouched = x.copy()
    volume = pack_glyph_volume(x, group)
    assert np.array_equal(unpack_glyph_volume(volume), x)
    assert np.array_equal(x, untouched)
    assert volume.coefficients.size == ((length + group - 1) // group) * group * 1024


def test_overlay_collision_is_distinguished_by_ordered_volume():
    x = np.zeros((2, 1, 4, 4), dtype=np.uint8)
    x[0, 0, 0, 0] = x[1, 0, 3, 3] = 1
    assert np.array_equal(x.sum(0), x[::-1].sum(0))
    a, b = pack_glyph_volume(x, 2), pack_glyph_volume(x[::-1], 2)
    assert not np.array_equal(a.coefficients, b.coefficients)
    assert not np.array_equal(unpack_glyph_volume(a, retain_details=False), x)


def test_completed_block_is_causal_but_partial_block_is_not():
    x = np.random.default_rng(8).random((8, 2, 5, 5), dtype=np.float32)
    y = x.copy(); y[4:] = 1 - y[4:]
    assert np.array_equal(pack_glyph_volume(x).coefficients[:1], pack_glyph_volume(y).coefficients[:1])
    y = x.copy(); y[1] = 1 - y[1]
    assert not np.array_equal(pack_glyph_volume(x).coefficients[:1], pack_glyph_volume(y).coefficients[:1])
    assert np.allclose(unpack_glyph_volume(pack_glyph_volume(x)), x, atol=2e-7, rtol=0)


def test_invalid_values_and_corrupt_metadata_fail_closed():
    x = np.zeros((3, 1, 4, 4), dtype=np.float32)
    for group in [0, 3, True]:
        with pytest.raises(ValueError): pack_glyph_volume(x, group)
    x[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError): pack_glyph_volume(x)
    with pytest.raises(TypeError): pack_glyph_volume(np.zeros((1, 1, 4, 4), dtype=np.float64))
    volume = pack_glyph_volume(np.zeros((3, 1, 4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        unpack_glyph_volume(GlyphVolume(volume.coefficients, 9, 1, 4))
