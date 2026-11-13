"""Physical support, causality and normalization of the prospective writer."""
import pytest
import torch

from ilm.visual_lm.glyph_joint_density import GlyphJointDensity, pack_glyphs, unpack_glyphs


def test_arbitrary_pixels_roundtrip_and_bit_order():
    torch.manual_seed(19)
    ink = torch.randint(0, 2, (32, 1, 32, 32), dtype=torch.uint8)
    ink[0].zero_(); ink[1].fill_(1)
    patterns = pack_glyphs(ink)
    assert patterns.shape == (32, 128)
    assert torch.equal(unpack_glyphs(patterns), ink)
    one = torch.zeros(8, 1, 32, 32, dtype=torch.uint8)
    for bit in range(8):
        one[bit, 0, bit//4, bit%4] = 1
    assert torch.equal(pack_glyphs(one)[:, 0], 1 << torch.arange(8))
    with pytest.raises(ValueError):
        pack_glyphs(ink.float() + .1)
    with pytest.raises(ValueError):
        unpack_glyphs(patterns.float() + .25)


def test_target_causality_cached_likelihood_and_real_gradient():
    torch.set_num_threads(2); torch.manual_seed(23)
    model = GlyphJointDensity(torch.zeros(4096), torch.ones(4096))
    state = torch.randn(2, 4096)
    ink = torch.randint(0, 2, (2, 1, 32, 32), dtype=torch.uint8)
    patterns = pack_glyphs(ink)
    logits = model(state, patterns)
    changed = patterns.clone(); changed[:, 64:] ^= 255
    other = model(state, changed)
    assert torch.equal(logits[:, :65], other[:, :65])
    assert not torch.equal(logits[:, 65:], other[:, 65:])
    assert torch.allclose(logits.softmax(-1).sum(-1), torch.ones(2, 128), atol=1e-6)
    with torch.no_grad():
        condition = model.condition_features(state); caches = None; serial = []
        for position in range(128):
            v, caches = model.step(condition, patterns[:, position-1] if position else None,
                                   position, caches)
            serial.append(v)
        assert (torch.stack(serial, 1) - logits.detach()).abs().max() < 1e-4
    losses = model.image_nll(state, ink)
    assert losses.shape == (2,) and bool(torch.isfinite(losses).all())
    losses.mean().backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    with pytest.raises(ValueError):
        model.image_nll(state[:1], ink)


def test_runtime_bridge_preserves_feedback_and_removes_old_writer():
    from ilm.visual_lm.glyph_joint_density import JointGlyphBridge
    from ilm.visual_lm.semantic_glyph_loop import SemanticGlyphBridge
    torch.set_num_threads(2); torch.manual_seed(31)
    base = SemanticGlyphBridge().eval()
    density = GlyphJointDensity(torch.zeros(4096), torch.ones(4096)).eval()
    bridge = JointGlyphBridge(base, density).eval()
    state = torch.randn(1, 4096)
    with torch.no_grad():
        logits, _, stop = bridge.predict(state)
        pixels = 1 - 2*(logits.sigmoid() >= .5).float()
        assert pixels.shape == (1, 1, 32, 32)
        assert torch.equal(bridge.feedback_embedding(pixels), base.feedback_embedding(pixels))
        assert torch.equal(stop, base.stop(state).flatten())
    assert not hasattr(bridge, 'decoder') and not hasattr(bridge, 'readout')
    assert not hasattr(bridge, 'decoder_input')
    assert not any(p.requires_grad for p in bridge.encoder.parameters())
    assert not any(p.requires_grad for p in bridge.stop.parameters())
