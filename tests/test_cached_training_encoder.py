import torch

from ilm.visual_lm.cached_training_encoder import CachedTrainingImageLanguage
from ilm.visual_lm.continuous_autoregressive_language import ContinuousImageLanguage
from ilm.visual_lm.continuous_glyph_codec import ContinuousGlyphCodec, ContinuousGlyphCodecConfig


def test_cache_overflow_weight_reload_and_inference():
    torch.set_num_threads(2)
    torch.manual_seed(32)
    model = CachedTrainingImageLanguage(ContinuousGlyphCodec(ContinuousGlyphCodecConfig()),
                                        components=1, model_dim=128, layers=2, heads=4,
                                        cache_entries=2).train()
    originals = torch.randint(0, 2, (4, 1, 32, 32), dtype=torch.uint8)
    ink = originals[[2, 0, 2, 3, 1, 0]]
    expected = ContinuousImageLanguage.encode_ink(model, ink)
    torch.testing.assert_close(model.encode_ink(ink), expected, atol=2e-5, rtol=2e-5)
    assert model.encoded_training_cells == 4 and len(model._indices) == 2
    torch.testing.assert_close(model.encode_ink(ink), expected, atol=2e-5, rtol=2e-5)
    assert model.encoded_training_cells == 6
    with torch.no_grad():
        model.codec.encoder.projection.bias.add_(torch.linspace(-.5, .5, 768))
    changed = model.encode_ink(ink)
    assert model.encoded_training_cells == 4 and not torch.allclose(changed, expected)
    torch.testing.assert_close(changed, ContinuousImageLanguage.encode_ink(model, ink), atol=2e-5, rtol=2e-5)
    model.eval()
    output = model.encode_ink(ink)
    assert model.encoded_training_cells == 4
    assert torch.equal(output, ContinuousImageLanguage.encode_ink(model, ink))
    assert all('_features' not in k and '_indices' not in k for k in model.state_dict())
