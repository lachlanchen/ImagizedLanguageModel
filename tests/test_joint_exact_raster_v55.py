import torch

from ilm.visual_lm.canonical_glyph_language import CanonicalGlyphLanguageConfig
from ilm.visual_lm.exact_raster_density_v54 import ExactRasterDensityConfig
from ilm.visual_lm.hierarchical_raster_language import CausalRasterReader
from ilm.visual_lm.joint_exact_raster_v55 import JointExactRasterLanguage


def build(joint):
    torch.manual_seed(55)
    reader = CausalRasterReader(
        CanonicalGlyphLanguageConfig(model_dim=384, layers=1, heads=6)
    )
    return JointExactRasterLanguage(
        reader,
        ExactRasterDensityConfig(width=128, layers=1, heads=4, pixel_hidden=64),
        reader_trainable=joint,
    )


def test_likelihood_updates_reader_only_in_joint_arm():
    context = torch.randint(0, 2, (2, 4, 1, 32, 32)).float()
    target = torch.randint(0, 2, (2, 1, 32, 32)).float()
    for joint in (False, True):
        model = build(joint).train()
        hidden, fields = model.read(context)
        model.image_nll(target, hidden, fields).mean().backward()
        grads = [p.grad for p in model.reader.parameters() if p.grad is not None]
        assert bool(grads) == joint
        if joint:
            assert sum(float(g.square().sum()) for g in grads) > 0
            assert all(torch.isfinite(g).all() for g in grads)
        assert model.reader.logit_scale.grad is None
        assert not model.reader.training
        assert all(not module.training for module in model.reader.modules())


def test_joint_reader_is_causal_and_uses_same_pixel_law():
    model = build(True).eval()
    context = torch.randint(0, 2, (2, 8, 1, 32, 32)).float()
    changed = context.clone()
    changed[:, 5:] = 1 - changed[:, 5:]
    state = model.reader.language(context)
    alternative = model.reader.language(changed)
    torch.testing.assert_close(
        state["hidden_states"][:, :5],
        alternative["hidden_states"][:, :5],
        atol=0,
        rtol=0,
    )
    image = torch.randint(0, 2, (2, 1, 32, 32)).float()
    fields = model.reader.field.encode_unit(image)
    copied = model.sample_state(
        torch.zeros(2, 384),
        fields,
        generator=torch.Generator().manual_seed(55),
        greedy=True,
    )
    torch.testing.assert_close(copied, image, atol=0, rtol=0)
