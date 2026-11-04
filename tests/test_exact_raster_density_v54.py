import torch

from ilm.visual_lm.canonical_glyph_language import CanonicalGlyphLanguageConfig
from ilm.visual_lm.exact_raster_density_v54 import (
    ConditionalRowMADE,
    ExactRasterDensity,
    ExactRasterDensityConfig,
)
from ilm.visual_lm.hierarchical_raster_language import CausalRasterReader


def make_model():
    return ExactRasterDensity(
        CausalRasterReader(CanonicalGlyphLanguageConfig()), ExactRasterDensityConfig()
    )


def test_made_does_not_see_current_or_future_pixel():
    model = ConditionalRowMADE(192, 256)
    torch.nn.init.normal_(model.output.weight, std=0.1)
    pixels = torch.randn(2, 32, requires_grad=True)
    condition = torch.randn(2, 192)
    for column in (0, 7, 19, 31):
        gradient = torch.autograd.grad(
            model(pixels, condition)[:, column].sum(), pixels
        )[0]
        assert torch.equal(gradient[:, column:], torch.zeros_like(gradient[:, column:]))
        if column:
            assert gradient[:, :column].abs().sum() > 0


def test_full_raster_does_not_see_current_or_future_pixel():
    model = make_model()
    torch.nn.init.normal_(model.pixel_density.output.weight, std=0.1)
    torch.nn.init.normal_(model.pixel_density.condition_out.weight, std=0.1)
    pixels = torch.rand(1, 1, 32, 32, requires_grad=True)
    logits = model.logits(pixels, torch.randn(1, 384), torch.randn(1, 1024))
    gradient = torch.autograd.grad(logits[0, 0, 12, 15], pixels)[0].flatten()
    assert gradient[: 12 * 32 + 15].abs().sum() > 0
    assert torch.equal(
        gradient[12 * 32 + 15 :], torch.zeros_like(gradient[12 * 32 + 15 :])
    )


def test_initial_mode_preserves_arbitrary_supplied_image():
    model = make_model().eval()
    images = torch.randint(0, 2, (2, 1, 32, 32)).float()
    fields = model.reader.field.encode_unit(images)
    logits = model.logits(images, torch.zeros(2, 384), fields)
    assert torch.equal((logits >= 0).float(), images)
    expected = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            logits, images, reduction="none"
        )
        .flatten(1)
        .sum(1)
    )
    assert torch.allclose(
        model.image_nll(images, torch.zeros(2, 384), fields), expected
    )
    assert torch.allclose(
        expected,
        torch.full_like(
            expected, 1024 * torch.nn.functional.softplus(torch.tensor(-6.0))
        ),
        atol=1e-4,
    )
    sample = model.sample_state(
        torch.zeros(2, 384),
        fields,
        generator=torch.Generator().manual_seed(54),
        greedy=True,
    )
    assert torch.equal(sample, images)


def test_density_backpropagates_without_changing_reader():
    model = make_model()
    image = torch.randint(0, 2, (2, 1, 32, 32)).float()
    loss = model.image_nll(image, torch.randn(2, 384), torch.randn(2, 1024)).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.pixel_density.output.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.reader.parameters())
    assert model.boundary_receipt()["density_parameters"] < 5_000_000
