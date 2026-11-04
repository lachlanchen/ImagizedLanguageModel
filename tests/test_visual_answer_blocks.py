import torch

from ilm.visual_lm.exact_raster_density_v54 import ExactRasterDensityConfig
from ilm.visual_lm.visual_answer_blocks import VisualAnswerBlocks
from ilm.visual_lm.visual_copy_language import VisualCopyReaderConfig


def model():
    torch.manual_seed(58)
    return VisualAnswerBlocks(
        VisualCopyReaderConfig(layers=1),
        ExactRasterDensityConfig(width=128, layers=1, heads=2),
    ).eval()


def test_block_probability_normalization_bounds_and_gradients():
    m = model()
    x = torch.randint(0, 2, (2, 6, 1, 32, 32)).float()
    lengths = torch.tensor([6, 4])
    logp = m.block_distribution(x, lengths)
    torch.testing.assert_close(logp.exp().flatten(1).sum(-1), torch.ones(2))
    assert torch.isneginf(logp[1, 4:]).all()
    assert torch.isneginf(logp[1, 3, 1:]).all()
    targets = x[:, 1:3]
    nll = m.block_nll(logp, x, lengths, targets, torch.tensor([2, 2]))
    nll.sum().backward()
    for layer in (m.reader.input_projection, m.block_start_query, m.block_end_query):
        assert (
            torch.isfinite(layer.weight.grad).all()
            and layer.weight.grad.abs().sum() > 0
        )


def test_duplicate_image_blocks_are_marginalized_for_likelihood():
    m = model()
    a = torch.zeros(1, 1, 1, 32, 32)
    a[..., 1, 1] = 1
    b = torch.zeros_like(a)
    b[..., 2, 2] = 1
    x = torch.cat((a, b, a, b), dim=1)
    scores = torch.full((1, 4, 16), -torch.inf)
    scores[0, 0, 1] = torch.log(torch.tensor(0.2))
    scores[0, 2, 1] = torch.log(torch.tensor(0.3))
    scores[0, 1, 0] = torch.log(torch.tensor(0.5))
    nll = m.block_nll(scores, x, torch.tensor([4]), x[:, :2], torch.tensor([2]))
    torch.testing.assert_close(nll, -torch.tensor([0.5]).log())


def test_padded_source_pixels_cannot_change_a_shorter_prompt():
    m = model()
    x = torch.randint(0, 2, (1, 6, 1, 32, 32)).float()
    a = m.block_distribution(x, torch.tensor([4]))
    x[:, 4:] = 1 - x[:, 4:]
    b = m.block_distribution(x, torch.tensor([4]))
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_native_decode_matches_complete_duplicate_marginal_mode():
    m = model()
    a = torch.zeros(1, 1, 32, 32)
    a[..., 1, 1] = 1
    b = torch.zeros_like(a)
    b[..., 2, 2] = 1
    x = torch.cat((a, b, a, b))
    scores = torch.full((1, 4, 16), -torch.inf)
    scores[0, 0, 1] = torch.log(torch.tensor(0.3))
    scores[0, 2, 1] = torch.log(torch.tensor(0.3))
    scores[0, 1, 0] = torch.log(torch.tensor(0.4))
    m.block_distribution = lambda *_: scores
    result = m.generate_block(x)
    assert torch.equal(result["images"], x[:2])
    assert result["stopped"] and not result["capped"]
    assert abs(result["events"][0]["marginal_block_probability"] - 0.6) < 1e-6
