import math

import torch

from ilm.visual_lm.exact_raster_density_v54 import ExactRasterDensityConfig
from ilm.visual_lm.visual_copy_language import (
    VisualCopyLanguage,
    VisualCopyReaderConfig,
    mixture_log_probability,
    raster_equality,
)


def small_model():
    torch.manual_seed(71)
    return VisualCopyLanguage(
        VisualCopyReaderConfig(layers=1),
        ExactRasterDensityConfig(width=128, layers=1, heads=2, pixel_hidden=64),
    ).eval()


def test_probability_normalized_and_duplicate_mass():
    # Three possible images; the first occurs twice in the observed prompt.
    attention = torch.tensor([[0.2, 0.3, 0.5]]).log()
    equality = torch.tensor(
        [[True, True, False], [False, False, True], [False, False, False]]
    )
    free = torch.tensor([0.1, 0.2, 0.7])
    logp = mixture_log_probability(
        torch.full((3,), math.log(0.6 / 0.4)),
        attention.expand(3, -1),
        equality,
        -free.log(),
    )
    expected = 0.6 * torch.tensor([0.5, 0.5, 0]) + 0.4 * free
    torch.testing.assert_close(logp.exp(), expected)
    torch.testing.assert_close(logp.exp().sum(), torch.tensor(1.0))
    stop = 0.27
    assert abs(float((1 - stop) * logp.exp().sum() + stop) - 1) < 1e-7
    gate = torch.tensor([0.0], requires_grad=True)
    att = torch.zeros(1, 3, requires_grad=True).log_softmax(-1)
    value = mixture_log_probability(
        gate, att, torch.zeros(1, 3, dtype=torch.bool), torch.ones(1)
    )
    value.sum().backward()
    assert torch.isfinite(gate.grad).all()
    masked_attention = torch.tensor([[0.0, -torch.inf, -torch.inf]], requires_grad=True)
    hidden_match = torch.tensor([[False, True, False]])
    density = torch.ones(1, requires_grad=True)
    value = mixture_log_probability(
        torch.zeros(1), masked_attention, hidden_match, density
    )
    value.sum().backward()
    assert (
        torch.isfinite(masked_attention.grad).all()
        and torch.isfinite(density.grad).all()
    )


def test_exact_equality_on_arbitrary_unencoded_forms():
    x = torch.randint(0, 2, (2, 4, 1, 32, 32)).float()
    sources = torch.cat((x, x[:, :1]), 1)
    e = raster_equality(x, sources)
    assert e[:, 0, -1].all() and e[:, range(4), range(4)].all()
    sources[0, 0, 0, 5, 9] = 1 - sources[0, 0, 0, 5, 9]
    assert not raster_equality(x, sources)[0, 0, 0]


def test_future_pixels_and_padding_cannot_change_earlier_query():
    m = small_model()
    x = torch.randint(0, 2, (1, 7, 1, 32, 32)).float()
    first = m.visual_condition(x, torch.tensor([4]), torch.tensor([[3, 4]]))
    y = x.clone()
    y[:, 5:] = 1 - y[:, 5:]
    second = m.visual_condition(y, torch.tensor([4]), torch.tensor([[3, 4]]))
    for key in ("hidden", "fields", "attention_logp", "copy_logits", "stop_logits"):
        torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
    # For natural-stream queries, even a longer packed prompt cannot expose future keys.
    c = m.visual_condition(x, torch.tensor([7]), torch.tensor([[2]]))
    assert torch.isneginf(c["attention_logp"][..., 3:]).all()


def test_exact_event_law_has_finite_reader_and_head_gradients():
    m = small_model()
    x = torch.randint(0, 2, (1, 5, 1, 32, 32)).float()
    c = m.visual_condition(x, torch.tensor([3]), torch.tensor([[2, 3]]))
    y = x[:, [0, 1]]
    loss = m.event_nll(
        c, y, torch.tensor([[True, False]]), torch.tensor([[False, True]])
    ).sum()
    loss.backward()
    for layer in (
        m.reader.input_projection,
        m.copy_query,
        m.copy_key,
        m.copy_gate,
        m.stop_head,
    ):
        g = layer.weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_native_copy_and_learned_stop_need_no_reference_length():
    m = small_model()
    with torch.no_grad():
        m.copy_gate.bias.fill_(30)
        m.copy_query.weight.zero_()
        m.copy_key.weight.zero_()
        m.stop_head.bias.fill_(-30)
    p = torch.randint(0, 2, (2, 1, 32, 32)).float()
    p = torch.cat((p, p[:1]), 0)
    result = m.generate(p, maximum_output_cells=2)
    assert result["capped"] and not result["stopped"]
    assert torch.equal(result["images"], p[:1].expand(2, -1, -1, -1))
    with torch.no_grad():
        m.stop_head.bias.fill_(30)
    result = m.generate(p)
    assert result["stopped"] and len(result["images"]) == 0
