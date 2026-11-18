import copy
import math

import torch

from ilm.visual_lm.continuous_autoregressive_language import (
    ContinuousDensityConfig, ContinuousWritingDensity, ContinuousImageLanguage,
    mixture_log_prob,
)
from ilm.visual_lm.continuous_glyph_codec import ContinuousGlyphCodec, ContinuousGlyphCodecConfig


torch.set_num_threads(2)


def test_normalized_group_mixture_matches_independent_distribution():
    torch.manual_seed(4)
    target = torch.randn(3, 2, dtype=torch.float64)
    means = torch.randn(3, 4, 2, dtype=torch.float64)
    scales = torch.rand_like(means) + .1
    logits = torch.randn(3, 4, dtype=torch.float64)
    dist = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(logits=logits),
        torch.distributions.Independent(torch.distributions.Normal(means, scales), 1))
    assert torch.allclose(mixture_log_prob(target, logits, means, scales), dist.log_prob(target), atol=1e-12)
    # A shared mixture puts mass on the two complete modes, not their average.
    logits = torch.zeros(1, 2, dtype=torch.float64)
    means = torch.tensor([[[-2., -2.], [2., 2.]]], dtype=torch.float64)
    scales = torch.full_like(means, .05)
    mode = mixture_log_prob(torch.tensor([[-2., -2.]]), logits, means, scales)
    midpoint = mixture_log_prob(torch.zeros(1, 2), logits, means, scales)
    assert float(mode - midpoint) > 1000


def test_local_cache_and_future_target_exclusion():
    torch.manual_seed(5)
    model = ContinuousWritingDensity(12, ContinuousDensityConfig(
        latent_width=32, group_width=4, local_width=32, local_layers=2, local_heads=4)).eval()
    states, target = torch.randn(2, 12), torch.randn(2, 32)
    full = model(states, target)
    groups = target.reshape(2, 8, 4)
    features, cache = model.condition(states), None
    for i in range(8):
        step, cache = model.step(features, None if i == 0 else groups[:, i - 1], i, cache)
        for a, b in zip(step, full):
            assert torch.allclose(a, b[:, i], atol=1e-6, rtol=1e-6)
    changed = target.clone()
    changed[:, 12:] += 30
    other = model(states, changed)
    for a, b in zip(full, other):
        assert torch.equal(a[:, :4], b[:, :4])
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = -model.log_prob(states, target).mean()
    loss.backward()
    assert float(model.condition.weight.grad.norm()) > 0
    assert float(model.blocks[0].qkv.weight.grad.norm()) > 0
    optimizer.step()
    assert not torch.equal(model.output.weight, initial['output.weight'])
    model.load_state_dict(initial)
    assert all(torch.equal(v, initial[k]) for k, v in model.state_dict().items())
    a = model.generate(states, stochastic=True, generator=torch.Generator().manual_seed(11))
    b = model.generate(states, stochastic=True, generator=torch.Generator().manual_seed(11))
    c = model.generate(states, stochastic=True, generator=torch.Generator().manual_seed(12))
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_native_history_causality_frozen_codec_and_actual_pixel_feedback():
    torch.manual_seed(6)
    codec = ContinuousGlyphCodec(ContinuousGlyphCodecConfig())
    model = ContinuousImageLanguage(codec, model_dim=128, layers=2, heads=4).eval()
    ink = torch.randint(0, 2, (4, 1, 32, 32), dtype=torch.uint8)
    other = ink.clone()
    other[2:] ^= 1
    assert torch.equal(model.history(ink)[:, :2], model.history(other)[:, :2])
    terms = model.training_terms(ink, 2)
    (terms['code_nll'] + terms['stop_bce']).backward()
    assert terms['target_positions'] == 2 and terms['stop_positions'] == 3
    assert model.training_terms(ink, 0)['target_positions'] == 3
    assert all(p.grad is None for p in codec.parameters())
    assert float(model.input_projection.weight.grad.norm()) > 0
    assert float(model.stop.weight.grad.norm()) > 0
    calls = []
    hook = codec.encoder.register_forward_pre_hook(lambda _m, inputs: calls.append(inputs[0].detach().clone()))
    with torch.no_grad():
        model.stop.weight.zero_()
        model.stop.bias.fill_(-10)
    emitted, receipt = model.generate(ink, maximum_new=2)
    hook.remove()
    assert receipt['capped'] and not receipt['stopped'] and emitted.shape == (2, 1, 32, 32)
    # The encoder normalizes white pixels to [-1,1]. Its second input contains
    # the actual first emitted image, not the predicted latent vector.
    assert len(calls) == 2
    assert torch.equal(calls[1][-1], 1 - 2 * emitted[0].float())
    assert bool(((emitted == 0) | (emitted == 1)).all())
