import torch

from ilm.visual_lm.frozen_visual_foundation import (
    FrozenFeatureBlockHead,
    even_cells,
    native_strip,
    semi_orthogonal,
)


def test_native_strip_order_sign_and_odd_padding():
    x = torch.zeros(3, 1, 32, 32)
    x[0, 0, 1, 2] = 1
    x[1, 0, 4, 7] = 1
    x[2, 0, 9, 10] = 1
    y = native_strip(x)
    assert y.shape == (1, 1, 32, 128)
    assert y[0, 0, 1, 2] == -1 and y[0, 0, 4, 39] == -1 and y[0, 0, 9, 74] == -1
    assert (y[:, :, :, 96:] == 1).all()
    assert torch.equal(even_cells(x)[:3], x)


def test_fixed_projection_has_orthonormal_columns():
    q = semi_orthogonal(20, 8, 17)
    assert torch.equal(q, semi_orthogonal(20, 8, 17))
    torch.testing.assert_close(q.T @ q, torch.eye(8), atol=1e-6, rtol=1e-6)


def test_readout_law_duplicate_mass_and_finite_gradient():
    torch.manual_seed(59)
    h = FrozenFeatureBlockHead()
    x = torch.zeros(1, 4, 1, 32, 32)
    x[:, [0, 2], 0, 1, 1] = 1
    x[:, [1, 3], 0, 2, 2] = 1
    f = torch.randn(1, 4, 514)
    logp = h(f, torch.tensor([4]))
    torch.testing.assert_close(logp.exp().flatten(1).sum(-1), torch.ones(1))
    nll = h.block_nll(logp, x, torch.tensor([4]), x[:, :2], torch.tensor([2]))
    expected = -torch.logaddexp(logp[0, 0, 1], logp[0, 2, 1])
    torch.testing.assert_close(nll[0], expected)
    nll.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in h.parameters())
