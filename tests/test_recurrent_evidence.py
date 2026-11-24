import torch
from ilm.visual_lm.recurrent_evidence import (
    PathMass,
    RUNS,
    run_matches,
    ComposingWriter,
    previous_marks,
)


def test_path_mass_and_gradient_equal_enumerated_censored_paths():
    torch.manual_seed(141)
    arcs = torch.randn(2, 5, 5, requires_grad=True)
    counts = torch.tensor([5, 3])

    def enum(i, j):
        if j >= counts[i]:
            return [arcs.new_zeros(())]
        return [
            arcs[i, j, k] + rest
            for k, length in enumerate((1, *RUNS))
            for rest in enum(i, j + length)
        ]

    expected = torch.stack([torch.stack(enum(i, 0)).logsumexp(0) for i in range(2)])
    got = PathMass.apply(arcs, counts)
    assert torch.allclose(got, expected, atol=1e-6)
    assert torch.allclose(
        torch.autograd.grad(got.sum(), arcs)[0],
        torch.autograd.grad(expected.sum(), arcs)[0],
        atol=1e-6,
    )


def test_copy_cannot_emit_stop_but_can_cross_a_censored_boundary():
    s = torch.tensor([[3, 4, 5, 6]] * 2)
    y = torch.tensor([[3, 4, 1], [3, 4, 0]])
    matches = list(run_matches(s, y))
    assert matches[1][:, 0, 0].all()
    assert not matches[2][0, 0, 0] and matches[2][1, 0, 0]


def test_run_distribution_has_unit_mass_for_short_and_long_pages():
    torch.manual_seed(14)
    torch.set_num_threads(1)
    net = ComposingWriter(width=8)
    q = torch.tensor([[3, 4], [2, 0]])
    s = torch.tensor([[3, 4, 3, 0], [4, 0, 0, 0]])
    y = torch.tensor([[3, 4], [4, 1]])
    bank = torch.randn(5, 8)
    page = net.page(bank, q, s)
    _, spell, runs = net.actions(bank[y], previous_marks(y), s, page)
    assert torch.allclose(
        spell.exp() + sum(v.exp().sum(-1) for v in runs),
        torch.ones_like(spell),
        atol=1e-6,
    )


def test_physical_decoder_agrees_with_full_training_density():
    from ilm.visual_lm.discourse_writer import DiscourseWriter
    from scripts.write_recurrent_evidence import generate
    from scripts.train_visual_cursor import base_batch

    torch.set_num_threads(1)
    torch.manual_seed(180)
    net = ComposingWriter(width=16).eval()
    base = DiscourseWriter(width=32, depth=1, shape_dim=16).eval()
    x = torch.randn(12, 256)
    q = torch.tensor([[2, 3], [3, 6]])
    s = torch.tensor([[2, 3, 2, 3, 4, 5, 5, 4], [4, 5, 2, 3, 3, 2, 4, 5]])
    y = torch.tensor([[2, 3, 4, 5, 1], [3, 6, 2, 4, 5]])
    ctx = torch.cat((s, q), 1).tolist()
    ids, pos = map(torch.as_tensor, base_batch(ctx, y.tolist()))
    lp = base.log_prob(base.encode(x), ids, 11, pos)
    expected, _ = net.observed(x, q, s, y, lp, 11)
    actual = generate(base, net, x, 11, q.tolist(), s.tolist(), ctx, 5, forced=y).sum(1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)
