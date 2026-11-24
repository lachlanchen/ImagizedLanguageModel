import numpy as np
import torch
from ilm.visual_lm.sentence_context import ContextWriter, scoring_positions


def test_context_budget_and_visible_prefix():
    for left in range(5):
        for fact in range(1, 6):
            for right in range(5):
                if not left + right:
                    continue
                y = np.arange(left + fact + right + 1) + 2
                y[-1] = 1
                a, p = scoring_positions(y, left, fact)
                assert a[left] == 0 and list(a[:left]) == list(y[:left])
                assert list(a[left + 1 :]) == list(y[left + fact :])
                assert len(p) == len(set(p)) == np.count_nonzero(a)
                assert p[0] == 0 and p[-1] == len(y) - 1


def test_original_loss_and_gradients_are_preserved():
    torch.manual_seed(120)
    net = ContextWriter(16)
    bank = torch.randn(12, 256)
    ctx, h = net.encode(
        bank, 9, torch.tensor([[2, 3], [2, 4]]), torch.tensor([[4, 10], [5, 6]])
    )
    y = torch.tensor([[2, 10, 1, 0], [3, 6, 4, 1]])
    old = net.loss(ctx, h, y).mean()
    empty = torch.empty(0, dtype=torch.long)
    new = net.objective("surround", ctx, h, y, empty, None, None)
    torch.testing.assert_close(old, new)
    for a, b in zip(
        torch.autograd.grad(old, tuple(net.parameters()), retain_graph=True),
        torch.autograd.grad(new, tuple(net.parameters())),
        strict=True,
    ):
        torch.testing.assert_close(a, b)
