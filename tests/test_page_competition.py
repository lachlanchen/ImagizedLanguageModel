from itertools import product
import numpy as np
import torch
from ilm.visual_lm.page_competition import best_reference, mistake_loss, spans


def test_global_reference_repeated_overlapping_and_alternative_placements():
    rng = np.random.default_rng(7301)
    options = [[[[0, 2], [2, 4]], [[1, 3], [3, 5]]], [[[1, 2], [4, 5]]]]
    for _ in range(50):
        r, c = rng.integers(-8, 10, (5, 3)), rng.integers(-8, 10, 4)
        y, s = best_reference(r, c, options)
        candidates = []
        for ref in options:
            for paths in product(*ref):
                t = np.zeros(5, int)
                for a, b in sorted(paths):
                    if t[a:b].any():
                        break
                    t[a:b] = 2
                    t[a] = 1
                else:
                    candidates.append(
                        (sum(r[i, x] for i, x in enumerate(t)) + c[len(ref)], spans(t))
                    )
        assert s == max(v for v, p in candidates) and (s, spans(y)) in candidates


def test_quantized_gap_and_path_gradients():
    role = torch.randn(2, 4, 3, requires_grad=True)
    c = torch.randn(2, 4, requires_grad=True)
    f = torch.tensor([[1, 2, 1, 0], [1, 2, 0, 0]])
    g = torch.tensor([[0, 1, 0, 0], [0, 0, 1, 0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    loss = mistake_loss(
        role, c, f, g, mask, torch.tensor([1.0, 0.0]), torch.tensor([1.5, 10.0])
    )
    loss.backward()
    assert (
        loss.item() == 1.25 and role.grad[0, 0, 1] == 0.5 and role.grad[0, 0, 0] == -0.5
    )
    assert not role.grad[1].any() and not role.grad[0, 3].any()
    assert c.grad[0, 2] == 0.5 and c.grad[0, 1] == -0.5
