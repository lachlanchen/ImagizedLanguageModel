from itertools import combinations
import numpy as np
import torch
from ilm.visual_lm.complete_fact_sets import source_runs, competing_margins


def test_whole_path_differences_and_gradients():
    torch.manual_seed(42)
    for tags, offset in (
        ([1, 2, 1, 2, 0, 0, 0, 0, 0], 0),
        ([0, 1, 2, 0, 1, 0, 0, 0, 0], 1),
        ([1, 0, 0, 0], 0),
        ([1, 2, 2], 0),
    ):
        n = len(tags)
        width = n + 2
        page = np.arange(n, dtype=np.int32)
        a, b, res = source_runs(page, np.array(tags))
        runs = list(zip(a.tolist(), b.tolist()))

        def row(x, padval=0):
            return torch.tensor([list(x) + [padval] * 2])

        keep = row([True] * n, False)
        u = torch.randn(1, width, 3, dtype=torch.float64, requires_grad=True)
        c = torch.randn(1, 17, dtype=torch.float64, requires_grad=True)
        margins, valid = competing_margins(
            u, c, keep, row(a), row(b), row(res, False), torch.tensor([offset])
        )

        def score(paths):
            y = torch.zeros(n, dtype=torch.long)
            for i, j in paths:
                y[i] = 1
                y[i + 1 : j] = 2
            return u[0, torch.arange(n), y].sum() + c[0, len(paths)]

        gold = score(runs)
        chosen = runs[offset]
        length = chosen[1] - chosen[0]
        others = [
            (i, i + length)
            for i in range(n - length + 1)
            if not res[i : i + length].any()
        ]
        fam = [
            [s for m in range(len(runs)) for s in combinations(runs, m)],
            [[r for r in runs if r != chosen] + [o] for o in others],
            [runs + [o] for o in others],
        ]
        obj = []
        for j, paths in enumerate(fam):
            assert valid[0, j].item() == bool(paths)
            if paths:
                obj.append(gold - torch.stack([score(p) for p in paths]).max())
        x = margins[valid]
        ref = torch.stack(obj)
        torch.testing.assert_close(x, ref)
        for actual, expected in zip(
            torch.autograd.grad(x.sum(), (u, c), retain_graph=True),
            torch.autograd.grad(ref.sum(), (u, c)),
        ):
            torch.testing.assert_close(actual, expected)


def test_repeated_and_overlapping_copies_are_reserved():
    a, b, res = source_runs(
        np.array([7, 7, 7, 3, 7, 7], np.int32), np.array([1, 2, 0, 0, 0, 0])
    )
    assert a.tolist() == [0] and b.tolist() == [2]
    assert res.tolist() == [True, True, True, False, True, True]
