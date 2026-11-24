import math
import torch
from torch.nn import functional as F

from ilm.visual_lm.phrase_region import PhraseRegionWriter, source_runs


def test_teacher_loss_matches_explicit_action_sum_and_gradients():
    torch.manual_seed(29)
    torch.set_num_threads(2)
    net = PhraseRegionWriter(width=12)
    features = torch.randn(12, 256)
    q = torch.tensor([[2, 3, 4, 5, 0], [6, 6, 0, 0, 0]])
    p = torch.tensor([[2, 3, 4, 5, 8, 9], [6, 6, 7, 0, 0, 0]])
    y = torch.tensor([[2, 3, 4, 5, 10, 9, 1], [6, 6, 7, 6, 1, 0, 0]])
    ctx, h = net.encode(features, 10, q, p)
    # Glyph 10 is outside the spelling bank and absent from either source.
    # Use an actual visible out-of-bank glyph in the physical copy branch.
    p[0, 4] = 10
    ctx, h = net.encode(features, 10, q, p)
    actual = net.loss(ctx, h, y)
    src = ctx[2]
    teacher = torch.cat((net.bos.expand(2, 1, -1), ctx[0][y[:, :-1]]), 1)
    states, _ = net.gru(teacher, h)
    past = torch.stack([F.pad(y, (i, 0))[:, : y.size(1)] for i in (1, 2, 3)], -1)
    lp, cp, gate = net.distribution(ctx, states, past)
    sizes = (1, 2, 4, 8, 16)
    prob = net.length_logprobs(
        states, [r.ge(2).all(-1) for r in source_runs(src)]
    ).exp()
    groups = []
    for b in range(len(y)):
        source = src[b].tolist()
        costs = []
        for t, target in enumerate(y[b].tolist()):
            if not target:
                break
            right = y[b, t:].tolist()
            mass = 0.0
            if target <= lp.size(-1):
                mass = (gate[b, t].sigmoid() * lp[b, t, target - 1].exp()).item()
            for k, n in enumerate(sizes):
                legal = [
                    j for j in range(len(source) - n + 1) if min(source[j : j + n]) >= 2
                ]
                hits = [
                    j
                    for j in legal
                    if source[j : j + n] == right[:n] and len(right) >= n
                ]
                if not hits:
                    continue
                m = (
                    (-gate[b, t]).sigmoid()
                    * prob[b, t, k]
                    * cp[b, t, hits].sum()
                    / cp[b, t, legal].sum()
                ).item()
                mass = m if k else mass + m
            costs.append(-math.log(mass))
        groups.append(sum(costs) / len(costs))
    torch.testing.assert_close(actual, torch.tensor(groups))
    actual.mean().backward()
    assert all(
        torch.isfinite(v.grad).all() for v in net.parameters() if v.grad is not None
    )
    assert net.run_length.weight.grad.abs().sum() > 0
