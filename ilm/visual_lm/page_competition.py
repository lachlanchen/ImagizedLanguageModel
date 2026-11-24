"""Select the highest scoring complete literal reference under a visual path model."""

import numpy as np
import torch


def best_reference(role, count, options):
    """Exact branch and bound. A candidate contains one occurrence per annotated member."""
    begin = role[:, 1] - role[:, 0]
    prefix = np.concatenate(([0], (role[:, 2] - role[:, 0]).cumsum()))
    best, chosen = None, None
    for ref in options:
        choices = []
        for occurrences in ref:
            choices.append(
                sorted(
                    (
                        (int(begin[i] + prefix[j] - prefix[i + 1]), i, j)
                        for i, j in occurrences
                    ),
                    reverse=True,
                )
            )
        # Fewest alternatives first to expose conflicts early.
        choices.sort(key=len)
        bound = [0] * (len(ref) + 1)
        for k in reversed(range(len(ref))):
            bound[k] = bound[k + 1] + choices[k][0][0]
        active = []

        def visit(k, score):
            nonlocal best, chosen
            if best is not None and score + bound[k] <= best:
                return
            if k == len(choices):
                best, chosen = score, sorted(active)
                return
            for gain, i, j in choices[k]:
                if best is not None and score + gain + bound[k + 1] <= best:
                    break
                if all(j <= a or b <= i for a, b in active):
                    active.append((i, j))
                    visit(k + 1, score + gain)
                    active.pop()

        visit(0, int(count[len(ref)]))
    if chosen is None:
        raise ValueError("No feasible placement of any supported reference")
    y = np.zeros(len(role), np.int64)
    for i, j in chosen:
        y[i] = 1
        y[i + 1 : j] = 2
    return y, int(role[:, 0].sum()) + best


def spans(tags):
    out = []
    for i, t in enumerate(tags):
        if t == 1:
            out.append([i, i + 1])
        elif t == 2:
            out[-1][1] = i + 1
    return out


def score_difference(role, count, free, gold, mask):
    """Differentiable difference for two detached, globally selected paths."""
    delta = (role.gather(2, free[..., None]) - role.gather(2, gold[..., None])).squeeze(
        2
    )
    k = (free == 1).sum(1, keepdim=True)
    m = (gold == 1).sum(1, keepdim=True)
    return (delta * mask).sum(1) + (count.gather(1, k) - count.gather(1, m)).flatten()


def mistake_loss(role, count, free, gold, mask, wrong, fixed_gap):
    gap = score_difference(role, count, free, gold, mask)
    # Evaluate at exactly the deployed millilogit scores, take the local subgradient.
    deployed = gap - gap.detach() + fixed_gap
    return (wrong * (1 + deployed).clamp_min(0)).mean()
