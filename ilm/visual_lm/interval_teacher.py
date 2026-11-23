"""Disintegrate a distribution over source intervals into copy-position and stop mass."""

import torch
from torch.nn import functional as F


def interval_targets(source, valid, answer, lengths, teacher=None, max_span=48):
    """Return [batch, step, source start] copy masses and [batch, step] stop masses.

    At step k a continuing occurrence starting at j copies source cell j+k.
    The observed answer length excludes the terminal action. Duplicate physical
    occurrences are all retained. No gradients pass through teacher scores.
    """
    steps = answer.shape[1]
    if steps > max_span + 1 or max_span < 1:
        raise ValueError("Answer exceeds the interval support")
    repeat = F.pad(source, (0, max_span), value=-1).unfold(1, max_span + 1, 1)[
        :, :, :steps
    ]
    compatible = torch.cat(
        (
            valid[:, :, None],
            (repeat[:, :, :-1] == answer[:, None, :-1]).cumprod(-1).bool()
            & valid[:, :, None],
        ),
        -1,
    )
    gold = compatible.gather(2, lengths[:, None, None].expand(-1, source.size(1), 1))
    if not gold.any(1).all():
        raise ValueError("Gold answer absent from source page")
    k = torch.arange(steps, device=source.device)
    if teacher is None:
        copies = (gold.float() / gold.sum(1, keepdim=True)) * (
            k[None, None, :] < lengths[:, None, None]
        )
        return copies.transpose(1, 2), (k == lengths[:, None]).float()
    start, end = teacher.float().masked_fill(~valid[:, :, None], -torch.inf).unbind(-1)
    windows = F.pad(end, (0, max_span - 1), value=-torch.inf).unfold(1, max_span, 1)
    suffix = windows.flip([-1]).logcumsumexp(-1).flip([-1])
    more = (
        start[:, :, None] + F.pad(suffix, (0, 1), value=-torch.inf)[:, :, :steps]
    ).masked_fill(~compatible, -torch.inf)
    ends = (
        start[:, :, None] + F.pad(windows, (1, 0), value=-torch.inf)[:, :, :steps]
    ).masked_fill(~compatible, -torch.inf)
    stop = ends.logsumexp(1)
    scale = torch.logaddexp(stop, more.logsumexp(1))
    if not torch.isfinite(scale[k <= lengths[:, None]]).all():
        raise ValueError("A true prefix has no finite teacher mass")
    scale = scale.masked_fill(~torch.isfinite(scale), 0)
    return (more - scale[:, None, :]).exp().transpose(1, 2), (stop - scale).exp()


def path_loss(pointer, gate, stop_logp, source, valid, targets, mask, teacher, weight):
    """Cross entropy over the writer's physical copy positions and terminal action."""
    lengths = mask.sum(1) - 1
    gold_copy, gold_stop = interval_targets(source, valid, targets, lengths)
    if weight:
        soft_copy, soft_stop = interval_targets(
            source, valid, targets, lengths, teacher
        )
        gold_copy = gold_copy.lerp(soft_copy, weight)
        gold_stop = gold_stop.lerp(soft_stop, weight)
    t, p = pointer.shape[1:]
    dest = (
        torch.arange(p, device=source.device)[None, :]
        + torch.arange(t, device=source.device)[:, None]
    ).clamp_max(p - 1)
    copies = (
        pointer.log_softmax(-1).gather(2, dest.expand(len(source), -1, -1))
        + F.logsigmoid(-gate)[:, :, None]
    )
    terminal = stop_logp + F.logsigmoid(gate)
    return -((gold_copy * copies).sum(-1) + gold_stop * terminal)[mask].mean()
