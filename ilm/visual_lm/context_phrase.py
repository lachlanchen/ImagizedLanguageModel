"""Score pictured phrase continuations with the visual writer's existing exemplar head."""

import torch
from torch.nn import functional as F


def phrase_scores(
    model, features, bank_ids, sources, lengths, targets, sizes, prefix_sizes
):
    """Return mean log similarity-softmax per supplied suffix. Inputs contain image-bank indices.

    A visible cell equal to the next planned anchor advances the forced cursor.
    The free writer still makes its own close, copy, retrieve and draw decisions.
    """
    mem, _, mask, h = model.read(features[sources], lengths)
    history = torch.cat(
        (model.start.expand(len(mem), 1, -1), model.embed(features[targets[:, :-1]])), 1
    )
    q, _ = model.history(history, h)
    with torch.no_grad():
        b = torch.arange(len(mem), device=mem.device)
        pos = torch.arange(mem.size(1), device=mem.device)
        plan = (model.keep(mem).squeeze(-1) >= 0) & ~mask
        following = (
            torch.where(plan, pos, lengths[:, None])
            .flip((1,))
            .cummin(1)
            .values.flip((1,))
        )
        following = torch.cat((following, lengths[:, None]), 1)
        left = torch.full_like(lengths, -1)
        offset = torch.zeros_like(left)
        states = []
        for t in range(targets.size(1)):
            right = following[b, left + 1]
            states.append(torch.stack((left, right, offset), 1))
            match = (right < lengths) & (
                sources[b, right.clamp_max(mem.size(1) - 1)] == targets[:, t]
            )
            left = torch.where(match, right, left)
            offset = torch.where(match, 0, offset + 1)
        lo, hi, off = torch.stack(states, 1).unbind(-1)
    z = model.boundary_state(q, mem, model.prefix_sum(mem), lengths, lo, hi, off)
    t = torch.arange(q.size(1), device=q.device)
    active = (t >= prefix_sizes[:, None]) & (t < sizes[:, None])
    keys = F.normalize(model.prototype_key(model.embed(features[bank_ids])), dim=-1)
    wanted = targets[active]
    index = torch.searchsorted(bank_ids, wanted)
    logits = model.prototype_scores(z[active], keys).float()
    values = -F.cross_entropy(logits, index, reduction="none")
    summed = values.new_zeros(len(mem)).index_add(0, active.nonzero()[:, 0], values)
    return summed / (sizes - prefix_sizes)
