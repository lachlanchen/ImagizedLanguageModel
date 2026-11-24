"""Propose complete replies from observed word images and the model's learned stop."""

import torch


def copy_groups(variants):
    """Number equal word images locally within each input, retaining original order."""
    ids = variants.flatten(1)
    sorted_ids, order = ids.sort(dim=1, stable=True)
    changes = torch.zeros_like(sorted_ids, dtype=torch.long)
    changes[:, 1:] = sorted_ids[:, 1:] != sorted_ids[:, :-1]
    return torch.empty_like(changes).scatter_(1, order, changes.cumsum(1))


def modal_copy_position(logits, groups, valid):
    """First position of the most probable nonblank emitted image."""
    scores = logits.double().masked_fill(~valid, -torch.inf)
    weights = (scores - scores.amax(dim=1, keepdim=True)).exp()
    masses = torch.zeros_like(weights).scatter_add_(1, groups, weights)
    return masses.gather(1, groups).masked_fill(~valid, -1).argmax(1)


@torch.inference_mode()
def copy_proposals(net, bank, inputs, variants, width=8, limit=32, max_words=None):
    if width < 1 or limit < 1:
        raise ValueError("Search width and list size must be positive")
    maximum = net.max_words if max_words is None else min(net.max_words, max_words)
    if maximum < 1:
        raise ValueError("At least one word is required")
    memory, _ = net.memory(*inputs)
    device = variants.device
    b = variants.size(0)
    ids = variants.flatten(1).long()
    groups = copy_groups(variants)
    n = ids.size(1)
    group_words = torch.zeros_like(ids).scatter_(1, groups, ids)
    expanded = tuple(v.repeat_interleave(width, 0) for v in memory)
    state = expanded[-1][None]
    prev = state.new_zeros(b * width, 1, 50)
    prefix = torch.zeros((b, width, maximum), device=device, dtype=torch.long)
    scores = torch.full((b, width), -torch.inf, device=device, dtype=torch.float64)
    scores[:, 0] = 0
    completed = []
    stops = []
    for depth in range(maximum + 1):
        ch, ptr, _, next_state = net.step(expanded, prev, state)
        channel = ch[:, 0].double().log_softmax(-1).view(b, width, 3)
        if depth:
            completed.append(prefix.clone())
            stops.append((scores + channel[:, :, 0]) / (depth + 1))
        if depth == maximum:
            break
        p = (
            ptr[:, 0]
            .reshape(b, width, n)
            .double()
            .masked_fill(ids[:, None] == 0, -torch.inf)
            .softmax(-1)
        )
        grouped = torch.zeros_like(p).scatter_add_(2, groups[:, None].expand_as(p), p)
        moves = scores[:, :, None] + channel[:, :, 1, None] + grouped.log()
        scores, idx = moves.flatten(1).topk(width, dim=1)
        parents = idx // n
        words = group_words.gather(1, idx % n)
        prefix = prefix.gather(1, parents[:, :, None].expand_as(prefix))
        prefix[:, :, depth] = words
        offset = torch.arange(b, device=device)[:, None] * width
        state = next_state[:, (parents + offset).flatten()].contiguous()
        prev = bank.features[words.flatten(), :50][:, None]
    final = torch.cat(stops, 1)
    paths = torch.cat(completed, 1)
    values, order = final.topk(min(limit, final.size(1)), dim=1)
    paths = paths.gather(1, order[:, :, None].expand(-1, -1, maximum))
    return [
        [
            (tuple(v for v in row if v), s)
            for row, s in zip(rows, ss)
            if s != float("-inf")
        ]
        for rows, ss in zip(paths.tolist(), values.tolist())
    ]
