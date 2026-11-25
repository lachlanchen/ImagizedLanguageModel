"""Bounded stopped beam search and source reconstruction in physical glyph space."""

import math
import numpy as np
import torch
from torch.nn import functional as F
from scripts.train_ordered_translation import pack


@torch.inference_mode()
def stopped_beams(writer, bank, source, inventory, direction, width=6, limit=128):
    if not 1 <= width < len(inventory) or not 0 < limit <= 128:
        raise ValueError("Invalid beam configuration")
    context, mask = writer.encode(bank, source)
    context = [tuple(v.repeat_interleave(width, 0) for v in layer) for layer in context]
    mask = mask.repeat_interleave(width, 0)
    size = len(source)
    device = source.device
    x = writer.begin[direction].expand(size * width, 1, -1)
    words = writer.prototypes(bank, inventory)
    active = bank.new_full((size, width), -torch.inf)
    active[:, 0] = 0
    prefix = torch.zeros((size, width, limit), device=device, dtype=torch.long)
    finals = active.new_full(active.shape, -torch.inf)
    images = prefix.clone()
    past = None
    base = torch.arange(size, device=device)[:, None] * width

    def select(seq, idx):
        return seq.gather(1, idx[:, :, None].expand(-1, -1, seq.shape[2]))

    for step in range(limit + 1):
        h, past = writer.decode(x, context, mask, past, step)
        logp = ((h[:, 0] @ words.T).float() / math.sqrt(h.size(-1))).log_softmax(-1)
        scores = active[:, :, None] + logp.reshape(size, width, -1)
        # Search length normalisation is fixed; all final comparisons use the same candidates.
        pool = torch.cat((finals, scores[:, :, 0] / ((step + 1) ** 0.8)), 1)
        finals, which = pool.topk(width, 1)
        images = select(torch.cat((images, prefix), 1), which)
        if step == limit:
            break
        scores[:, :, 0] = -torch.inf
        active, draw = scores.flatten(1).topk(width, 1)
        previous = draw // len(inventory)
        next_cell = inventory[draw % len(inventory)]
        prefix = select(prefix, previous)
        prefix[:, :, step] = next_cell
        indices = (base + previous).flatten()
        past = [tuple(v[indices] for v in layer) for layer in past]
        x = bank[next_cell.flatten()][:, None, :]
        if (active[:, 0] / ((limit + 1) ** 0.8) <= finals[:, -1]).all():
            break
    return [
        [list(filter(None, v)) for v, sc in zip(group, s) if np.isfinite(sc)]
        for group, s in zip(images.cpu().tolist(), finals.cpu().tolist())
    ]


@torch.inference_mode()
def sequence_scores(writer, features, sources, answers, inventory, direction):
    """Mean log probability per output cell including stop. Empty source is undefined."""
    out = [-math.inf] * len(sources)
    local = torch.full((len(features),), -100, device=features.device)
    local[inventory] = torch.arange(len(inventory), device=features.device)
    idx = [i for i, x in enumerate(sources) if x]
    for start in range(0, len(idx), 24):
        batch = idx[start : start + 24]
        x = pack([sources[i] for i in batch], features.device)
        y = pack([answers[i] + [1] for i in batch], features.device)
        logits = writer(features, x, y, inventory, direction)
        labels = local[y]
        # An impossible visible source cell means zero reconstruction probability.
        supported = ((y == 0) | (labels != -100)).all(1)
        loss = F.cross_entropy(
            logits.float().transpose(1, 2), labels, reduction="none", ignore_index=-100
        )
        values = (-loss.sum(1) / (y != 0).sum(1)).masked_fill(~supported, -torch.inf)
        for i, s in zip(batch, values.tolist(), strict=True):
            out[i] = s
    return out


@torch.inference_mode()
def rerank(writer, features, sources, inventories, direction, greedy):
    """Return forward, round-trip, and all stopped candidates without reference answers."""
    bank = writer.optical(features)
    groups = stopped_beams(
        writer, bank, pack(sources, features.device), inventories[direction], direction
    )
    original = []
    proposed = []
    for x, opts, (g, done) in zip(sources, groups, greedy, strict=True):
        if done and g not in opts:
            opts.append(g)
        original.extend([x] * len(opts))
        proposed.extend(opts)
    f = sequence_scores(
        writer, features, original, proposed, inventories[direction], direction
    )
    r = sequence_scores(
        writer, features, proposed, original, inventories[1 - direction], 1 - direction
    )
    result = []
    offset = 0
    for opts in groups:
        n = len(opts)
        forward = f[offset : offset + n]
        reverse = r[offset : offset + n]
        offset += n
        combined = [a + 0.5 * b for a, b in zip(forward, reverse)]
        if not any(math.isfinite(v) for v in combined):
            combined = forward
        result.append(
            dict(
                forward=opts[int(np.argmax(forward))],
                roundtrip=opts[int(np.argmax(combined))],
                candidates=opts,
                forward_scores=forward,
                reverse_scores=[v if math.isfinite(v) else None for v in reverse],
            )
        )
    return result
