"""Beam search with one pool per completed prefix of the model's own field plan."""

import torch
from ilm.visual_lm.semantic_page import BLOCK


@torch.inference_mode()
def ordered_copy_beam(
    net, memory, visible, palette, markers, width=8, limit=120, forbidden=()
):
    """Return the best finished beam visiting each supplied copy action exactly once."""
    if (
        not markers
        or len(set(markers)) != len(markers)
        or min(markers) < 1
        or max(markers) >= len(palette)
    ):
        raise ValueError("Expected distinct printable pointer indices")
    if (
        width < 1
        or not len(markers) < limit <= len(net.position) * BLOCK
        or not visible.any(1).all()
    ):
        raise ValueError("Invalid beam, limit or visible rows")
    batch = len(memory)
    states = len(markers) + 1
    beams = states * width
    device = memory.device
    vocab = len(palette)
    embed = net.feedback(palette)
    mem = memory.repeat_interleave(beams, 0)
    vis = visible.repeat_interleave(beams, 0)
    previous = embed.new_zeros((batch * beams, 1, BLOCK, embed.size(1)))
    live = memory.new_full((batch, states, width), -torch.inf, dtype=torch.float32)
    live[:, 0, 0] = 0
    paths = torch.zeros((batch, beams, limit), device=device, dtype=torch.long)
    best = live.new_full((batch,), -torch.inf)
    answer = paths[:, 0].clone()
    lengths = torch.zeros(batch, device=device, dtype=torch.long)
    off = torch.arange(batch, device=device)[:, None] * beams
    rows = torch.arange(batch, device=device)
    for t in range(limit):
        phase = t % BLOCK
        if phase == 0:
            outer = net.context(previous, mem, vis)[:, -1:]
            block = torch.zeros_like(previous[:, :1])
        lp = net.letters(outer, block)[:, 0, phase].log_softmax(-1)
        scores = live[:, :, :, None] + lp.reshape(batch, states, width, vocab)
        terminal, idx = scores[:, -1, :, 0].max(1)
        terminal = terminal / (t + 1) ** 0.7
        improved = terminal > best
        answer = torch.where(
            improved[:, None], paths[rows, (states - 1) * width + idx], answer
        )
        lengths = torch.where(improved, t, lengths)
        best = torch.maximum(best, terminal)
        if t == limit - 1 or (
            t % 8 == 7 and (live.amax((1, 2)) / limit**0.7 <= best).all()
        ):
            break
        advances = [scores[:, i, :, m].clone() for i, m in enumerate(markers)]
        scores[:, :, :, 0] = -torch.inf
        scores[:, :, :, list(markers) + list(forbidden)] = -torch.inf
        new_scores = []
        parents = []
        glyphs = []
        for state in range(states):
            options = scores[:, state].flatten(1)
            if state:
                options = torch.cat((options, advances[state - 1]), 1)
            prob, index = options.topk(width, dim=1)
            prev = state * width + index // vocab
            glyph = index % vocab
            if state:
                advancing = index >= width * vocab
                prev = torch.where(
                    advancing, (state - 1) * width + index - width * vocab, prev
                )
                glyph = torch.where(advancing, markers[state - 1], glyph)
            new_scores.append(prob)
            parents.append(prev)
            glyphs.append(glyph)
        live = torch.stack(new_scores, 1)
        local = torch.cat(parents, 1)
        token = torch.cat(glyphs, 1)
        take = (local + off).flatten()
        paths = paths.gather(1, local[:, :, None].expand(-1, -1, limit))
        paths[:, :, t] = token
        previous = previous[take]
        block = block[take]
        block[:, 0, phase] = embed[token.flatten()]
        if phase == BLOCK - 1:
            previous = torch.cat((previous, block), 1)
        else:
            outer = outer[take]
    if not best.isfinite().all():
        raise RuntimeError("No finite stopped path covers the predicted plan")
    return [s[:n] for s, n in zip(answer.tolist(), lengths.tolist())]


def realize(net, memory, visible, palette, actions, first_id, second_id, limit=120):
    """Preserve already compliant outputs. Repair a missing or disordered copy sequence."""
    result = list(actions)
    for count, required in ((1, [first_id]), (2, [first_id, second_id])):
        indices = [
            i
            for i, (s, has_second) in enumerate(zip(actions, visible[:, 3].tolist()))
            if 1 + has_second == count
            and [c for c in s if c in (first_id, second_id)] != required
        ]
        if indices:
            # On one-field rows the absent second pointer must remain illegal.
            repaired = ordered_copy_beam(
                net,
                memory[indices],
                visible[indices],
                palette,
                required,
                limit=limit,
                forbidden=() if count == 2 else (second_id,),
            )
            for i, seq in zip(indices, repaired):
                result[i] = seq
    return result
