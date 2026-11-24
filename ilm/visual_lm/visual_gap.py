"""Complete a gap between two physical word strips without encoding the missing word."""

import torch
from ilm.visual_lm.context_cloze import neighbour_features
from ilm.visual_lm.optical_passage_qa import word_runs


@torch.inference_mode()
def fill(net, left, right=None):
    if net.mode != "ink":
        raise ValueError("This entry reads pictured words")
    words = [
        [x[a:b] for a, b in word_runs(x)] if x is not None else []
        for x in (left, right)
    ]
    before, after = words[0][-2:], words[1][:2]
    observed = before + after
    if not observed:
        raise ValueError("Supply at least one observed word")
    f = net.encode_words(net.reader, None, observed)[:, :50]
    # A missing right field hides two unknown future slots. A supplied field
    # has a known boundary, including when it contains just one word.
    n = len(before) + 1 + (2 if right is None else len(after))
    ink = f.new_zeros(1, n, 50)
    seen = torch.zeros(1, n, device=f.device, dtype=torch.bool)
    i = len(before)
    ink[0, :i] = f[:i]
    seen[0, :i] = True
    if after:
        ink[0, i + 1 :] = f[i:]
        seen[0, i + 1 :] = True
    pos = torch.tensor([[i]], device=f.device)
    local = neighbour_features(ink, seen, pos.new_tensor([n]), pos)
    state = torch.cat((local.new_zeros(1, 1, net.width), local), -1)
    idx = int(net.retrieve_logits(state).argmax())
    return net.prototype_words[idx]


@torch.inference_mode()
def continue_words(net, prefix, count):
    """Greedily emit a fixed number of word images from two previous observations."""
    if prefix.ndim != 3 or prefix.shape[1:] != (2, 50) or count < 1:
        raise ValueError("Expected two observed word features and a positive length")
    batch = len(prefix)
    history = prefix
    emitted = []
    for _ in range(count):
        shown = history.new_ones(batch, 2, 1)
        flags = torch.cat((shown, torch.zeros_like(shown)), -1)
        before = torch.cat((history * 6, flags), -1).flatten(1)
        state = torch.cat(
            (before.new_zeros(batch, net.width), before, before.new_zeros(batch, 104)),
            -1,
        )
        choice = net.retrieve_logits(state).argmax(1)
        emitted.append(choice)
        history = torch.stack((history[:, 1], net.prototype_features[choice, :50]), 1)
    return torch.stack(emitted, 1)
