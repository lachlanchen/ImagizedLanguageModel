"""Predict target strip incidence from the visual source and guide first emissions."""
import torch
from torch import nn
from torch.nn import functional as F


class SourceSupport(nn.Module):
    def __init__(self, choices, width=192):
        super().__init__()
        self.local = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
        self.outputs = nn.ModuleList(nn.Linear(width, n) for n in choices)
        for t, layer in enumerate(self.outputs):
            nn.init.normal_(layer.weight, std=.005)
            nn.init.zeros_(layer.bias)
            self.register_buffer(f"prior_{t}", torch.zeros(len(layer.bias)))

    def logits(self, memory, valid, t):
        # The frozen decoder's value field retains one vector per source window.
        values = memory[-1][1].transpose(1, 2).flatten(2)
        local = self.outputs[t](self.local(values))
        maximum = local.masked_fill(~valid[:, 0, 0, :, None], -torch.inf).amax(1)
        return maximum.float() + getattr(self, f"prior_{t}")

    def bonus(self, memory, valid, t):
        logp = F.logsigmoid(self.logits(memory, valid, t))
        change = logp - F.logsigmoid(getattr(self, f"prior_{t}"))
        bonus = .8 * change.clamp(0, 3)
        bonus[:, 0] = 0
        return bonus

    def loss(self, memory, valid, gold, t):
        logits = self.logits(memory, valid, t)
        present = torch.zeros_like(logits).scatter_(1, gold, 1.)
        # Inverse-root frequency gives rare strips weight without changing the
        # population optimum for each individual Bernoulli probability.
        weights = getattr(self, f"prior_{t}")[1:].sigmoid().clamp_min(.0001).rsqrt()
        loss = F.binary_cross_entropy_with_logits(logits[:, 1:], present[:, 1:], reduction="none")
        return (loss * weights).mean() / weights.mean()


@torch.inference_mode()
def write_with_support(body, head, pictured, lengths, tiles, table, t, limit=128):
    if not 1 <= limit <= 128:
        raise ValueError("Supported writing limit is 1–128 cells")
    source, mask = body.encode(pictured, lengths)
    bonus = head.bonus(source, mask, t)
    feedback, keys = body.actions(tiles, table)
    state = body.begin[t].expand(len(lengths), 1, -1)
    past = None
    counts = torch.zeros_like(lengths)
    stopped = torch.zeros_like(lengths, dtype=torch.bool)
    overflow = stopped.clone()
    output = torch.zeros((len(lengths), limit), dtype=torch.long, device=lengths.device)
    offsets = torch.arange(table.size(1), device=lengths.device)
    for step in range(limit + 1):
        latent, past = body.decode(state, source, mask, step, past)
        scores = latent[:,0] @ keys.T * keys.size(-1)**-.5
        choice = (scores + bonus).argmax(-1)
        bonus.scatter_(1, choice[:, None], 0.)
        active = ~(stopped | overflow)
        stopped |= active & choice.eq(0)
        strip = table[choice]
        size = strip.ne(0).sum(-1)
        dest = counts[:, None] + offsets
        row, col = (active[:, None] & strip.ne(0) & dest.lt(limit)).nonzero(as_tuple=True)
        output[row, dest[row,col]] = strip[row,col]
        overflow |= active & (counts + size > limit)
        counts = (counts + active.long() * size).clamp_max(limit)
        if (overflow | stopped).all():
            break
        state = feedback[choice,None]
    return output, counts, stopped
