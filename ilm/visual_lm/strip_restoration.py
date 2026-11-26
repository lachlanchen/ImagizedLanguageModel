"""Conditional parallel generation over pictured strips with a source-predicted length."""
import torch
from torch import nn

from ilm.visual_lm.optical_actions import OpticalActions


def corrupt(gold, displace, rng):
    """Choose identical supervision in both arms. Shuffle only selected physical strips."""
    n, width = gold.shape
    def random(shape):
        return torch.rand(shape, device=gold.device, generator=rng)
    regime = random((n, 1))
    blind = regime < 1/3
    permuted = (regime >= 1/3) & (regime < 7/12)
    rank = random(gold.shape).masked_fill(gold.eq(0), 2)
    selected = ((rank < .25 + .70 * random((n, 1))) | blind | permuted |
                (rank == rank.amin(1, keepdim=True))) & gold.ne(0)
    displaced = selected & ~blind & (permuted | (random(gold.shape) < .65))
    # Sort occupied destination positions spatially and pair with shuffled occupied sources.
    positions = torch.arange(width, device=gold.device).expand(n, -1)
    dest = (positions + (~displaced).long()*width).argsort(1)
    src = (random(gold.shape) + (~displaced).float()*2).argsort(1)
    mixed = torch.zeros_like(gold).scatter_(1, dest, gold.gather(1, src))
    observed = gold.masked_fill(selected, 0)
    if displace:
        observed = torch.where(displaced, mixed, observed)
    return observed, selected


class RestoringWriter(OpticalActions):
    def __init__(self, width=192):
        super().__init__(width)
        self.mask_form = nn.Parameter(torch.randn(width) * .03)
        self.length_head = nn.Sequential(nn.Linear(width+1, width), nn.GELU(), nn.Linear(width, 96))

    def length_scores(self, memory, valid, lengths, direction):
        value = memory[-1][1].transpose(1, 2).flatten(2)
        occupied = valid[:, 0, 0, :, None]
        pooled = (value * occupied).sum(1) / occupied.sum(1)
        return self.length_head(torch.cat((pooled + self.begin[direction], lengths[:, None]/96), 1))

    def position_scores(self, observed, occupied, memory, valid, actions, direction):
        feedback, keys = actions
        x = torch.where(observed.ne(0).unsqueeze(-1), feedback[observed], self.mask_form)
        x = x + self.position[:x.size(1)] + self.begin[direction]
        for layer, source in zip(self.decoder, memory, strict=True):
            norm = layer.norm(x)
            x = x + layer.selfatt(norm, layer.selfatt.keys(norm), occupied[:, None, None, :])
            x = x + layer.cross(layer.cnorm(x), source, valid)
            x = x + layer.ff(layer.ffnorm(x))
        return (self.query(x) @ keys[1:].T) * keys.size(-1)**-.5

    def forward(self, pictured, lengths, observed, occupied, tiles, table, direction):
        memory, valid = self.encode(pictured, lengths)
        logits = self.position_scores(observed, occupied, memory, valid, self.actions(tiles, table), direction)
        return logits, self.length_scores(memory, valid, lengths, direction)

    @torch.inference_mode()
    def write(self, pictured, lengths, tiles, table, direction, limit=128):
        memory, valid = self.encode(pictured, lengths)
        actions = self.actions(tiles, table)
        count = self.length_scores(memory, valid, lengths, direction).argmax(1) + 1
        pos = torch.arange(int(count.max()), device=count.device)[None]
        occupied = pos < count[:, None]
        observed = torch.zeros_like(occupied, dtype=torch.long)
        confidence = torch.full_like(observed, -float('inf'), dtype=torch.float)
        for step in range(1, 7):
            logits = self.position_scores(observed, occupied, memory, valid, actions, direction)
            values, indices = logits.float().log_softmax(-1).max(-1)
            empty = observed.eq(0) & occupied
            observed = torch.where(empty, indices+1, observed)
            confidence = torch.where(empty, values, confidence)
            remaining = count*(6-step)//6
            rank = confidence.masked_fill(~occupied, float('inf')).argsort(1).argsort(1)
            observed = observed.masked_fill(rank < remaining[:, None], 0)
        ordered = table[observed].flatten(1)
        size = ordered.ne(0).sum(1)
        index = ordered.ne(0).cumsum(1)-1
        rows, cols = (ordered.ne(0) & index.lt(limit)).nonzero(as_tuple=True)
        canvas = ordered.new_zeros(len(ordered), limit)
        canvas[rows, index[rows, cols]] = ordered[rows, cols]
        return canvas, size.clamp_max(limit), size.le(limit)
