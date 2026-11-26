"""A bilingual image encoder and autoregressive writer over finite visual strips."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.ordered_translation import Block


def pool_tiles(ink):
    """Preserve ordered cells; ink is one and paper is zero."""
    return F.avg_pool2d(ink.reshape(-1, 1, 32, 32).float(), 2).reshape(*ink.shape[:-2], 256)


def strip_indices(strips, device):
    table = torch.zeros((len(strips), 8), dtype=torch.long, device=device)
    for i, strip in enumerate(strips):
        if strip:
            table[i, :len(strip)] = torch.tensor(strip, device="cpu")
    return table


class OpticalActions(nn.Module):
    def __init__(self, width=192):
        super().__init__()
        self.eye = nn.Sequential(nn.Linear(256, width), nn.LayerNorm(width), nn.GELU(),
                                 nn.Linear(width, width))
        self.pair = nn.Linear(2 * width, width)
        self.encoder = nn.ModuleList(Block(width) for _ in range(3))
        self.decoder = nn.ModuleList(Block(width, True) for _ in range(3))
        self.compose = nn.Linear(8 * width, width)
        self.key = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
        self.query = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
        self.feedback = nn.LayerNorm(width)
        self.stop = nn.Parameter(torch.randn(1, width) * .03)
        self.begin = nn.Parameter(torch.randn(2, width) * .03)
        phase = torch.arange(130)[:, None] * torch.exp(
            -math.log(10000) * torch.arange(0, width, 2) / width)
        self.register_buffer("position", torch.stack((phase.sin(), phase.cos()), -1).flatten(1),
                             persistent=False)

    def actions(self, tiles, table):
        pictured = self.eye(tiles)[table] * table.ne(0).unsqueeze(-1)
        meaning = self.compose(pictured.flatten(1))
        forms = torch.cat((self.stop, meaning[1:]), 0)
        return self.feedback(forms), self.key(forms)

    def encode(self, pictured, lengths):
        cells = self.eye(pictured)
        valid = torch.arange(cells.size(1), device=cells.device)[None] < lengths[:, None]
        cells = cells * valid.unsqueeze(-1)
        if cells.size(1) % 2:
            cells = F.pad(cells, (0, 0, 0, 1))
        x = self.pair(cells.flatten(1).reshape(cells.size(0), -1, self.pair.in_features))
        x = x + self.position[:x.size(1)]
        mask = valid[:, None, None, ::2]
        for layer in self.encoder:
            x, _ = layer(x, mask)
        return [layer.cross.keys(x) for layer in self.decoder], mask

    def decode(self, x, source, valid, offset=0, past=None):
        x = x + self.position[offset:offset + x.size(1)]
        cache = []
        for i, layer in enumerate(self.decoder):
            x, kv = layer(x, valid, source[i], None if past is None else past[i])
            cache.append(kv)
        return self.query(x), cache

    def forward(self, pictured, lengths, target, tiles, table, direction):
        source, valid = self.encode(pictured, lengths)
        history, keys = self.actions(tiles, table)
        begin = self.begin[direction].expand(target.size(0), 1, -1)
        x = torch.cat((begin, history[target]), 1)
        latent, _ = self.decode(x, source, valid)
        return latent @ keys.T * (keys.size(-1) ** -.5)

    @torch.inference_mode()
    def write(self, pictured, lengths, tiles, table, direction, limit=128):
        """Return an ordered canvas of physical tile addresses plus observed stops."""
        if not 1 <= limit <= 128:
            raise ValueError("The supported output limit is 1–128 glyph cells")
        source, valid = self.encode(pictured, lengths)
        history, keys = self.actions(tiles, table)
        x = self.begin[direction].expand(pictured.size(0), 1, -1)
        past = None
        out = torch.zeros((len(x), limit), dtype=torch.long, device=x.device)
        n = torch.zeros_like(lengths)
        stop = torch.zeros_like(lengths, dtype=torch.bool)
        overflow = stop.clone()
        offsets = torch.arange(table.size(1), device=x.device)
        for step in range(limit + 1):
            latent, past = self.decode(x, source, valid, step, past)
            choice = (latent[:, 0] @ keys.T).argmax(-1)
            strip = table[choice]
            count = strip.ne(0).sum(-1)
            active = ~(overflow | stop)
            stop |= active & choice.eq(0)
            dest = n[:, None] + offsets
            take = active[:, None] & strip.ne(0) & dest.lt(limit)
            row, col = take.nonzero(as_tuple=True)
            out[row, dest[row, col]] = strip[row, col]
            overflow |= active & (n + count > limit)
            n = (n + active.long() * count).clamp_max(limit)
            if (stop | overflow).all():
                break
            x = history[choice, None]
        return out, n, stop
