"""Matched continuous writers with or without frozen source descriptors."""
import torch
from torch import nn
from torch.nn import functional as F

from .parallel_canvas import ParallelCanvasWriter


class SourceCanvasWriter(nn.Module):
    def __init__(self, white_code):
        super().__init__()
        self.canvas = ParallelCanvasWriter(white_code)
        self.geometry = nn.Linear(4, self.canvas.config.width)
        self.types = nn.Parameter(torch.randn(2, self.canvas.config.width) * .02)

    @property
    def device(self):
        return self.types.device

    def forward(self, summary, visual, geometry, lengths, *, source):
        if (summary.ndim != 3 or summary.shape[1:] != (52, 4096)
                or visual.ndim != 3 or visual.shape[0] != len(summary)
                or visual.shape[2] != 4096 or geometry.shape != (*visual.shape[:2], 4)
                or lengths.shape != (len(summary),)
                or bool(((lengths < 1) | (lengths > visual.shape[1])).any())):
            raise ValueError("complete prompt-only summary/source/geometry required")
        c = self.canvas
        memory = c.memory(summary.float()) + self.types[0]
        mask = None
        if source:
            v = c.memory(visual.float()) + self.geometry(geometry.float()) + self.types[1]
            memory = torch.cat((memory, v), 1)
            padding = torch.arange(visual.shape[1], device=self.device)[None] >= lengths[:, None]
            mask = torch.cat((torch.zeros(len(summary), 52, dtype=torch.bool, device=self.device), padding), 1)
        x = c.queries.expand(len(summary), -1, -1)
        for layer in c.layers:
            x = layer(x, memory, memory_key_padding_mask=mask)
        residual = c.output(c.norm(x))
        return F.layer_norm(c.white_code + residual, (768,), eps=c.config.epsilon)

    def loss(self, inputs, target, tile_lengths, *, source):
        z = self(*inputs, source=source)
        error = (z-target).square().mean(-1)
        active = torch.arange(z.shape[1], device=self.device)[None] < tile_lengths[:, None]
        a = (error*active).sum(-1)/active.sum(-1)
        count = (~active).sum(-1)
        p = (error*~active).sum(-1)/count.clamp_min(1)
        return torch.where(count > 0, .5*(a+p), a).mean()

    @torch.no_grad()
    def generate(self, inputs, codec, *, source):
        z = self(*inputs, source=source)
        n, slots, _ = z.shape
        tiles = (codec.decode(z.reshape(-1,768)) >= 0).to(torch.uint8).mul(255)
        image = tiles.reshape(n,slots,32,32).permute(0,2,1,3).reshape(n,32,slots*32)
        return image.cpu(), z.cpu()
