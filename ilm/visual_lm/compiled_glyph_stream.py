"""Compile the frozen glyph motor's finite image and feedback functions for inference."""
from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class GlyphTable:
    pixels: torch.Tensor
    feedback: torch.Tensor
    keys: torch.Tensor
    direction: int


class CompiledGlyphStream:
    """A deployment snapshot. Keep the model in evaluation mode with immutable weights."""
    def __init__(self, model, directions=(0,1)):
        self.model = model.eval().requires_grad_(False)
        self.tables = {direction:self._compile(direction) for direction in directions}

    @torch.inference_mode()
    def _compile(self, direction):
        if direction not in (0,1):
            raise ValueError("Unknown writing direction")
        model = self.model
        codes = getattr(model, f"codes_{direction}")
        images, vectors = [], []
        for batch in codes.split(256):
            ink = model.motor(batch).gt(0).reshape(-1,32,32)
            images.append(ink)
            vectors.append(model.writer.optical(F.avg_pool2d(ink[:,None].float(),2).flatten(1)))
        keys = torch.cat((model.writer.end[None], model.writer.target(codes)))
        return GlyphTable(torch.cat(images), torch.cat(vectors), keys, direction)

    @torch.inference_mode()
    def write(self, tiles, lengths, direction, limit=128):
        table = self.tables[direction]
        ctx, valid = self.model.scene(tiles,lengths,direction)
        writer = self.model.writer
        x = writer.begin[direction].expand(len(tiles),1,-1)
        past = None
        finished = torch.zeros(len(tiles),dtype=torch.bool,device=tiles.device)
        sizes = torch.zeros_like(finished,dtype=torch.long)
        outputs = []
        for i in range(limit+1):
            h, past = writer.decode(x,ctx,valid,past,i)
            decision = (h[:,0] @ table.keys.T).argmax(-1)
            finished = finished | decision.eq(0)
            if i==limit or finished.all():
                break
            index = (decision-1).clamp_min(0)
            outputs.append(table.pixels[index] & ~finished[:,None,None])
            sizes += ~finished
            x = table.feedback[index,None,:]
        return torch.stack(outputs,1) if outputs else tiles[:,:0].bool(), sizes, finished
