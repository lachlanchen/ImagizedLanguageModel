"""A recurrent visual span editor conditioned on two predicted regions of a page."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.composed_answer_writer import COPY, WRITE, STOP

DIM = 64


def pooled(pixels):
    return F.avg_pool2d(pixels.reshape(-1, 1, 32, 32).float(), 4).flatten(1)


class GroundedInkEditor(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(64 + 8, DIM)
        self.context = nn.Conv1d(DIM, DIM, 5, padding=2)
        self.style_map = nn.Sequential(
            nn.Linear(64, DIM), nn.Tanh(), nn.Linear(DIM, DIM)
        )
        self.initial = nn.Linear(2 * DIM, DIM)
        self.recurrence = nn.GRUCell(DIM, DIM)
        self.action = nn.Linear(DIM, 3)
        self.boundaries = nn.Linear(DIM, 2 * DIM)
        self.readout = nn.Sequential(
            nn.Linear(DIM, DIM * 2), nn.GELU(), nn.Linear(DIM * 2, 1024)
        )
        self.history = nn.Linear(64 + 3, DIM)

    def encode(self, pixels, valid, regions, instruction, style_mask):
        pos = torch.arange(pixels.shape[1], device=pixels.device)[None, :, None, None]
        # Distances are to both boundaries of both predicted answer regions.
        delta = pos - regions[:, None]
        flags = torch.cat(
            (delta.eq(0).flatten(2), delta.clamp(-12, 12).flatten(2) / 12.0), dim=-1
        )
        x = self.input(torch.cat((pixels, flags), -1)).tanh() * valid.unsqueeze(-1)
        x = (
            x + self.context(x.transpose(1, 2)).transpose(1, 2)
        ).tanh() * valid.unsqueeze(-1)
        style = (self.style_map(instruction) * style_mask.unsqueeze(-1)).sum(
            1
        ) / style_mask.sum(1, keepdim=True)
        state = self.initial(
            torch.cat((style, x.sum(1) / valid.sum(1, keepdim=True)), 1)
        ).tanh()
        return x, state

    def decide(self, x, valid, state):
        q = self.boundaries(state).reshape(len(x), 2, DIM)
        bounds = (q @ x.transpose(1, 2)) / DIM**0.5
        return (
            self.action(state),
            bounds.masked_fill(~valid[:, None], -1e4),
            self.readout(state),
        )

    def advance(self, state, choice, shape):
        v = self.history(torch.cat((shape, F.one_hot(choice, 3).float()), 1)).tanh()
        return self.recurrence(v, state)

    @torch.inference_mode()
    def emit(self, page, regions, instruction, max_steps=24):
        """Consume packed images; emit arbitrary 32×32 bitmaps or copy contiguous ink."""
        if not page or not instruction or len(page) % 128 or len(instruction) % 128:
            raise ValueError("Expected nonempty packed page and instruction strips")
        if len(regions) != 2 or any(
            not 0 <= a <= b < len(page) // 128 for a, b in regions
        ):
            raise ValueError("Two inclusive source regions are required")
        device = next(self.parameters()).device

        def features(raw):
            bits = np.unpackbits(np.frombuffer(raw, np.uint8)).reshape(-1, 1024)
            return pooled(torch.from_numpy(bits).to(device))[None]

        page_features = features(page)
        style = features(instruction)
        valid = torch.ones(page_features.shape[:2], dtype=torch.bool, device=device)
        x, state = self.encode(
            page_features,
            valid,
            torch.tensor([regions], device=device),
            style,
            torch.ones_like(style[:, :, 0], dtype=torch.bool),
        )
        chunks = []
        for _ in range(max_steps):
            acts, bounds, raster = self.decide(x, valid, state)
            action = acts.argmax(1)
            if action.item() == STOP:
                return b"".join(chunks), True
            if action.item() == COPY:
                a = int(bounds[0, 0].argmax())
                b = a + int(bounds[0, 1, a:].argmax())
                chunks.append(page[128 * a : 128 * (b + 1)])
                feedback = page_features[:, a : b + 1].mean(1)
            else:
                bits = raster > 0
                chunks.append(np.packbits(bits[0].cpu().numpy()).tobytes())
                feedback = pooled(bits)
            state = self.advance(state, action, feedback)
        return b"".join(chunks), False
