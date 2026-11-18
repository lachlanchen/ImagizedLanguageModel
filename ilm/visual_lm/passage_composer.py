"""Compose pictured answers from ordered question regions and passage spans."""
import numpy as np
import torch
from torch import nn

from ilm.visual_lm.idiom_ink import visual_input
from ilm.visual_lm.mixed_ink_reader import MixedInkReader

IGNORE, READ, JOIN, READ_STOP = range(4)


class PassageComposer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("landmarks", torch.zeros(48, dtype=torch.long))
        self.reader = MixedInkReader()
        self.planner = nn.Sequential(nn.Conv1d(114, 48, 5, padding=2), nn.GELU(),
                                     nn.Conv1d(48, 48, 5, padding=2), nn.GELU(),
                                     nn.Conv1d(48, 4, 1))

    def features(self, pixels):
        bits = torch.as_tensor(np.unpackbits(pixels, axis=1), device=self.landmarks.device)
        return visual_input(bits, self.landmarks)

    def plan(self, features, handles, lengths):
        pos = torch.arange(handles.shape[1], device=handles.device)
        valid = pos < lengths[:, None]
        distance = lengths[:, None] - 1 - pos
        location = torch.stack((pos.expand_as(distance).clamp(max=20),
                                distance.clamp(0, 20)), -1).float() / 20
        state = torch.cat((features[handles], location), -1) * valid.unsqueeze(-1)
        # Mask intermediate layers too, so padding and batch composition are irrelevant.
        state = state.transpose(1, 2)
        for layer in self.planner:
            state = layer(state) * valid[:, None, :]
        return state.transpose(1, 2)

    @staticmethod
    def regions(tags):
        result = []
        start = None
        for pos, tag in enumerate(tags):
            if tag in (READ, READ_STOP):
                if start is None:
                    start = pos
            elif start is not None:
                result.append(("span", start, pos))
                start = None
            if tag == JOIN:
                result.append(("prompt", pos, pos + 1))
            if tag == READ_STOP:
                result.append(("span", start, pos + 1))
                return result, True
        if start is not None:
            result.append(("span", start, len(tags)))
        return result, False

    @torch.inference_mode()
    def answer(self, passage, prompt):
        for tiles in (passage, prompt):
            if tiles.dtype != np.uint8 or tiles.ndim != 2 or tiles.shape[1] != 128 or not len(tiles):
                raise ValueError("Expected nonempty rows of packed 32 by 32 glyph images")
        bank, ids = np.unique(np.concatenate((passage, prompt)), axis=0, return_inverse=True)
        features = self.features(bank)
        device = features.device
        p, q = torch.as_tensor(ids, device=device).split([len(passage), len(prompt)])
        length = lambda seq: torch.tensor([len(seq)], device=device)
        tags = self.plan(features, q[None], length(q))[0].argmax(-1).tolist()
        plan, stopped = self.regions(tags)
        output = []
        for kind, a, b in plan:
            if kind == "prompt":
                output.append(prompt[a:b].tobytes())
            else:
                part = q[a:b]
                result = self.reader(features, p[None], part[None], length(p), length(part))
                _, l, r = self.reader.decode(result, length(p))
                output.append(passage[int(l):int(r)].tobytes())
        return b"".join(output), stopped
