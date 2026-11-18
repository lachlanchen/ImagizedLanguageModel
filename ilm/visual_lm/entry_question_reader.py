"""Read a pictured instruction while writing evidence from a dictionary card."""

from functools import lru_cache

import numpy as np
import torch
from torch import nn

from ilm.visual_lm.entry_span_writer import EntryWriter, appearance, cells, VIS


@lru_cache(maxsize=16000)
def retina(glyph):
    ink = np.unpackbits(np.frombuffer(glyph, np.uint8)).reshape(8, 4, 8, 4)
    return ink.mean((1, 3), dtype=np.float32).ravel()


def anchors(records, literal_count):
    labels = []
    for head, fields in records:
        labels.append((head, ()))
        for key, _ in fields:
            labels.extend([(head, key), (head, key)])
    return labels + [((), ())] * literal_count


def offsets(question, labels):
    positions = {}
    seq = cells(question)
    for part in {part for pair in labels for part in pair if part}:
        first = next((i for i in range(len(seq)) if seq[i : i + len(part)] == part), -1)
        positions[part] = first
    return np.array(
        [(positions.get(h, -1), positions.get(k, -1)) for h, k in labels],
        dtype=np.int16,
    )


class QuestionReader(EntryWriter):
    def __init__(self):
        super().__init__()
        self.retina_conv = nn.Sequential(
            nn.Conv1d(64, 32, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 32, 5, padding=2),
            nn.GELU(),
        )
        self.context = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 64))
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)

    def read(self, features, positions, images, valid):
        length = valid.sum(1).clamp_min(1)
        match = positions >= 0
        locations = positions.clamp_min(0) / length[:, None, None]
        features = features.clone()
        features[:, :, 4:8:2] = match
        features[:, :, 5:8:2] = locations
        base = self.encode(features)
        strip = self.retina_conv(images.transpose(1, 2)).transpose(1, 2)
        strip = strip * valid.unsqueeze(-1)
        summary = strip.sum(1) / length[:, None]
        anchor = torch.where(match[:, :, 1], positions[:, :, 1], positions[:, :, 0])
        around = strip[
            torch.arange(len(strip), device=strip.device)[:, None],
            anchor.clamp_min(0).long(),
        ]
        around = around * (anchor >= 0).unsqueeze(-1)
        residual = self.context(
            torch.cat((around, summary[:, None].expand_as(around)), dim=-1)
        )
        return torch.cat((base[:, :-1] + residual, base[:, -1:]), dim=1)

    @torch.inference_mode()
    def emit(self, keys, options, limit=72):
        feedback = keys.new_zeros(len(keys), 1, VIS)
        streams = [bytearray() for _ in options]
        stopped = [False] * len(options)
        state = None
        for _ in range(limit):
            scores, state = self(keys, feedback, state)
            actions = scores[:, 0].argmax(-1).tolist()
            previous = []
            for i, (o, a) in enumerate(zip(options, actions, strict=True)):
                stopped[i] |= a == len(o.payload)
                span = () if stopped[i] else o.payload[a]
                streams[i].extend(b"".join(span))
                previous.append(appearance(span))
            if all(stopped):
                break
            feedback = torch.as_tensor(
                np.stack(previous), device=keys.device
            ).unsqueeze(1)
        return list(map(bytes, streams)), stopped
