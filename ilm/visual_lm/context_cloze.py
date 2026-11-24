"""Learn a conditional word-image distribution from relative ink and a pictured question."""

import torch
from torch import nn
from torch.nn import functional as F
from ilm.visual_lm.word_memory import WordMemory


def neighbour_features(images, visible, lengths, positions):
    """Read four offsets without ever reading the centre; mark boundaries and masks separately."""
    offsets = positions.new_tensor([-2, -1, 1, 2])
    idx = positions[:, :, None] + offsets
    inside = (idx >= 0) & (idx < lengths[:, None, None])
    safe = idx.clamp(0, images.size(1) - 1)
    b = torch.arange(len(images), device=images.device)[:, None, None]
    shown = inside & visible[b, safe]
    ink = images[b, safe] * shown[:, :, :, None]
    return torch.cat((ink * 6, shown[..., None], (~inside)[..., None]), -1).flatten(2)


class ContextCloze(WordMemory):
    def __init__(self, palette, mode, images, lengths, features):
        super().__init__(palette, mode, images, lengths, features)
        self.local_query = nn.Sequential(
            nn.Linear(4 * 52 + self.width, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 96),
        )
        self.local_gate = nn.Linear(4 * 52 + self.width, 1)
        nn.init.zeros_(self.local_query[-1].weight)
        nn.init.zeros_(self.local_query[-1].bias)
        nn.init.zeros_(self.local_gate.weight)
        nn.init.zeros_(self.local_gate.bias)

    def paint(self, memory, appearance, visible, length):
        route, pointer, z = super().paint(memory, appearance, visible, length)
        pos = torch.arange(appearance.size(1), device=z.device)[None].expand(len(z), -1)
        local = neighbour_features(appearance, visible, length, pos)
        return route, pointer, torch.cat((z, local), -1)

    def retrieve_logits(self, state):
        q = self.retrieve(state[..., : self.width]) + self.local_query(state)
        return F.linear(q, self.keys, self.prior) / self.keys.shape[1] ** 0.5

    def gate_logits(self, state):
        return self.gate(state[..., : self.width]) + self.local_gate(state)

    def draw_letters(self, state):
        return super().draw_letters(state[..., : self.width].contiguous())
