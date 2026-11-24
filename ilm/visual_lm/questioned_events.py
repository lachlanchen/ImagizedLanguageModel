"""Score each visible verb and its object jointly with a pictured question."""

import math
import torch
from torch import nn


class QuestionedEvents(nn.Module):
    def __init__(self):
        super().__init__()
        self.content = nn.Sequential(nn.Linear(160, 160), nn.LayerNorm(160), nn.GELU())
        self.verb = nn.Linear(160, 48)
        self.object = nn.Linear(160, 48)
        self.unary = nn.Linear(160, 2)
        self.read_key = nn.Linear(160, 146)
        self.distance = nn.Embedding(63, 1)
        for layer in (self.object, self.unary, self.read_key, self.distance):
            nn.init.zeros_(layer.weight)
            if getattr(layer, "bias", None) is not None:
                nn.init.zeros_(layer.bias)

    def page(self, states, syntax, keys):
        hidden = self.content(states)
        endpoint = self.unary(hidden)
        link = self.verb(hidden) @ self.object(hidden).transpose(1, 2) / math.sqrt(48)
        positions = torch.arange(states.shape[1], device=states.device)
        offset = (positions[None, :] - positions[:, None]).clamp(-31, 31) + 31
        arc = (
            0.2 * syntax
            + endpoint[:, :, 0, None]
            + endpoint[:, None, :, 1]
            + link
            + self.distance(offset).squeeze(-1)
        )
        return arc, keys + self.read_key(hidden)

    @staticmethod
    def question_scores(keys, question, mask):
        return (
            (keys @ question.transpose(1, 2))
            .masked_fill(~mask[:, None, :], -torch.inf)
            .amax(-1)
        )

    @classmethod
    def scores(cls, arcs, keys, question, mask):
        return arcs + 8 * cls.question_scores(keys, question, mask)[:, :, None]

    @classmethod
    def select(cls, arcs, keys, question, mask):
        """Compare per-verb maxima, reusing the best object for every question."""
        strength, objects = arcs.max(dim=-1)
        verbs = (strength + 8 * cls.question_scores(keys, question, mask)).argmax(
            dim=-1
        )
        objects = objects.expand(len(verbs), -1).gather(1, verbs[:, None])[:, 0]
        return verbs, objects
