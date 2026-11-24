"""Predict one or two distinct pictured fields and condition a glyph writer on their order."""

import torch
from torch import nn

from ilm.visual_lm.semantic_page import SemanticPage, WIDTH


class FactSequence(SemanticPage):
    def __init__(self, classes, fields, columns):
        super().__init__(classes, fields, columns)
        self.fields = fields
        self.second = nn.Sequential(
            nn.Linear(2 * WIDTH, WIDTH), nn.GELU(), nn.Linear(WIDTH, fields + 1)
        )
        self.second_role = nn.Parameter(torch.randn(WIDTH) * 0.03)

    def plan(self, question, label_states, domain_ids, available, teacher=None):
        query = self.encode_rows(question)
        domain = label_states[self.fields + domain_ids]
        state = query + domain
        first_scores = self.concept(state).float().masked_fill(~available, -torch.inf)
        first = first_scores.argmax(1) if teacher is None else teacher[:, 0]
        first_state = label_states[first]
        remaining = available.clone()
        remaining.scatter_(1, first[:, None], False)
        legal = torch.cat((remaining, torch.ones_like(remaining[:, :1])), 1)
        second_scores = (
            self.second(torch.cat((state, first_state), 1))
            .float()
            .masked_fill(~legal, -torch.inf)
        )
        second = second_scores.argmax(1) if teacher is None else teacher[:, 1]
        memory = torch.stack(
            (
                query,
                domain,
                first_state,
                label_states[second.clamp_max(self.fields - 1)] + self.second_role,
            ),
            1,
        )
        visible = torch.ones(memory.shape[:2], device=memory.device, dtype=torch.bool)
        visible[:, -1] = second != self.fields
        return (
            first_scores,
            second_scores,
            torch.stack((first, second), 1),
            memory,
            visible,
        )
