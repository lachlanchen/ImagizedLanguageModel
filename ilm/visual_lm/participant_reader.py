"""Learn an additive question-conditioned score for complete visual source spans."""

import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.interval_writer import IntervalWriter, MAX_RUN


class ParticipantReader(IntervalWriter):
    """Keep the copy/composition decision fixed while adapting visual answer selection."""

    def __init__(self):
        super().__init__()
        self.requires_grad_(False)
        width = 64
        self.source_projection = nn.Linear(128, width)
        self.query_projection = nn.Linear(128, width)
        self.question_values = nn.Linear(128, width)
        self.interaction = nn.Linear(3 * width, width)
        self.context_layers = nn.ModuleList(
            nn.Sequential(nn.Conv1d(width, width, 5, dilation=d, padding=2 * d), nn.GELU())
            for d in (1, 3, 9)
        )
        self.context_norms = nn.ModuleList(nn.LayerNorm(width) for _ in self.context_layers)
        self.source_start = nn.Linear(width, 1)
        self.source_end = nn.Linear(width, 1)
        self.run_length = nn.Linear(width, MAX_RUN)
        for layer in (self.source_start, self.source_end, self.run_length):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, encoded, passage):
        with torch.no_grad():
            action, prior = super().forward(encoded, passage)
        n = passage.shape[1]
        source = self.source_projection(encoded[0][:, :n])
        query = encoded[0][:, n:]
        alignment = source @ self.query_projection(query).transpose(1, 2) / source.shape[-1]**0.5
        weights = alignment.masked_fill(encoded[1][:, None, n:] == 0, -1e4).softmax(-1)
        attended = weights @ self.question_values(query)
        states = self.interaction(torch.cat((source, attended, source * attended), -1))
        mask = passage.ne(0).unsqueeze(-1)
        for layer, norm in zip(self.context_layers, self.context_norms):
            states = norm(states + layer((states * mask).transpose(1, 2)).transpose(1, 2))
        end = self.source_end(states).squeeze(-1)
        end_by_length = F.pad(end, (0, MAX_RUN - 1)).unfold(1, MAX_RUN, 1)
        return action, prior + self.source_start(states) + end_by_length + self.run_length(states)
