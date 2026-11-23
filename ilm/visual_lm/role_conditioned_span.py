"""Select a participant directly from page pixels under a question's predicted role."""

from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file

from ilm.visual_lm.bound_fact_reader import best_bounds, features_batch
from ilm.visual_lm.ordered_page_features import contextualize


class ParticipantHead(nn.Module):
    """Learn a correction to the frozen span prior from nearby ink and the requested role."""

    def __init__(self, roles, width=96):
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, width), nn.GELU()
        )
        self.role = nn.Embedding(roles, width)
        self.context = nn.Conv1d(width, width, 5, padding=2)
        self.score = nn.Linear(width, 2)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, page, valid, role, prior):
        local = self.project(page[:, :, :768]) * valid.unsqueeze(2)
        neighborhood = self.context(local.transpose(1, 2)).transpose(1, 2)
        conditional = (local + neighborhood) * (1 + self.role(role)[:, None].tanh())
        return (prior + self.score(F.gelu(conditional))).masked_fill(
            ~valid[:, :, None], -1e4
        )


def span_scores(logits, valid, limit=48):
    """Joint scores indexed by start and inclusive distance from start."""
    width = min(limit, logits.shape[1])
    first, last = logits.unbind(-1)
    right = F.pad(last, (0, width - 1), value=-torch.inf).unfold(1, width, 1)
    allowed = F.pad(valid, (0, width - 1), value=False).unfold(1, width, 1)
    return (first[:, :, None] + right).masked_fill(
        ~(valid[:, :, None] & allowed), -torch.inf
    )


class ParticipantReader:
    def __init__(self, base, question_channel, weights):
        self.base = base
        self.channel = question_channel
        self.head = ParticipantHead(len(base.frames)).to(base.device)
        self.head.load_state_dict(load_file(str(Path(weights))))
        self.head.eval()

    @torch.inference_mode()
    def fields_many(self, pairs):
        values = contextualize(self.base.reader, pairs) if pairs else []
        return self.from_features(pairs, values)

    @torch.inference_mode()
    def from_features(self, pairs, values):
        result = []
        for start in range(0, len(pairs), 32):
            chunk = pairs[start : start + 32]
            data = features_batch(values[start : start + 32], self.base.device)
            roles, bounds = self.base.head(*data)
            subjects = best_bounds(bounds[:, :, :2]).tolist()
            names = [
                p[a * 128 : b * 128]
                for (p, _), (a, b) in zip(chunk, subjects, strict=True)
            ]
            chosen = [
                self.channel.choose(q, s, r)[0]
                for (_, q), s, r in zip(
                    chunk, names, roles.argmax(1).tolist(), strict=True
                )
            ]
            out = self.head(
                data[1],
                data[2],
                torch.tensor(chosen, device=bounds.device),
                bounds[:, :, 2:],
            )
            objects = best_bounds(out).tolist()
            result.extend(
                (s, r, p[a * 128 : b * 128])
                for s, r, (p, _), (a, b) in zip(
                    names, chosen, chunk, objects, strict=True
                )
            )
        return result
