"""Choose an answer by pooling the probability assigned to identical source images."""

from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file

from ilm.visual_lm.bound_fact_reader import best_bounds, features_batch
from ilm.visual_lm.ordered_page_features import contextualize
from ilm.visual_lm.role_conditioned_span import ParticipantHead, span_scores

EXPERTS = ("generic", "factual", "likelihood", "margin")
INPUT_WIDTH = 768 + 4 * 5 + 6 + 2


class ChoiceHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 48), nn.GELU())
        self.score = nn.Sequential(
            nn.Linear(48 + INPUT_WIDTH - 768, 48), nn.GELU(), nn.Linear(48, 4)
        )

    def forward(self, features):
        context = self.project(features[:, :768])
        return self.score(torch.cat((context, features[:, 768:]), dim=1))


def pooled_choice(logits, agreement):
    """Each row of agreement identifies all proposals with the same physical output."""
    mass = (agreement * logits.softmax(-1).unsqueeze(1)).sum(-1)
    return mass.argmax(1)


class SpanChoiceReader:
    def __init__(self, base, channel, experts, weights=None):
        self.base, self.channel = base, channel
        self.heads = []
        for name in ("likelihood.safetensors", "participant.safetensors"):
            head = ParticipantHead(len(base.frames)).to(base.device)
            head.load_state_dict(load_file(str(Path(experts) / name)))
            self.heads.append(head.eval())
        self.choice = ChoiceHead().to(base.device)
        if weights:
            self.choice.load_state_dict(load_file(str(weights)))
        self.choice.eval()

    @torch.inference_mode()
    def proposals(self, pairs):
        if not pairs:
            raise ValueError(
                "At least one pictured question and nonempty passage required"
            )
        summary, page, valid = data = features_batch(
            contextualize(self.base.reader, pairs), self.base.device
        )
        roles, bounds = self.base.head(*data)
        subjects = [
            p[a * 128 : b * 128]
            for (p, _), (a, b) in zip(
                pairs, best_bounds(bounds[:, :, :2]).tolist(), strict=True
            )
        ]
        chosen = [
            self.channel.choose(q, s, r)
            for (_, q), s, r in zip(
                pairs, subjects, roles.argmax(1).tolist(), strict=True
            )
        ]
        role = torch.tensor([v[0] for v in chosen], device=page.device)
        prior = bounds[:, :, 2:]
        ends = [page[:, :, 768:770], prior] + [
            h(page, valid, role, prior) for h in self.heads
        ]
        strips = [[] for _ in pairs]
        lengths = valid.sum(1)
        features = [summary]
        for logits in ends:
            scores = span_scores(logits, valid)
            flat = scores.flatten(1)
            winner = flat.argmax(1)
            top = flat.gather(1, winner[:, None]).squeeze(1)
            runner = (
                flat.masked_fill(
                    torch.zeros_like(flat, dtype=torch.bool).scatter_(
                        1, winner[:, None], True
                    ),
                    -torch.inf,
                )
                .max(1)
                .values
            )
            start = winner // scores.shape[2]
            size = winner % scores.shape[2] + 1
            for options, (p, _), a, n in zip(
                strips, pairs, start.tolist(), size.tolist(), strict=True
            ):
                options.append(p[a * 128 : (a + n) * 128])
            features.append(
                torch.stack(
                    (
                        (top / 20).clamp(-5, 5),
                        ((top - runner) / 8).clamp(0, 5),
                        (top - flat.logsumexp(1)).exp(),
                        start / lengths,
                        size / 48,
                    ),
                    dim=1,
                )
            )
        agree = torch.tensor(
            [[[a == b for b in options] for a in options] for options in strips],
            device=page.device,
        )
        features.extend(
            [agree[:, i, j, None] for i in range(4) for j in range(i + 1, 4)]
        )
        features.append(
            torch.tensor(
                [
                    [confidence, len(q) / 128 / 128]
                    for (_, confidence), (_, q) in zip(chosen, pairs, strict=True)
                ],
                device=page.device,
            )
        )
        fields = [
            (s, r, options)
            for s, (r, _), options in zip(subjects, chosen, strips, strict=True)
        ]
        return torch.cat(features, dim=1), agree, fields

    @torch.inference_mode()
    def fields_many(self, pairs):
        result = []
        for start in range(0, len(pairs), 24):
            features, agree, fields = self.proposals(pairs[start : start + 24])
            take = pooled_choice(self.choice(features), agree).tolist()
            result.extend(
                (s, r, options[i])
                for (s, r, options), i in zip(fields, take, strict=True)
            )
        return result
