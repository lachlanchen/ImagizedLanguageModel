"""Score distinct physical positions and pool probability on identical answer strips."""

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file

from ilm.visual_lm.bound_fact_reader import best_bounds, features_batch
from ilm.visual_lm.ordered_page_features import contextualize
from ilm.visual_lm.role_conditioned_span import span_scores

DIM = 48
WIDTH = 4 * DIM + 4 + 3
COUNT = 32


class SpanRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(DIM, 32), nn.GELU(), nn.Linear(32, 4))
        self.residual = nn.Sequential(
            nn.LayerNorm(WIDTH), nn.Linear(WIDTH, 96), nn.GELU(), nn.Linear(96, 1)
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, features, valid):
        x = features.float()
        mixture = self.gate(x[:, :, :DIM]).log_softmax(-1)
        # Candidate evidence carries the full-span log posterior under each frozen reader.
        prior = (mixture + x[:, :, 4 * DIM : 4 * DIM + 4] * 8).logsumexp(-1)
        return (prior + self.residual(x).squeeze(-1)).masked_fill(~valid, -torch.inf)


def image_choice(scores, classes):
    mass = torch.zeros_like(scores).scatter_add(1, classes, scores.softmax(1))
    return mass.argmax(1)


class SpanListReader:
    def __init__(self, experts, weights=None):
        self.experts = experts
        self.device = experts.base.device
        self.head = SpanRanker().to(self.device)
        if weights:
            self.head.load_state_dict(load_file(str(weights)))
        self.head.eval()
        generator = torch.Generator().manual_seed(241196)
        self.projection = (
            torch.randint(2, (768, DIM), generator=generator).float() * 2 - 1
        ).to(self.device) / 768**0.5

    @torch.inference_mode()
    def proposals(self, pairs):
        base = self.experts.base
        cls, page, mask = data = features_batch(
            contextualize(base.reader, pairs), self.device
        )
        roles, bounds = base.head(*data)
        sb = best_bounds(bounds[:, :, :2]).tolist()
        subjects = [
            p[a * 128 : b * 128] for (p, _), (a, b) in zip(pairs, sb, strict=True)
        ]
        chosen = [
            self.experts.channel.choose(q, s, r)[0]
            for (_, q), s, r in zip(
                pairs, subjects, roles.argmax(1).tolist(), strict=True
            )
        ]
        role = torch.tensor(chosen, device=self.device)
        prior = bounds[:, :, 2:]
        logits = [page[:, :, 768:770], prior] + [
            h(page, mask, role, prior) for h in self.experts.heads
        ]
        flat = torch.stack([span_scores(z, mask).flatten(1) for z in logits], 1)
        width = min(48, page.size(1))
        indices = flat.argsort(dim=2, descending=True, stable=True)[:, :, :8]
        if indices.size(2) < 8:
            indices = F.pad(indices, (0, 8 - indices.size(2)), value=0)
        # Put all four first choices before lower ranks for depth-one evaluation.
        indices = indices.transpose(1, 2).reshape(-1, COUNT)
        where = torch.arange(COUNT, device=self.device)
        previous = where[None, :] < where[:, None]
        duplicate = ((indices[:, :, None] == indices[:, None, :]) & previous).any(2)
        a = indices // width
        n = indices % width + 1
        b = a + n
        valid = (~duplicate) & (b <= mask.sum(1)[:, None])
        row = torch.arange(len(pairs), device=self.device)[:, None]
        b = b.clamp_max(page.size(1))
        logp = (
            flat.log_softmax(-1)
            .gather(2, indices[:, None].expand(-1, 4, -1))
            .transpose(1, 2)
            .clamp_min(-64)
        )
        context = page[:, :, :768] @ self.projection
        summed = F.pad(context.cumsum(1), (0, 0, 1, 0))
        features = torch.cat(
            (
                (cls @ self.projection)[:, None].expand(-1, COUNT, -1),
                context[row, a],
                context[row, b - 1],
                (summed[row, b] - summed[row, a]) / n[:, :, None],
                logp / 8,
                (a / mask.sum(1)[:, None])[:, :, None],
                (n / 48)[:, :, None],
                (n.log() / 4)[:, :, None],
            ),
            dim=-1,
        )
        strips = []
        classes = []
        # A group is represented by the first physical instance of the same bitmap.
        for (p, _), starts, ends, allowed in zip(
            pairs, a.tolist(), b.tolist(), valid.tolist(), strict=True
        ):
            unique = {}
            options = [
                p[128 * i : 128 * j] if ok else b""
                for i, j, ok in zip(starts, ends, allowed, strict=True)
            ]
            classes.append(
                [unique.setdefault(o, k) if o else 0 for k, o in enumerate(options)]
            )
            strips.append(options)
        fields = list(zip(subjects, chosen, strips, strict=True))
        return features.half(), valid, torch.tensor(classes, device=self.device), fields

    @torch.inference_mode()
    def fields_many(self, pairs):
        result = []
        for start in range(0, len(pairs), 24):
            x, mask, classes, fields = self.proposals(pairs[start : start + 24])
            picks = image_choice(self.head(x, mask), classes).tolist()
            result.extend(
                (s, r, opts[k]) for (s, r, opts), k in zip(fields, picks, strict=True)
            )
        return result
