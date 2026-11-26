"""A source-conditioned optical writer with an optional copy channel to a pictured draft."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from ilm.visual_lm.full_image_roles import FullImageWriter


class GlyphDraftEditor(FullImageWriter):
    def __init__(self):
        super().__init__()
        width = self.pos.shape[1]
        self.draft_context = nn.Conv1d(width, width, 3, padding=1)
        self.draft_query = nn.Linear(width, width, bias=False)
        self.draft_key = nn.Linear(width, width, bias=False)
        self.draft_gate = nn.Linear(2 * width, 1)
        nn.init.constant_(self.draft_gate.bias, -2.)

    def draft_memory(self, bank, draft, local):
        if draft is None:
            return None
        valid = draft.ne(0)
        # Sinusoids also cover phrase drafts longer than the source position table.
        width = bank.shape[-1]
        frequencies = torch.exp(torch.arange(0, width, 2, device=bank.device) * (-math.log(10000) / width))
        angles = torch.arange(draft.shape[1], device=bank.device)[:, None] * frequencies
        pos = torch.stack((angles.sin(), angles.cos()), -1).flatten(1)
        x = (bank[draft] + pos) * valid.unsqueeze(-1)
        x = x + F.gelu(self.draft_context(x.transpose(1, 2)).transpose(1, 2))
        return self.draft_key(x), x, local[draft].clamp_min(0), valid

    def distribution(self, h, prototypes, memory):
        logits = (h @ prototypes.T).float() / math.sqrt(h.shape[-1])
        if memory is None:
            return logits.log_softmax(-1)
        keys, values, local_ids, valid = memory
        affinity = (self.draft_query(h) @ keys.transpose(1, 2)).float() / math.sqrt(h.shape[-1])
        attention = affinity.masked_fill(~valid[:, None], -1e30).softmax(-1) * valid[:, None]
        summary = attention.to(values.dtype) @ values
        gate = self.draft_gate(torch.cat((h, summary), -1)).float().sigmoid() * valid.any(1)[:, None, None]
        mass = (1 - gate) * logits.softmax(-1)
        mapped = local_ids[:, None].expand(-1, h.shape[1], -1)
        return mass.scatter_add(2, mapped, gate * attention).clamp_min(1e-30).log()

    def forward(self, features, source, target, inventory, direction, draft, local):
        bank = self.optical(features)
        context, valid = self.encode(bank, source)
        previous = torch.cat((self.begin[direction].expand(len(source), 1, -1), bank[target[:, :-1]]), 1)
        hidden, _ = self.decode(previous, context, valid)
        return self.distribution(hidden, self.prototypes(bank, inventory), self.draft_memory(bank, draft, local))

    @torch.inference_mode()
    def write(self, features, source, inventory, direction, draft, local, limit=128):
        bank = self.optical(features)
        context, valid = self.encode(bank, source)
        memory = self.draft_memory(bank, draft, local)
        prototypes = self.prototypes(bank, inventory)
        current = self.begin[direction].expand(len(source), 1, -1)
        finished = torch.zeros(len(source), device=source.device, dtype=torch.bool)
        result, past = [], None
        for offset in range(limit + 1):
            h, past = self.decode(current, context, valid, past, offset)
            ids = inventory[self.distribution(h, prototypes, memory)[:, 0].argmax(-1)]
            result.append(ids.masked_fill(finished, 0))
            finished = finished | ids.eq(1)
            if finished.all():
                break
            current = bank[ids][:, None]
        return torch.stack(result, 1)
