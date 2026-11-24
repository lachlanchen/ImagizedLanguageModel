"""Small recurrent glyph writer conditioned on the current and preceding pictured fields."""

import torch
from torch import nn
from torch.nn import functional as F


class FieldCursor(nn.Module):
    def __init__(self, classes, copy_id):
        super().__init__()
        self.copy_id = copy_id
        self.label = nn.Sequential(
            nn.Linear(256, 48),
            nn.GELU(),
            nn.Flatten(1),
            nn.Linear(8 * 48, 48),
            nn.LayerNorm(48),
        )
        self.glyph = nn.Sequential(nn.Linear(256, 48), nn.GELU())
        self.body = nn.GRU(48 * 5, 128, num_layers=1, batch_first=True)
        self.readout = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, classes))

    def encode_labels(self, label_cells):
        # All fixed field/domain labels fit in eight 32-pixel cells.
        if label_cells[:, 8:].any():
            raise ValueError("Field label exceeds the learned window")
        return self.label(label_cells[:, :8])

    @staticmethod
    def field_states(keys, domains, plans, cursor):
        n = keys.shape[0]
        null = n
        bank = F.pad(keys, (0, 0, 0, 1))
        first = plans[:, 0, None].expand_as(cursor)
        second = plans[:, 1, None].expand_as(cursor)
        current = torch.where(
            cursor == 0, first, torch.where(cursor == 1, second, null)
        )
        previous = torch.where(
            cursor == 0, null, torch.where(cursor == 1, first, second)
        )
        upcoming = torch.where(cursor == 0, second, null)
        dom = bank[domains][:, None].expand(-1, cursor.size(1), -1)
        return torch.cat((dom, bank[previous], bank[current], bank[upcoming]), -1)

    def forward(self, keys, domains, plans, previous, embedding):
        cursor = (previous == self.copy_id).long().cumsum(1)
        conditions = self.field_states(keys, domains, plans, cursor)
        state, _ = self.body(torch.cat((embedding[previous], conditions), -1))
        return self.readout(state)

    @torch.inference_mode()
    def write(self, keys, domains, plans, embedding, limit=120):
        cursor = torch.zeros_like(domains)
        length = (plans < len(keys)).sum(1)
        if (length + 1 > limit).any():
            raise ValueError("Output limit must allow each source copy and a stop")
        previous = torch.zeros_like(cursor)
        finished = torch.zeros_like(cursor, dtype=torch.bool)
        outputs = []
        hidden = None
        for step in range(limit):
            conditions = self.field_states(keys, domains, plans, cursor[:, None])
            state, hidden = self.body(
                torch.cat((embedding[previous, None], conditions), -1), hidden
            )
            logits = self.readout(state[:, 0])
            logits[:, 0].masked_fill_(cursor < length, -torch.inf)
            logits[:, self.copy_id].masked_fill_(cursor >= length, -torch.inf)
            # Reserve one transition per remaining pointer and the final stop.
            must = (limit - step) == (length - cursor + 1)
            choice = torch.where(
                must, torch.where(cursor < length, self.copy_id, 0), logits.argmax(1)
            )
            choice = choice.masked_fill(finished, 0)
            outputs.append(choice)
            finished |= choice == 0
            cursor += choice == self.copy_id
            previous = choice
            if finished.all():
                break
        if not finished.all():
            raise RuntimeError("Invalid cursor termination")
        return [[i for i in s if i] for s in torch.stack(outputs, 1).tolist()]
