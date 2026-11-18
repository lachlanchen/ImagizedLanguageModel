"""A source-position transducer with a physical ink output and closed-loop feedback."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def visual_features(packed):
    """Intensity in each of 64 square regions, measured directly on the image."""
    pixels = np.unpackbits(packed, axis=-1).reshape(-1, 8, 4, 8, 4)
    return pixels.mean(axis=(2, 4), dtype=np.float32).reshape(-1, 64)


def article_memory(passage, question):
    tiles = np.concatenate([passage, question])
    # Position-local group numbers only combine identical physical alternatives.
    # They are never embedded, shared between articles or exposed to the reader.
    group = {}
    ids = np.asarray([group.setdefault(t.tobytes(), len(group)) for t in tiles])
    position = np.arange(len(tiles), dtype=np.float32)
    features = np.column_stack([visual_features(tiles), np.log1p(position)/8,
                                position / max(1, len(tiles)-1),
                                position >= len(passage)]).astype(np.float32)
    return tiles, features, ids, group


class TitleTransducer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(67, 64)
        self.neighbours = nn.Conv1d(64, 64, 5, padding=2)
        self.query = nn.Linear(64, 64)
        self.initial = nn.Linear(128, 64)
        self.causal = nn.GRUCell(128, 64)
        self.decision = nn.Linear(128, 3)  # select a picture, draw, stop
        self.continue_span = nn.Linear(64, 1)
        self.ink = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1024))

    def encode(self, features, valid):
        h = torch.tanh(self.input(features)) * valid.unsqueeze(-1)
        mem = torch.tanh(h + self.neighbours(h.transpose(1, 2)).transpose(1, 2))
        mean = (mem * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        hidden = torch.tanh(self.initial(torch.cat([mem[:, 0], mean], -1)))
        return mem, hidden

    def step(self, mem, valid, hidden, focus, feedback):
        attended = torch.bmm(focus[:, None], mem).squeeze(1)
        hidden = self.causal(torch.cat([feedback, attended], -1), hidden)
        scores = torch.bmm(mem, self.query(hidden).unsqueeze(-1)).squeeze(-1) / 8
        jump = scores.masked_fill(~valid, -torch.inf).softmax(-1)
        shifted = F.pad(focus[:, :-1], (1, 0)) * valid
        surviving = shifted.sum(-1, keepdim=True)
        keep = self.continue_span(hidden).sigmoid()
        positions = (1 - keep * surviving) * jump + keep * shifted
        read = torch.bmm(positions[:, None], mem).squeeze(1)
        state = torch.cat([hidden, read], -1)
        return hidden, positions, self.decision(state), self.ink(state)

    @torch.inference_mode()
    def write(self, passage, question, contiguous=False, limit=32):
        device = self.input.weight.device
        tiles, features, ids, group = article_memory(passage, question)
        features = torch.as_tensor(features, device=device)[None]
        valid = torch.ones(features.shape[:2], dtype=torch.bool, device=device)
        mem, hidden = self.encode(features, valid)
        groups = torch.as_tensor(ids, device=device)
        focus = torch.zeros_like(valid, dtype=features.dtype)
        feedback = torch.zeros_like(hidden)
        result, actions = [], []
        cursor = None
        for _ in range(limit + 1):
            hidden, positions, gate, ink = self.step(mem, valid, hidden, focus, feedback)
            action = int(gate.argmax(-1).item())
            if action == 2:
                return result, actions, True
            if len(result) == limit:
                break
            if contiguous:
                cursor = int(positions.argmax().item()) if cursor is None else cursor + 1
                if cursor == len(tiles):
                    break
                mask = torch.arange(len(tiles), device=device) == cursor
                cell = tiles[cursor]
                action = 0
            elif action == 0:
                masses = positions.new_zeros(len(group)).scatter_add_(0, groups, positions[0])
                mask = groups == masses.argmax()
                cell = tiles[int(mask.nonzero()[0, 0].item())]
            else:
                cell = np.packbits((ink[0] > 0).cpu().numpy())
                mask = groups == group.get(cell.tobytes(), -1)
            posterior = positions * mask
            focus = posterior / posterior.sum(-1, keepdim=True).clamp_min(1e-12)
            feedback = torch.as_tensor(visual_features(cell[None]), device=device)
            result.append(cell)
            actions.append(action)
        return result, actions, False
