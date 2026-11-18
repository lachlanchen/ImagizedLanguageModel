"""Autoregressive copy-or-write model for pictured, field-labelled entries.

A cell is 128 packed bytes (32 by 32 ink bits). The only document grammar
specifies two header brackets, a field separator and a field terminator.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from torch import nn

CELL = 128
VIS = 33
FEATURES = 4 + 4 + 32 + VIS


def cells(stream):
    if len(stream) % CELL:
        raise ValueError("Incomplete packed glyph")
    return tuple(stream[i : i + CELL] for i in range(0, len(stream), CELL))


def parse_card(raw, marks):
    opening, closing, colon, end = marks
    records = []
    seq = cells(raw)
    i = 0
    while i < len(seq):
        if seq[i] != opening:
            raise ValueError("Expected record header")
        j = seq.index(closing, i + 1)
        head = seq[i + 1 : j]
        fields = []
        i = j + 1
        while i < len(seq) and seq[i] != opening:
            sep = seq.index(colon, i)
            stop = seq.index(end, sep + 1)
            if sep == i or stop == sep + 1:
                raise ValueError("Empty field")
            fields.append((seq[i:sep], seq[sep + 1 : stop]))
            i = stop + 1
        if not head or not fields:
            raise ValueError("Empty record")
        records.append((head, fields))
    return records


@lru_cache(maxsize=32000)
def thumbnail(glyph):
    ink = np.unpackbits(np.frombuffer(glyph, np.uint8)).reshape(4, 8, 4, 8)
    return ink.mean((1, 3)).astype(np.float32).ravel()


@lru_cache(maxsize=32000)
def appearance(span):
    if not span:
        return np.zeros(VIS, np.float32)
    return np.r_[
        thumbnail(span[0]), thumbnail(span[-1]), np.float32(np.log1p(len(span)) / 4)
    ]


def location(part, query):
    for i in range(len(query) - len(part) + 1):
        if query[i : i + len(part)] == part:
            return 1.0, i / max(1, len(query))
    return 0.0, 0.0


@dataclass
class Options:
    payload: list
    features: np.ndarray


def alternatives(records, query, literals):
    query = cells(query)
    actions = []
    for head, fields in records:
        actions.append((0, head, (), head))
        for key, value in fields:
            actions.extend([(1, head, key, key), (2, head, key, value)])
    actions.extend((3, (), (), (glyph,)) for glyph in literals)
    # Stop is a separate learned output and has no physical payload.
    features = np.zeros((len(actions), FEATURES), np.float32)
    for i, (kind, head, key, payload) in enumerate(actions):
        f = features[i]
        f[kind] = 1
        if head:
            f[4:6] = location(head, query)
        if key:
            f[6:8] = location(key, query)
            f[8:40] = appearance(key)[:32]
        f[40:] = appearance(payload)
    return Options([v[-1] for v in actions], features)


def alignment(options, answer):
    """Shortest segmentation; sum supervision over equal physical emissions."""
    goal = cells(answer)
    by_first = {}
    for i, span in enumerate(options.payload):
        by_first.setdefault(span[0], {}).setdefault(span, []).append(i)
    costs = [len(goal) + 1] * len(goal) + [0]
    path = {}
    for i in reversed(range(len(goal))):
        for span, choices in by_first.get(goal[i], {}).items():
            j = i + len(span)
            if goal[i:j] == span and costs[j] + 1 < costs[i]:
                costs[i] = costs[j] + 1
                path[i] = span, choices
    if goal and 0 not in path:
        raise ValueError("Unreachable training target")
    i = 0
    feedback, labels = [appearance(())], []
    while i < len(goal):
        span, choices = path[i]
        feedback.append(appearance(span))
        labels.append(choices)
        i += len(span)
    labels.append([len(options.payload)])
    return np.stack(feedback), labels


class EntryWriter(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.keys = nn.Sequential(
            nn.Linear(FEATURES, width), nn.Tanh(), nn.Linear(width, width)
        )
        self.read_history = nn.GRU(VIS, width, batch_first=True)
        self.start = nn.Parameter(torch.zeros(width))
        self.stop_key = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.stop_key, std=0.1)

    def encode(self, features):
        stop = self.stop_key.expand(len(features), 1, -1)
        return torch.cat((self.keys(features), stop), 1)

    def forward(self, keys, feedback, state=None):
        if state is None:
            state = self.start.expand(1, len(keys), -1).contiguous()
        h, state = self.read_history(feedback, state)
        return h @ keys.transpose(1, 2), state

    @torch.inference_mode()
    def write(self, options, max_steps=72):
        """Greedy execution; never receives a reference span or expected length."""
        device = self.start.device
        keys = self.encode(
            torch.from_numpy(np.stack([o.features for o in options])).to(device)
        )
        feedback = torch.zeros((len(keys), 1, VIS), device=device)
        outputs = [[] for _ in options]
        stopped = [False] * len(options)
        state = None
        for _ in range(max_steps):
            logits, state = self(keys, feedback, state)
            choices = logits[:, 0].argmax(-1).tolist()
            previous = []
            for i, (o, a) in enumerate(zip(options, choices)):
                stopped[i] |= a == len(o.payload)
                span = () if stopped[i] else o.payload[a]
                outputs[i].extend(span)
                previous.append(appearance(span))
            if all(stopped):
                break
            feedback = torch.from_numpy(np.stack(previous)).to(device).unsqueeze(1)
        return [b"".join(v) for v in outputs], stopped
