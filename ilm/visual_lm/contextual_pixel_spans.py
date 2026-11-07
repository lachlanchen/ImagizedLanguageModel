"""Question-conditioned spans of original ink columns; no language inventory."""
import hashlib

import numpy as np
import torch
from torch import nn


def units(image):
    a = np.asarray(image)
    if a.ndim != 2 or a.shape[0] != 32 or not np.isin(a, [0, 255]).all():
        raise ValueError("binary 32-pixel writing strip required")
    ink = (a == 0).any(0)
    edges = np.diff(np.r_[False, ink, False].astype(np.int8))
    spans = np.stack((np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)), 1)
    if len(spans) == 0:
        raise ValueError("visible source ink required")
    return spans


def normalized(image):
    a = np.asarray(image)
    _y, x = np.nonzero(a == 0)
    if not len(x):
        return np.full((32, 0), 255, np.uint8)
    return a[:, x.min():x.max() + 1].copy()


def image_key(image):
    a = normalized(image)
    return a.shape, hashlib.sha256(a.T.copy().tobytes()).hexdigest()


def candidate_keys(image, spans, maximum=128):
    columns = memoryview(np.asarray(image).T.copy()).cast("B")
    for start in range(len(spans)):
        h = hashlib.sha256()
        previous = int(spans[start, 0])
        for length in range(1, min(maximum, len(spans) - start) + 1):
            end = int(spans[start + length - 1, 1])
            h.update(columns[previous * 32:end * 32])
            previous = end
            yield start, length, ((32, end - int(spans[start, 0])), h.digest().hex())


def copied(image, spans, start, length):
    if start < 0 or length < 1 or start + length > len(spans):
        raise ValueError("selected source span is unavailable")
    crop = np.asarray(image)[:, spans[start, 0]:spans[start + length - 1, 1]]
    return np.pad(crop, ((0, 0), (2, 2)), constant_values=255)


def matches(image, spans, targets, maximum=128):
    """Targets enter training/scoring only; compare complete physical rasters."""
    keys = {image_key(t) for t in targets}
    widths = {key[0][1] for key in keys}
    result = []
    for start, length, key in candidate_keys(image, spans, maximum):
        if key[0][1] in widths and key in keys:
            result.append((start, length - 1))
    if not result:
        raise ValueError("target image is not an admissible visible source span")
    return result


class ContextualPixelSpans(nn.Module):
    maximum = 128

    def __init__(self, width=256):
        super().__init__()
        self.visual = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, width))
        self.query = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, width))
        self.score = nn.Sequential(nn.Linear(3 * width + 4, width), nn.GELU(), nn.Linear(width, width),
                                   nn.GELU(), nn.Linear(width, 2))
        self.length_bias = nn.Parameter(torch.zeros(self.maximum))

    def forward(self, visual, memory, geometry, lengths):
        v = self.visual(visual.float())
        q = self.query(memory.float()).mean(1)[:, None].expand_as(v)
        scores = self.score(torch.cat((v, q, v * q, geometry.float()), -1))
        n = visual.shape[1]
        starts = torch.arange(n, device=visual.device)[:, None]
        ends = starts + torch.arange(self.maximum, device=visual.device)[None]
        s = scores[:, :, 0, None] + scores[:, ends.clamp(max=n - 1), 1] + self.length_bias
        valid = ends[None] < lengths[:, None, None]
        return s.masked_fill(~valid, -torch.inf).flatten(1).log_softmax(-1).reshape(-1, n, self.maximum)

    @staticmethod
    def loss(logp, correct):
        return torch.stack([-p[torch.tensor(c, device=p.device)[:, 0], torch.tensor(c, device=p.device)[:, 1]].logsumexp(0)
                            for p, c in zip(logp, correct)]).mean()

    @torch.no_grad()
    def select(self, logp, image, spans):
        """Sum duplicate complete images; never inspect an answer/reference."""
        p = logp.double().exp().cpu().numpy()
        mass, representatives = {}, {}
        for start, length, k in candidate_keys(image, spans, self.maximum):
            mass[k] = mass.get(k, 0.) + float(p[start, length - 1])
            representatives.setdefault(k, (start, length))
        key = max(mass, key=mass.get)
        start, length = representatives[key]
        return copied(image, spans, start, length), dict(start=start, length=length, marginal_probability=mass[key])
