"""Read a marked cell in its visual context and write a variable-length gloss."""

import torch
from torch import nn
from torch.nn import functional as F

RADIUS = 8
WIDTH = 160


def retinal_features(pixels):
    """Pool each 32×32 cell independently, preserving its position in the stream."""
    return F.avg_pool2d(pixels.reshape(-1, 1, 32, 32).float(), 2).reshape(
        *pixels.shape[:-1], 256
    )


class VisualGlossWriter(nn.Module):
    def __init__(self, landmarks):
        super().__init__()
        self.register_buffer("landmarks", torch.as_tensor(landmarks, dtype=torch.long))
        k = len(landmarks)
        self.register_buffer(
            "bits", ((torch.arange(2**k)[:, None] >> torch.arange(k)) & 1).float()
        )
        self.stop = 2**k
        self.cell = nn.Sequential(
            nn.Linear(256, WIDTH),
            nn.LayerNorm(WIDTH),
            nn.GELU(),
            nn.Linear(WIDTH, WIDTH),
        )
        self.local = nn.Conv1d(WIDTH, WIDTH, 3, padding=1)
        self.wide = nn.Conv1d(WIDTH, WIDTH, 5, padding=4, dilation=2)
        self.context = nn.Sequential(
            nn.LayerNorm(7 * WIDTH),
            nn.Linear(7 * WIDTH, 2 * WIDTH),
            nn.GELU(),
            nn.Linear(2 * WIDTH, WIDTH),
            nn.LayerNorm(WIDTH),
        )
        self.prefix = nn.Linear(k + 1, 48)
        self.speller = nn.GRU(WIDTH + 48, WIDTH, batch_first=True)
        self.distribution = nn.Linear(WIDTH, self.stop + 1)
        self.motor = nn.Sequential(
            nn.Linear(k, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 1024),
        )

    def encode(self, features):
        x = self.cell(features)
        loc = F.gelu(self.local(x.transpose(1, 2))).transpose(1, 2)
        far = F.gelu(self.wide(loc.transpose(1, 2))).transpose(1, 2)
        z = x + far + loc
        center = features.shape[1] // 2
        v = torch.cat(
            (x[:, center], z[:, center - 2 : center + 3].flatten(1), z.mean(1)), dim=-1
        )
        return self.context(v)

    def feedback(self, previous):
        bit = self.bits[previous.clamp_max(self.stop - 1)]
        begin = (previous == self.stop).unsqueeze(-1)
        return self.prefix(
            torch.cat((bit.masked_fill(begin, 0), begin.float()), -1)
        ).tanh()

    def teacher(self, context, codes):
        initial = torch.full_like(codes[:, :1], self.stop)
        prior = self.feedback(torch.cat((initial, codes[:, :-1]), 1))
        state, _ = self.speller(
            torch.cat((context[:, None, :].expand(-1, codes.shape[1], -1), prior), -1),
            context.unsqueeze(0).contiguous(),
        )
        return self.distribution(state)

    @torch.inference_mode()
    def write(self, features, limit=32):
        context = self.encode(features)
        state = context.unsqueeze(0).contiguous()
        prev = torch.full((len(features),), self.stop, device=features.device)
        result = []
        ended = torch.zeros_like(prev, dtype=torch.bool)
        for _ in range(limit):
            inputs = torch.cat((context, self.feedback(prev)), -1)[:, None]
            out, state = self.speller(inputs, state)
            prev = self.distribution(out[:, 0]).argmax(-1).masked_fill(ended, self.stop)
            result.append(prev)
            ended |= prev == self.stop
            if ended.all():
                break
        return torch.stack(result, 1), ended

    def draw(self, codes):
        return self.motor(self.bits[codes.clamp_max(self.stop - 1)]) > 0
