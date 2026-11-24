"""Question-conditioned latent focus for an independent pictured gloss writer."""

import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.visual_word_sense import RADIUS, WIDTH, retinal_features


def cell_features(packed):
    shifts = torch.arange(7, -1, -1, device=packed.device)
    pixels = ((packed[..., None].long() >> shifts) & 1).flatten(-2)
    return retinal_features(pixels)


def encoding_windows(writer, features):
    padded = F.pad(features, (0, 0, RADIUS, RADIUS))
    windows = padded.unfold(1, 2*RADIUS+1, 1).transpose(-2, -1)
    batch, cells = features.shape[:2]
    return torch.cat([
        writer.encode(chunk) for chunk in windows.reshape(-1, 2*RADIUS+1, 256).split(384)
    ]).reshape(batch, cells, WIDTH)


class GlossFocus(nn.Module):
    def __init__(self):
        super().__init__()
        self.page = nn.GRU(WIDTH + 2, 48, bidirectional=True, batch_first=True)
        self.keys = nn.Linear(96 + WIDTH, 96)
        self.q_input = nn.Linear(256, 64)
        self.q_conv = nn.Conv1d(64, 64, 3, padding=1)
        self.query = nn.Linear(128, 96)

    def read_questions(self, features, lengths):
        mask = torch.arange(features.shape[1], device=features.device) < lengths[:, None]
        x = F.gelu(self.q_input(features)) * mask[:, :, None]
        x = x + F.gelu(self.q_conv(x.transpose(1, 2))).transpose(1, 2)
        mean = (x*mask[:, :, None]).sum(1) / lengths[:, None]
        peak = x.masked_fill(~mask[:, :, None], -torch.inf).amax(1)
        return self.query(torch.cat((mean, peak), -1))

    def forward(self, windows, lengths, questions):
        pos = torch.arange(windows.shape[1], device=windows.device)[None]
        valid = pos < lengths[:,None]
        geom = torch.stack((pos.expand_as(valid)/31, pos/(lengths[:,None]-1).clamp_min(1)), -1)
        x = torch.cat((windows, geom), -1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        hidden, _ = self.page(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True, total_length=windows.shape[1])
        keys = self.keys(torch.cat((windows, h), -1))
        return (torch.einsum("bld,bqd->bql",keys,questions)/96**0.5).masked_fill(~valid[:,None], -torch.inf)
