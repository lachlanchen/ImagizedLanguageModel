"""Shared visual representations for printed question streams in different scripts."""

from collections import defaultdict
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


class PicturedQuestionEncoder(nn.Module):
    def __init__(self, width=160):
        super().__init__()
        self.ink = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, width))
        self.local = nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(width, width, 3, padding=1),
        )
        self.context = nn.ModuleList(
            nn.TransformerEncoderLayer(
                width,
                4,
                3 * width,
                dropout=0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(2)
        )
        self.pool = nn.Linear(width, 1)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 128))
        i = torch.arange(512)[:, None]
        freq = torch.exp(-math.log(10000) * torch.arange(0, width, 2) / width)
        self.register_buffer(
            "position", torch.stack(((i * freq).sin(), (i * freq).cos()), -1).flatten(1)
        )

    def forward(self, questions, pixels):
        lengths = torch.tensor(list(map(len, questions)), device=pixels.device)
        if (
            not len(questions)
            or lengths.min() < 1
            or lengths.max() > len(self.position)
        ):
            raise ValueError("Expected questions of one to 512 glyph images")
        ids = pad_sequence(questions, batch_first=True)
        mask = (
            torch.arange(ids.shape[1], device=pixels.device)[None, :] < lengths[:, None]
        )
        # Only ink values enter the optical map. Index values are local storage offsets.
        unique, inverse = ids.unique(return_inverse=True)
        z = (
            self.ink(pixels.index_select(0, unique))[inverse]
            + self.position[: ids.shape[1]] * 0.1
        )
        z = z.masked_fill(~mask[:, :, None], 0)
        z = z + self.local(z.transpose(1, 2)).transpose(1, 2)
        for layer in self.context:
            z = layer(z, src_key_padding_mask=~mask)
        weights = (
            self.pool(z).squeeze(-1).float().masked_fill(~mask, -torch.inf).softmax(1)
        )
        return F.normalize(self.out((weights[:, :, None] * z).sum(1)).float(), dim=1)

    @staticmethod
    def pack(streams, device):
        cells = {}
        sequences = []
        for stream in streams:
            if not stream or len(stream) % 128:
                raise ValueError("Expected a stream of packed 32 by 32 glyphs")
            sequences.append(
                torch.tensor(
                    [
                        cells.setdefault(stream[i : i + 128], len(cells))
                        for i in range(0, len(stream), 128)
                    ],
                    device=device,
                )
            )
        raw = np.frombuffer(b"".join(cells), np.uint8).reshape(-1, 128)
        bits = (
            torch.from_numpy(np.unpackbits(raw, axis=1))
            .to(device)
            .float()
            .reshape(-1, 1, 32, 32)
        )
        return sequences, F.avg_pool2d(bits, 2).flatten(1)

    @torch.inference_mode()
    def encode(self, streams, batch_size=64):
        if not streams or batch_size < 1:
            raise ValueError("Provide at least one question and a positive batch size")
        ids, pixels = self.pack(streams, self.position.device)
        by_length = defaultdict(list)
        for i, seq in enumerate(ids):
            by_length[len(seq)].append(i)
        result = pixels.new_empty((len(ids), self.out[-1].out_features))
        for group in by_length.values():
            for begin in range(0, len(group), batch_size):
                selected = group[begin : begin + batch_size]
                result[selected] = self([ids[i] for i in selected], pixels)
        return result
