"""An order-sensitive pictured question on the same visual focus and writing path."""

import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.gloss_focus import GlossFocus


class OrderedGlossFocus(GlossFocus):
    def __init__(self):
        super().__init__()
        del self.q_conv
        self.q_sequence = nn.GRU(64, 64, batch_first=True, bidirectional=True)

    def read_questions(self, features, lengths):
        x = F.gelu(self.q_input(features))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        sequence, _ = self.q_sequence(packed)
        sequence, _ = nn.utils.rnn.pad_packed_sequence(sequence, batch_first=True)
        mask = torch.arange(sequence.shape[1], device=x.device) < lengths[:, None]
        return self.query(sequence.masked_fill(~mask[...,None], -torch.inf).amax(1))
