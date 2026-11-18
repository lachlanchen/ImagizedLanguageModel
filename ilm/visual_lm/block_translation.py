"""A visual transducer that assigns zero or more physical drawings to each source cell."""

import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.idiom_ink import InkMotor, nibbles, visual_input
from ilm.visual_lm.visual_edit_writer import InkPatterns


class BlockTranslator(nn.Module):
    slots = 8

    def __init__(self, landmarks, width=128):
        super().__init__()
        self.register_buffer("landmarks", landmarks.clone())
        self.motor = InkMotor().requires_grad_(False)
        self.shape = nn.Sequential(
            nn.Linear(112, width), nn.GELU(), nn.Linear(width, width)
        )
        self.pattern = nn.Embedding(12 * 16, width)
        self.register_buffer("pattern_ids", torch.arange(12) * 16)
        self.context = nn.GRU(
            width, width // 2, num_layers=2, bidirectional=True, batch_first=True
        )
        self.norm = nn.LayerNorm(width)
        self.length = nn.Linear(width, self.slots + 1)
        self.slot = nn.Embedding(self.slots, width)
        self.plan = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, width)
        )
        self.copy = nn.Linear(width, 1)
        self.draw = InkPatterns(width)

    def read(self, features, lengths):
        observed = nibbles(features[..., 64:]) + self.pattern_ids
        x = self.shape(features) + self.pattern(observed).mean(-2)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.context(packed)
        state, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True, total_length=x.shape[1]
        )
        return self.norm(x + state)

    def decision(self, state, slot):
        z = state + self.slot(slot)
        return z + self.plan(z)

    def loss(self, features, lengths, counts, target_bits, same_image):
        state = self.read(features, lengths)
        valid = (
            torch.arange(features.shape[1], device=state.device)[None]
            < lengths[:, None]
        )
        length_loss = F.cross_entropy(
            self.length(state)[valid], counts[valid].clamp_max(self.slots)
        )
        active = torch.arange(self.slots, device=state.device) < counts[:, :, None]
        b, c, s = active.nonzero(as_tuple=True)
        if not len(b):
            return length_loss
        plan = self.decision(state[b, c], s)
        copy = same_image[b, c, s]
        copy_loss = F.binary_cross_entropy_with_logits(
            self.copy(plan)[:, 0], copy.float()
        )
        if copy.all():
            return length_loss + copy_loss
        gold = target_bits[b[~copy], c[~copy], s[~copy]]
        draw = F.cross_entropy(
            self.draw.teacher(plan[~copy], gold).transpose(1, 2), nibbles(gold)
        )
        return length_loss + copy_loss + draw

    @torch.inference_mode()
    def generate(self, pictures, lengths, *, reverse_context=False):
        """Return unpadded streams of 32-by-32 cells. Empty translations are retained."""
        if (
            pictures.ndim != 3
            or pictures.size(-1) != 1024
            or len(lengths) != len(pictures)
        ):
            raise ValueError(
                "Expected a batch of source images and one length per source"
            )
        if (lengths < 1).any() or (lengths > pictures.size(1)).any():
            raise ValueError("Invalid visible source length")
        features = visual_input(pictures, self.landmarks)
        if reverse_context:
            position = torch.arange(pictures.size(1), device=pictures.device)[None, :]
            order = torch.where(
                position < lengths[:, None], lengths[:, None] - 1 - position, position
            )
            features = features.gather(1, order[..., None].expand_as(features))
        state = self.read(features, lengths)
        if reverse_context:
            state = state.gather(1, order[..., None].expand_as(state))
        counts = self.length(state).argmax(-1)
        counts.masked_fill_(
            torch.arange(pictures.size(1), device=state.device)[None]
            >= lengths[:, None],
            0,
        )
        b, c, s = (
            torch.arange(self.slots, device=state.device) < counts[:, :, None]
        ).nonzero(as_tuple=True)
        plan = self.decision(state[b, c], s)
        ink = pictures[b, c].clone()
        paint = self.copy(plan)[:, 0] < 0
        if paint.any():
            bits = self.draw.choose(plan[paint])
            z = self.motor(bits) > 0
            z[:, self.landmarks] = bits.bool()
            ink[paint] = z
        return ink.split(counts.sum(1).tolist())
