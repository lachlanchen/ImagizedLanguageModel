"""Controlled optical and byte observation models for the same dialogue writer."""

import numpy as np
import torch
from torch.nn import functional as F
from ilm.visual_lm.dialogue_ink import DialogueInk
from ilm.visual_lm.gloss_focus import cell_features


def interval_occupancy(endpoints):
    """Marginal P(left <= i <= right | left <= right) for every source word."""
    left, right = endpoints.unbind(-1)
    before = torch.logcumsumexp(left, dim=1)
    after = torch.logcumsumexp(right.flip((1,)), dim=1).flip((1,))
    normalizer = (left + after).logsumexp(1, keepdim=True)
    return (before + after - normalizer).clamp(max=0).exp()


class SurfaceDialogue(DialogueInk):
    def __init__(self, palette, mode):
        super().__init__(palette)
        if mode not in ("ink", "byte"):
            raise ValueError(mode)
        self.mode = mode
        seed = torch.Generator().manual_seed(73161)
        self.register_buffer(
            "observation_projection", torch.randn(256, 66, generator=seed)
        )
        self.register_buffer(
            "position_sign", torch.randint(2, (128, 50), generator=seed).float() * 2 - 1
        )

    def memory(
        self,
        passage,
        passage_length,
        question,
        question_length,
        history,
        history_length,
    ):
        memory, endpoints = super().memory(
            passage, passage_length, question, question_length, history, history_length
        )
        if getattr(self, "use_predicted_interval", False):
            values, keys, edits, mask, preceding, state = memory
            bonus = (3 * interval_occupancy(endpoints)).log1p()
            bonus = F.pad(bonus, (0, edits.size(1) - passage.size(1)))
            memory = values, keys, edits + bonus[..., None], mask, preceding, state
        return memory, endpoints

    def observe(self, packed):
        if self.mode == "byte":
            byte = packed[..., 0].long()
            return F.one_hot(byte, 256).float() * (byte != 0)[..., None]
        return F.normalize(cell_features(packed), dim=-1)

    def letter_inputs(self):
        return self.letter_in(self.observe(self.palette))

    def surface(self, packed, lengths, unused=None):
        obs = self.observe(packed) @ self.observation_projection
        time = torch.arange(obs.size(1), device=obs.device)
        weights = self.position_sign[time % len(self.position_sign)]
        word = F.normalize(
            (obs[:, :, :50] * weights * (time[None] < lengths[:, None])[..., None]).sum(
                1
            ),
            dim=-1,
        )
        last = lengths - 1
        places = torch.stack(
            (last * 0, last.clamp(max=1), (last - 1).clamp(min=0), last), 1
        )
        edge = obs[torch.arange(len(obs), device=obs.device)[:, None], places, 50:]
        return torch.cat(
            (word, edge.flatten(1), lengths[:, None].float().log1p() / 3), 1
        )

    @torch.no_grad()
    def encode_words(self, reader, unused, words):
        lengths = [len(w) for w in words]
        packed = np.zeros((len(words), max(lengths), words[0].shape[1]), np.uint8)
        for j, w in enumerate(words):
            packed[j, : len(w)] = w
        device = reader.sketch.device
        return self.surface(
            torch.from_numpy(packed).to(device), torch.tensor(lengths, device=device)
        )
