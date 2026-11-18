"""A recurrent page writer whose question-dependent update can differ by candidate."""

import torch
from torch import nn
from ilm.visual_lm.entry_question_reader import QuestionReader


class ConditionedEntryReader(QuestionReader):
    def __init__(self, candidate_dependent):
        super().__init__()
        self.candidate_dependent = candidate_dependent
        self.gain = nn.Parameter(torch.zeros(()))

    def read(self, features, positions, images, valid):
        length = valid.sum(1).clamp_min(1)
        features = features.clone()
        match = positions >= 0
        features[:, :, 4:8:2] = match
        features[:, :, 5:8:2] = positions.clamp_min(0) / length[:, None, None]
        keys = self.encode(features)
        strip = self.retina_conv(images.transpose(1, 2)).transpose(1, 2)
        strip = strip * valid.unsqueeze(-1)
        global_view = strip.sum(1) / length[:, None]
        anchor = torch.where(match[:, :, 1], positions[:, :, 1], positions[:, :, 0])
        local = strip[
            torch.arange(len(strip), device=strip.device)[:, None],
            anchor.clamp_min(0).long(),
        ]
        local = local * (anchor >= 0).unsqueeze(-1)
        residual = self.context(
            torch.cat((local, global_view[:, None].expand_as(local)), dim=-1)
        )
        # At gain zero both arms reproduce the shared checkpoint. The scalar control
        # can rescale the question signal but cannot separate candidates at one anchor.
        scale = (
            self.gain * torch.tanh(keys[:, :-1])
            if self.candidate_dependent
            else self.gain
        )
        return torch.cat((keys[:, :-1] + residual * (1 + scale), keys[:, -1:]), dim=1)
