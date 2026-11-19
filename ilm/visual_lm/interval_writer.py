"""A learned reading action for copying a complete run of source glyph images."""
import torch
from torch import nn
from torch.nn import functional as F

MAX_RUN = 64


class IntervalWriter(nn.Module):
    """Condition source boundaries on question-aligned visual features from the frozen writer."""

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(128, 64)
        self.context = nn.ModuleList(nn.Conv1d(64, 64, 5, dilation=d, padding=2*d) for d in (1, 4, 12))
        self.norm = nn.ModuleList(nn.LayerNorm(64) for _ in self.context)
        self.bounds = nn.Linear(64, 2)
        self.length = nn.Linear(64, MAX_RUN)
        self.ends = nn.Linear(64, 8)
        self.starts = nn.Linear(64, 8)
        self.action = nn.Sequential(nn.Linear(128+64, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, encoded, passage):
        valid = passage.ne(0)
        states = self.input(encoded[0][:,:passage.shape[1]])
        for layer, norm in zip(self.context, self.norm):
            neighbors = layer((states * valid[...,None]).transpose(1,2)).transpose(1,2)
            states = norm(states + F.gelu(neighbors))
        pooled = states.masked_fill(~valid[...,None], -1e4).amax(1)
        action = self.action(torch.cat((pooled, encoded[-1]), -1)).squeeze(-1)
        left, right = self.bounds(states).unbind(-1)
        legal = F.pad(valid, (0,MAX_RUN-1)).unfold(1,MAX_RUN,1)
        last = F.pad(self.ends(states).transpose(1,2), (0,MAX_RUN-1)).unfold(2,MAX_RUN,1)
        agreement = torch.einsum("bld,bdlk->blk", self.starts(states), last) / 8**.5
        logits = left[:,:,None] + F.pad(right,(0,MAX_RUN-1)).unfold(1,MAX_RUN,1)
        logits = logits + self.length(states) + agreement
        return action.float(), logits.float().masked_fill(~legal, -torch.inf)

    @torch.inference_mode()
    def generate(self, writer, ink, passage, question, memory_keys, memory, limit=64):
        action, scores = self(writer.encode(ink,passage,question), passage)
        choices = scores[:,:,:min(limit,MAX_RUN)].contiguous()
        if choices.shape[-1] == 0:
            return writer.generate(ink,passage,question,memory_keys,memory,limit=limit)
        pos = choices.flatten(1).argmax(1)
        start = pos // choices.shape[-1]
        length = pos % choices.shape[-1] + 1
        step = torch.arange(limit,device=passage.device)
        rows = passage.gather(1,(start[:,None]+step).clamp_max(passage.shape[1]-1))
        rows = rows.masked_fill(step >= length[:,None],0)
        read = action >= 0
        stopped = read.clone()
        if (~read).any():
            rows[~read],stopped[~read] = writer.generate(ink,passage[~read],question[~read],memory_keys,memory,limit=limit)
        return rows, stopped
