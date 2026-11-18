"""Read a pictured question and choose a source span or the recurrent ink writer."""
import torch
from torch import nn
from torch.nn import functional as F


class MixedInkReader(nn.Module):
    def __init__(self, width=48):
        super().__init__()
        self.shape = nn.Sequential(nn.Linear(112, width), nn.GELU())
        self.question = nn.Conv1d(width, width, 3, padding=1)
        self.route = nn.Linear(width * 2, 1)
        self.local = nn.Linear(4 * width + 14, width)
        self.context = nn.ModuleList(nn.Conv1d(width, width, 5, dilation=d, padding=2*d)
                                     for d in (1, 4, 12))
        self.boundaries = nn.Linear(width, 2)
        self.duration = nn.Linear(width * 2, 49)

    def forward(self, features, passage, question, lengths, qlengths):
        # Handles index input pixels only. Padding is distinguished using lengths.
        pos = torch.arange(passage.shape[1], device=passage.device)
        qpos = torch.arange(question.shape[1], device=question.device)
        valid = pos[None] < lengths[:, None]
        qvalid = qpos[None] < qlengths[:, None]
        p = self.shape(features[passage]) * valid.unsqueeze(-1)
        q = self.shape(features[question]) * qvalid.unsqueeze(-1)
        q = q + F.gelu(self.question(q.transpose(1, 2)).transpose(1, 2))
        summary = torch.cat(((q*qvalid.unsqueeze(-1)).sum(1)/qlengths[:,None],
                             q.masked_fill(~qvalid.unsqueeze(-1), -1e4).amax(1)), -1)
        same = (passage[:,:,None] == question[:,None,:]) & qvalid[:,None] & valid[:,:,None]
        weights = (p @ q.transpose(1,2)) / p.shape[-1]**.5 + 3 * same
        attended = weights.masked_fill(~qvalid[:,None], -1e4).softmax(-1) @ q
        # Exact physical runs of one, two and three consecutive matching images.
        match = same
        run = []
        for offset in range(3):
            if offset:
                match = match[:,:-1,:-1] & same[:,offset:,offset:]
            if match.shape[1] and match.shape[2]:
                covered = match.any(-1).float()
                run.extend([F.pad(covered,(0,offset)), F.pad(covered,(offset,0))])
            else:
                run.extend([torch.zeros_like(p[:,:,0])] * 2)
        single = run[0]
        for radius in (3, 12):
            cumulative = F.pad(single.cumsum(1), (1,0))
            run.extend([(cumulative[:,pos] - cumulative[:,(pos-radius).clamp_min(0)])/radius,
                        (cumulative[:,(pos+radius+1).clamp_max(len(pos))] - cumulative[:,pos+1])/radius])
        relative = pos[None] - lengths[:,None]//2
        run.extend([pos[None]/(lengths[:,None]-1).clamp_min(1), (relative/8).clamp(-1,1),
                    (-relative.abs()/4).exp(), same.sum(-1)/qlengths[:,None]])
        mixed = torch.cat((p, attended, summary[:,None].expand(-1,len(pos),-1), torch.stack(run,-1)), -1)
        state = F.gelu(self.local(mixed)) * valid.unsqueeze(-1)
        for conv in self.context:
            update = F.gelu(conv(state.transpose(1,2)).transpose(1,2))
            state = (state + update) * valid.unsqueeze(-1)
        score = self.boundaries(state).masked_fill(~valid.unsqueeze(-1), -1e4)
        return self.route(summary).squeeze(1), score[:,:,0], score[:,:,1], self.duration(summary)

    @staticmethod
    def end_scores(end, duration, starts, lengths):
        # The learned tail bin shares a bias; every legal end still competes.
        pos = torch.arange(end.shape[1], device=end.device)
        distance = pos - starts[...,None]
        bins = distance.clamp(0, 48)
        rows = torch.arange(len(end), device=end.device)[:,None,None]
        logits = end[:,None] + duration[rows,bins]
        return logits.masked_fill((distance<0) | (pos >= lengths[:,None,None]), -1e4)

    def decode(self, outputs, lengths):
        gate, start, end, duration = outputs
        initial, starts = start.log_softmax(1).topk(min(12,start.shape[1]), dim=1)
        endings = self.end_scores(end, duration, starts, lengths).log_softmax(-1)
        stop_score, stops = endings.max(-1)
        choice = (initial + stop_score).argmax(1,keepdim=True)
        return gate > 0, starts.gather(1,choice).flatten(), stops.gather(1,choice).flatten()+1
