"""Read a pictured question and jointly locate the two ends of a clause relation."""

import torch
from torch import nn
from ilm.visual_lm.ordered_gloss_question import OrderedGlossFocus


class VisualClauseRoles(OrderedGlossFocus):
    def __init__(self):
        super().__init__()
        self.role = nn.Linear(96, 2)
        self.endpoints = nn.Linear(96, 2)
        self.verb_link = nn.Linear(96, 24, bias=False)
        self.object_link = nn.Linear(96, 24, bias=False)

    def arc_scores(self, windows, lengths):
        pos = torch.arange(windows.shape[1], device=windows.device)[None]
        valid = pos < lengths[:, None]
        geometry = torch.stack((pos.expand_as(valid)/31, pos/(lengths[:,None]-1).clamp_min(1)), -1)
        packed = nn.utils.rnn.pack_padded_sequence(torch.cat((windows,geometry),-1),
                   lengths.cpu(),batch_first=True,enforce_sorted=False)
        hidden, _ = self.page(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(hidden,batch_first=True,total_length=windows.shape[1])
        keys = self.keys(torch.cat((windows,h),-1)).tanh()
        unary = self.endpoints(keys)
        link = self.verb_link(keys) @ self.object_link(keys).transpose(1,2)
        scores = unary[:,:,0,None] + unary[:,None,:,1] + link / 24**0.5
        # A singleton has only one possible focus for the generic image API.
        diagonal = (pos[:,:,None]==pos[:,None,:]) & (lengths[:,None,None]>1)
        return scores.masked_fill(~valid[:,:,None] | ~valid[:,None,:] | diagonal,-torch.inf)

    def forward(self, windows, lengths, questions):
        scores = self.arc_scores(windows,lengths)
        width = scores.shape[-1]
        arc = scores.flatten(1).argmax(-1)
        endpoints = torch.stack((arc//width,arc%width),-1)
        positions = endpoints.gather(1,self.role(questions).argmax(-1))
        return scores.new_full((*positions.shape,width),-torch.inf).scatter_(2,positions[...,None],0.)
