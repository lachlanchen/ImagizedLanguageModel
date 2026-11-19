"""Autoregressive actions that copy a variable source span, emit stored ink, or stop."""
import torch
from torch import nn
from torch.nn import functional as F

COPY, WRITE, STOP = range(3)
MAX_SPAN = 64


class OrderedPageWriter(nn.Module):
    def __init__(self, memory_size, width=128):
        super().__init__()
        self.width = width
        self.input = nn.Sequential(nn.LayerNorm(773), nn.Linear(773,width))
        self.neighbors = nn.Conv1d(width,width,7,padding=3)
        self.local = nn.LayerNorm(width)
        self.initial = nn.Sequential(nn.LayerNorm(768),nn.Linear(768,width),nn.Tanh())
        self.history = nn.GRU(width,width,batch_first=True)
        self.start_key = nn.Linear(width,width)
        self.start_query = nn.Linear(width,width)
        self.end_key = nn.Linear(width,width)
        self.end_query = nn.Linear(2*width,width)
        self.length = nn.Linear(2*width,MAX_SPAN)
        self.choose = nn.Linear(2*width,3)
        self.emit = nn.Linear(2*width,memory_size)
        self.written = nn.Embedding(memory_size,width)
        self.copied = nn.Linear(2*width,width)
        self.begin = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.begin,std=.03)

    def encode(self, global_state, source, mask):
        features = self.input(source) * mask.unsqueeze(-1)
        context = self.neighbors(features.transpose(1,2)).transpose(1,2)
        values = self.local(features + F.gelu(context))
        return self.initial(global_state).unsqueeze(0), values, self.start_key(values), self.end_key(values)

    def choice(self, hidden, values, keys, prior, mask):
        starts = (self.start_query(hidden) @ keys.transpose(1,2)) / self.width**.5
        starts = (starts + prior[:,None,:]).masked_fill(~mask[:,None,:], -1e4)
        state = torch.cat((hidden,starts.softmax(-1) @ values),dim=-1)
        return self.choose(state),self.emit(state),starts

    def sizes(self, hidden, values, keys, prior, mask, batch, starts):
        q = torch.cat((hidden,values[batch,starts]),dim=-1)
        positions = starts.unsqueeze(-1) + torch.arange(MAX_SPAN,device=values.device)
        legal = positions < mask.sum(1)[batch].unsqueeze(-1)
        positions = positions.clamp_max(values.shape[1]-1)
        at = batch.unsqueeze(-1)
        scores = (keys[at,positions] * self.end_query(q).unsqueeze(-2)).sum(-1) / self.width**.5
        return (scores + self.length(q) + prior[at,positions]).masked_fill(~legal,-1e4)

    def feedback(self, values, action, left, right, glyph):
        b = torch.arange(len(values),device=values.device)[:,None]
        copied = self.copied(torch.cat((values[b,left],values[b,right]),dim=-1))
        return torch.where((action==WRITE)[:,:,None],self.written(glyph),copied)

    def loss(self, global_state, source, mask, targets):
        action,left,right,glyph = targets.unbind(-1)
        h0,v,ks,ke = self.encode(global_state,source,mask)
        fb = self.feedback(v,action,left,right,glyph)
        x = torch.cat((self.begin.expand(len(v),1,-1),fb[:,:-1]),dim=1)
        hidden,_ = self.history(x,h0)
        mode,writes,starts = self.choice(hidden,v,ks,source[:,:,768],mask)
        n = (action>=0).sum()
        loss = F.cross_entropy(mode.transpose(1,2),action,ignore_index=-1,reduction="sum")
        copy = action==COPY
        if copy.any():
            # Gather only copy actions, avoiding long padded B x T x 64 x D endpoint tensors.
            b,t = copy.nonzero(as_tuple=True)
            at = left[b,t]
            scores = self.sizes(hidden[b,t],v,ke,source[:,:,769],mask,b,at)
            loss = loss + F.cross_entropy(starts[b,t],at,reduction="sum")
            loss = loss + F.cross_entropy(scores,right[b,t]-at,reduction="sum")
        write = action==WRITE
        if write.any():
            loss = loss + F.cross_entropy(writes[write],glyph[write],reduction="sum")
        return loss/n

    @torch.inference_mode()
    def generate(self, global_state, source, mask, max_steps=200):
        state,v,ks,ke = self.encode(global_state,source,mask)
        x = self.begin.expand(len(v),1,-1)
        complete = torch.zeros(len(v),dtype=torch.bool,device=v.device)
        history = []
        batch = torch.arange(len(v),device=v.device)[:,None]
        for _ in range(max_steps):
            h,state = self.history(x,state)
            mode,writes,starts = self.choice(h,v,ks,source[:,:,768],mask)
            action = mode.argmax(-1).masked_fill(complete[:,None],STOP)
            left = starts.argmax(-1)
            right = left + self.sizes(h,v,ke,source[:,:,769],mask,batch,left).argmax(-1)
            glyph = writes.argmax(-1)
            history.append(torch.stack((action,left,right,glyph),dim=-1)[:,0])
            complete |= action[:,0]==STOP
            if complete.all():
                break
            x = self.feedback(v,action,left,right,glyph)
        return torch.stack(history,dim=1).cpu().numpy()
