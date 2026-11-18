"""A recurrent visual answer distribution over complete familiar shapes and pictured source cells."""
import torch
from torch import nn
from torch.nn import functional as F

WIDTH = 128


class WindowEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(WIDTH, WIDTH, 3, padding=d, dilation=d) for d in (1, 3))
        self.norms = nn.ModuleList(nn.LayerNorm(WIDTH) for _ in self.convs)

    def forward(self, x, valid):
        for conv, norm in zip(self.convs, self.norms):
            x = x * valid[..., None]
            x = norm(x + F.gelu(conv(x.transpose(1, 2)).transpose(1, 2)))
        return x * valid[..., None]


class GroundedGlyphDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ink = nn.Sequential(nn.Linear(256, WIDTH*2), nn.GELU(), nn.Linear(WIDTH*2, WIDTH), nn.LayerNorm(WIDTH))
        self.question = WindowEncoder()
        self.passage = WindowEncoder()
        self.q_pool = nn.Linear(WIDTH, 1)
        self.q_key = nn.Linear(WIDTH, WIDTH)
        self.p_key = nn.Linear(WIDTH, WIDTH)
        self.merge = nn.Linear(3*WIDTH + 3, WIDTH)
        self.summary = nn.Linear(2*WIDTH, WIDTH)
        self.start = nn.Parameter(torch.zeros(WIDTH))
        self.recurrence = nn.GRU(WIDTH, WIDTH, batch_first=True)
        self.address = nn.Linear(WIDTH, WIDTH)
        self.source_key = nn.Linear(WIDTH, WIDTH)
        self.salience = nn.Linear(WIDTH, 1)
        self.copy_history = nn.Parameter(torch.tensor([1.5, .7, .3]))
        self.focus = nn.Sequential(nn.Linear(3*WIDTH, WIDTH), nn.GELU(), nn.LayerNorm(WIDTH))
        self.action = nn.Linear(WIDTH, 3)  # stop, write, copy
        self.write_key = nn.Linear(WIDTH, WIDTH)
        self.write_query = nn.Linear(WIDTH, WIDTH)

    def encode(self, ink, passage, question):
        pv, qv = passage.ne(0), question.ne(0)
        p = self.passage(ink[passage], pv)
        q = self.question(ink[question], qv)
        qweight = self.q_pool(q).squeeze(-1).masked_fill(~qv, -1e4).softmax(-1)
        summary = (qweight[...,None]*q).sum(1)
        same = (passage[:,:,None]==question[:,None,:]) & pv[:,:,None] & qv[:,None,:]
        affinities = self.p_key(p) @ self.q_key(q).transpose(1, 2) / WIDTH**.5 + 2*same
        aligned = affinities.masked_fill(~qv[:,None], -1e4).softmax(-1) @ q
        pos = torch.arange(p.shape[1], device=p.device)[None].expand_as(passage)
        location = pos / pv.sum(1, keepdim=True).sub(1).clamp_min(1)
        features = torch.stack((same.any(-1), location, pos==0), dim=-1).to(p.dtype)
        p = p + F.gelu(self.merge(torch.cat((p,aligned,summary[:,None].expand_as(p),features), -1)))
        pweight = self.salience(p).squeeze(-1).masked_fill(~pv,-1e4).softmax(-1)
        context = self.summary(torch.cat((summary,(pweight[...,None]*p).sum(1)), -1)).tanh()
        source = torch.cat((p,q),dim=1)
        handles = torch.cat((passage,question),dim=1)
        return source, handles, self.source_key(source), context

    def states(self, previous_ink, context, hidden=None):
        if hidden is None:
            hidden = context.unsqueeze(0).contiguous()
        return self.recurrence(previous_ink + context[:,None], hidden)

    def predict(self, encoded, state, history_bias, memory_keys):
        source, handles, keys, context = encoded
        pointer = self.address(state) @ keys.transpose(1,2) / WIDTH**.5
        pointer = (pointer + history_bias).masked_fill(handles[:,None]==0, -1e4)
        read = pointer.softmax(-1) @ source
        focus = self.focus(torch.cat((state,read,context[:,None].expand_as(state)),-1))
        shape = self.write_query(focus) @ memory_keys.T / WIDTH**.5
        return self.action(focus), shape, pointer

    def history_bias(self, handles, previous, passage_extent):
        # At position j count consecutive preceding source tiles matching the output prefix.
        # Prevent a match from continuing across the boundary between the two images.
        same = (previous[:,:,None]==handles[:,None]) & (previous[:,:,None]>0)
        one = F.pad(same[:,:,:-1], (1,0))
        one[:,:,passage_extent] = False
        run = one
        bias = self.copy_history[0]*run
        for weight in self.copy_history[1:]:
            run = one & F.pad(run[:,:-1,:-1],(1,0,1,0))
            bias = bias + weight*run
        return bias

    def likelihood(self, ink, passage, question, answer, memory_keys, memory_ids):
        encoded = self.encode(ink,passage,question)
        previous = F.pad(answer,(1,0))
        valid = F.pad(answer.ne(0),(1,0),value=True)
        target = F.pad(answer,(0,1))
        feedback = torch.cat((self.start.expand(len(answer),1,-1),ink[answer]),1)
        state,_ = self.states(feedback,encoded[-1])
        bias = self.history_bias(encoded[1],previous,passage.shape[1])
        gate, shape, pointer = self.predict(encoded,state,bias,memory_keys)
        actions = gate.float().log_softmax(-1)
        choice = memory_ids[target]
        write = shape.float().log_softmax(-1).gather(-1,choice.clamp_min(0)[...,None]).squeeze(-1)
        write = (actions[...,1]+write).masked_fill(choice<0,-1e4)
        match = (target[...,None]==encoded[1][:,None]) & (target[...,None]!=0)
        copy = pointer.float().log_softmax(-1).masked_fill(~match,-1e4).logsumexp(-1)
        ll = torch.where(target==0,actions[...,0],torch.logaddexp(write,actions[...,2]+copy))
        return ll, valid

    @torch.inference_mode()
    def generate(self, ink, p, q, memory_keys, memory, limit=96):
        encoded = self.encode(ink,p,q)
        handles = encoded[1]
        b = len(p)
        hidden = None
        prev = torch.zeros(b,1,dtype=torch.long,device=p.device)
        feedback = self.start.expand(b,1,-1)
        output = torch.zeros(b,limit,device=p.device,dtype=torch.long)
        stopped = torch.zeros(b,device=p.device,dtype=torch.bool)
        for t in range(limit+1):
            state,hidden = self.states(feedback,encoded[-1],hidden)
            bias = self.history_bias(handles,prev,p.shape[1])[:,-1:]
            gate,shape,pointer = self.predict(encoded,state,bias,memory_keys)
            gate = gate[:,0].float().softmax(-1)
            # Sum every realization of the same visible bitmap before choosing an output.
            prob = torch.zeros(b,len(ink),device=p.device)
            prob[:,memory] = gate[:,1:2]*shape[:,0].float().softmax(-1)
            prob.scatter_add_(1,handles,gate[:,2:3]*pointer[:,0].float().softmax(-1))
            prob[:,0] = gate[:,0]
            tile = prob.argmax(1)
            stopped |= tile==0
            if t==limit or stopped.all():
                break
            tile = tile.masked_fill(stopped,0)
            output[:,t] = tile
            feedback = ink[tile,None]
            prev = torch.cat((prev[:,-2:],tile[:,None]),1)
        return output,stopped
