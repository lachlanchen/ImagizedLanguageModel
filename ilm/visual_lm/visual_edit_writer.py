"""Monotone visual editing with exact source copying and a physical ink generator."""
import torch
from torch import nn
from torch.nn import functional as F

from ilm.visual_lm.idiom_ink import GROUPS, InkMotor, bit_image, nibbles, visual_input


COPY, REPLACE, INSERT, DROP, STOP = range(5)
START = 5


class InkPatterns(nn.Module):
    """Predict twelve successive four-pixel patterns inside a single glyph."""
    def __init__(self,width,hidden=128):
        super().__init__()
        self.plan = nn.Linear(width,hidden)
        self.prefix = nn.Parameter(torch.randn(GROUPS,16,hidden)*.03)
        self.position = nn.Parameter(torch.randn(GROUPS,hidden)*.03)
        self.output = nn.Linear(hidden,16)
        self.bias = nn.Parameter(torch.zeros(GROUPS,16))

    def teacher(self,state,bits):
        codes = nibbles(bits)
        positions = torch.arange(GROUPS,device=state.device)
        terms = self.prefix[positions,codes]
        prefix = F.pad(terms[:,:-1].cumsum(1),(0,0,1,0))
        return self.output(F.gelu(self.plan(state)[:,None]+self.position+prefix))+self.bias

    def choose(self,state,beam=4,*,return_score=False):
        context = self.plan(state)
        scores = state.new_zeros(len(state),1)
        history = torch.empty(len(state),1,0,device=state.device,dtype=torch.long)
        prefix = torch.zeros_like(context)[:,None]
        rows = torch.arange(len(state),device=state.device)[:,None]
        for j in range(GROUPS):
            logits = self.output(F.gelu(context[:,None]+self.position[j]+prefix)) + self.bias[j]
            scores,choices = (scores[:,:,None]+logits.log_softmax(-1)).flatten(1).topk(beam,dim=1)
            parents,code = choices//16,choices%16
            prefix = prefix[rows,parents]+self.prefix[j,code]
            history = torch.cat((history[rows,parents],code[:,:,None]),dim=-1)
        bits = bit_image(history[:,0])
        return (bits,scores[:,0]) if return_score else bits


class VisualEditWriter(nn.Module):
    def __init__(self,landmarks,width=128):
        super().__init__()
        self.register_buffer('landmarks',torch.as_tensor(landmarks,dtype=torch.long))
        self.motor = InkMotor()
        self.shape = nn.Linear(112,width)
        self.question = nn.Sequential(nn.Linear(112,width),nn.GELU(),nn.Linear(width,width))
        self.source_convs = nn.ModuleList(nn.Conv1d(width,width,5,padding=2*d,dilation=d) for d in (1,3,8))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in self.source_convs)
        self.eos = nn.Parameter(torch.zeros(width))
        self.initial = nn.Linear(width*2,width)
        self.act = nn.Embedding(6,16)
        self.step = nn.GRU(width*2+112+16,width,batch_first=True)
        self.action = nn.Linear(width,5)
        self.draw = InkPatterns(width)

    def encode(self,source,valid,question,qvalid):
        q = (self.question(question)*qvalid[:,:,None]).sum(1)/qvalid.sum(1,keepdim=True)
        v = valid[:,:,None]
        h = self.shape(source)*v
        for conv,norm in zip(self.source_convs,self.norms):
            h = norm(h + F.gelu(conv(h.transpose(1,2)).transpose(1,2)))*v
        global_state = torch.tanh(self.initial(torch.cat(((h*v).sum(1)/v.sum(1),q),dim=1)))
        # The EOF feature is available at the first padded position, irrespective of batch width.
        memory = F.pad(h,(0,0,0,1))
        memory = torch.where(F.pad(valid,(0,1))[:,:,None],memory,self.eos)
        return memory,global_state

    def decoder_inputs(self,current,global_state,prev_ink,prev_action):
        return torch.cat((current,global_state,prev_ink,self.act(prev_action)),dim=-1)

    def teacher(self,memory,global_state,cursors,prev_ink,prev_action):
        rows = torch.arange(len(memory),device=memory.device)[:,None]
        inputs = self.decoder_inputs(memory[rows,cursors],global_state[:,None].expand(-1,cursors.shape[1],-1),
                                     prev_ink,prev_action)
        return self.step(inputs,global_state[None].contiguous())[0]

    def action_logits(self,states,at_end):
        score = self.action(states)
        invalid = at_end[...,None] & torch.tensor([True,True,False,True,False],device=states.device)
        return score.masked_fill(invalid,-1e4)

    @torch.inference_mode()
    def generate(self,source_ink,lengths,question,qlengths,*,max_output=192,insertion_limit=None,joint_choice=False):
        if max_output < 1 or (lengths<1).any() or (qlengths<1).any():
            raise ValueError('Nonempty pictured inputs and a positive output limit are required')
        device = source_ink.device
        source = visual_input(source_ink,self.landmarks)
        valid = torch.arange(source.shape[1],device=device)[None]<lengths[:,None]
        qvalid = torch.arange(question.shape[1],device=device)[None]<qlengths[:,None]
        memory,global_state = self.encode(source,valid,question,qvalid)
        state = global_state[None].contiguous()
        b = len(source)
        rows = torch.arange(b,device=device)
        cursor = torch.zeros_like(lengths)
        n = torch.zeros_like(lengths)
        prev = torch.full_like(n,START)
        feedback = source.new_zeros(b,112)
        complete = torch.zeros(b,dtype=torch.bool,device=device)
        capped = torch.zeros_like(complete)
        output = torch.zeros(b,max_output,1024,device=device,dtype=torch.bool)
        budgets = insertion_limit(memory,global_state) if insertion_limit is not None else None
        used = torch.zeros_like(lengths)
        actions = []
        # Each non-stop operation consumes a source tile or emits a tile.
        for _ in range(max_output+source.shape[1]+1):
            inputs = self.decoder_inputs(memory[rows,cursor],global_state,feedback,prev)
            h,state = self.step(inputs[:,None],state)
            logits = self.action_logits(h[:,0],cursor==lengths)
            if budgets is not None:
                logits[:,INSERT].masked_fill_(used >= budgets[rows,cursor],-1e4)
            action = logits.argmax(-1)
            live = ~(complete|capped)
            proposed_bits = None
            if joint_choice:
                proposed = live & ((action==REPLACE)|(action==INSERT))
                if proposed.any():
                    bits,logp = self.draw.choose(h[proposed,0],return_score=True)
                    proposed_bits = source.new_zeros(b,len(self.landmarks))
                    proposed_bits[proposed] = bits
                    logits[proposed,REPLACE] += logp
                    logits[proposed,INSERT] += logp
                    action = logits.argmax(-1)
            emitting = (action<=INSERT) & live
            capped |= emitting & (n==max_output)
            live &= ~capped
            actions.append(torch.where(live,action,-1))
            complete |= live & (action==STOP)
            copy = live & (action==COPY)
            draw = live & ((action==REPLACE)|(action==INSERT))
            emit = copy|draw
            picture = source_ink[rows,cursor.clamp_max(source.shape[1]-1)].bool().clone()
            if draw.any():
                bits = self.draw.choose(h[draw,0]) if proposed_bits is None else proposed_bits[draw]
                picture[draw] = self.motor(bits)>0
            output[rows[emit],n[emit]] = picture[emit]
            feedback = torch.where(emit[:,None],visual_input(picture,self.landmarks),feedback)
            n += emit.long()
            if budgets is not None:
                used = torch.where(action==INSERT,used+1,0)
            cursor += (live & ((action==COPY)|(action==REPLACE)|(action==DROP))).long()
            prev = action
            if (complete|capped).all():
                break
        return output,n,complete,torch.stack(actions,dim=1)
