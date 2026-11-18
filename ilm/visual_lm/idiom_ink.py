"""Predict physical landmark pixels recurrently and draw each completed glyph."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LANDMARKS = 48
GROUPS = LANDMARKS // 4
STOP = 16


def choose_landmarks(images):
    """Greedily separate training shapes with the same observed bit prefix."""
    masks = np.asarray(images, dtype=np.uint8).reshape(-1,1024)
    partitions = [np.arange(len(masks))]
    entropy = masks.mean(0) * (1-masks.mean(0))
    chosen = []
    for _ in range(LANDMARKS):
        gains = np.zeros(1024, dtype=np.int64)
        for ids in partitions:
            ones = masks[ids].sum(0, dtype=np.int64)
            gains += ones * (len(ids)-ones)
        gains[chosen] = -1
        if gains.max() == 0:
            remaining = entropy.copy()
            remaining[chosen] = -1
            column = int(remaining.argmax())
        else:
            column = int(gains.argmax())
        chosen.append(column)
        children = []
        for ids in partitions:
            bits = masks[ids,column].astype(bool)
            children.extend(part for part in (ids[bits],ids[~bits]) if len(part)>1)
        partitions = children
    return np.asarray(chosen)


def visual_input(pixels, landmarks):
    flat = pixels.reshape(*pixels.shape[:-1],8,4,8,4)
    return torch.cat((flat.float().mean((-3,-1)).flatten(-2),pixels[...,landmarks].float()),-1)


def nibbles(bits):
    groups = bits.long().unflatten(-1,(GROUPS,4))
    return (groups * groups.new_tensor([8,4,2,1])).sum(-1)


def bit_image(symbols):
    return ((symbols.unsqueeze(-1) >> symbols.new_tensor([3,2,1,0])) & 1).flatten(-2).float()


class InkMotor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(LANDMARKS,768),nn.GELU(),nn.Linear(768,768),
                                  nn.GELU(),nn.Linear(768,1024))

    def forward(self,bits):
        return self.layers(2*bits.float()-1)


class PicturedIdiom(nn.Module):
    def __init__(self,landmarks):
        super().__init__()
        self.register_buffer('landmarks',torch.as_tensor(landmarks,dtype=torch.long))
        self.motor=InkMotor()
        self.shape=nn.Sequential(nn.Linear(112,192),nn.GELU(),nn.Linear(192,128))
        self.position=nn.Parameter(torch.randn(48,128)*.02)
        layer=nn.TransformerEncoderLayer(128,4,256,.08,batch_first=True,activation='gelu',norm_first=False)
        self.reader=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.attention=nn.Linear(128,2)
        self.initial=nn.Linear(4*128,192)
        self.ink_group=nn.Embedding(18,32)  # 16 ink patterns, stop and start
        self.group_position=nn.Embedding(GROUPS,16)
        self.feedback=nn.Linear(LANDMARKS,48)
        self.causal=nn.GRU(32+16+48+192,192,batch_first=True)
        self.next_pattern=nn.Linear(192,17)

    def read(self,features):
        h=self.reader(self.shape(features)+self.position)
        pooled=torch.bmm(self.attention(h).transpose(1,2).softmax(-1),h)
        return self.initial(torch.cat([pooled.flatten(1),h[:,23],h[:,24]],-1)).tanh()

    def inputs(self,previous,position,prior_ink,context):
        return torch.cat((self.ink_group(previous),self.group_position(position),
                          self.feedback(prior_ink),context),-1)

    def teacher_logits(self,features,target_bits):
        b,n,_=target_bits.shape
        context=self.read(features)
        codes=nibbles(target_bits).reshape(b,-1)
        labels=F.pad(codes,(0,1),value=STOP)
        previous=F.pad(codes,(1,0),value=17)
        pos=torch.arange(previous.shape[1],device=previous.device)
        prior=F.pad(target_bits,(0,0,1,0))[:,pos//GROUPS]
        seq=self.inputs(previous,(pos%GROUPS).expand_as(previous),prior,context[:,None].expand(-1,len(pos),-1))
        states,_=self.causal(seq,context[None].contiguous())
        return self.next_pattern(states),labels

    @torch.inference_mode()
    def generate(self,features,beam=4,limit=8):
        """A separate beam per prompt; no inventory, corpus or font is queried."""
        return self.generate_from_context(self.read(features),beam,limit)

    @torch.inference_mode()
    def generate_from_context(self,ctx,beam=4,limit=8,*,observe_step=None,later_context=None):
        b=len(ctx);device=ctx.device
        ctx=ctx[:,None].expand(-1,beam,-1)
        hidden=ctx.contiguous()
        scores=torch.full((b,beam),-torch.inf,device=device);scores[:,0]=0
        done=torch.zeros_like(scores,dtype=torch.bool)
        length=torch.zeros_like(scores,dtype=torch.long)
        previous=torch.full_like(length,17)
        feedback=ctx.new_zeros(b,beam,LANDMARKS)
        codes=torch.zeros(b,beam,limit,GROUPS,device=device,dtype=torch.long)
        rows=torch.arange(b,device=device)[:,None]
        for t in range(limit*GROUPS+1):
            slot,offset=divmod(t,GROUPS)
            if later_context is not None and t==2*GROUPS:
                ctx=later_context[:,None].expand(-1,beam,-1)
            inputs=self.inputs(previous,torch.full_like(previous,offset),feedback,ctx)
            h,_=self.causal(inputs.reshape(b*beam,1,-1),hidden.reshape(1,b*beam,-1).contiguous())
            h=h.reshape(b,beam,-1)
            logits=self.next_pattern(h).float()
            if offset:
                logits[:,:,STOP]=-torch.inf
            probs=logits.log_softmax(-1)
            probs.masked_fill_(done.unsqueeze(-1),-torch.inf)
            probs[:,:,STOP]=torch.where(done,torch.zeros_like(scores),probs[:,:,STOP])
            winners,choices=(scores[:,:,None]+probs).flatten(1).topk(beam)
            parent=choices//17;symbol=choices%17
            if observe_step is not None:
                observe_step(t,parent,symbol,winners)
            codes=codes[rows,parent];feedback=feedback[rows,parent];length=length[rows,parent]
            done=done[rows,parent] | (symbol==STOP)
            scores=winners;hidden=h[rows,parent];previous=symbol
            if t==limit*GROUPS or bool(done.all()):
                break
            codes[:,:,slot,offset]=symbol.clamp_max(15)
            if offset==GROUPS-1:
                # Read measurements from the actual emitted pixels for the next glyph.
                picture=self.motor(bit_image(codes[:,:,slot]))>0
                feedback=torch.where(done[:,:,None],feedback,picture[...,self.landmarks].float())
                length=length+(~done).long()
        pixels=(self.motor(bit_image(codes[:,0]))>0).cpu().numpy()
        return np.packbits(pixels,axis=-1),length[:,0].cpu().numpy(),done[:,0].cpu().numpy()
