"""Draw alternate physical responses and compare them to their pictured context."""
import torch
from torch import nn
from .idiom_ink import GROUPS,LANDMARKS,STOP,bit_image,visual_input


def answer_encoder():
    return nn.Sequential(nn.Linear(4*112,384),nn.GELU(),nn.Linear(384,192))


@torch.inference_mode()
def propose(model,features,count=32,limit=8,*,rng):
    context=model.read(features)
    b=len(context);device=context.device
    ctx=context.repeat_interleave(count,dim=0);h=ctx[None].contiguous()
    k=len(ctx);last=torch.full((k,),17,device=device,dtype=torch.long)
    feedback=ctx.new_zeros(k,LANDMARKS)
    length=torch.zeros_like(last);ended=torch.zeros(k,device=device,dtype=torch.bool)
    output=torch.zeros(k,limit,1024,device=device,dtype=torch.bool)
    patterns=torch.zeros(k,GROUPS,device=device,dtype=torch.long)
    for t in range(limit*GROUPS+1):
        cell,offset=divmod(t,GROUPS)
        inputs=model.inputs(last,torch.full_like(last,offset),feedback,ctx)
        step,h=model.causal(inputs[:,None],h)
        logits=model.next_pattern(step[:,0]).float()
        if offset:logits[:,STOP]=-torch.inf
        last=torch.multinomial(logits.softmax(-1),1,generator=rng).flatten()
        ended=ended | (last==STOP)
        if cell==limit or ended.all().item():break
        patterns[:,offset]=last.clamp_max(15)
        if offset==GROUPS-1:
            ink=model.motor(bit_image(patterns))>0
            output[:,cell]=ink & ~ended[:,None]
            length=length+(~ended).long()
            feedback=torch.where(ended[:,None],feedback,ink[:,model.landmarks].float())
    return output.reshape(b,count,limit,1024),length.reshape(b,count),ended.reshape(b,count)


@torch.inference_mode()
def select(model,encoder,features,outputs,length,stopped):
    state=nn.functional.normalize(model.read(features).float(),dim=-1)
    pictures=visual_input(outputs[:,:,:4],model.landmarks).flatten(-2)
    keys=nn.functional.normalize(encoder(pictures).float(),dim=-1)
    scores=(state[:,None]*keys).sum(-1)
    allowed=(length==4)&stopped
    choice=scores.masked_fill(~allowed,-torch.inf).argmax(-1)
    return torch.where(allowed.any(-1),choice,torch.zeros_like(choice))
