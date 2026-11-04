"""Conditional whole-bitmap mixtures; no character dictionary or image bank."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def bitmap_log_prob(component_logits, mixture_logits, targets):
    """Exact normalized joint probability of a binary bitmap under a mixture."""
    if component_logits.shape[:-2] != mixture_logits.shape[:-1]:
        raise ValueError("mixture batch dimensions differ")
    if component_logits.shape[-2] != mixture_logits.shape[-1]:
        raise ValueError("component count differs")
    if targets.shape != component_logits.shape[:-2]+component_logits.shape[-1:]:
        raise ValueError("target bitmap shape differs")
    logits=component_logits.float()
    expanded=targets.float().unsqueeze(-2).expand_as(logits)
    component_logp=-F.binary_cross_entropy_with_logits(logits,expanded,reduction="none").sum(-1)
    return (mixture_logits.float().log_softmax(-1)+component_logp).logsumexp(-1)


def expert_mode_candidates(component_logits, mixture_logits):
    """Score the experts' complete modes; this is not guaranteed global MAP."""
    candidates=(component_logits>0).float()
    count=component_logits.shape[-2]
    scores=[]
    for i in range(count):
        scores.append(bitmap_log_prob(component_logits,mixture_logits,candidates[...,i,:]))
    scores=torch.stack(scores,-1)
    selected=scores.argmax(-1)
    index=selected[...,None,None].expand(*selected.shape,1,candidates.shape[-1])
    bitmap=candidates.gather(-2,index).squeeze(-2)
    return bitmap,scores.gather(-1,selected[...,None]).squeeze(-1)


class ConditionalBitmapMixture(nn.Module):
    def __init__(self,width,pixels=2048,components=8):
        super().__init__()
        if min(width,pixels,components)<1:raise ValueError("positive dimensions required")
        self.width,self.pixels,self.components=width,pixels,components
        self.projection=nn.Linear(width,components*pixels,bias=False)
        self.gate=nn.Linear(width,components,bias=False) if components>1 else None
        if self.gate is not None:nn.init.zeros_(self.gate.weight)

    @torch.no_grad()
    def initialize_native_projection(self,native_weight,*,seed=20266210,noise=.01):
        # Native ConvTranspose2d stores [state,channel,height,width]. Repeating
        # this physical-pixel projection creates conditional experts, not a
        # catalogue of writing symbols or stored candidate images.
        source=native_weight.flatten(1).T
        if source.shape!=(self.pixels,self.width):raise ValueError("native pixel projection differs")
        self.projection.weight.copy_(source.repeat(self.components,1))
        if self.components>1 and noise:
            generator=torch.Generator(device="cpu").manual_seed(seed)
            delta=torch.randn(self.projection.weight.shape,generator=generator)
            delta-=delta.reshape(self.components,self.pixels,self.width).mean(0).repeat(self.components,1)
            self.projection.weight.add_(delta.to(self.projection.weight),alpha=noise)

    def forward(self,states):
        if states.shape[-1]!=self.width:raise ValueError("context state width differs")
        pixels=self.projection(states).reshape(*states.shape[:-1],self.components,self.pixels)
        gate=self.gate(states) if self.gate is not None else states.new_zeros(*states.shape[:-1],1)
        return pixels,gate


def strip_blocks(pixels):
    if pixels.ndim!=4 or pixels.shape[1:3]!=(1,32) or pixels.shape[-1]%64:
        raise ValueError("native images must be [B,1,32,T*64]")
    b= len(pixels);t=pixels.shape[-1]//64
    return pixels.reshape(b,1,32,t,64).permute(0,3,1,2,4).flatten(2).add(1).mul(.5)


class CorrelatedNativeLanguage(nn.Module):
    def __init__(self,backbone,head):
        super().__init__();self.backbone=backbone;self.head=head

    @classmethod
    def from_author(cls,author,components=8,*,seed=20266210,noise=.01):
        weight=author.out_proj.weight
        head=ConditionalBitmapMixture(author.config.hidden_size,weight[0].numel(),components).to(weight)
        head.initialize_native_projection(weight,seed=seed,noise=noise)
        return cls(author.model,head)

    def forward(self,pixels,attention_mask,*,use_cache=False,past_key_values=None):
        result=self.backbone(inputs_embeds=pixels,attention_mask=attention_mask,
                             use_cache=use_cache,past_key_values=past_key_values)
        components,gate=self.head(result.last_hidden_state)
        return components,gate,result.past_key_values

    def answer_nll(self,pixels,prompt_blocks):
        blocks=strip_blocks(pixels)
        if not 1<=prompt_blocks<blocks.shape[1]:raise ValueError("invalid source boundary")
        mask=torch.ones(pixels.shape[0],blocks.shape[1],device=pixels.device)
        components,gate,_=self(pixels,mask)
        logp=bitmap_log_prob(components[:,prompt_blocks-1:-1],
                            gate[:,prompt_blocks-1:-1],blocks[:,prompt_blocks:])
        return -logp.mean()/self.head.pixels
