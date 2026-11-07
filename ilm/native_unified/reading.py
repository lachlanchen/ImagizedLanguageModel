"""Train visual access with a frozen auxiliary native language head.

These adapters operate only on the clean input-image path. They are not an
image-output ILM, and their text readout remains a training/diagnostic helper.
"""
from contextlib import contextmanager
import functools
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ilm.image_edit.paired import LowRankLinear
from .showo2_source import reading_prefix


class GatedLowRank(LowRankLinear):
    enabled = False

    def forward(self, value):
        return super().forward(value) if self.enabled else self.base(value)


def add_reading_adapters(bundle, rank=16):
    model=bundle.model;model.requires_grad_(False)
    names=[name for name,layer in model.named_modules() if isinstance(layer,nn.Linear)
           and name.startswith(("und_trans.","fusion_proj."))]
    if not names:raise ValueError("native semantic/fusion layers missing")
    for name in names:
        parent,_,child=name.rpartition(".")
        setattr(model.get_submodule(parent),child,GatedLowRank(model.get_submodule(name),rank))
    return names


@contextmanager
def reader_gate(model):
    layers=[m for m in model.modules() if isinstance(m,GatedLowRank)]
    old=[m.enabled for m in layers]
    try:
        for m in layers:m.enabled=True
        yield
    finally:
        for m,value in zip(layers,old):m.enabled=value


def checkpoint_reading(bundle):
    bundle.model.und_trans.gradient_checkpointing=True
    bundle.model.und_trans._gradient_checkpointing_func=functools.partial(checkpoint,use_reentrant=False)
    bundle.model.showo.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    bundle.model.showo.config.use_cache=False
    bundle.model.und_trans.train();bundle.model.showo.train()


def supervised_loss(bundle, latent, expected, instruction):
    # Deliberately unwrap only the inference-mode decorator. The source
    # prefix construction itself is the authenticated native computation.
    prefix,mask=reading_prefix.__wrapped__(bundle,latent,instruction)
    answer=torch.tensor(bundle.tokenizer(expected,add_special_tokens=False).input_ids+
        [bundle.tokenizer.eos_token_id],device=bundle.device,dtype=torch.long)
    history=bundle.model.showo.model.embed_tokens(answer[:-1])[None]
    embeddings=torch.cat([prefix,history],dim=1)
    attention=bundle.native.masks.causal_attn_mask_naive(1,embeddings.shape[1],bundle.device).to(bundle.dtype)
    n=prefix.shape[1];attention[:,:,:n,:n]=mask
    outputs=bundle.model.showo(inputs_embeds=embeddings,attention_mask=attention,
        use_cache=False,output_hidden_states=False,num_logits_to_keep=answer.numel())
    loss=F.cross_entropy(outputs.logits[0].float(),answer)
    return loss,dict(prefix_positions=n,target_decisions=answer.numel())


@torch.inference_mode()
def cached_decisions(bundle, prefix, mask, *, seed, cap):
    """Native last-logit and incremental-cache path; sampling convention unchanged."""
    from transformers.cache_utils import DynamicCache
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    cache=DynamicCache()
    output=bundle.model.showo(inputs_embeds=prefix,attention_mask=mask,
        past_key_values=cache,use_cache=True,num_logits_to_keep=1,output_hidden_states=False)
    result=[]
    for index in range(cap):
        logits=output.logits[:,-1,:]
        threshold=torch.topk(logits,1).values[:,-1,None]
        logits=logits.masked_fill(logits<threshold,-float("inf"))
        next_id=torch.multinomial(F.softmax(logits,dim=-1),num_samples=1)
        value=int(next_id.item());result.append(value)
        if value==bundle.tokenizer.eos_token_id or index==cap-1:break
        embedding=bundle.model.showo.model.embed_tokens(next_id)
        current_length=cache.get_seq_length()+1
        next_mask=torch.zeros((1,1,1,current_length),device=bundle.device,dtype=bundle.dtype)
        output=bundle.model.showo(inputs_embeds=embedding,attention_mask=next_mask,
            past_key_values=cache,use_cache=True,num_logits_to_keep=1,output_hidden_states=False)
    return result


def trainable_state(model):
    return {name:value.detach().cpu().clone() for name,value in model.named_parameters() if value.requires_grad}


def restore(model,state):
    parameters={name:value for name,value in model.named_parameters() if value.requires_grad}
    if parameters.keys()!=state.keys():raise ValueError("reader adapter architecture differs")
    with torch.no_grad():
        for name,value in parameters.items():
            if value.shape!=state[name].shape or not bool(torch.isfinite(state[name]).all()):
                raise ValueError("invalid learned residual")
            value.copy_(state[name].to(value))
