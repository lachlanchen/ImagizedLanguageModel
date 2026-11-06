"""Continuous image history and writing over an inherited contextual body.

The deployed module has no tokenizer, vocabulary table or language output
projection. Its inputs are frozen numeric prompt KV, continuous image codes,
and structural positions. It writes pixels and rereads the emitted pixels.
"""

from dataclasses import dataclass
import hashlib

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .clean_writing_flow import CleanWritingConfig, CleanWritingFlow
from .suffix_attention_mask import causal_suffix_mask


@dataclass(frozen=True)
class NativeImagePrefix:
    key_values: tuple
    initial_state: torch.Tensor
    rotary_offsets: torch.Tensor
    blocked_before: int = 0
    valid_lengths: torch.Tensor | None = None

    @property
    def length(self):
        return self.key_values[0][0].shape[-2]

    def validate(self,context):
        if (len(self.key_values)!=len(context.layers) or self.initial_state.ndim!=3
                or self.initial_state.shape[1:]!=(1,4096) or self.rotary_offsets.shape!=(len(self.initial_state),)
                or self.rotary_offsets.dtype!=torch.long or self.rotary_offsets.device!=self.initial_state.device
                or not 0<=self.blocked_before<self.length):
            raise ValueError("native prompt KV, initial state and structural offsets required")
        if self.valid_lengths is not None and (self.valid_lengths.shape!=self.rotary_offsets.shape
                or self.valid_lengths.device!=self.initial_state.device or self.valid_lengths.dtype!=torch.long
                or bool(((self.valid_lengths<=self.blocked_before)|(self.valid_lengths>self.length)).any())):
            raise ValueError("valid source lengths must identify real prefix keys before padding")
        for k,v in self.key_values:
            if (k.shape!=v.shape or k.ndim!=4 or k.shape[0]!=len(self.initial_state)
                    or k.shape[-2]!=self.length or k.device!=self.initial_state.device or v.device!=k.device
                    or k.dtype!=torch.float32 or v.dtype!=torch.float32 or k.requires_grad or v.requires_grad):
                raise ValueError("frozen full-precision prefix KV must share the batch/device/length")
        return self

    def to(self,device):
        return NativeImagePrefix(tuple((k.to(device),v.to(device)) for k,v in self.key_values),
            self.initial_state.to(device),self.rotary_offsets.to(device),self.blocked_before,
            self.valid_lengths.to(device) if self.valid_lengths is not None else None)


class FunctionalPrefixCache:
    """Concatenate immutable old KV with this call's new KV, never append twice.

    Checkpoint recomputation can call a layer repeatedly. A mutable dynamic
    cache would append the same teacher-forced future again during backward.
    Only inference requests collection of the completed new cache.
    """

    def __init__(self,values,*,collect=False):
        self.values=values
        self.updated=[None]*len(values) if collect else None

    def update(self,key_states,value_states,layer_idx,cache_kwargs=None):
        k,v=self.values[layer_idx]
        if key_states.shape[:2]!=k.shape[:2] or key_states.shape[-1]!=k.shape[-1]:
            raise ValueError("new native attention states differ from the frozen prefix")
        pair=(torch.cat((k,key_states),dim=-2),torch.cat((v,value_states),dim=-2))
        if self.updated is not None:
            self.updated[layer_idx]=pair
        return pair


class NativeContextContinuation(nn.Module):
    """Checkpoint only new image-history queries through a frozen native body."""

    def __init__(self,context):
        super().__init__()
        if context.config.vocab_size!=0 or context.embed_tokens is not None or context.config.hidden_size!=4096:
            raise ValueError("the vocabulary-free inherited contextual body is required")
        self.context=context.requires_grad_(False).eval()

    def train(self,mode=True):
        super().train(mode)
        self.context.eval()
        return self

    def forward(self,embeddings,prefix,*,checkpoint_layers=True,collect_cache=False):
        prefix.validate(self.context)
        if (embeddings.ndim!=3 or embeddings.shape[0]!=len(prefix.initial_state)
                or embeddings.shape[-1]!=4096 or not len(embeddings[0])
                or embeddings.device!=prefix.initial_state.device or embeddings.dtype!=torch.float32):
            raise ValueError("floating image-history embeddings [B,L,4096] required")
        if collect_cache and torch.is_grad_enabled():
            raise ValueError("cache collection is inference-only; training prefix remains immutable")
        new_length=embeddings.shape[1]
        physical=torch.arange(prefix.length,prefix.length+new_length,device=embeddings.device)
        rotary=prefix.rotary_offsets[:,None]+torch.arange(new_length,device=embeddings.device)[None]
        coordinates=rotary[None].expand(3,-1,-1)
        mask=causal_suffix_mask(prefix.length,new_length,prefix.blocked_before,device=embeddings.device)
        if prefix.valid_lengths is not None:
            mask=mask.expand(len(embeddings),-1,-1,-1).clone()
            keys=torch.arange(prefix.length+new_length,device=embeddings.device)
            padding=(keys[None]>=prefix.valid_lengths[:,None])&(keys[None]<prefix.length)
            mask=mask.masked_fill(padding[:,None,None,:],torch.finfo(mask.dtype).min)
        cache=FunctionalPrefixCache(prefix.key_values,collect=collect_cache)
        with torch.autocast(device_type=embeddings.device.type,enabled=False):
            hidden=embeddings
            position_embeddings=self.context.rotary_emb(hidden,coordinates)
            for layer in self.context.layers:
                def block(value,current_layer=layer):
                    return current_layer(value,attention_mask=mask,position_ids=physical[None].expand(len(value),-1),
                        past_key_values=cache,cache_position=physical,position_embeddings=position_embeddings)
                if checkpoint_layers and torch.is_grad_enabled() and hidden.requires_grad:
                    hidden=checkpoint(block,hidden,use_reentrant=False,preserve_rng_state=True)
                else:
                    hidden=block(hidden)
            hidden=self.context.norm(hidden)
        return hidden,tuple(cache.updated) if collect_cache else None


class NativeImageLanguageModel(nn.Module):
    """Complete causal image-code training and autonomous physical writing.

    A learned per-tile encoder supplies contextual input embeddings; clean
    image-code prediction and a learned structural stop supply output. Native
    prompt KV is retained, rather than replaced by a fresh small language body.
    """

    def __init__(self,context,writer,*,input_scale=1.,flow_steps=32,maximum_blocks=128):
        super().__init__()
        if input_scale<=0 or flow_steps!=32 or maximum_blocks!=128:
            raise ValueError("positive image embedding scale and declared generation limits required")
        self.continuation=NativeContextContinuation(context)
        self.writer=writer.requires_grad_(False).eval()
        self.image_input=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,1024),nn.SiLU(),nn.Linear(1024,4096))
        self.register_buffer("input_scale",torch.tensor(float(input_scale)))
        self.flow=CleanWritingFlow(CleanWritingConfig(width=4096))
        self.stop=nn.Linear(4096,1)
        nn.init.zeros_(self.stop.weight);nn.init.constant_(self.stop.bias,-3.)
        self.flow_steps=flow_steps
        self.maximum_blocks=maximum_blocks

    @property
    def device(self):
        return self.stop.weight.device

    def embed_images(self,codes):
        if codes.ndim!=4 or codes.shape[2:]!=(2,768) or codes.device!=self.device or not codes.is_floating_point():
            raise ValueError("continuous physical image codes [B,T,2,768] required")
        with torch.autocast(device_type=self.device.type,enabled=False):
            values=self.image_input(codes.float().reshape(len(codes),-1,768))
            return F.layer_norm(values,(4096,))*self.input_scale

    def conditions(self,prefix,codes,*,checkpoint_layers=True):
        if codes.shape[0]!=len(prefix.initial_state) or codes.shape[1]>self.maximum_blocks:
            raise ValueError("image histories must match prompt batch and capacity")
        if not codes.shape[1]:
            return prefix.initial_state.float()
        hidden,_=self.continuation(self.embed_images(codes),prefix,checkpoint_layers=checkpoint_layers)
        # Each completed block contains two tile observations. Its second
        # state conditions the following block, never the block being read.
        return torch.cat((prefix.initial_state.float(),hidden[:,1::2]),dim=1)

    def training_loss(self,prefix,codes,lengths,*,generator,writer_blocks=16):
        if (lengths.shape!=(len(codes),) or bool(((lengths<1)|(lengths>codes.shape[1])).any())
                or writer_blocks<1):
            raise ValueError("complete nonempty image targets and per-example lengths required")
        states=self.conditions(prefix,codes)
        batches=[];positions=[];weights=[];continuing=[];terminal=[]
        with torch.autocast(device_type=self.device.type,enabled=False):
            logits=self.stop(states.float()).squeeze(-1)
        for b,n in enumerate(lengths.tolist()):
            chosen=torch.randperm(n,device=self.device,generator=generator)[:writer_blocks]
            batches.append(torch.full_like(chosen,b));positions.append(chosen)
            weights.append(torch.full((len(chosen),),1/(len(codes)*len(chosen)),device=self.device))
            continuing.append(F.softplus(logits[b,:n]).mean());terminal.append(F.softplus(-logits[b,n]))
        batch,position=torch.cat(batches),torch.cat(positions)
        target=codes[batch,position].float().flatten(1)
        noise=torch.randn(target.shape,device=self.device,generator=generator)
        time=torch.rand((len(target),),device=self.device,generator=generator)
        with torch.autocast(device_type=self.device.type,dtype=torch.bfloat16,enabled=self.device.type=="cuda"):
            clean=self.flow((1-time[:,None])*noise+time[:,None]*target,time,states[batch,position])
        error=(clean-target).square().mean(-1)
        writing=(error/(1-time).clamp_min(.05).square()*torch.cat(weights)).sum()
        stopping=.5*(torch.stack(continuing).mean()+torch.stack(terminal).mean())
        return writing+stopping,dict(flow=writing.detach(),clean=error.mean().detach(),
            stop=stopping.detach(),sampled_blocks=len(target))

    @torch.no_grad()
    def generate(self,prefix,*,generator,maximum_blocks=None):
        if len(prefix.initial_state)!=1:
            raise ValueError("one actual prompt prefix required for generation")
        if prefix.valid_lengths is not None:
            raise ValueError("generation uses an unpadded owning prompt cache")
        cap=self.maximum_blocks if maximum_blocks is None else maximum_blocks
        if type(cap) is not int or not 1<=cap<=self.maximum_blocks:
            raise ValueError("valid uniform generation cap required")
        current=prefix
        images=[];probabilities=[];hashes=[];stopped=False
        for position in range(cap+1):
            condition=current.initial_state[:,0].float()
            probability=self.stop(condition).float().sigmoid().item()
            probabilities.append(probability)
            if probability>=.5:
                stopped=True;break
            if position==cap:
                break
            _,endpoint=self.flow.sample_endpoint(condition,generator=generator,steps=self.flow_steps)
            coordinates=F.layer_norm(endpoint.reshape(-1,768),(768,),eps=self.writer.codec.config.layer_norm_epsilon).reshape(1,2,768)
            soft=self.writer.predict(coordinates)
            image=torch.where(soft<0,-torch.ones_like(soft),torch.ones_like(soft))
            actual=self.writer.encode_targets(image)
            hidden,updated=self.continuation(self.embed_images(actual[None]),current,
                checkpoint_layers=False,collect_cache=True)
            current=NativeImagePrefix(updated,hidden[:,-1:],current.rotary_offsets+2,current.blocked_before)
            images.append(image.cpu())
            hashes.append(hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest())
        return dict(images=torch.cat(images) if images else torch.empty(0,1,32,64),
            stopped=stopped,capped=not stopped,stop_probabilities=probabilities,
            feedback_sha256=hashes,feedback_from_emitted_pixels=True,
            native_context_kv_retained=True,tokenizer=False,language_vocabulary=False)

    def trainable_state_dict(self):
        return {name:value.detach().cpu().clone() for name,value in self.state_dict().items()
            if name.startswith(("image_input.","flow.","stop.")) or name=="input_scale"}

    @torch.no_grad()
    def load_trainable_state_dict(self,state):
        expected=self.trainable_state_dict()
        if state.keys()!=expected.keys() or any(state[n].shape!=v.shape or state[n].dtype!=v.dtype for n,v in expected.items()):
            raise ValueError("native image-adapter/head checkpoint architecture differs")
        current=self.state_dict()
        for name,value in state.items():
            current[name].copy_(value.to(current[name].device))
