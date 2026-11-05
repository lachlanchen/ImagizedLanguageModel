"""Incremental continuous-image writing states, without recomputing history.

This is an inference-only cache for SemanticWritingModel.conditions. It does
not change weights, output distributions, image codes, stopping or sampling.
Floating-point equivalence needs verification on the intended execution
backend before using it in a qualified generation path.
"""

import torch
from torch.nn import functional as F

from .semantic_writing_model import sinusoidal


class CausalWritingCache:
    def __init__(self, model):
        if model.training:
            raise ValueError("incremental writing requires evaluation mode")
        self.model=model
        self.versions=tuple((parameter,parameter._version) for parameter in model.parameters())
        self.keys,self.values=[],[]
        self.blocks=0
        self.prefix_length=None
        self.state=None

    def _unchanged(self):
        if self.model.training or any(parameter._version!=version for parameter,version in self.versions):
            raise ValueError("model mode or weights changed; construct a fresh cache")

    def _layer(self, layer, hidden, index, allowed):
        batch,length,width=hidden.shape
        q,k,v=layer.qkv(layer.norm1(hidden)).reshape(
            batch,length,3,layer.heads,width//layer.heads).permute(2,0,3,1,4)
        if self.prefix_length is not None:
            if k.dtype!=self.keys[index].dtype or k.device!=self.keys[index].device:
                raise ValueError("use one precision and device context throughout the cache")
            k=torch.cat((self.keys[index],k),dim=2)
            v=torch.cat((self.values[index],v),dim=2)
            self.keys[index],self.values[index]=k,v
        else:
            self.keys.append(k.contiguous())
            self.values.append(v.contiguous())
        attention=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed,dropout_p=0.)
        hidden=hidden+layer.output(attention.transpose(1,2).reshape(batch,length,width))
        return hidden+layer.ffn(layer.norm2(hidden))

    @torch.inference_mode()
    def start(self, memory):
        self._unchanged()
        if self.state is not None or self.keys:
            raise ValueError("a cache can only be initialized once")
        if (memory.ndim!=3 or memory.shape[0]!=1 or memory.shape[-1]!=self.model.config.memory_width
                or memory.shape[1]<1 or memory.device!=self.model.device):
            raise ValueError("one owning prompt memory on the model device required")
        prefix=self.model.memory(memory.float())
        hidden=torch.cat((prefix,self.model.start),dim=1)
        positions=torch.arange(hidden.shape[1],device=hidden.device)
        hidden=hidden+sinusoidal(positions,self.model.config.width)[None]
        allowed=(positions[None,:]<=positions[:,None])[None,None]
        for index,layer in enumerate(self.model.layers):
            hidden=self._layer(layer,hidden,index,allowed)
        self.prefix_length=hidden.shape[1]
        self.state=self.model.final_norm(hidden[:,-1:])[:,0]
        return self.state

    @torch.inference_mode()
    def append(self, image_codes):
        self._unchanged()
        if self.state is None:
            raise ValueError("initialize the prompt memory first")
        if self.blocks>=self.model.config.maximum_blocks:
            raise ValueError("writing history exceeds registered capacity")
        if image_codes.shape!=(1,2,768) or image_codes.device!=self.model.device:
            raise ValueError("one physical block's continuous image codes required")
        hidden=self.model.history(image_codes.reshape(1,1,-1).float())
        position=torch.tensor([self.prefix_length+self.blocks],device=self.model.device)
        hidden=hidden+sinusoidal(position,self.model.config.width)[None]
        # This is the rightmost query: every stored key is causally visible.
        # is_causal=True would mask past keys incorrectly for a single query.
        for index,layer in enumerate(self.model.layers):
            hidden=self._layer(layer,hidden,index,None)
        self.blocks+=1
        self.state=self.model.final_norm(hidden)[:,0]
        return self.state

    @property
    def storage_bytes(self):
        return sum(t.numel()*t.element_size() for t in (*self.keys,*self.values))

    @property
    def sequence_length(self):
        return 0 if self.prefix_length is None else self.prefix_length+self.blocks
