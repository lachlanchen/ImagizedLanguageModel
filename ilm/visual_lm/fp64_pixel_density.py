"""Small physical head with FP64 arithmetic and unchanged FP32 parameters."""
import torch
from torch import nn
from torch.nn import functional as F

from .conditional_pixel_density import ConditionalPixelDensity,PixelDensityConfig,pattern_bits


class CastLinear(nn.Module):
    def __init__(self, original):
        super().__init__();self.weight,self.bias=original.weight,original.bias

    def forward(self, values):
        return F.linear(values.double(),self.weight.double(),self.bias.double())


class CastLayerNorm(nn.Module):
    def __init__(self, original):
        super().__init__();self.weight,self.bias=original.weight,original.bias
        self.normalized_shape,self.eps=original.normalized_shape,original.eps

    def forward(self, values):
        return F.layer_norm(values.double(),self.normalized_shape,self.weight.double(),self.bias.double(),self.eps)


class FP64ConditionalPixelDensity(ConditionalPixelDensity):
    def __init__(self, mean, scale, config=PixelDensityConfig()):
        super().__init__(mean,scale,config)
        def replace(parent):
            for name,child in list(parent.named_children()):
                if isinstance(child,nn.Linear):setattr(parent,name,CastLinear(child))
                elif isinstance(child,nn.LayerNorm):setattr(parent,name,CastLayerNorm(child))
                else:replace(child)
        replace(self)

    def configuration(self):return dict(super().configuration(),decoder_arithmetic='float64',parameter_storage='float32')

    def condition_features(self, states):
        values=states.reshape(len(states),self.config.condition_width).double()
        return self.condition((values-self.condition_mean.double())/self.condition_scale.double())

    def forward(self, states, targets):
        if targets.ndim!=2 or not 1<=targets.shape[1]<=self.config.positions:
            raise ValueError('a nonempty physical target prefix is required')
        previous=self.bits(pattern_bits(targets[:,:-1]).double()*2-1)
        x=torch.cat((self.start.double().expand(len(targets),1,-1),previous),dim=1)
        x=x+self.positions[:targets.shape[1]].double()+self.condition_features(states)[:,None]
        for block in self.blocks:x,_=block(x)
        return self.output(self.final_norm(x))

    def step(self, condition, previous, position, caches):
        if not 0<=position<self.config.positions:raise ValueError('physical position out of bounds')
        if position==0:
            if previous is not None or caches is not None:raise ValueError('first step has no preceding piece or cache')
            x=self.start.double().expand(len(condition),1,-1)
        else:
            if previous is None or caches is None:raise ValueError('later steps require the emitted prefix')
            x=self.bits(pattern_bits(previous).double()*2-1)[:,None]
        x=x+self.positions[position].double()+condition[:,None]
        updated=[]
        for i,block in enumerate(self.blocks):
            x,cache=block(x,cache=None if caches is None else caches[i],incremental=True);updated.append(cache)
        return self.output(self.final_norm(x[:,0])),updated
