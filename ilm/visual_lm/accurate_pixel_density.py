"""FP32 physical decoder with higher-precision LayerNorm accumulation.

Parameter names, shapes and storage remain identical to the original density.
Only normalization arithmetic changes; no tolerances or logits are altered.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .conditional_pixel_density import ConditionalPixelDensity, PixelDensityConfig


class AccurateLayerNorm(nn.LayerNorm):
    def forward(self, values):
        if values.dtype!=torch.float32:
            raise ValueError('FP32 physical decoder activations required')
        with torch.autocast(device_type=values.device.type,enabled=False):
            return F.layer_norm(values.double(),self.normalized_shape,
                self.weight.double(),self.bias.double(),self.eps).float()


class AccurateConditionalPixelDensity(ConditionalPixelDensity):
    def __init__(self, mean, scale, config=PixelDensityConfig()):
        super().__init__(mean, scale, config)
        for block in self.blocks:
            for name in ('norm1','norm2'):
                original=getattr(block,name)
                replacement=AccurateLayerNorm(original.normalized_shape,eps=original.eps)
                replacement.weight,replacement.bias=original.weight,original.bias
                setattr(block,name,replacement)
        original=self.final_norm
        self.final_norm=AccurateLayerNorm(original.normalized_shape,eps=original.eps)
        self.final_norm.weight,self.final_norm.bias=original.weight,original.bias

    def configuration(self):
        return dict(super().configuration(),normalization_accumulation='float64')
