"""Training-only whole-form supervision for a physical image condition.

The helper is separate from the physical density and is never called by its
generator. Form labels describe exact training masks, not Unicode identities.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def build_form_index(packed):
    """Canonical equivalence labels from source-only 128-byte binary masks."""
    if (not isinstance(packed,torch.Tensor) or packed.dtype!=torch.uint8
            or packed.ndim!=2 or packed.shape[1]!=128 or not len(packed)):
        raise ValueError('Require complete packed 32x32 source masks')
    forms=sorted({row.tobytes() for row in packed.detach().cpu().numpy()})
    return {bits:index for index,bits in enumerate(forms)}


def form_labels(ink,index):
    """Fail on unsupported source masks; never drop or approximate a target."""
    if ink.ndim!=4 or ink.shape[1:]!=(1,32,32) or not len(ink) or not bool(((ink==0)|(ink==1)).all()):
        raise ValueError('Require complete binary target images')
    packed=np.packbits(ink.detach().cpu().numpy().reshape(len(ink),1024),axis=1,bitorder='big')
    try:labels=[index[row.tobytes()] for row in packed]
    except KeyError as error:raise ValueError('Target form is outside the bound training-only source index') from error
    return torch.tensor(labels,dtype=torch.long)


class GlobalFormSupervisor(nn.Module):
    """A discarded classifier on the actual shared pre-generation condition."""
    def __init__(self,width,classes,*,seed=202611191):
        super().__init__()
        if type(width) is not int or type(classes) is not int or min(width,classes)<2:
            raise ValueError('Require a condition width and at least two source forms')
        self.width,self.classes=width,classes
        with torch.random.fork_rng(devices=[]),torch.device('cpu'):
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
            self.classifier=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,classes))

    def forward(self,features):
        if features.ndim!=2 or features.shape[1]!=self.width or not bool(torch.isfinite(features).all()):
            raise ValueError('Finite shared conditions required')
        return self.classifier(features.float())

    def logits(self,density,states,glyph_index=None,span_length=None):
        # This is exactly the function used by the density's forward/step path.
        # There is no target mask, image latent, or physical prefix argument.
        return self(density.condition_features(states,glyph_index,span_length))

    def losses(self,density,states,labels,glyph_index=None,span_length=None):
        if labels.shape!=(len(states),) or labels.dtype!=torch.long or not bool(((labels>=0)&(labels<self.classes)).all()):
            raise ValueError('Complete source-form labels required')
        logits=self.logits(density,states,glyph_index,span_length)
        return F.cross_entropy(logits,labels,reduction='none'),logits

    @property
    def loss_scale(self):return math.log(self.classes)
