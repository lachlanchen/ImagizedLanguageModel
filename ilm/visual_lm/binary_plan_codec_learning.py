"""Matched continuous/binary codec learning and physical-only source partition."""
import hashlib

import torch
from torch.nn import functional as F

from .binary_plan_glyph_density import binary_images
from .continuous_glyph_codec_training import glyph_sobel_edges


def physical_partition(packed):
    """Keep exact aliases together; reserve hash bucket zero, train on empty."""
    if packed.dtype!=torch.uint8 or packed.ndim!=2 or packed.shape[1]!=128 or not len(packed):
        raise ValueError('All packed binary source images required')
    digest=[hashlib.sha256(row.tobytes()).digest() for row in packed.detach().cpu().numpy()]
    empty=hashlib.sha256(bytes(128)).digest()
    held=torch.tensor([d!=empty and int.from_bytes(d[:8],'big')%10==0 for d in digest],dtype=torch.bool)
    return held,[d.hex() for d in digest]


def codec_forward(codec,ink,mode):
    """Same tensors and drawing network; only the global quantization changes."""
    if mode=='binary':return codec(ink)
    if mode!='continuous':raise ValueError('Fixed continuous/binary comparison only')
    ink=binary_images(ink);values=codec.encoder(ink.float()*2-1).tanh()
    return dict(plan_values=values,image_logits=codec._draw(values))


def reconstruction_terms(logits,ink):
    """One complete-image objective per row, invariant to microbatch grouping.

    Foreground is one. Boundary BCE + .5 Sobel L1 + .5 foreground Dice reuse
    the earlier codec's physical terms, with explicit per-image normalization.
    This is reconstruction supervision, not a normalized image likelihood.
    """
    target=binary_images(ink).float()
    if logits.shape!=target.shape or not bool(torch.isfinite(logits).all()):raise ValueError('Finite matching image logits required')
    probability=logits.float().sigmoid();target_edges=glyph_sobel_edges(target)
    strength=target_edges.square().sum(1,keepdim=True).sqrt().clamp(0,1)
    weights=1+2*strength
    bce=(F.binary_cross_entropy_with_logits(logits.float(),target,reduction='none')*weights).flatten(1).sum(1)/weights.flatten(1).sum(1)
    edge=(glyph_sobel_edges(probability)-target_edges).abs().flatten(1).mean(1)
    overlap=2*(probability*target).flatten(1).sum(1)
    dice=1-(overlap+1e-6)/(probability.flatten(1).sum(1)+target.flatten(1).sum(1)+1e-6)
    return dict(loss=bce+.5*edge+.5*dice,bce=bce,edge=edge,dice=dice)
