"""Late-time decoded endpoint supervision for a frozen continuous glyph motor."""
import torch
from torch.nn import functional as F


def decoder_endpoint_loss(velocity, mixed, times, mean, scale, decoder, ink):
    """Per-image foreground/background balanced BCE for t>=0.75.

    Decoder parameters stay frozen; gradients still reach the velocity through
    its predicted endpoint. This is training supervision, not an ODE sampler.
    """
    if (velocity.shape != mixed.shape or velocity.ndim != 2 or velocity.shape[1] != 1024
            or times.shape != (len(velocity),) or ink.shape != (len(velocity),1,32,32)
            or mean.shape != (1024,) or scale.shape != (1024,)):
        raise ValueError('Full motor coordinates, per-image times and32x32 ink targets required')
    if any(p.requires_grad for p in decoder.parameters()):
        raise ValueError('Freeze the decoder while keeping the velocity gradient')
    eligible = times >= .75
    count = int(eligible.sum())
    if not count:
        return velocity.sum()*0, 0
    endpoint = mixed[eligible].float()+(1-times[eligible,None])*velocity[eligible].float()
    logits = decoder((endpoint*scale+mean).reshape(-1,64,4,4)).float()
    target = ink[eligible].float()
    loss = F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    axes = (1,2,3)
    black = (loss*target).sum(axes)/target.sum(axes).clamp_min(1)
    white = (loss*(1-target)).sum(axes)/(1-target).sum(axes).clamp_min(1)
    return (.5*(black+white)).mean(), count
