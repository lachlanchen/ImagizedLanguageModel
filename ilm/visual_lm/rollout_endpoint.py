"""Full finite-solver gradient for the existing continuous glyph motor."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def heun_endpoint(flow, state, noise, *, recompute=True):
    """Exactly32 deployed intervals, differentiating every one of the63 fields.

    Inputs contain no target. Recomputed intervals have no random operation.
    This is reverse differentiation of the finite solver, not an ODE adjoint.
    """
    if (state.ndim != 2 or state.shape[1] != 4096 or
            noise.shape != (len(state), 1024) or state.device != noise.device):
        raise ValueError('Actual4096-wide states and paired1024-wide noise required')
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=noise.is_cuda):
        condition = flow.condition(state)
        x = noise.clone()
        for i in range(32):
            # Bind i now: backward recomputation must use this interval's time.
            def interval(value, context, step=i):
                t = torch.full((len(value),), step/32, device=value.device)
                v = flow.velocity(value, t, context)
                proposal = value + v/32
                if step == 31:
                    return proposal
                v2 = flow.velocity(proposal, t+1/32, context)
                return value + .5*(v+v2)/32
            x = (checkpoint(interval, x, condition, use_reentrant=False,
                            preserve_rng_state=False) if recompute else interval(x, condition))
        return x


def balanced_ink_bce(logits, ink):
    """Equal foreground/background mass per image, then equal image mass."""
    if logits.shape != ink.shape or logits.ndim != 4 or logits.shape[1:] != (1,32,32):
        raise ValueError('One32-square ink target per decoded image required')
    target = ink.float()
    bce = F.binary_cross_entropy_with_logits(logits.float(), target, reduction='none')
    axes = (1,2,3)
    black = (bce*target).sum(axes)/target.sum(axes).clamp_min(1)
    white = (bce*(1-target)).sum(axes)/(1-target).sum(axes).clamp_min(1)
    return (.5*(black+white)).mean()
