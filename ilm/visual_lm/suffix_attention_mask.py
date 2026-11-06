"""Explicit causal visibility for retained contextual suffixes and new images."""

import torch


def causal_suffix_mask(past_length, new_length, blocked_before, *, device, dtype=torch.float32):
    """Hide old keys, retain suffix/history, and exclude future current queries.

    Positions are physical cache indices. Rotary coordinates are deliberately
    separate: callers keep the source coordinates while testing visibility.
    The returned additive mask is accepted directly by native eager attention.
    """
    if (type(past_length) is not int or type(new_length) is not int or type(blocked_before) is not int
            or past_length<1 or new_length<1 or not 0<=blocked_before<past_length
            or dtype not in (torch.float32,torch.float64)):
        raise ValueError("valid prefix, new-query count, retained boundary and floating mask required")
    keys=torch.arange(past_length+new_length,device=device)
    queries=past_length+torch.arange(new_length,device=device)
    allowed=(keys[None]>=blocked_before)&(keys[None]<=queries[:,None])
    result=torch.full((new_length,past_length+new_length),torch.finfo(dtype).min,device=device,dtype=dtype)
    return result.masked_fill(allowed,0.)[None,None]
