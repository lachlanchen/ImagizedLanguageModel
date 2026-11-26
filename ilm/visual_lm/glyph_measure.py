"""Conservative expansion of semantic states over a one-position-per-glyph visual grid."""

from contextlib import contextmanager

import torch


def positions(ends, glyphs):
    """One state per teacher piece plus EOS; grid includes leading and trailing slots."""
    n = glyphs + 2
    if len(ends) + 1 > n:
        raise ValueError("More teacher pieces than available glyph positions")
    if any(a < 0 or a > glyphs for a in ends) or list(ends) != sorted(ends):
        raise ValueError("Invalid written offsets")
    mapping = []
    for j, edge in enumerate([*ends, glyphs+1]):
        # Reserve one slot for each remaining piece, including EOS.
        last = min(max(len(mapping)+1, edge+1), n-(len(ends)-j))
        mapping.extend([j]*(last-len(mapping)))
    if len(mapping) != n:
        raise ValueError("Incomplete optical lattice")
    return mapping


def expand_states(hidden, assignments):
    n = max(map(len, assignments))
    batch = len(assignments)
    ix = torch.zeros((batch,n),device=hidden.device,dtype=torch.long)
    valid = torch.zeros_like(ix)
    mass = torch.zeros_like(ix,dtype=hidden.dtype)
    for j, seq in enumerate(assignments):
        cells = torch.tensor(seq,device=hidden.device)
        ix[j,:len(seq)] = cells
        valid[j,:len(seq)] = 1
        mass[j,:len(seq)] = -cells.bincount()[cells].to(hidden.dtype).log()
    states = hidden.gather(1,ix.unsqueeze(-1).expand(-1,-1,hidden.size(-1)))
    return states, valid, mass


@contextmanager
def weighted_cross_attention(net, log_mass):
    """Add a per-source log measure in every pretrained decoder cross-attention layer."""
    def add_mass(module, args, kwargs):
        bias = log_mass[:,None,None,:]
        base = kwargs.get("attention_mask")
        kwargs["attention_mask"] = bias if base is None else bias + base
        return args, kwargs
    handles = []
    try:
        for block in net.model.decoder.layers:
            handles.append(block.encoder_attn.register_forward_pre_hook(add_mass,with_kwargs=True))
        yield
    finally:
        for h in handles:
            h.remove()
