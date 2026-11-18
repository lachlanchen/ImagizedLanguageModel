"""Exact likelihood of a pictured target with path-independent edit probabilities."""
import torch
from torch.nn import functional as F


def alignment_log_likelihood(insert, drop, diagonal, stop, *, target_lengths=None):
    """Marginalize all insert, drop and diagonal paths to the bottom-right corner.

    A source has n glyphs and a target has m glyphs. The log edge probabilities
    have final dimensions (n+1,m), (n,m+1), and (n,m). Stop is the log
    probability of ending at (n,m). Equal-sized examples may share batch axes.
    Insertion and deletion log probabilities must be finite.
    Distinct copy and replace actions with the same endpoint belong together in
    diagonal via logaddexp. The neural state at (i,j) must be path-independent.
    """
    n, m = diagonal.shape[-2:]
    batch = diagonal.shape[:-2]
    if (insert.shape != (*batch, n+1, m) or drop.shape != (*batch, n, m+1)
            or stop.shape != batch):
        raise ValueError('Incompatible source or target grid dimensions')
    # Prefix sums permit a vectorized scan over every insertion within a row.
    # Accumulate in double precision because long runs subtract similar log scores.
    prefix = F.pad(insert.double().cumsum(-1), (1, 0))
    mass = prefix[..., 0, :]
    for i in range(n):
        deleted = mass + drop[..., i, :]
        advanced = mass[..., :-1] + diagonal[..., i, :]
        entry = torch.cat((deleted[..., :1], torch.logaddexp(deleted[..., 1:], advanced)), -1)
        cost = prefix[..., i+1, :]
        mass = cost + torch.logcumsumexp(entry - cost, -1)
    if target_lengths is not None:
        if target_lengths.shape != batch:
            raise ValueError('Target lengths must match the batch axes')
        return mass.gather(-1, target_lengths.long().unsqueeze(-1)).squeeze(-1) + stop
    return mass[..., -1] + stop


def observed_prefix_logp(action, glyph, source_bits, target_bits, lengths):
    """Probability of reaching each observed glyph prefix before further edits or stopping."""
    from ilm.visual_lm.visual_edit_writer import COPY, DROP, INSERT, REPLACE, STOP

    batch, sites, _ = glyph.shape
    log_cursor = glyph.new_full((batch,sites),-torch.inf,dtype=torch.float64)
    log_cursor[:,0]=0.
    result=log_cursor.new_zeros(batch)
    terminal=result.clone()
    for j in range(int(lengths.max())+1):
        drop=action[:,:-1,j,DROP].double()
        offset=F.pad(drop.cumsum(1),(1,0))
        arrival=offset+torch.logcumsumexp(log_cursor-offset,1)
        reached=lengths==j
        result=torch.where(reached,log_cursor.logsumexp(1),result)
        terminal=torch.where(reached,arrival[:,-1]+action[:,-1,j,STOP],terminal)
        if j==glyph.shape[2]:
            break
        insert=arrival+action[:,:,j,INSERT]+glyph[:,:,j]
        equal=(source_bits==target_bits[:,None,j]).all(-1)
        replace=action[:,:-1,j,REPLACE]+glyph[:,:-1,j]
        advance=arrival[:,:-1]+torch.logaddexp(replace,action[:,:-1,j,COPY].masked_fill(~equal,-torch.inf))
        log_cursor=torch.logaddexp(insert,F.pad(advance,(1,0),value=-torch.inf))
    return result,terminal
