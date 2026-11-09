"""Read-only numerical diagnosis of the physical raster decoder.

The double-precision reference is a separate functional implementation. It
scores the same true physical prefix and has no tokenizer or character table.
These checks do not assess language quality.
"""
import torch
from torch.nn import functional as F


@torch.no_grad()
def trace_density(model, states, targets):
    """Retain both teacher paths and both autonomous greedy trajectories."""
    model.eval()
    full = model(states, targets)
    condition = model.condition_features(states)
    parts, caches = [], None
    for j in range(targets.shape[1]):
        logits, caches = model.step(condition, None if j == 0 else targets[:, j-1], j, caches)
        parts.append(logits)
    cached = torch.stack(parts, 1)
    changed = targets.clone(); changed[:, 128:] ^= 255
    altered = model(states, changed)
    greedy_cached = model.generate(states)
    greedy_full = targets.new_zeros((len(states), 1))
    for j in range(256):
        greedy_full[:, j] = model(states, greedy_full)[:, -1].argmax(-1)
        if j < 255:
            greedy_full = torch.cat((greedy_full, greedy_full.new_zeros((len(states), 1))), 1)
    result = dict(states=states, targets=targets, teacher_full=full, teacher_cached=cached,
        greedy_cached=greedy_cached, greedy_full=greedy_full,
        suffix_causality_max_error=float((full[:, :129]-altered[:, :129]).abs().max()),
        altered_future_effect=float((full[:, 129:]-altered[:, 129:]).abs().max()))
    return {key:value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        for key,value in result.items()}


def differences(full, cached):
    """Compare distribution and decision errors without changing logits."""
    full, cached = full.double(), cached.double()
    error = (full-cached).abs()
    worst = torch.unravel_index(error.argmax(), error.shape)
    logp, logq = full.log_softmax(-1), cached.log_softmax(-1)
    return dict(max_absolute=float(error.max()), rms=float(error.square().mean().sqrt()),
        normwise_relative=float(torch.linalg.vector_norm(full-cached)/torch.linalg.vector_norm(full).clamp_min(1e-30)),
        maximum_logit_magnitude=float(full.abs().max()),
        teacher_argmax_changes=int((full.argmax(-1)!=cached.argmax(-1)).sum()),
        maximum_log_probability_error=float((logp-logq).abs().max()),
        maximum_total_variation=float((.5*(logp.exp()-logq.exp()).abs().sum(-1)).max()),
        worst_coordinate=[int(i) for i in worst],
        full_at_worst=float(full[worst]),cached_at_worst=float(cached[worst]))


@torch.no_grad()
def reference_teacher(weights, config, states, targets, *, incremental=False):
    """Independent FP64 CPU equations for the complete trained architecture."""
    if states.device.type!='cpu' or targets.device.type!='cpu':
        raise ValueError('the independent reference is CPU only')
    state={name:value.detach().cpu().double() for name,value in weights.items()}
    heads, width, layers = config['heads'], config['width'], config['layers']
    def linear(x, name):
        return F.linear(x, state[name+'.weight'], state[name+'.bias'])
    def norm(x, name):
        return F.layer_norm(x, (width,), state[name+'.weight'], state[name+'.bias'], 1e-5)
    def bits(values):
        return (((values[...,None] >> torch.arange(8)) & 1).double()*2-1)
    condition=linear((states.double()-state['condition_mean'])/state['condition_scale'], 'condition')
    def run(x, cache=None):
        new=[]
        for i in range(layers):
            name=f'blocks.{i}'; batch,length,_=x.shape
            q,k,v=linear(norm(x,name+'.norm1'),name+'.qkv').reshape(batch,length,3,heads,width//heads).permute(2,0,3,1,4).unbind(0)
            if cache is not None:
                k=torch.cat((cache[i][0],k),2);v=torch.cat((cache[i][1],v),2)
            # Explicit attention and softmax, rather than the production SDPA.
            score=(q@k.transpose(-2,-1))/(width//heads)**.5
            if not incremental:
                score=score.masked_fill(torch.ones(length,length,dtype=torch.bool).triu(1),-float('inf'))
            attention=score.softmax(-1)@v
            x=x+linear(attention.transpose(1,2).reshape(batch,length,width),name+'.project')
            x=x+linear(F.gelu(linear(norm(x,name+'.norm2'),name+'.feedforward.0')),name+'.feedforward.2')
            new.append((k,v))
        return linear(norm(x,'final_norm'),'output'), new
    if not incremental:
        previous=linear(bits(targets[:,:-1]),'bits')
        x=torch.cat((state['start'].expand(len(targets),1,-1),previous),1)
        # Preserve the production addition order: x + position, then condition.
        x=x+state['positions'][:targets.shape[1]]+condition[:,None]
        return run(x)[0]
    cache=None;parts=[]
    for j in range(targets.shape[1]):
        x=state['start'].expand(len(targets),1,-1) if j==0 else linear(bits(targets[:,j-1]),'bits')[:,None]
        logits,cache=run(x+state['positions'][j]+condition[:,None],cache)
        parts.append(logits[:,0])
    return torch.stack(parts,1)
