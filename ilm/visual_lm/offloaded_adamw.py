"""FP32 AdamW with host-resident moments and bounded device update chunks.

This is a memory-placement implementation of ordinary AdamW, not a new learning
rule. Model weights and gradients stay on their original device. No numerical
quantization or CPU-side model inference is introduced.
"""
from collections import defaultdict
import math

import torch
from torch.optim import Optimizer


class OffloadedAdamW(Optimizer):
    def __init__(self, params, lr=2e-4, betas=(.9,.999), eps=1e-8, weight_decay=.01,
                 chunk_elements=16_777_216, guard=lambda: None):
        if lr < 0 or eps <= 0 or weight_decay < 0 or not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError('Invalid AdamW hyperparameters')
        if type(chunk_elements) is not int or chunk_elements < 1:
            raise ValueError('Positive device chunk size required')
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay,
                                    chunk_elements=chunk_elements))
        self.guard = guard

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        # Validate all gradients before mutating any parameter or moment.
        for group in self.param_groups:
            for p in group['params']:
                if p.dtype != torch.float32 or not p.is_contiguous() or p.is_complex():
                    raise ValueError('Offloaded AdamW requires contiguous real FP32 parameters')
                g = p.grad
                if g is not None and (g.is_sparse or g.dtype != torch.float32 or g.shape != p.shape
                                      or not bool(torch.isfinite(g).all())):
                    raise ValueError('Finite dense FP32 gradients required before an update')
        for group in self.param_groups:
            beta1,beta2 = group['betas']
            for p in group['params']:
                if p.grad is None: continue
                self.guard()
                state = self.state[p]
                if not state:
                    state.update(step=0,exp_avg=torch.zeros(p.numel(),dtype=torch.float32,device='cpu'),
                                 exp_avg_sq=torch.zeros(p.numel(),dtype=torch.float32,device='cpu'))
                state['step'] += 1
                step = state['step']
                step_size = group['lr'] / (1-beta1**step)
                variance_correction = math.sqrt(1-beta2**step)
                weights = p.view(-1); gradient = p.grad.reshape(-1)
                for first in range(0,p.numel(),group['chunk_elements']):
                    self.guard()
                    last = min(first+group['chunk_elements'],p.numel())
                    mean = state['exp_avg'][first:last].to(p.device)
                    square = state['exp_avg_sq'][first:last].to(p.device)
                    grad = gradient[first:last]
                    weights[first:last].mul_(1-group['lr']*group['weight_decay'])
                    mean.lerp_(grad,1-beta1)
                    square.mul_(beta2).addcmul_(grad,grad,value=1-beta2)
                    denominator = square.sqrt().div_(variance_correction).add_(group['eps'])
                    weights[first:last].addcdiv_(mean,denominator,value=-step_size)
                    state['exp_avg'][first:last].copy_(mean.cpu())
                    state['exp_avg_sq'][first:last].copy_(square.cpu())
        return loss

    def load_state_dict(self, state_dict):
        # The parent loader moves moments to the parameter device. Keep them on
        # CPU explicitly, avoiding a full extra pair of GPU-sized tensors.
        groups = state_dict['param_groups']
        if len(groups) != len(self.param_groups): raise ValueError('Optimizer group count changed')
        mapping = {}
        replacement_groups = []
        for saved,current in zip(groups,self.param_groups):
            if len(saved['params']) != len(current['params']): raise ValueError('Optimizer parameter count changed')
            mapping.update(zip(saved['params'],current['params']))
            replacement_groups.append(dict(saved,params=current['params']))
        replacement_state = defaultdict(dict)
        for identity,stored in state_dict['state'].items():
            if identity not in mapping: raise ValueError('Unknown optimizer state identity')
            p = mapping[identity]; step = stored['step']
            if torch.is_tensor(step): step = int(step.item())
            if type(step) is not int or step < 0: raise ValueError('Invalid Adam counter')
            values = dict(step=step)
            for key in ('exp_avg','exp_avg_sq'):
                v = stored[key]
                if v.dtype != torch.float32 or v.numel() != p.numel() or not bool(torch.isfinite(v).all()):
                    raise ValueError('Invalid FP32 host moment state')
                values[key] = v.detach().reshape(-1).cpu().clone()
            replacement_state[p] = values
        self.param_groups = replacement_groups
        self.state = replacement_state

    def moment_bytes(self):
        return sum(s[k].numel()*s[k].element_size() for s in self.state.values()
                   for k in ('exp_avg','exp_avg_sq') if k in s)
