"""Add a learned, question-conditioned context correction to optical span scores."""
from threading import Lock
import torch
from torch import nn
from torch.nn import functional as F


def side_means(values, valid, radius):
    """Mean of valid preceding/following cells, excluding the candidate cell."""
    pos = torch.arange(values.size(1), device=values.device)
    lo = (pos - radius).clamp_min(0)
    hi = (pos + 1 + radius).clamp_max(values.size(1))
    weights = valid.unsqueeze(-1).float()
    total = F.pad((values.float() * weights).cumsum(1), (0, 0, 1, 0))
    count = F.pad(weights.cumsum(1), (0, 0, 1, 0))
    left = (total[:, pos] - total[:, lo]) / (count[:, pos] - count[:, lo]).clamp_min(1)
    right = (total[:, hi] - total[:, pos+1]) / (count[:, hi] - count[:, pos+1]).clamp_min(1)
    return left, right


class DirectionalCorrection(nn.Module):
    radii = (4, 16, 64, 256)

    def __init__(self, mode, width=32):
        super().__init__()
        if mode not in ("signed", "unsigned"):
            raise ValueError(mode)
        self.mode = mode
        self.cell = nn.Sequential(nn.Linear(128, width), nn.LayerNorm(width), nn.GELU())
        self.query = nn.Linear(64, width)
        self.combine = nn.Sequential(nn.Linear((1+2*len(self.radii))*width, 64),
                                     nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 2))
        nn.init.zeros_(self.combine[-1].weight)
        nn.init.zeros_(self.combine[-1].bias)

    def forward(self, state, question, valid):
        cell = self.cell(state.float()) * (1 + self.query(question.float()).tanh()[:, None, :])
        blocks = [cell]
        for radius in self.radii:
            left, right = side_means(cell, valid, radius)
            delta = right-left
            blocks.extend((left+right, delta if self.mode == "signed" else delta.abs()))
        return self.combine(torch.cat(blocks, -1)).unbind(-1)


class ContextualSpanReader(nn.Module):
    def __init__(self, frozen, correction):
        super().__init__()
        self.frozen = frozen.requires_grad_(False).eval()
        self.correction = correction
        self._capture_lock = Lock()

    def lattice(self, features, query, page):
        # The preserved reader has a hook-only feature interface. Scope both captures
        # to this invocation including concurrent calls to the same deployed instance.
        boundaries, pool_logits, encoded = [], [], []
        with self._capture_lock, torch.no_grad():
            handles = [
                self.frozen.boundaries.register_forward_pre_hook(lambda _, args: boundaries.append(args[0])),
                self.frozen.pool.register_forward_pre_hook(lambda _, args: encoded.append(args[0])),
                self.frozen.pool.register_forward_hook(lambda _, args, output: pool_logits.append(output)),
            ]
            try:
                base = self.frozen(features, query, page)
            finally:
                for hook in handles:
                    hook.remove()
            qweight = pool_logits[0].squeeze(-1).float().masked_fill(query.eq(0), -torch.inf).softmax(-1)
            question = (qweight[:, :, None] * encoded[0]).sum(1)
        state = boundaries[0]
        start, end = self.correction(state, question, page.ne(0))
        length = base.size(-1)
        tails = F.pad(end, (0, length - 1)).unfold(1, length, 1)
        return base+start[:, :, None]+tails, state

    @torch.inference_mode()
    def select(self, features, query, page, ranker):
        scores, state = self.lattice(features, query, page)
        flat = scores.flatten(1)
        ix = flat.argsort(descending=True, stable=True)[:, :32]
        base = flat.gather(1, ix)
        start = (ix // scores.size(-1)).masked_fill(~base.isfinite(), 0)
        size = (ix % scores.size(-1)+1).masked_fill(~base.isfinite(), 1)
        end = start + size - 1
        batch = torch.arange(len(query), device=query.device)[:, None]
        totals = F.pad(state.float().cumsum(1), (0,0,1,0))
        inside = (totals[batch,end+1]-totals[batch,start]) / size.unsqueeze(-1)
        rep = torch.stack((state[batch,start], state[batch,end], inside), 2).half()
        selected = ranker(rep,base,size).argmax(1,keepdim=True)
        return start.gather(1,selected).flatten(), size.gather(1,selected).flatten()
