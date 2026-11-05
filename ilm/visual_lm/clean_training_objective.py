"""Clean writing with a declared logit-normal noise schedule for full training."""

import torch
from torch.nn import functional as F

from .clean_writing_flow import CleanSemanticWritingModel


class CleanTrainingModel(CleanSemanticWritingModel):
    """Same inference architecture and state keys; only the training law differs."""

    def training_loss(self, memory, codes, lengths, *, generator, writer_blocks=16):
        if writer_blocks < 1 or bool((lengths < 1).any()):
            raise ValueError("complete nonempty training answers required")
        states = self.conditions(memory, codes, lengths)
        logits = self.stop(states).squeeze(-1).float()
        batches, positions, weights, continuing, terminal = [], [], [], [], []
        for b, n in enumerate(lengths.tolist()):
            chosen = torch.randperm(n, device=self.device, generator=generator)[:writer_blocks]
            batches.append(torch.full_like(chosen, b))
            positions.append(chosen)
            weights.append(torch.full((len(chosen),), 1/(len(lengths)*len(chosen)), device=self.device))
            continuing.append(F.softplus(logits[b, :n]).mean())
            terminal.append(F.softplus(-logits[b, n]))
        batch, position = torch.cat(batches), torch.cat(positions)
        target = codes[batch, position].float().flatten(1)
        noise = torch.randn(target.shape, device=self.device, generator=generator)
        time = (-.8+.8*torch.randn((len(target),), device=self.device, generator=generator)).sigmoid()
        noisy = (1-time[:, None])*noise+time[:, None]*target
        clean = self.flow(noisy, time, states[batch, position])
        error = (clean-target).square().mean(-1)
        writing = (error/(1-time).clamp_min(self.config.loss_noise_minimum).square()*torch.cat(weights)).sum()
        stop = .5*(torch.stack(continuing).mean()+torch.stack(terminal).mean())
        return writing+stop, dict(flow=writing.detach(), clean=error.mean().detach(),
            stop=stop.detach(), sampled_blocks=len(target))
