"""Keep native image generation fixed while testing decoder-aware regression."""
import math

import torch
from torch.nn import functional as F

from .adaptive_image_continuation import AdaptiveImageLanguageModel
from .decoder_pullback import decoder_pullback_error


class MetricImageLanguageModel(AdaptiveImageLanguageModel):
    """Same learned tensors and autonomous sampler; a training-only metric."""

    def __init__(self, context, writer, *, input_scale, query_seed, metric_strength):
        super().__init__(context, writer, input_scale=input_scale, query_seed=query_seed)
        if not math.isfinite(metric_strength) or metric_strength < 0:
            raise ValueError("finite nonnegative writing metric strength required")
        self.metric_strength = metric_strength

    def training_loss(self, prefix, codes, lengths, *, generator, writer_blocks=16):
        if (lengths.shape != (len(codes),) or bool(((lengths<1)|(lengths>codes.shape[1])).any())
                or writer_blocks<1):
            raise ValueError("complete nonempty image targets and per-example lengths required")
        states = self.conditions(prefix, codes)
        batches, positions, weights, continuing, terminal = [], [], [], [], []
        with torch.autocast(device_type=self.device.type, enabled=False):
            logits = self.stop(states.float()).squeeze(-1)
        for b, n in enumerate(lengths.tolist()):
            chosen = torch.randperm(n, device=self.device, generator=generator)[:writer_blocks]
            batches.append(torch.full_like(chosen,b));positions.append(chosen)
            weights.append(torch.full((len(chosen),),1/(len(codes)*len(chosen)),device=self.device))
            continuing.append(F.softplus(logits[b,:n]).mean());terminal.append(F.softplus(-logits[b,n]))
        batch, position = torch.cat(batches), torch.cat(positions)
        target = codes[batch,position].float().flatten(1)
        noise = torch.randn(target.shape, device=self.device, generator=generator)
        time = torch.rand((len(target),), device=self.device, generator=generator)
        noisy = (1-time[:,None])*noise+time[:,None]*target
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type=="cuda"):
            clean = self.flow(noisy, time, states[batch,position])
        error, components = decoder_pullback_error(self.writer.codec, noisy, clean-target,
            strength=self.metric_strength)
        writing = (error/(1-time).clamp_min(.05).square()*torch.cat(weights)).sum()
        stopping = .5*(torch.stack(continuing).mean()+torch.stack(terminal).mean())
        return writing+stopping, dict(flow=writing.detach(), clean=components["code"].mean().detach(),
            decoder_tangent=components["decoder_tangent"].mean().detach(), stop=stopping.detach(),
            sampled_blocks=len(target))
