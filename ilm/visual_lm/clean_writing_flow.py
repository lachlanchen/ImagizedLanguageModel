"""Predict clean continuous writing codes; contract noise analytically."""

from dataclasses import dataclass
import hashlib

import torch
from torch.nn import functional as F

from .full_rank_writing_flow import FullRankWritingFlow
from .semantic_writing_model import SemanticWritingConfig, SemanticWritingModel, sinusoidal


@dataclass(frozen=True)
class CleanWritingConfig(SemanticWritingConfig):
    flow_width: int = 1536
    prediction_kind: str = "clean_endpoint"
    loss_noise_minimum: float = .05

    def __post_init__(self):
        super().__post_init__()
        if (self.flow_width < self.block_width or self.prediction_kind != "clean_endpoint"
                or not 0 < self.loss_noise_minimum < 1):
            raise ValueError("invalid clean writing configuration")


class CleanWritingFlow(FullRankWritingFlow):
    def __init__(self, config=None):
        super().__init__(CleanWritingConfig() if config is None else config)

    def forward(self, noisy, time, condition):
        hidden = self.noisy(noisy)+self.condition(condition)+self.time(sinusoidal(time*1000, 128))
        hidden = self.blocks(hidden)
        # Casting after a BF16 projection would retain its quantization error.
        # The physical writing coordinates receive an actual FP32 projection.
        with torch.autocast(device_type=noisy.device.type, enabled=False):
            return F.linear(hidden.float(), self.output.weight.float(), self.output.bias.float())

    @torch.no_grad()
    def sample_endpoint(self, condition, *, generator, steps=32):
        if type(steps) is not int or steps < 1 or condition.ndim != 2 or condition.shape[1] != self.condition.in_features:
            raise ValueError("positive step count and continuous conditions required")
        device = condition.device
        noise = torch.randn((len(condition), 1536), device=device, generator=generator)
        value = noise.clone()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for index in range(steps):
                clean = self(value, torch.full((len(value),), index/steps, device=device), condition)
                # Exact linear-noise contraction when the clean estimate is
                # constant on this interval. The final update has zero noise
                # coefficient; no division is evaluated at t=1.
                remaining = steps-index
                value = ((remaining-1)/remaining)*value+clean/remaining
        return noise, value


class CleanSemanticWritingModel(SemanticWritingModel):
    """Complete causal writer with image history and clean-code prediction.

    The extra configuration fields deliberately prevent a velocity-only
    checkpoint loader from silently accepting these weights.
    """

    def __init__(self, config=None):
        config = CleanWritingConfig() if config is None else config
        super().__init__(config)
        self.flow = CleanWritingFlow(config)

    def training_loss(self, memory, codes, lengths, *, generator, writer_blocks=16):
        if writer_blocks < 1 or bool((lengths < 1).any()):
            raise ValueError("complete nonempty training answers required")
        states = self.conditions(memory, codes, lengths)
        logits = self.stop(states).squeeze(-1).float()
        batches, positions, weights = [], [], []
        continuing, terminal = [], []
        for b, n in enumerate(lengths.tolist()):
            chosen = torch.randperm(n, device=self.device, generator=generator)[:writer_blocks]
            batches.append(torch.full_like(chosen, b))
            positions.append(chosen)
            weights.append(torch.full((len(chosen),), 1/(len(lengths)*len(chosen)), device=self.device))
            continuing.append(F.softplus(logits[b, :n]).mean())
            terminal.append(F.softplus(-logits[b, n]))
        b, index = torch.cat(batches), torch.cat(positions)
        target = codes[b, index].float().flatten(1)
        noise = torch.randn(target.shape, device=self.device, generator=generator)
        time = torch.rand((len(target),), device=self.device, generator=generator)
        noisy = (1-time[:, None])*noise+time[:, None]*target
        clean = self.flow(noisy, time, states[b, index])
        error = (clean-target).square().mean(-1)
        denominator = (1-time).clamp_min(self.config.loss_noise_minimum)
        weighted = (error/denominator.square()*torch.cat(weights)).sum()
        stop = .5*(torch.stack(continuing).mean()+torch.stack(terminal).mean())
        return weighted+stop, dict(flow=weighted.detach(), clean=error.mean().detach(),
            stop=stop.detach(), sampled_blocks=len(target))

    @torch.no_grad()
    def generate(self, memory, writer, *, generator, maximum_blocks=None):
        if memory.ndim != 3 or len(memory) != 1:
            raise ValueError("one prompt memory required")
        cap = self.config.maximum_blocks if maximum_blocks is None else maximum_blocks
        if not 1 <= cap <= self.config.maximum_blocks:
            raise ValueError("invalid generation cap")
        history = torch.empty((1, 0, 2, 768), device=self.device)
        images, probabilities, hashes = [], [], []
        stopped = False
        for position in range(cap+1):
            condition = self.conditions(memory, history)[:, -1]
            probability = self.stop(condition).float().sigmoid().item()
            probabilities.append(probability)
            if probability >= .5:
                stopped = True
                break
            if position == cap:
                break
            _, endpoint = self.flow.sample_endpoint(condition, generator=generator, steps=self.config.flow_steps)
            coordinates = F.layer_norm(endpoint.reshape(-1, 768), (768,),
                eps=writer.codec.config.layer_norm_epsilon).reshape(1, 2, 768)
            soft = writer.predict(coordinates)
            image = torch.where(soft < 0, -torch.ones_like(soft), torch.ones_like(soft))
            reread = writer.encode_targets(image).to(self.device)
            history = torch.cat((history, reread[None]), dim=1)
            images.append(image.cpu())
            hashes.append(hashlib.sha256(reread.cpu().numpy().tobytes()).hexdigest())
        return dict(images=torch.cat(images) if images else torch.empty(0, 1, 32, 64),
            stopped=stopped, capped=not stopped, stop_probabilities=probabilities,
            feedback_sha256=hashes, feedback_from_emitted_pixels=True)
