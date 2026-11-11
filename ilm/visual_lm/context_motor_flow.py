"""Context-conditioned continuous glyph motor with no lexical runtime table."""
import math

import torch
from torch import nn


def time_features(t):
    frequencies = torch.exp(-math.log(10000.)*torch.arange(64, device=t.device).float()/63)
    phase = 1000*t.float()[:, None]*frequencies[None]
    return torch.cat([phase.cos(), phase.sin()], dim=-1)


class MotorBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.path = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(),
                                  nn.Linear(width, width))

    def forward(self, x):
        return x + .1*self.path(x)


class ContextMotorFlow(nn.Module):
    """Velocity in all1024 motor coordinates, conditional on the actual reader h."""
    def __init__(self):
        super().__init__()
        width = 1536
        self.noisy = nn.Linear(1024, width)
        self.context = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, width))
        self.time = nn.Sequential(nn.Linear(128, 512), nn.SiLU(), nn.Linear(512, width))
        self.blocks = nn.Sequential(*(MotorBlock(width) for _ in range(3)))
        self.output = nn.Linear(width, 1024)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def condition(self, state):
        if state.ndim != 2 or state.shape[-1] != 4096:
            raise ValueError('A complete4096-dimensional actual contextual state is required')
        return self.context(state.float())

    def velocity(self, x, t, condition):
        if (x.ndim != 2 or x.shape[-1] != 1024 or t.shape != (len(x),)
                or condition.shape != (len(x), 1536)):
            raise ValueError('One full motor state, time and context per sample required')
        hidden = self.noisy(x.float()) + condition + self.time(time_features(t))
        return self.output(self.blocks(hidden)).float()

    def forward(self, x, t, state):
        return self.velocity(x, t, self.condition(state))

    @torch.no_grad()
    def sample(self, state, *, steps=32, generator):
        if steps != 32:
            raise ValueError('The registered candidate fixes32Heun intervals, last Euler')
        noise = torch.randn((len(state), 1024), generator=generator, device=state.device, dtype=torch.float32)
        x = noise.clone()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=x.is_cuda):
            condition = self.condition(state)
            for i in range(steps):
                t = torch.full((len(x),), i/steps, device=x.device)
                v = self.velocity(x, t, condition)
                proposal = x + v/steps
                if i+1 == steps:
                    x = proposal
                else:
                    v2 = self.velocity(proposal, t+1/steps, condition)
                    x = x + .5*(v+v2)/steps
        return x, noise


class ContextMotorBridge(nn.Module):
    """Sample pixels, and reread those pixels with the unchanged neural encoder.

    Only the encoder, decoder, stopping and semantic-affine modules are retained
    from the base bridge. Its deterministic readout/decoder_input are not used
    or installed here. Motor-coordinate statistics are1024 scalars, not a glyph
    inventory. The per-case RNG belongs to the caller and is saved in traces.
    """
    def __init__(self, base, flow, mean, scale):
        super().__init__()
        if mean.shape != (1024,) or scale.shape != (1024,) or not bool((scale > 0).all()):
            raise ValueError('Positive, full motor-coordinate normalization required')
        self.encoder, self.decoder, self.stop = base.encoder, base.decoder, base.stop
        self.flow = flow
        self.semantic_width = 4096
        self.register_buffer('semantic_mean', base.semantic_mean.detach().clone())
        self.register_buffer('semantic_scale', base.semantic_scale.detach().clone())
        self.register_buffer('motor_mean', mean.detach().float().clone())
        self.register_buffer('motor_scale', scale.detach().float().clone())
        self.generator = None
        self.motor_trace = []

    def begin_case(self, seed):
        self.generator = torch.Generator(device=self.motor_mean.device).manual_seed(seed)
        self.motor_trace = []

    def feedback_embedding(self, images):
        return self.encoder(images.float()).float()*self.semantic_scale + self.semantic_mean

    @torch.no_grad()
    def predict(self, state):
        from .semantic_glyph_loop import tensor_hash
        stop = self.stop(state.float()).flatten()
        if bool((stop.sigmoid() >= .5).all()):
            return torch.zeros(len(state), 1, 32, 32, device=state.device), state, stop
        if self.generator is None:
            raise ValueError('Set a prospectively fixed per-case seed before sampling')
        normalized, noise = self.flow.sample(state, generator=self.generator)
        motor = normalized*self.motor_scale + self.motor_mean
        logits = self.decoder(motor.reshape(-1, 64, 4, 4)).float()
        self.motor_trace.append(dict(noise=noise.cpu(), normalized_motor=normalized.cpu(),
            noise_sha256=tensor_hash(noise), motor_sha256=tensor_hash(motor),
            image_sha256=tensor_hash(1-2*(logits.sigmoid() >= .5).float())))
        return logits, motor, stop
