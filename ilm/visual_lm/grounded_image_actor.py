"""Continuous image reader/context/writer; no language alphabet in the actor.

Inherited modules are supplied as native, vocabulary-free vision/context
bodies. Physical images use -1 for black and +1 for white. Structural queries
and a binary stop event do not identify characters or words.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GroundedImageConfig:
    context_width: int = 4096
    adapter_rank: int = 8
    adapter_alpha: float = 16.
    writer_width: int = 384
    writer_layers: int = 6
    writer_heads: int = 6
    writer_patch: int = 4

    def __post_init__(self):
        if (self.context_width != 4096 or self.adapter_rank < 1
                or self.writer_width % self.writer_heads or self.writer_patch != 4
                or self.writer_layers < 1):
            raise ValueError("incompatible grounded image architecture")


def validate_image(image):
    if (not isinstance(image, torch.Tensor) or image.device.type != "cpu"
            or image.dtype != torch.float32 or image.ndim != 4
            or image.shape[:2] != (1, 1) or min(image.shape[-2:]) < 32
            or any(n % 32 for n in image.shape[-2:])):
        raise ValueError("CPU float32 [1,1,H,W] image, dimensions divisible by 32, required")
    if not bool(torch.isfinite(image).all()) or float(image.abs().max()) > 1:
        raise ValueError("finite physical pixels in [-1,1] required")


def native_image_patches(image):
    """Torch equivalent of pinned Qwen2VLImageProcessor, without resizing.

    Repeat the still image along the two-frame kernel dimension; preserve its
    spatial layout and the archive's RGB mean/std of 0.5. Returns owning packed
    pixels. The signed input already has exactly this normalization.
    """
    validate_image(image)
    height, width = image.shape[-2:]
    rgb = image[0].expand(3, height, width)
    frames = rgb.unsqueeze(0).expand(2, -1, -1, -1)
    packed = frames.reshape(1, 2, 3, height//32, 2, 16, width//32, 2, 16)
    packed = packed.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(-1, 1536).clone()
    return packed, (1, height//16, width//16)


class ImageAdapterLinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)
        device = base.weight.device
        self.down = nn.Parameter(torch.empty(rank, base.in_features, device=device, dtype=torch.float32))
        self.up = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        self.scale = alpha/rank

    def forward(self, x):
        # Trainable parameters and optimizer states stay FP32; the frozen base
        # uses BF16 storage. Autocast may still accelerate both contractions.
        delta = F.linear(F.linear(x.float(), self.down), self.up)*self.scale
        result = self.base(x)
        return result+delta.to(result.dtype)


def time_features(t, width=128):
    frequencies = torch.exp(-math.log(10000)*torch.arange(width//2, device=t.device).float()/(width//2-1))
    phase = t.float()[:, None]*frequencies[None]*1000
    return torch.cat((phase.cos(), phase.sin()), dim=-1)


class ImageFlowBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.qkv = nn.Linear(width, width*3)
        self.attention_output = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, width*4), nn.GELU(approximate="tanh"), nn.Linear(width*4, width))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, width*6))

    def forward(self, x, condition):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(condition).chunk(6, -1)
        h = self.norm1(x)*(1+scale1[:, None])+shift1[:, None]
        b, n, d = h.shape
        q, k, v = self.qkv(h).reshape(b, n, 3, self.heads, d//self.heads).permute(2, 0, 3, 1, 4)
        attention = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
        x = x+gate1[:, None]*self.attention_output(attention.transpose(1, 2).reshape(b, n, d))
        h = self.norm2(x)*(1+scale2[:, None])+shift2[:, None]
        return x+gate2[:, None]*self.mlp(h)


class ContinuousBlockWriter(nn.Module):
    """Conditional flow over all 2,048 physical pixels of a writing block."""

    def __init__(self, config):
        super().__init__()
        d = config.writer_width
        self.input = nn.Conv2d(1, d, 4, stride=4)
        self.position = nn.Parameter(torch.randn(1, 128, d)*.02)
        self.context_norm = nn.LayerNorm(config.context_width)
        self.context_projection = nn.Linear(2*config.context_width, d)
        self.time = nn.Sequential(nn.Linear(128, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(ImageFlowBlock(d, config.writer_heads) for _ in range(config.writer_layers))
        self.norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, 16)

    def forward(self, pixels, times, context):
        if (pixels.ndim != 4 or pixels.shape[1:] != (1, 32, 64)
                or times.shape != pixels.shape[:1] or context.shape != (pixels.shape[0], 2, 4096)):
            raise ValueError("flow needs [B,1,32,64] pixels, [B] times and [B,2,4096] context")
        x = self.input(pixels).flatten(2).transpose(1, 2)+self.position
        condition = self.context_projection(self.context_norm(context.float()).flatten(1))+self.time(time_features(times))
        for block in self.blocks:
            x = block(x, condition)
        return self.output(self.norm(x)).reshape(-1, 8, 16, 4, 4).permute(0, 1, 3, 2, 4).reshape(-1, 1, 32, 64).float()

    @torch.no_grad()
    def sample(self, context, *, steps, generator):
        if type(steps) is not int or steps < 1:
            raise ValueError("a positive flow step count is required")
        x = torch.randn((context.shape[0], 1, 32, 64), device=context.device, generator=generator)
        # Explicit midpoint integration. No external decoder or image bank.
        for i in range(steps):
            t = x.new_full((len(x),), i/steps)
            v = self(x, t, context)
            x = x+self(x+v/(2*steps), t+.5/steps, context)/steps
        return x.clamp(-1, 1)


class GroundedImageActor(nn.Module):
    def __init__(self, vision, context, config=GroundedImageConfig()):
        super().__init__()
        if context.config.vocab_size != 0:
            raise ValueError("contextual body must have zero language vocabulary")
        context.embed_tokens = None
        self.vision = vision.requires_grad_(False).eval()
        self.context = context.requires_grad_(False)
        self.config = config
        for layer in self.context.layers:
            for name in ("q_proj", "v_proj", "o_proj"):
                parent = layer.self_attn
                setattr(parent, name, ImageAdapterLinear(getattr(parent, name), config.adapter_rank, config.adapter_alpha))
        device = next(context.parameters()).device
        self.queries = nn.Parameter(torch.randn(2, config.context_width, device=device)*.02)
        self.writer = ContinuousBlockWriter(config).to(device)
        self.stop = nn.Linear(config.context_width, 1).to(device)
        nn.init.zeros_(self.stop.weight)
        nn.init.constant_(self.stop.bias, -3.)

    @property
    def device(self):
        return self.queries.device

    def train(self, mode=True):
        super().train(mode)
        self.vision.eval()
        return self

    @torch.no_grad()
    def encode_images(self, images):
        if not images:
            raise ValueError("at least one observed image is required")
        packed, grids = zip(*(native_image_patches(image) for image in images))
        values, deep = self.vision(torch.cat(packed).to(self.device, torch.bfloat16),
            grid_thw=torch.tensor(grids, device=self.device, dtype=torch.long))
        counts = [g[0]*g[1]*g[2]//4 for g in grids]
        return list(values.split(counts)), [list(level.split(counts)) for level in deep], grids

    def assemble(self, features, deep, grids):
        """Place query pairs before the corresponding, separately encoded answer.

        Only image features occupy DeepStack positions. The fourth positional
        plane carries strictly sequential mask positions, independent of the
        three spatial/temporal rotary coordinates.
        """
        embeddings, positions, visual, queries = [], [], [], []
        offset = 0
        for index, (feature, grid) in enumerate(zip(features, grids)):
            if index:
                queries.append(sum(len(e) for e in embeddings))
                embeddings.append(self.queries.to(feature.dtype))
                positions.append(torch.tensor([[offset, offset], [offset, offset], [offset, offset+1]], device=self.device))
                visual.extend((False, False))
                offset += 2
            _, gh, gw = grid
            h, w = gh//2, gw//2
            row = torch.arange(h, device=self.device).repeat_interleave(w)+offset
            col = torch.arange(w, device=self.device).repeat(h)+offset
            positions.append(torch.stack((torch.full_like(row, offset), row, col)))
            embeddings.append(feature)
            visual.extend([True]*len(feature))
            offset += max(h, w)
        queries.append(sum(len(e) for e in embeddings))
        embeddings.append(self.queries.to(features[0].dtype))
        positions.append(torch.tensor([[offset, offset], [offset, offset], [offset, offset+1]], device=self.device))
        visual.extend((False, False))
        x = torch.cat(embeddings)[None]
        coordinates = torch.cat(positions, dim=1)
        position_ids = torch.cat((torch.arange(x.shape[1], device=self.device)[None], coordinates), dim=0)[:, None]
        return dict(inputs_embeds=x, position_ids=position_ids,
            visual_pos_masks=torch.tensor(visual, device=self.device)[None],
            deepstack_visual_embeds=[torch.cat(level) for level in deep]), queries, offset+2

    def conditions(self, prompt, completed_blocks=(), *, use_cache=False):
        for block in completed_blocks:
            if block.shape != (1, 1, 32, 64):
                raise ValueError("completed writing must consist of 32-by-64 physical blocks")
        features, deep, grids = self.encode_images([prompt, *completed_blocks])
        inputs, queries, offset = self.assemble(features, deep, grids)
        outputs = self.context(**inputs, use_cache=use_cache)
        indices = torch.tensor([[i, i+1] for i in queries], device=self.device)
        return outputs.last_hidden_state[0, indices], outputs.past_key_values, offset

    def training_loss(self, prompt, targets, *, generator, writer_blocks=16):
        """Teacher forcing over a complete answer; sample only writer losses.

        `targets` excludes the cache's two white termination blocks. Their
        replacement is a supervised structural stop event after the full answer.
        Sampling a uniform subset of blocks gives an unbiased block-mean loss.
        """
        if not targets or writer_blocks < 1:
            raise ValueError("at least one actual target block and writer loss required")
        conditions, _, _ = self.conditions(prompt, targets)
        count = len(targets)
        chosen = torch.randperm(count, device=self.device, generator=generator)[:writer_blocks]
        y = torch.cat([targets[i].to(self.device) for i in chosen.tolist()])
        noise = torch.randn(y.shape, device=self.device, generator=generator)
        t = torch.rand((len(y),), device=self.device, generator=generator)
        noisy = (1-t[:, None, None, None])*noise+t[:, None, None, None]*y
        velocity = self.writer(noisy, t, conditions[chosen])
        flow = F.mse_loss(velocity.float(), y-noise)
        logits = self.stop(conditions.float().mean(dim=1)).flatten().float()
        # Equal weight to actual stop and mean continue events, independent of
        # answer length. White space itself does not determine this decision.
        stop = .5*(F.softplus(-logits[-1])+F.softplus(logits[:-1]).mean())
        return flow+stop, dict(flow=flow.detach(), stop=stop.detach(),
            selected_blocks=chosen.detach(), conditioning_states=conditions.detach())

    @torch.no_grad()
    def append_generated(self, block, past, *, offset):
        if block.shape != (1, 1, 32, 64):
            raise ValueError("one completed physical writing block required")
        features, deep, grids = self.encode_images([block])
        h, w = grids[0][1]//2, grids[0][2]//2
        feature = features[0]
        positions = torch.tensor([[offset]*2, [offset]*2, [offset, offset+1]], device=self.device)
        query_offset = offset+max(h, w)
        positions = torch.cat((positions, torch.tensor([[query_offset]*2, [query_offset]*2,
            [query_offset, query_offset+1]], device=self.device)), dim=1)
        length = past.get_seq_length()
        position_ids = torch.cat((torch.arange(length, length+4, device=self.device)[None], positions), dim=0)[:, None]
        result = self.context(inputs_embeds=torch.cat((feature, self.queries.to(feature.dtype)))[None],
            position_ids=position_ids, visual_pos_masks=torch.tensor([[True, True, False, False]], device=self.device),
            deepstack_visual_embeds=[level[0] for level in deep], past_key_values=past, use_cache=True)
        return result.last_hidden_state[:, -2:], result.past_key_values, query_offset+2

    @torch.no_grad()
    def generate(self, prompt, *, maximum_blocks=128, flow_steps=16, seed=0):
        if type(maximum_blocks) is not int or maximum_blocks < 1:
            raise ValueError("positive output capacity required")
        self.eval()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        states, past, offset = self.conditions(prompt, use_cache=True)
        condition = states[-1:]
        images, events = [], []
        for index in range(maximum_blocks):
            probability = float(self.stop(condition.float().mean(dim=1)).float().sigmoid())
            if probability >= .5:
                events.append(dict(block=index, stop_probability=probability, stopped=True))
                break
            pixels = self.writer.sample(condition, steps=flow_steps, generator=generator)
            block = torch.where(pixels >= 0, 1., -1.).cpu()
            images.append(block)
            events.append(dict(block=index, stop_probability=probability, stopped=False))
            if index+1 < maximum_blocks:
                condition, past, offset = self.append_generated(block, past, offset=offset)
        return dict(images=tuple(images), events=events,
            stopped=bool(events[-1]["stopped"]), capped=not events[-1]["stopped"])

    def trainable_state(self):
        return {name: p.detach().cpu().clone() for name, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state(self, state):
        parameters = {name: p for name, p in self.named_parameters() if p.requires_grad}
        if state.keys() != parameters.keys():
            raise ValueError("trainable actor identity differs")
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(state[name])
