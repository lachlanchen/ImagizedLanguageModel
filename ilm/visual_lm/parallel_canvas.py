"""Predict an entire writing image without a text-continuation interface."""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CanvasConfig:
    slots: int = 64
    width: int = 384
    layers: int = 6
    heads: int = 6
    memory_width: int = 4096
    code_width: int = 768
    epsilon: float = 1e-5

    def __post_init__(self):
        if (self.slots < 2 or self.slots % 2 or self.width % self.heads
                or self.layers < 1 or self.memory_width != 4096 or self.code_width != 768):
            raise ValueError("invalid continuous canvas architecture")


class ParallelCanvasWriter(nn.Module):
    """A continuous encoder-decoder; inference accepts prompt memory only.

    Positions identify image tiles, never characters or words. The codec is
    separate and frozen. No target or history argument exists in forward.
    """
    def __init__(self, white_code, config=CanvasConfig()):
        super().__init__()
        if white_code.shape != (768,) or not torch.isfinite(white_code).all():
            raise ValueError("the actual white-image codec coordinate is required")
        self.config = config
        self.register_buffer("white_code", white_code.detach().float().clone())
        self.memory = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, config.width))
        self.queries = nn.Parameter(torch.randn(1, config.slots, config.width)*.02)
        self.layers = nn.ModuleList(nn.TransformerDecoderLayer(config.width, config.heads,
            dim_feedforward=4*config.width, dropout=0., activation="gelu", batch_first=True,
            norm_first=True) for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, 768)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    @property
    def device(self):
        return self.queries.device

    def configuration(self):
        return asdict(self.config)

    def forward(self, memory):
        if (memory.ndim != 3 or memory.shape[-1] != 4096 or memory.shape[1] < 1
                or memory.device != self.device or not memory.is_floating_point()):
            raise ValueError("prompt-only visual memory [B,N,4096] required")
        m = self.memory(memory.float())
        x = self.queries.expand(len(memory), -1, -1)
        for layer in self.layers:
            x = layer(x, m)
        with torch.autocast(device_type=self.device.type, enabled=False):
            residual = F.linear(self.norm(x).float(), self.output.weight.float(), self.output.bias.float())
            return F.layer_norm(self.white_code+residual, (768,), eps=self.config.epsilon)

    def training_loss(self, memory, target, tile_lengths):
        if (target.shape != (len(memory), self.config.slots, 768)
                or tile_lengths.shape != (len(memory),)
                or bool(((tile_lengths < 1) | (tile_lengths > self.config.slots)).any())):
            raise ValueError("complete canvas targets and active tile lengths required")
        prediction = self(memory)
        errors = (prediction-target.float()).square().mean(-1)
        active = torch.arange(self.config.slots, device=self.device)[None] < tile_lengths[:,None]
        a = (errors*active).sum(-1)/active.sum(-1)
        pads = (~active).sum(-1)
        p = (errors*~active).sum(-1)/pads.clamp_min(1)
        loss = torch.where(pads > 0, .5*(a+p), a).mean()
        return loss, dict(active_mse=a.mean().detach(), padding_mse=p.mean().detach())

    @torch.no_grad()
    def generate(self, memory, codec):
        """Return the full fixed canvas and its predicted continuous coordinates."""
        with torch.autocast(device_type=self.device.type, enabled=False):
            coordinates = self(memory)
            logits = codec.decode(coordinates.reshape(-1, 768))
        tiles = (logits >= 0).to(torch.uint8).mul(255).reshape(len(memory), self.config.slots, 32, 32)
        canvas = tiles.permute(0,2,1,3).reshape(len(memory),32,32*self.config.slots)
        return canvas.cpu(), coordinates.cpu()


class CanvasImageActor(nn.Module):
    def __init__(self, observer, writing, codec):
        super().__init__()
        self.observer = observer.requires_grad_(False).eval()
        self.writing = writing.eval()
        self.codec = codec.requires_grad_(False).eval()
        if observer.actor.context.embed_tokens is not None or observer.actor.context.config.vocab_size:
            raise ValueError("language vocabulary must be absent from image inference")

    @torch.inference_mode()
    def generate(self, prompt):
        memory, geometry = self.observer(prompt)
        canvas, coordinates = self.writing.generate(memory, self.codec)
        return dict(image=canvas[0], coordinates=coordinates[0], geometry=geometry,
                    memory=memory.cpu(), image_input_image_output=True,
                    teacher_service=False, language_vocabulary=False, fixed_canvas=True)


def load_canvas_actor(bundle, *, controls, codec_path, ollama_root, gguf_source, device):
    from .codec_block_writer import load_qualified_writing_codec
    from .grounded_image_weights import load_grounded_image_actor, file_sha
    from .semantic_image_actor import CONTROL_SHA
    from .semantic_image_memory import SemanticImageMemory
    saved = torch.load(bundle, map_location="cpu", weights_only=True)
    if not saved.get("training_complete") or saved.get("updates") != 2000 or saved.get("purpose") != "parallel_canvas_fit":
        raise ValueError("a completed declared canvas inference bundle is required")
    if file_sha(controls) != CONTROL_SHA:
        raise ValueError("source control vectors changed")
    writing = ParallelCanvasWriter(saved["model"]["white_code"], CanvasConfig(**saved["config"]))
    writing.load_state_dict(saved["model"], strict=True); writing = writing.to(device).eval()
    base, provenance = load_grounded_image_actor(ollama_root=ollama_root, gguf_source=gguf_source, device=device)
    base.writer = nn.Identity(); base.requires_grad_(False).eval()
    observer = SemanticImageMemory(base, **torch.load(controls, map_location="cpu", weights_only=True))
    codec = load_qualified_writing_codec(codec_path, device)
    actor = CanvasImageActor(observer, writing, codec).eval()
    return actor, dict(source=provenance, complete_parameters=sum(p.numel() for p in actor.parameters()),
        trained_parameters=sum(p.numel() for p in writing.parameters()), bundle_sha256=file_sha(bundle),
        external_pretraining=True, fixed_canvas=True, general_ilm_goal_complete=False)
