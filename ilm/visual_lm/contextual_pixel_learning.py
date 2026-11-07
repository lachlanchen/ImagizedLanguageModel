"""Differentiable contextual span reader with the completed forward interface."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .causal_visual_prefix import fold_writing_prompt
from .contextual_pixel_encoder import ContextualPixelReader
from .contextual_pixel_spans import units
from .grounded_image_actor import ImageAdapterLinear
from .semantic_image_memory import native_budget_image


def pooling_matrix(grid, height, width):
    """Fixed bilinear sampling map, averaged over each unit's sample points.

    Construct on CPU in double precision. Border handling and coordinates
    follow grid_sample(align_corners=False, padding_mode='border').
    """
    if grid.ndim != 4 or grid.shape[0] != 1 or grid.shape[-1] != 2:
        raise ValueError("one fixed [1,units,samples,2] grid required")
    xy = grid.detach().cpu().double()[0]
    x = ((xy[..., 0] + 1) * width / 2 - .5).clamp(0, width - 1)
    y = ((xy[..., 1] + 1) * height / 2 - .5).clamp(0, height - 1)
    x0, y0 = x.floor().long(), y.floor().long()
    dx, dy = x - x0, y - y0
    matrix = torch.zeros(x.shape[0], height * width, dtype=torch.float64)
    for iy, wy in ((0, 1 - dy), (1, dy)):
        for ix, wx in ((0, 1 - dx), (1, dx)):
            columns = (y0 + iy).clamp(max=height - 1) * width + (x0 + ix).clamp(max=width - 1)
            matrix.scatter_add_(1, columns, wy * wx / x.shape[1])
    return matrix


class _FixedGridPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, field, grid, matrix):
        ctx.save_for_backward(matrix)
        ctx.field_shape = field.shape
        return F.grid_sample(field, grid, align_corners=False, padding_mode="border").mean(-1).transpose(1, 2)[0]

    @staticmethod
    def backward(ctx, gradient):
        matrix, = ctx.saved_tensors
        return (gradient.transpose(0, 1) @ matrix).reshape(ctx.field_shape), None, None


def fixed_grid_pool(field, grid):
    if field.ndim != 4 or field.shape[0] != 1 or field.dtype != grid.dtype:
        raise ValueError("one field and matching fixed-grid dtype required")
    matrix = pooling_matrix(grid, *field.shape[-2:]).to(field.device, field.dtype)
    return _FixedGridPool.apply(field, grid.detach(), matrix)


class LearningPixelReader(nn.Module):
    """Reuse the original source/head, exposing contextual adapters to image loss."""
    def __init__(self, **source):
        super().__init__()
        reader = ContextualPixelReader(**source)
        self.observer, self.head = reader.observer, reader.head
        self.provenance = reader.provenance
        self.observer.requires_grad_(False)
        for module in self.observer.actor.context.modules():
            if isinstance(module, ImageAdapterLinear):
                module.down.requires_grad_(True)
                module.up.requires_grad_(True)
        self.head.requires_grad_(True)
        self.observer.actor.context.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        self.eval()

    @property
    def device(self):
        return next(self.head.parameters()).device

    def train(self, mode=True):
        super().train(mode)
        self.observer.actor.vision.eval()
        return self

    def encode(self, image):
        pixels = np.asarray(image)
        spans = units(pixels)
        physical = torch.from_numpy(pixels.copy()).float().div(127.5).sub(1)[None, None]
        folded = fold_writing_prompt(physical)
        arguments, metadata = self.observer.assemble(native_budget_image(folded.pixels))
        with torch.autocast(device_type=self.device.type, enabled=False):
            output = self.observer.actor.context(**arguments, use_cache=False)
        start = metadata["before_positions"]
        h, w = metadata["grid"][1] // 2, metadata["grid"][2] // 2
        field = output.last_hidden_state[:, start:start + h*w].transpose(1, 2).reshape(1, 4096, h, w).float()
        ends = np.array([r[1] for r in folded.source_ranges])
        row = np.searchsorted(ends, spans.mean(1), side="right")
        left = np.array([r[0] for r in folded.source_ranges])[row]
        points = []
        for fy in (.25, .5, .75):
            for fx in (.25, .5, .75):
                xx = spans[:, 0] + fx * (spans[:, 1] - spans[:, 0]) - left
                yy = row * 32 + fy * 32
                points.append(np.stack((2 * xx / folded.pixels.shape[-1] - 1,
                                        2 * yy / folded.pixels.shape[-2] - 1), -1))
        grid = torch.from_numpy(np.stack(points, 1)).to(self.device, torch.float32)[None]
        visual = fixed_grid_pool(field, grid)
        geometry = torch.tensor(np.stack((spans[:, 0] / pixels.shape[1], spans[:, 1] / pixels.shape[1],
            row / max(1, len(folded.source_ranges)), (spans[:, 1] - spans[:, 0]) / 32), -1), dtype=torch.float32)
        # Retain the exact BF16 source/head boundary, now with autograd attached.
        return dict(visual=visual.to(torch.bfloat16),
            memory=output.last_hidden_state[0, -len(self.observer.after):].to(torch.bfloat16),
            geometry=geometry.to(self.device), source_units=torch.from_numpy(spans.copy()), metadata=metadata)

    def forward(self, image):
        values = self.encode(image)
        logp = self.head(values["visual"][None].float(), values["memory"][None].float(),
            values["geometry"][None], torch.tensor([len(values["visual"])], device=self.device))[0]
        return logp, values

    def trained_parameters(self):
        return {n:p for n,p in self.named_parameters() if p.requires_grad}

    def trained_state(self):
        return {n:p.detach().cpu().clone() for n,p in self.trained_parameters().items()}

    def restore_trained_state(self, state):
        params = self.trained_parameters()
        if params.keys() != state.keys():
            raise ValueError("trainable parameter set changed")
        with torch.no_grad():
            for name, parameter in params.items():
                parameter.copy_(state[name])

    def optimizer(self):
        groups = [{"params":list(self.head.parameters()), "lr":1e-4},
                  {"params":[p for n,p in self.trained_parameters().items() if not n.startswith("head.")], "lr":1e-5}]
        return torch.optim.AdamW(groups, betas=(.9,.95), weight_decay=.01, foreach=False)
