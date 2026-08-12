from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class FovealResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.expand = nn.Conv2d(channels, channels * 3, 1)
        self.contract = nn.Conv2d(channels * 3, channels, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        hidden = self.depthwise(image)
        hidden = self.contract(F.silu(self.expand(self.norm(hidden))))
        return image + hidden


@dataclass(frozen=True)
class VisualSaccadeConfig:
    fovea_size: int = 32
    visual_dim: int = 192
    state_dim: int = 384
    state_layers: int = 3
    retina_base_channels: int = 64
    ink_base_channels: int = 96
    dropout: float = 0.05

    def __post_init__(self) -> None:
        if self.fovea_size < 16 or self.fovea_size % 8:
            raise ValueError("fovea_size must be a multiple of eight and at least 16")
        if self.visual_dim < 64 or self.state_dim < 128:
            raise ValueError("visual saccade state is underspecified")
        if self.state_layers < 1:
            raise ValueError("state_layers must be positive")


class FovealRetina(nn.Module):
    """Encode a continuous ink fixation without a glyph or character lookup."""

    def __init__(self, config: VisualSaccadeConfig):
        super().__init__()
        base = config.retina_base_channels
        self.config = config
        self.stem = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(4, base, 3, padding=1),
            nn.GroupNorm(_groups(base), base),
            nn.SiLU(),
            FovealResidualBlock(base),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1),
            FovealResidualBlock(base * 2),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base * 2, base * 3, 4, stride=2, padding=1),
            FovealResidualBlock(base * 3),
            FovealResidualBlock(base * 3),
        )
        self.output = nn.Sequential(
            nn.Linear(base * 6, config.visual_dim * 2),
            nn.SiLU(),
            nn.Linear(config.visual_dim * 2, config.visual_dim),
            nn.LayerNorm(config.visual_dim),
        )

    def forward(self, fovea: torch.Tensor) -> torch.Tensor:
        expected = (1, self.config.fovea_size, self.config.fovea_size)
        if fovea.ndim != 4 or tuple(fovea.shape[1:]) != expected:
            raise ValueError(f"expected continuous foveal images [batch, {expected}]")
        if not torch.is_floating_point(fovea):
            raise TypeError("FovealRetina accepts continuous image tensors only")
        hidden = self.down2(self.down1(self.stem(fovea.clamp(0, 1))))
        summary = torch.cat((hidden.mean(dim=(2, 3)), hidden.amax(dim=(2, 3))), dim=-1)
        return self.output(summary)


class NextFoveaInkHead(nn.Module):
    def __init__(self, config: VisualSaccadeConfig):
        super().__init__()
        base = config.ink_base_channels
        seed = config.fovea_size // 8
        self.seed = seed
        self.base = base
        self.input = nn.Sequential(
            nn.LayerNorm(config.state_dim + config.visual_dim),
            nn.Linear(config.state_dim + config.visual_dim, base * 4 * seed * seed),
            nn.SiLU(),
        )
        self.decode = nn.Sequential(
            FovealResidualBlock(base * 4),
            nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1),
            FovealResidualBlock(base * 2),
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1),
            FovealResidualBlock(base),
            nn.ConvTranspose2d(base, base // 2, 4, stride=2, padding=1),
            nn.GroupNorm(_groups(base // 2), base // 2),
            nn.SiLU(),
            nn.Conv2d(base // 2, 1, 3, padding=1),
        )
        nn.init.constant_(self.decode[-1].bias, -2.5)

    def forward(self, state: torch.Tensor, current_visual: torch.Tensor) -> torch.Tensor:
        batch, length, _ = state.shape
        hidden = self.input(torch.cat((state, current_visual), dim=-1))
        hidden = hidden.reshape(batch * length, self.base * 4, self.seed, self.seed)
        ink = self.decode(hidden)
        return ink.reshape(batch, length, 1, ink.shape[-2], ink.shape[-1])


class VisualSaccadeLM(nn.Module):
    """Causal language dynamics over continuous visual fixations."""

    def __init__(self, config: VisualSaccadeConfig):
        super().__init__()
        self.config = config
        self.online_retina = FovealRetina(config)
        self.target_retina = copy.deepcopy(self.online_retina)
        self.target_retina.requires_grad_(False).eval()
        self.dynamics = nn.GRU(
            config.visual_dim,
            config.state_dim,
            num_layers=config.state_layers,
            batch_first=True,
            dropout=config.dropout if config.state_layers > 1 else 0.0,
        )
        self.next_visual = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, config.state_dim),
            nn.SiLU(),
            nn.Linear(config.state_dim, config.visual_dim),
        )
        self.ink = NextFoveaInkHead(config)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.08)))

    def train(self, mode: bool = True) -> "VisualSaccadeLM":
        super().train(mode)
        self.target_retina.eval()
        return self

    def encode_sequence(self, foveas: torch.Tensor, *, target: bool = False) -> torch.Tensor:
        if foveas.ndim != 5:
            raise ValueError("foveal sequence must have shape [batch, length, 1, size, size]")
        batch, length = foveas.shape[:2]
        retina = self.target_retina if target else self.online_retina
        encoded = retina(foveas.reshape(batch * length, *foveas.shape[2:]))
        return encoded.reshape(batch, length, -1)

    def predict(
        self,
        context_foveas: torch.Tensor,
        *,
        initial_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        visual = self.encode_sequence(context_foveas)
        state, final_state = self.dynamics(visual, initial_state)
        return {
            "current_visual": visual,
            "state": state,
            "final_state": final_state,
            "predicted_visual": self.next_visual(state),
            "predicted_ink_logits": self.ink(state, visual),
        }

    def forward(
        self,
        context_foveas: torch.Tensor,
        target_foveas: torch.Tensor,
        current_reference_foveas: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        output = self.predict(context_foveas)
        with torch.no_grad():
            output["target_visual"] = self.encode_sequence(target_foveas, target=True)
            if current_reference_foveas is not None:
                output["current_reference_visual"] = self.encode_sequence(
                    current_reference_foveas,
                    target=True,
                )
        return output

    @property
    def contrastive_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("EMA momentum must be in [0, 1]")
        online = dict(self.online_retina.named_parameters())
        for name, target in self.target_retina.named_parameters():
            target.lerp_(online[name], 1.0 - momentum)
        online_buffers = dict(self.online_retina.named_buffers())
        for name, target in self.target_retina.named_buffers():
            target.copy_(online_buffers[name])


def _sample_rows(
    count: int,
    maximum: int,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if count <= maximum:
        return torch.arange(count, device=device)
    return torch.randperm(count, device=device, generator=generator)[:maximum]


def visual_saccade_loss(
    outputs: dict[str, torch.Tensor],
    target_ink: torch.Tensor,
    *,
    contrastive_scale: torch.Tensor,
    maximum_contrastive: int = 768,
    visual_weight: float = 1.0,
    contrastive_weight: float = 0.50,
    ink_weight: float = 0.45,
    invariance_weight: float = 0.20,
    retina_contrastive_weight: float = 0.30,
    variance_weight: float = 0.20,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    predicted = outputs["predicted_visual"].reshape(-1, outputs["predicted_visual"].shape[-1]).float()
    target = outputs["target_visual"].reshape(-1, outputs["target_visual"].shape[-1]).float().detach()
    normalized_prediction = F.normalize(predicted, dim=-1)
    normalized_target = F.normalize(target, dim=-1)
    visual = (1.0 - (normalized_prediction * normalized_target).sum(dim=-1)).mean()

    selected = _sample_rows(
        predicted.shape[0],
        maximum_contrastive,
        device=predicted.device,
        generator=generator,
    )
    selected_prediction = normalized_prediction[selected]
    selected_target = normalized_target[selected]
    labels = torch.arange(selected.shape[0], device=predicted.device)
    logits = contrastive_scale * selected_prediction @ selected_target.transpose(0, 1)
    contrastive = 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels)
    )

    target_ink = target_ink.float().clamp(0, 1)
    ink_logits = outputs["predicted_ink_logits"].float()
    positive_weight = ((1.0 - target_ink).sum() / target_ink.sum().clamp_min(1.0)).clamp(1.0, 12.0)
    ink = F.binary_cross_entropy_with_logits(ink_logits, target_ink, pos_weight=positive_weight)

    if "current_reference_visual" in outputs:
        current = outputs["current_visual"].reshape(-1, predicted.shape[-1]).float()
        reference = outputs["current_reference_visual"].reshape(-1, predicted.shape[-1]).float()
        normalized_current = F.normalize(current, dim=-1)
        normalized_reference = F.normalize(reference, dim=-1)
        invariance = (1.0 - (normalized_current * normalized_reference).sum(dim=-1)).mean()
        retina_selected = _sample_rows(
            current.shape[0],
            maximum_contrastive,
            device=current.device,
            generator=generator,
        )
        retina_logits = contrastive_scale * (
            normalized_current[retina_selected]
            @ normalized_reference[retina_selected].transpose(0, 1)
        )
        retina_labels = torch.arange(retina_selected.shape[0], device=current.device)
        retina_contrastive = 0.5 * (
            F.cross_entropy(retina_logits, retina_labels)
            + F.cross_entropy(retina_logits.transpose(0, 1), retina_labels)
        )
    else:
        invariance = visual.new_zeros(())
        retina_contrastive = visual.new_zeros(())
        retina_logits = logits.new_zeros((1, 1))
        retina_labels = labels.new_zeros((1,))

    centered = F.layer_norm(predicted, (predicted.shape[-1],))
    feature_std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    variance = F.relu(0.75 - feature_std).mean()
    total = (
        visual_weight * visual
        + contrastive_weight * contrastive
        + ink_weight * ink
        + invariance_weight * invariance
        + retina_contrastive_weight * retina_contrastive
        + variance_weight * variance
    )
    with torch.no_grad():
        retrieval = (logits.argmax(dim=1) == labels).float().mean()
        retina_retrieval = (retina_logits.argmax(dim=1) == retina_labels).float().mean()
        ink_binary = ink_logits.sigmoid() >= 0.5
        target_binary = target_ink >= 0.5
        true_positive = (ink_binary & target_binary).sum().float()
        ink_f1 = 2.0 * true_positive / (ink_binary.sum() + target_binary.sum()).clamp_min(1)
    return total, {
        "visual_cosine_loss": visual.detach(),
        "contrastive_loss": contrastive.detach(),
        "contrastive_top1": retrieval.detach(),
        "ink_bce": ink.detach(),
        "ink_f1": ink_f1.detach(),
        "cross_render_invariance": invariance.detach(),
        "retina_contrastive_loss": retina_contrastive.detach(),
        "retina_contrastive_top1": retina_retrieval.detach(),
        "variance_penalty": variance.detach(),
        "predicted_feature_std": feature_std.mean().detach(),
        "target_ink_fraction": target_ink.mean().detach(),
    }


def visual_saccade_config_payload(config: VisualSaccadeConfig) -> dict[str, Any]:
    return asdict(config)


def visual_saccade_config_from_payload(payload: dict[str, Any]) -> VisualSaccadeConfig:
    return VisualSaccadeConfig(**payload)
