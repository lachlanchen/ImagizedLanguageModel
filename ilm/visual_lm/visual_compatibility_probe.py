from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


V26_PROBE_STATES = (
    "appearance_state",
    "history_residual",
    "fused_state",
)


@dataclass(frozen=True)
class VisualCompatibilityProbeConfig:
    context_dim: int = 384
    candidate_dim: int = 192
    hidden_dim: int = 384
    projection_dim: int = 192
    temperature: float = 0.07

    def __post_init__(self) -> None:
        if min(
            self.context_dim,
            self.candidate_dim,
            self.hidden_dim,
            self.projection_dim,
        ) < 8:
            raise ValueError("visual compatibility dimensions are underspecified")
        if not 0.01 <= self.temperature <= 1.0:
            raise ValueError("visual compatibility temperature is invalid")


class VisualCandidateCompatibilityProbe(nn.Module):
    """Score arbitrary candidate images against an image-derived context state."""

    def __init__(self, config: VisualCompatibilityProbeConfig) -> None:
        super().__init__()
        self.config = config
        self.context_projection = nn.Sequential(
            nn.LayerNorm(config.context_dim),
            nn.Linear(config.context_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.projection_dim, bias=False),
        )
        self.candidate_projection = nn.Sequential(
            nn.LayerNorm(config.candidate_dim),
            nn.Linear(config.candidate_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.projection_dim, bias=False),
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / config.temperature))
        )
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def forward(
        self,
        context_states: torch.Tensor,
        candidate_states: torch.Tensor,
    ) -> torch.Tensor:
        if context_states.ndim != 3 or context_states.shape[1] != 2:
            raise ValueError("context states must have shape [B,2,C]")
        if candidate_states.ndim != 3 or candidate_states.shape[1] != 2:
            raise ValueError("candidate states must have shape [B,2,V]")
        if context_states.shape[0] != candidate_states.shape[0]:
            raise ValueError("context and candidate pair counts must match")
        if context_states.shape[2] != self.config.context_dim:
            raise ValueError("context state dimension does not match the probe")
        if candidate_states.shape[2] != self.config.candidate_dim:
            raise ValueError("candidate state dimension does not match the probe")
        queries = F.normalize(self.context_projection(context_states), dim=-1)
        keys = F.normalize(self.candidate_projection(candidate_states), dim=-1)
        return self.scale.float() * torch.einsum(
            "bqd,bkd->bqk", queries.float(), keys.float()
        )


def paired_compatibility_loss(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train the exact two-context/two-image assignment in both directions."""

    if logits.ndim != 3 or tuple(logits.shape[1:]) != (2, 2):
        raise ValueError("paired compatibility logits must have shape [B,2,2]")
    labels = torch.arange(2, device=logits.device).repeat(logits.shape[0])
    row_loss = F.cross_entropy(logits.reshape(-1, 2), labels)
    column_loss = F.cross_entropy(
        logits.transpose(1, 2).reshape(-1, 2), labels
    )
    loss = 0.5 * (row_loss + column_loss)
    margin_a = logits[:, 0, 0] - logits[:, 0, 1]
    margin_b = logits[:, 1, 1] - logits[:, 1, 0]
    margins = torch.stack((margin_a, margin_b), dim=1)
    ties = margins == 0
    accuracy_credit = (margins > 0).float() + 0.5 * ties.float()
    return loss, {
        "loss": loss.detach(),
        "arm_accuracy": accuracy_credit.mean().detach(),
        "strict_arm_accuracy": (margins > 0).float().mean().detach(),
        "arm_tie_rate": ties.float().mean().detach(),
        "both_correct_rate": (margins > 0).all(dim=1).float().mean().detach(),
        "mean_margin": margins.mean().detach(),
    }


def visual_compatibility_probe_config_payload(
    config: VisualCompatibilityProbeConfig,
) -> dict[str, Any]:
    return asdict(config)


def visual_compatibility_probe_boundary_receipt() -> dict[str, bool | str]:
    return {
        "architecture": "v26-frozen-visual-compatibility-probe",
        "input_context_is_image_derived": True,
        "input_candidates_are_images": True,
        "output_is_candidate_compatibility": True,
        "v26_backbone_is_frozen": True,
        "uses_strings": False,
        "uses_token_ids": False,
        "uses_unicode_ids": False,
        "uses_character_ids": False,
        "uses_ocr": False,
        "uses_external_language_model": False,
        "diagnostic_candidate_images_required": True,
        "deployed_language_model": False,
    }
