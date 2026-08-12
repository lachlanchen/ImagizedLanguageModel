from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .saccade_lm import FovealRetina, VisualSaccadeConfig
from .visual_cell_stream import CausalVisualBlock, RMSNorm


V26_HORIZONS = (1, 2, 4, 8)


@dataclass(frozen=True)
class FactorizedVisualContextConfig:
    cell_size: int = 32
    maximum_cells: int = 64
    visual_dim: int = 192
    model_dim: int = 384
    layers: int = 8
    heads: int = 6
    mlp_ratio: float = 3.0
    dropout: float = 0.05
    retina_base_channels: int = 64
    particle_count: int = 8
    particle_noise_dim: int = 64
    horizons: tuple[int, ...] = V26_HORIZONS
    evaluation_noise_seed: int = 20260909

    def __post_init__(self) -> None:
        if self.cell_size != 32:
            raise ValueError("V26 requires 32x32 visual cells")
        if self.maximum_cells != 64:
            raise ValueError("V26 fixes the context to at most 64 visual cells")
        if self.visual_dim < 64 or self.model_dim < 128:
            raise ValueError("V26 visual dimensions are underspecified")
        if self.layers < 1 or self.heads < 1 or self.model_dim % self.heads:
            raise ValueError("V26 causal layer/head configuration is invalid")
        if self.mlp_ratio < 2.0:
            raise ValueError("V26 MLP ratio must be at least two")
        if self.retina_base_channels < 8:
            raise ValueError("V26 retina must have at least eight base channels")
        if self.particle_count < 2 or self.particle_noise_dim < 8:
            raise ValueError("V26 requires multiple continuous particles")
        if tuple(self.horizons) != V26_HORIZONS:
            raise ValueError(f"V26 fixes future horizons to {V26_HORIZONS}")


class FactorizedVisualContextModel(nn.Module):
    """Predict continuous next-glyph distributions from image-only context."""

    def __init__(
        self,
        config: FactorizedVisualContextConfig,
        *,
        freeze_retina: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        retina_config = VisualSaccadeConfig(
            fovea_size=config.cell_size,
            visual_dim=config.visual_dim,
            state_dim=config.model_dim,
            state_layers=1,
            retina_base_channels=config.retina_base_channels,
            ink_base_channels=32,
            dropout=0.0,
        )
        self.retina = FovealRetina(retina_config)
        self.target_retina = copy.deepcopy(self.retina)
        self.target_retina.requires_grad_(False).eval()
        self.freeze_retina = bool(freeze_retina)
        if self.freeze_retina:
            self.retina.requires_grad_(False).eval()

        self.history_input = nn.Linear(
            config.visual_dim, config.model_dim, bias=False
        )
        self.history_blocks = nn.ModuleList(
            [CausalVisualBlock(config) for _ in range(config.layers)]
        )
        self.history_norm = RMSNorm(config.model_dim)
        self.history_adapter = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.appearance_adapter = nn.Sequential(
            nn.Linear(config.visual_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.context_gate = nn.Linear(config.model_dim * 2, config.model_dim)
        self.fusion_norm = RMSNorm(config.model_dim)

        self.horizon_embedding = nn.Parameter(
            torch.empty(len(config.horizons), config.model_dim)
        )
        self.noise_projection = nn.Linear(
            config.particle_noise_dim, config.model_dim, bias=False
        )
        self.proposal = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.visual_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.08)))
        generator = torch.Generator().manual_seed(config.evaluation_noise_seed)
        evaluation_noise = torch.randn(
            len(config.horizons),
            config.particle_count,
            config.particle_noise_dim,
            generator=generator,
        )
        self.register_buffer(
            "evaluation_noise", evaluation_noise, persistent=True
        )
        self._initialize_language_layers()

    def _initialize_language_layers(self) -> None:
        roots: list[nn.Module] = [
            self.history_input,
            self.history_blocks,
            self.history_adapter,
            self.appearance_adapter,
            self.context_gate,
            self.noise_projection,
            self.proposal,
        ]
        for root in roots:
            for module in root.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        nn.init.normal_(self.horizon_embedding, std=0.02)
        residual_scale = 1.0 / math.sqrt(2 * max(1, self.config.layers))
        for block in self.history_blocks:
            block.attention.output.weight.data.mul_(residual_scale)
            block.down.weight.data.mul_(residual_scale)

    def train(self, mode: bool = True) -> "FactorizedVisualContextModel":
        super().train(mode)
        self.target_retina.eval()
        if self.freeze_retina:
            self.retina.eval()
        return self

    @property
    def contrastive_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def _validate_cells(self, cells: torch.Tensor) -> None:
        if not torch.is_floating_point(cells):
            raise TypeError("V26 accepts floating image tensors only")
        if cells.ndim != 5 or tuple(cells.shape[2:]) != (1, 32, 32):
            raise ValueError("V26 cells must have shape [B,T,1,32,32]")
        if not 1 <= cells.shape[1] <= self.config.maximum_cells:
            raise ValueError("V26 context length must be in [1,64]")

    def encode_cells(self, cells: torch.Tensor, *, target: bool = False) -> torch.Tensor:
        self._validate_cells(cells)
        batch, length = cells.shape[:2]
        retina = self.target_retina if target else self.retina
        encoded = retina(cells.reshape(batch * length, 1, 32, 32).clamp(0, 1))
        return F.normalize(encoded.float(), dim=-1).reshape(batch, length, -1)

    def fuse_parts(
        self,
        appearance_state: torch.Tensor,
        history_residual: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        expected = (appearance_state.shape[0], self.config.model_dim)
        if appearance_state.shape != expected or history_residual.shape != expected:
            raise ValueError("V26 factorized states have incompatible shapes")
        gate = torch.sigmoid(
            self.context_gate(torch.cat((appearance_state, history_residual), dim=-1))
        )
        fused = self.fusion_norm(appearance_state + gate * history_residual)
        return {"history_gate": gate, "fused_state": fused}

    def factorize(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_cells(context)
        visual = self.encode_cells(context)
        appearance_visual = visual[:, -1]
        appearance_state = self.appearance_adapter(
            appearance_visual.to(self.appearance_adapter[0].weight.dtype)
        )
        if context.shape[1] == 1:
            history_residual = torch.zeros_like(appearance_state)
        else:
            history = self.history_input(
                visual[:, :-1].to(self.history_input.weight.dtype)
            )
            for block in self.history_blocks:
                history = block(history)
            history_residual = self.history_adapter(
                self.history_norm(history[:, -1])
            )
        fused = self.fuse_parts(appearance_state, history_residual)
        return {
            "context_visual": visual,
            "appearance_visual": appearance_visual,
            "appearance_state": appearance_state,
            "history_residual": history_residual,
            **fused,
        }

    def _horizon_indices(self, horizons: Sequence[int] | None) -> list[int]:
        values = self.config.horizons if horizons is None else tuple(horizons)
        index = {value: position for position, value in enumerate(self.config.horizons)}
        try:
            return [index[int(value)] for value in values]
        except KeyError as exc:
            raise ValueError(f"unknown V26 horizon {exc.args[0]}") from exc

    def predict_particles_from_state(
        self,
        fused_state: torch.Tensor,
        *,
        horizons: Sequence[int] | None = None,
        particle_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if fused_state.ndim != 2 or fused_state.shape[1] != self.config.model_dim:
            raise ValueError("V26 fused state must have shape [B,model_dim]")
        indices = self._horizon_indices(horizons)
        batch = fused_state.shape[0]
        horizon_count = len(indices)
        if particle_noise is None:
            if self.training:
                particle_noise = torch.randn(
                    batch,
                    horizon_count,
                    self.config.particle_count,
                    self.config.particle_noise_dim,
                    device=fused_state.device,
                    dtype=fused_state.dtype,
                )
            else:
                particle_noise = self.evaluation_noise[indices].to(
                    device=fused_state.device, dtype=fused_state.dtype
                )[None].expand(batch, -1, -1, -1)
        expected = (
            batch,
            horizon_count,
            self.config.particle_count,
            self.config.particle_noise_dim,
        )
        if tuple(particle_noise.shape) != expected:
            raise ValueError(f"V26 particle noise must have shape {expected}")
        horizon_state = self.horizon_embedding[indices].to(fused_state.dtype)
        hidden = (
            fused_state[:, None, None]
            + horizon_state[None, :, None]
            + self.noise_projection(particle_noise)
        )
        return F.normalize(self.proposal(hidden).float(), dim=-1)

    def language(
        self,
        context: torch.Tensor,
        *,
        horizons: Sequence[int] | None = None,
        particle_noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        output = self.factorize(context)
        output["particles"] = self.predict_particles_from_state(
            output["fused_state"],
            horizons=horizons,
            particle_noise=particle_noise,
        )
        return output

    @torch.no_grad()
    def encode_target_cells(self, target_cells: torch.Tensor) -> torch.Tensor:
        return self.encode_cells(target_cells, target=True)

    @torch.no_grad()
    def update_target_retina(self, momentum: float) -> None:
        if self.freeze_retina:
            return
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("retina EMA momentum must be in [0,1]")
        online = dict(self.retina.named_parameters())
        for name, target in self.target_retina.named_parameters():
            target.lerp_(online[name], 1.0 - momentum)
        online_buffers = dict(self.retina.named_buffers())
        for name, target in self.target_retina.named_buffers():
            target.copy_(online_buffers[name])


def _unit_distance_from_similarity(similarity: torch.Tensor) -> torch.Tensor:
    return (2.0 - 2.0 * similarity.clamp(-1.0, 1.0)).clamp_min(1e-12).sqrt()


def particle_candidate_scores(
    particles: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    """Score a shared continuous candidate set for each particle distribution."""

    if particles.ndim != 3:
        raise ValueError("particles must have shape [B,K,D]")
    if candidates.ndim != 2 or candidates.shape[1] != particles.shape[2]:
        raise ValueError("candidates must have shape [N,D]")
    similarity = torch.einsum(
        "bkd,nd->bkn", particles.float(), candidates.float()
    )
    return -_unit_distance_from_similarity(similarity).mean(dim=1)


def particle_target_scores(
    particles: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Score one aligned continuous target for each particle distribution."""

    if particles.ndim != 3:
        raise ValueError("particles must have shape [B,K,D]")
    if targets.ndim != 2 or targets.shape != (
        particles.shape[0],
        particles.shape[2],
    ):
        raise ValueError("per-example targets must have shape [B,D]")
    similarity = torch.einsum(
        "bkd,bd->bk", particles.float(), targets.float()
    )
    return -_unit_distance_from_similarity(similarity).mean(dim=1)


def particle_energy_score(
    particles: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return the empirical energy score for each batch and horizon."""

    if particles.ndim != 4:
        raise ValueError("particles must have shape [B,H,K,D]")
    if target.shape != (particles.shape[0], particles.shape[1], particles.shape[3]):
        raise ValueError("energy targets must have shape [B,H,D]")
    target_similarity = torch.einsum(
        "bhkd,bhd->bhk", particles.float(), target.float()
    )
    fidelity = _unit_distance_from_similarity(target_similarity).mean(dim=-1)
    pair_similarity = torch.einsum(
        "bhkd,bhjd->bhkj", particles.float(), particles.float()
    )
    diversity = _unit_distance_from_similarity(pair_similarity).mean(dim=(-1, -2))
    return fidelity - 0.5 * diversity


def multi_positive_particle_contrastive_loss(
    particles: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
    *,
    scale: torch.Tensor,
    duplicate_similarity: float = 0.985,
    own_candidate_offset: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if particles.ndim != 3 or targets.ndim != 2 or candidates.ndim != 2:
        raise ValueError("V26 contrastive inputs have invalid ranks")
    if particles.shape[0] != targets.shape[0]:
        raise ValueError("V26 particles and positive targets must align")
    scores = scale.float() * particle_candidate_scores(particles, candidates)
    target_similarity = targets.float() @ candidates.float().transpose(0, 1)
    positives = target_similarity >= duplicate_similarity
    own = own_candidate_offset + torch.arange(
        targets.shape[0], device=targets.device
    )
    if int(own.max()) >= candidates.shape[0]:
        raise ValueError("own candidate positions are outside the candidate set")
    positives[torch.arange(targets.shape[0], device=targets.device), own] = True
    positive_scores = scores.masked_fill(~positives, -torch.inf)
    loss = -(
        torch.logsumexp(positive_scores, dim=1)
        - torch.logsumexp(scores, dim=1)
    ).mean()
    predicted = scores.argmax(dim=1)
    accuracy = positives[
        torch.arange(targets.shape[0], device=targets.device), predicted
    ].float().mean()
    return loss, {
        "contrastive_loss": loss.detach(),
        "in_batch_visual_accuracy": accuracy.detach(),
        "contrastive_scale": scale.detach(),
    }


def suffix_pair_ranking_loss(
    particles_a: torch.Tensor,
    particles_b: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
    *,
    margin: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    score_aa = particle_target_scores(particles_a, target_a)
    score_ab = particle_target_scores(particles_a, target_b)
    score_bb = particle_target_scores(particles_b, target_b)
    score_ba = particle_target_scores(particles_b, target_a)
    margin_a = score_aa - score_ab
    margin_b = score_bb - score_ba
    loss = 0.5 * (
        F.softplus(margin - margin_a).mean()
        + F.softplus(margin - margin_b).mean()
    )
    return loss, {
        "pair_loss": loss.detach(),
        "pair_margin": torch.cat((margin_a, margin_b)).mean().detach(),
        "pair_ranking_accuracy": torch.cat(
            (margin_a > 0, margin_b > 0)
        ).float().mean().detach(),
    }


def factorized_visual_context_config_payload(
    config: FactorizedVisualContextConfig,
) -> dict[str, Any]:
    payload = asdict(config)
    payload["horizons"] = list(config.horizons)
    return payload


def factorized_visual_context_config_from_payload(
    payload: dict[str, Any],
) -> FactorizedVisualContextConfig:
    values = payload.copy()
    values["horizons"] = tuple(values.get("horizons", V26_HORIZONS))
    return FactorizedVisualContextConfig(**values)


def factorized_visual_context_boundary_receipt(
    config: FactorizedVisualContextConfig,
) -> dict[str, bool | str | int | list[int]]:
    return {
        "architecture": "factorized-visual-context-v26",
        "input_shape": [config.maximum_cells, 1, 32, 32],
        "output_shape": [len(config.horizons), config.particle_count, config.visual_dim],
        "output_is_continuous_visual_distribution": True,
        "last_appearance_is_factorized": True,
        "earlier_history_is_factorized": True,
        "history_can_be_zeroed_or_swapped": True,
        "uses_continuous_particle_noise": True,
        "uses_strings": False,
        "uses_token_ids": False,
        "uses_unicode_ids": False,
        "uses_character_ids": False,
        "uses_vocabulary_embedding": False,
        "uses_ocr": False,
        "uses_visual_codebook": False,
        "uses_glyph_lookup": False,
        "uses_external_language_model": False,
        "candidate_bank_deployed": False,
    }
