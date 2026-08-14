from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence, cast

import torch
import torch.nn as nn

from .canonical_glyph_language import OrthonormalGlyphField


V45_ARCHITECTURE = "noise-limited-retinal-field-v45"
V45_PROTOCOL = "references/noise_limited_retinal_field_v45_protocol.md"


@dataclass(frozen=True)
class NoiseLimitedRetinalFieldV45Config:
    size: int = 32
    fit_bank_size: int = 8_000
    identity_bank_size: int = 1_024
    whitening_power: float = 0.10
    ridge_ratio: float = 0.50
    radius_epsilon: float = 1e-8
    binary_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.size != 32:
            raise ValueError("V45 requires 32x32 raster cells")
        if self.fit_bank_size < 16:
            raise ValueError("V45 fit bank must contain at least 16 raster forms")
        if not 2 <= self.identity_bank_size <= self.fit_bank_size:
            raise ValueError("V45 identity bank must fit inside the statistic bank")
        if not 0.0 < self.whitening_power <= 0.5:
            raise ValueError("V45 matrix power must lie in (0, 0.5]")
        if self.ridge_ratio <= 0.0:
            raise ValueError("V45 covariance ridge must be positive")
        if not 0.0 < self.radius_epsilon < 1e-4:
            raise ValueError("V45 radius epsilon is invalid")
        if not 0.0 < self.binary_threshold < 1.0:
            raise ValueError("V45 binary threshold must lie in (0,1)")

    @property
    def field_dim(self) -> int:
        return self.size**2

    @property
    def is_production(self) -> bool:
        return self == NoiseLimitedRetinalFieldV45Config()


@dataclass(frozen=True)
class RetinalFieldComponents:
    direction: torch.Tensor
    log_radius: torch.Tensor
    radius: torch.Tensor
    field: torch.Tensor


class NoiseLimitedRetinalFieldV45(nn.Module):
    """Invertible, raster-only, noise-limited symmetric whitening field."""

    def __init__(
        self,
        config: NoiseLimitedRetinalFieldV45Config,
        *,
        mean_dct: torch.Tensor,
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor,
    ) -> None:
        super().__init__()
        dimension = config.field_dim
        if mean_dct.shape != (dimension,):
            raise ValueError("V45 mean DCT has the wrong shape")
        if eigenvectors.shape != (dimension, dimension):
            raise ValueError("V45 eigenvector matrix has the wrong shape")
        if eigenvalues.shape != (dimension,):
            raise ValueError("V45 eigenvalues have the wrong shape")
        for name, value in (
            ("mean_dct", mean_dct),
            ("eigenvectors", eigenvectors),
            ("eigenvalues", eigenvalues),
        ):
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"V45 {name} must be a finite floating tensor")

        mean = mean_dct.detach().to(device="cpu", dtype=torch.float64).contiguous()
        vectors = (
            eigenvectors.detach().to(device="cpu", dtype=torch.float64).contiguous()
        )
        values = eigenvalues.detach().to(device="cpu", dtype=torch.float64).contiguous()
        mean_variance = values.clamp_min(0.0).sum() / dimension
        if not bool(mean_variance > 0.0):
            raise ValueError("V45 covariance must have positive total variance")
        regularized = values.clamp_min(0.0) + config.ridge_ratio * mean_variance
        forward_scale = regularized.pow(-config.whitening_power)
        inverse_scale = regularized.pow(config.whitening_power)
        forward = (vectors * forward_scale) @ vectors.transpose(0, 1)
        inverse = (vectors * inverse_scale) @ vectors.transpose(0, 1)

        self.config = config
        self.dct = OrthonormalGlyphField(
            size=config.size,
            binary_threshold=config.binary_threshold,
        )
        self.register_buffer("mean_dct", mean, persistent=True)
        self.register_buffer("eigenvectors", vectors, persistent=True)
        self.register_buffer("eigenvalues", values, persistent=True)
        self.register_buffer("mean_variance", mean_variance, persistent=True)
        self.register_buffer("forward_matrix", forward.float(), persistent=False)
        self.register_buffer("inverse_matrix", inverse.float(), persistent=False)
        self.requires_grad_(False)

    def _validate_dct(self, fields: torch.Tensor) -> None:
        if not fields.is_floating_point():
            raise TypeError("V45 DCT fields must be floating point")
        if fields.ndim < 2 or fields.shape[-1] != self.config.field_dim:
            raise ValueError("V45 DCT fields must end in 1024 coefficients")
        if not bool(torch.isfinite(fields).all()):
            raise ValueError("V45 DCT fields must be finite")

    def _matrix_components(
        self,
        fields: torch.Tensor,
        *,
        exact: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if exact:
            work = fields.to(dtype=torch.float64)
            mean = self.mean_dct.to(device=fields.device, dtype=torch.float64)
            vectors = self.eigenvectors.to(device=fields.device, dtype=torch.float64)
            values = self.eigenvalues.to(device=fields.device, dtype=torch.float64)
            mean_variance = values.clamp_min(0.0).sum() / self.config.field_dim
            regularized = (
                values.clamp_min(0.0) + self.config.ridge_ratio * mean_variance
            )
            scale = regularized.pow(-self.config.whitening_power)
            transformed = ((work - mean) @ vectors * scale) @ vectors.transpose(0, 1)
            return transformed, mean, vectors
        work = fields.float()
        mean = self.mean_dct.to(device=fields.device, dtype=torch.float32)
        matrix = self.forward_matrix.to(device=fields.device, dtype=torch.float32)
        return (work - mean) @ matrix, mean, matrix

    def encode_dct(
        self,
        fields: torch.Tensor,
        *,
        exact: bool = False,
    ) -> RetinalFieldComponents:
        self._validate_dct(fields)
        transformed, _, _ = self._matrix_components(fields, exact=exact)
        radius = transformed.norm(dim=-1)
        clamped = radius.clamp_min(self.config.radius_epsilon)
        direction = transformed / clamped[..., None]
        return RetinalFieldComponents(
            direction=direction,
            log_radius=clamped.log(),
            radius=radius,
            field=transformed,
        )

    def encode_pixels(
        self,
        pixels: torch.Tensor,
        *,
        exact: bool = False,
    ) -> RetinalFieldComponents:
        return self.encode_dct(self.dct.encode(pixels), exact=exact)

    def decode_dct(
        self,
        direction: torch.Tensor,
        log_radius: torch.Tensor,
        *,
        exact: bool = False,
    ) -> torch.Tensor:
        self._validate_dct(direction)
        if log_radius.shape != direction.shape[:-1]:
            raise ValueError("V45 log radius must match the field leading shape")
        if not log_radius.is_floating_point() or not bool(
            torch.isfinite(log_radius).all()
        ):
            raise ValueError("V45 log radius must be finite and floating point")
        if exact:
            work = direction.to(torch.float64) * log_radius.to(torch.float64).exp()[
                ..., None
            ]
            vectors = self.eigenvectors.to(
                device=direction.device, dtype=torch.float64
            )
            values = self.eigenvalues.to(
                device=direction.device, dtype=torch.float64
            )
            mean = self.mean_dct.to(device=direction.device, dtype=torch.float64)
            mean_variance = values.clamp_min(0.0).sum() / self.config.field_dim
            regularized = (
                values.clamp_min(0.0) + self.config.ridge_ratio * mean_variance
            )
            inverse_scale = regularized.pow(self.config.whitening_power)
            return (work @ vectors * inverse_scale) @ vectors.transpose(0, 1) + mean
        work = direction.float() * log_radius.float().exp()[..., None]
        inverse = self.inverse_matrix.to(
            device=direction.device, dtype=torch.float32
        )
        mean = self.mean_dct.to(device=direction.device, dtype=torch.float32)
        return work @ inverse + mean

    def binary(
        self,
        direction: torch.Tensor,
        log_radius: torch.Tensor,
        *,
        exact: bool = False,
    ) -> torch.Tensor:
        dct = self.decode_dct(direction, log_radius, exact=exact)
        return (self.dct.signed_spatial(dct.float()) >= 0.0).to(direction.dtype)


@torch.no_grad()
def fit_noise_limited_retinal_field_v45(
    pixels: torch.Tensor,
    counts: Sequence[int] | torch.Tensor,
    *,
    config: NoiseLimitedRetinalFieldV45Config,
) -> NoiseLimitedRetinalFieldV45:
    if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (1, 32, 32):
        raise ValueError("V45 fit pixels must be [N,1,32,32]")
    if len(pixels) != config.fit_bank_size:
        raise ValueError("V45 fit pixels do not match the configured bank size")
    count_tensor = torch.as_tensor(counts, dtype=torch.float64, device="cpu")
    if count_tensor.shape != (len(pixels),):
        raise ValueError("V45 fit counts must align with raster forms")
    if not bool(torch.isfinite(count_tensor).all()) or not bool((count_tensor > 0).all()):
        raise ValueError("V45 fit counts must be finite and positive")

    dct = OrthonormalGlyphField(
        size=config.size,
        binary_threshold=config.binary_threshold,
    ).encode(pixels.detach().to(device="cpu", dtype=torch.float32))
    fields = dct.to(torch.float64)
    weights = count_tensor / count_tensor.sum()
    mean = (weights[:, None] * fields).sum(dim=0)
    centered = fields - mean
    covariance = (centered.transpose(0, 1) * weights[None]) @ centered
    covariance = 0.5 * (covariance + covariance.transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    return NoiseLimitedRetinalFieldV45(
        config,
        mean_dct=mean,
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
    )


def noise_limited_retinal_field_v45_config_from_payload(
    payload: Mapping[str, Any],
) -> NoiseLimitedRetinalFieldV45Config:
    return NoiseLimitedRetinalFieldV45Config(**dict(payload))


def noise_limited_retinal_field_v45_from_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    verify_sha256: bool = True,
) -> NoiseLimitedRetinalFieldV45:
    architecture = payload.get("architecture", payload.get("experiment"))
    if architecture != V45_ARCHITECTURE:
        raise ValueError("checkpoint is not a V45 noise-limited retinal field")
    config_payload = payload.get("config")
    state_payload = payload.get("field")
    if not isinstance(config_payload, Mapping):
        raise ValueError("V45 checkpoint is missing its configuration")
    if not isinstance(state_payload, Mapping) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state_payload.items()
    ):
        raise ValueError("V45 checkpoint is missing a tensor field state")
    state = cast(Mapping[str, torch.Tensor], state_payload)
    required = ("mean_dct", "eigenvectors", "eigenvalues")
    if any(name not in state for name in required):
        raise ValueError("V45 checkpoint field state is incomplete")
    config = noise_limited_retinal_field_v45_config_from_payload(config_payload)
    field = NoiseLimitedRetinalFieldV45(
        config,
        mean_dct=state["mean_dct"],
        eigenvectors=state["eigenvectors"],
        eigenvalues=state["eigenvalues"],
    )
    field.load_state_dict(state, strict=True)
    if verify_sha256:
        expected = payload.get("field_state_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("V45 checkpoint is missing its field-state digest")
        measured = noise_limited_retinal_field_v45_state_sha256(field)
        if measured != expected:
            raise ValueError("V45 checkpoint field-state digest differs")
    return field


def noise_limited_retinal_field_v45_boundary_receipt(
    field: NoiseLimitedRetinalFieldV45,
) -> dict[str, Any]:
    parameter_count = sum(parameter.numel() for parameter in field.parameters())
    return {
        "architecture": V45_ARCHITECTURE,
        "config": asdict(field.config),
        "input_is_continuous_raster": True,
        "output_is_continuous_direction_and_radius": True,
        "fixed_dct": True,
        "fixed_invertible_matrix_power": True,
        "exact_radial_side_channel": True,
        "statistics_are_buffers": True,
        "trainable_parameters": parameter_count,
        "uses_strings": False,
        "uses_token_ids": False,
        "uses_unicode_ids": False,
        "uses_character_ids": False,
        "uses_vocabulary_embedding": False,
        "uses_vocabulary_output": False,
        "uses_ocr": False,
        "uses_visual_codebook": False,
        "uses_quantization": False,
        "uses_glyph_lookup": False,
        "uses_external_language_model": False,
        "candidate_bank_deployed": False,
    }


def noise_limited_retinal_field_v45_boundary_is_clean(
    field: NoiseLimitedRetinalFieldV45,
) -> bool:
    receipt = noise_limited_retinal_field_v45_boundary_receipt(field)
    required_true = (
        "input_is_continuous_raster",
        "output_is_continuous_direction_and_radius",
        "fixed_dct",
        "fixed_invertible_matrix_power",
        "exact_radial_side_channel",
        "statistics_are_buffers",
    )
    required_false = (
        "uses_strings",
        "uses_token_ids",
        "uses_unicode_ids",
        "uses_character_ids",
        "uses_vocabulary_embedding",
        "uses_vocabulary_output",
        "uses_ocr",
        "uses_visual_codebook",
        "uses_quantization",
        "uses_glyph_lookup",
        "uses_external_language_model",
        "candidate_bank_deployed",
    )
    return (
        receipt["architecture"] == V45_ARCHITECTURE
        and receipt["trainable_parameters"] == 0
        and all(receipt[key] is True for key in required_true)
        and all(receipt[key] is False for key in required_false)
    )


def noise_limited_retinal_field_v45_state_sha256(
    field: NoiseLimitedRetinalFieldV45,
) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(field.state_dict().items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
