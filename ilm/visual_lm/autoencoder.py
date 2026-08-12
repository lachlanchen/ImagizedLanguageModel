from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class SpatialAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4, window_size: int = 8):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("SpatialAttention channels must be divisible by heads")
        self.heads = heads
        self.head_dim = channels // heads
        self.window_size = window_size
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        window = self.window_size
        pad_height = (-height) % window
        pad_width = (-width) % window
        normalized = self.norm(x)
        if pad_height or pad_width:
            normalized = F.pad(normalized, (0, pad_width, 0, pad_height), mode="replicate")
        padded_height, padded_width = normalized.shape[-2:]
        rows = padded_height // window
        columns = padded_width // window
        q, k, v = self.qkv(normalized).chunk(3, dim=1)

        def to_windows(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor.reshape(
                batch,
                self.heads,
                self.head_dim,
                rows,
                window,
                columns,
                window,
            )
            tensor = tensor.permute(0, 3, 5, 1, 4, 6, 2)
            return tensor.reshape(batch * rows * columns, self.heads, window * window, self.head_dim)

        attended = F.scaled_dot_product_attention(to_windows(q), to_windows(k), to_windows(v))
        attended = attended.reshape(
            batch,
            rows,
            columns,
            self.heads,
            window,
            window,
            self.head_dim,
        )
        attended = attended.permute(0, 3, 6, 1, 4, 2, 5).reshape(
            batch, channels, padded_height, padded_width
        )
        attended = attended[:, :, :height, :width]
        return x + self.proj(attended)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


@dataclass(frozen=True)
class VisualVAEConfig:
    image_channels: int = 3
    latent_channels: int = 8
    base_channels: int = 32
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 4)
    blocks_per_level: int = 2
    attention_heads: int = 4

    @property
    def downsample_factor(self) -> int:
        return 2 ** (len(self.channel_multipliers) - 1)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "VisualVAEConfig":
        values = dict(values)
        if "channel_multipliers" in values:
            values["channel_multipliers"] = tuple(values["channel_multipliers"])
        return cls(**values)


class DiagonalGaussian:
    def __init__(self, moments: torch.Tensor):
        self.mean, self.logvar = moments.chunk(2, dim=1)
        self.logvar = self.logvar.clamp(-20.0, 10.0)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.randn(
            self.mean.shape,
            dtype=self.mean.dtype,
            device=self.mean.device,
            generator=generator,
        )
        return self.mean + torch.exp(0.5 * self.logvar) * noise

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        value = 0.5 * (self.mean.square() + self.logvar.exp() - 1.0 - self.logvar)
        return value.flatten(1).mean(1)


class VisualPageVAE(nn.Module):
    """KL-regularized page autoencoder used by the image-native language model.

    The model never receives character IDs or text tokens. Its only model-facing
    representation is an RGB writing canvas and a continuous spatial latent.
    """

    def __init__(self, config: VisualVAEConfig | None = None):
        super().__init__()
        self.config = config or VisualVAEConfig()
        cfg = self.config
        widths = [cfg.base_channels * mult for mult in cfg.channel_multipliers]

        self.encoder_in = nn.Conv2d(cfg.image_channels, widths[0], 3, padding=1)
        self.encoder_levels = nn.ModuleList()
        channels = widths[0]
        for index, width in enumerate(widths):
            blocks = nn.ModuleList()
            for _ in range(cfg.blocks_per_level):
                blocks.append(ResidualBlock(channels, width))
                channels = width
            self.encoder_levels.append(
                nn.ModuleDict(
                    {
                        "blocks": blocks,
                        "downsample": Downsample(channels) if index < len(widths) - 1 else nn.Identity(),
                    }
                )
            )
        self.encoder_mid = nn.Sequential(
            ResidualBlock(channels, channels),
            SpatialAttention(channels, cfg.attention_heads),
            ResidualBlock(channels, channels),
        )
        self.encoder_out = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, cfg.latent_channels * 2, 3, padding=1),
        )

        self.decoder_in = nn.Conv2d(cfg.latent_channels, widths[-1], 3, padding=1)
        channels = widths[-1]
        self.decoder_mid = nn.Sequential(
            ResidualBlock(channels, channels),
            SpatialAttention(channels, cfg.attention_heads),
            ResidualBlock(channels, channels),
        )
        self.decoder_levels = nn.ModuleList()
        for reverse_index, width in enumerate(reversed(widths)):
            blocks = nn.ModuleList()
            for _ in range(cfg.blocks_per_level + 1):
                blocks.append(ResidualBlock(channels, width))
                channels = width
            is_last = reverse_index == len(widths) - 1
            self.decoder_levels.append(
                nn.ModuleDict(
                    {
                        "blocks": blocks,
                        "upsample": nn.Identity() if is_last else Upsample(channels),
                    }
                )
            )
        self.decoder_out = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, cfg.image_channels, 3, padding=1),
            nn.Tanh(),
        )

        self.register_buffer(
            "latent_mean",
            torch.zeros(1, cfg.latent_channels, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "latent_std",
            torch.ones(1, cfg.latent_channels, 1, 1),
            persistent=True,
        )

    def encode_distribution(self, image: torch.Tensor) -> DiagonalGaussian:
        hidden = self.encoder_in(image)
        for level in self.encoder_levels:
            for block in level["blocks"]:
                hidden = block(hidden)
            hidden = level["downsample"](hidden)
        hidden = self.encoder_mid(hidden)
        return DiagonalGaussian(self.encoder_out(hidden))

    def normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self.latent_mean) / self.latent_std.clamp_min(1e-6)

    def denormalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return latent * self.latent_std + self.latent_mean

    def encode(
        self,
        image: torch.Tensor,
        *,
        sample: bool = False,
        normalize: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        posterior = self.encode_distribution(image)
        latent = posterior.sample(generator=generator) if sample else posterior.mode()
        return self.normalize_latent(latent) if normalize else latent

    def decode(self, latent: torch.Tensor, *, normalized: bool = True) -> torch.Tensor:
        if normalized:
            latent = self.denormalize_latent(latent)
        hidden = self.decoder_in(latent)
        hidden = self.decoder_mid(hidden)
        for level in self.decoder_levels:
            for block in level["blocks"]:
                hidden = block(hidden)
            hidden = level["upsample"](hidden)
        return self.decoder_out(hidden)

    def forward(
        self,
        image: torch.Tensor,
        *,
        sample: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, DiagonalGaussian]:
        posterior = self.encode_distribution(image)
        latent = posterior.sample(generator=generator) if sample else posterior.mode()
        return self.decode(latent, normalized=False), posterior

    @torch.no_grad()
    def set_latent_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        mean = mean.reshape(1, self.config.latent_channels, 1, 1)
        std = std.reshape(1, self.config.latent_channels, 1, 1).clamp_min(1e-5)
        self.latent_mean.copy_(mean.to(self.latent_mean))
        self.latent_std.copy_(std.to(self.latent_std))


def sobel_edges(image: torch.Tensor) -> torch.Tensor:
    gray = image.mean(dim=1, keepdim=True)
    kernel_x = image.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).unsqueeze(0)
    kernel_y = kernel_x.transpose(-1, -2)
    edge_x = F.conv2d(gray, kernel_x, padding=1)
    edge_y = F.conv2d(gray, kernel_y, padding=1)
    return torch.sqrt(edge_x.square() + edge_y.square() + 1e-6)


def visual_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    posterior: DiagonalGaussian,
    *,
    edge_weight: float = 0.20,
    multiscale_weight: float = 0.15,
    kl_weight: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pixel = F.l1_loss(prediction, target)
    edge = F.l1_loss(sobel_edges(prediction), sobel_edges(target))
    coarse_prediction = F.avg_pool2d(prediction, kernel_size=4, stride=4)
    coarse_target = F.avg_pool2d(target, kernel_size=4, stride=4)
    multiscale = F.l1_loss(coarse_prediction, coarse_target)
    kl = posterior.kl().mean()
    total = pixel + edge_weight * edge + multiscale_weight * multiscale + kl_weight * kl
    return total, {
        "pixel": pixel,
        "edge": edge,
        "multiscale": multiscale,
        "kl": kl,
    }
