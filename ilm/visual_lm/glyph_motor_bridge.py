from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from .continuous_glyph_codec import (
    V34_ARCHITECTURE,
    ContinuousGlyphCodec,
    ContinuousGlyphCodecConfig,
)
from .continuous_glyph_codec_data import file_sha256


MXFONT_REPOSITORY = "https://github.com/clovaai/mxfont"
MXFONT_REVISION = "93f3c88517f7c904f16da6333adb2588dcdf3cce"
MXFONT_CHECKPOINT_SHA256 = (
    "dcbcb6438d9b1e3230551bc78fcf64ec5454a01734502bdeac410d2f5c404119"
)
MXFONT_SOURCE_SHA256 = {
    "models/__init__.py": "7e3f79444c534626ab05b41a4735e71d762d05e6e89511593e89462c78043990",
    "models/aux_classifier.py": "78f159ea3213c502fda75032d16e29cd6f475576f7599b872e7cf204b5c181dc",
    "models/decoder.py": "c3782fb270a22c89f0ee6571b7651b60c11ec48cf7e38405566c1e14f2c41b83",
    "models/discriminator.py": "08e9d3142422b46ae9fd553428f4bc81619d4825ae0d9b296b75dce52a4b3155",
    "models/experts.py": "8c361d653a533a991549dac2c7337b4fdf076a396091f181428df01cdb3fefa9",
    "models/generator.py": "aa0a358297b5c9913033dfff9616caa11bb79e049812f8e4bab97d15efc1b05b",
    "models/modules/__init__.py": "a15543efbb92e724d92e64ea326b75d7c53fe429899f3c19aab201aaf6158915",
    "models/modules/blocks.py": "c230abcf44b66e141c5172135236d35425827faec6a2997d33d80c0777331f32",
    "models/modules/cbam.py": "893f33cb8ab9e9eb92846701e040efb14c25302a360028e60b0eb1279d735bf8",
    "models/modules/frn.py": "9d3a31f68988a1d7c412cc8e008c2043792fd1e9433f9d1409ce90b29dbf003b",
    "models/modules/globalcontext.py": "5ec5fe071c2f1e07e5b57471f0c6b5edd77e229e43425cb362383798920a89c2",
    "models/modules/modules.py": "50c23ced26b6123f247b9fe41dea30079f14ea92138af975a74e13df428a1f16",
    "models/style_encoder.py": "0e9b4afa91f8f7b0071905fffe162607e32495abf38ab99a0396702314634cd7",
    "utils/__init__.py": "02f1abf2ff4153cfd27bdc3e1a49e28b24ece2a5df62b09cd5f416a5a3aeddf4",
    "utils/logger.py": "900b52cd171040b218621319d6cbc5f920af3ee84465b714d07ed38970a24d21",
    "utils/utils.py": "1cb3031b58772410f434813d743a9c13ad6bb923dc8f27442954c616f70c6374",
    "utils/visualize.py": "5fd634da277000e613804004e7aa23dd07ad3f0e3c968cd63ace424e9b8ad75b",
    "utils/writer.py": "b57f2adb2c1032e2c4bacca736edcfbb134853124593e9a935ce08242f56aafd",
}
V34_CODEC_CHECKPOINT_SHA256 = (
    "a138c9cb3b0502e43d1227f689c020893d56b468742c32e1840e44d299662f33"
)


def render_centered_glyph(
    font_path: str | Path,
    glyph: str,
    *,
    canvas_size: int = 128,
    font_size: int = 150,
    padding: int = 20,
) -> torch.Tensor:
    """Render one evaluation glyph as a unit-range grayscale tensor."""
    if len(glyph) != 1:
        raise ValueError("glyph-motor audit rendering requires one code point")
    if min(canvas_size, font_size) < 8 or padding < 0:
        raise ValueError("glyph-motor audit rendering geometry is invalid")
    font = ImageFont.truetype(str(Path(font_path)), size=font_size)
    left, top, right, bottom = font.getbbox(glyph)
    width, height = right - left, bottom - top
    if min(width, height) <= 0:
        raise ValueError(f"font produced an empty glyph for {glyph!r}")
    side = max(width, height)
    image = Image.new("L", (side + 2 * padding, side + 2 * padding), 255)
    origin = (
        padding + (side - width) // 2 - left,
        padding + (side - height) // 2 - top,
    )
    ImageDraw.Draw(image).text(origin, glyph, font=font, fill=0)
    image = image.resize(
        (canvas_size, canvas_size),
        Image.Resampling.BILINEAR,
    )
    pixels = torch.from_numpy(np.array(image, dtype=np.float32, copy=True))
    return pixels.unsqueeze(0).div_(255.0)


def load_unit_grayscale(path: str | Path, *, size: int = 128) -> torch.Tensor:
    if size < 8:
        raise ValueError("glyph-motor image size is too small")
    with Image.open(path) as image:
        grayscale = image.convert("L").resize(
            (size, size),
            Image.Resampling.BILINEAR,
        )
        pixels = torch.from_numpy(
            np.array(grayscale, dtype=np.float32, copy=True)
        )
    return pixels.unsqueeze(0).div_(255.0)


def unit_grayscale_to_pil(pixels: torch.Tensor) -> Image.Image:
    if pixels.ndim == 3 and pixels.shape[0] == 1:
        pixels = pixels[0]
    if pixels.ndim != 2 or not bool(torch.isfinite(pixels).all()):
        raise ValueError("glyph-motor image must be one finite grayscale plane")
    values = pixels.detach().float().cpu().clamp(0.0, 1.0).mul(255).round()
    return Image.fromarray(values.to(torch.uint8).numpy(), mode="L")


def binary_ink_f1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("glyph-motor F1 tensors must be aligned [B,1,H,W]")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("glyph-motor F1 threshold must be in [0,1]")
    predicted_ink = prediction.float() < threshold
    target_ink = target.float() < threshold
    overlap = 2.0 * (predicted_ink & target_ink).flatten(1).sum(dim=1)
    scale = predicted_ink.flatten(1).sum(dim=1) + target_ink.flatten(1).sum(
        dim=1
    )
    return (overlap + 1e-6) / (scale + 1e-6)


def load_qualified_v34_codec(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    verify_hash: bool = True,
) -> tuple[ContinuousGlyphCodec, dict[str, Any]]:
    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint_hash = file_sha256(path)
    if verify_hash and checkpoint_hash != V34_CODEC_CHECKPOINT_SHA256:
        raise ValueError(f"unexpected V34 checkpoint SHA-256: {checkpoint_hash}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("architecture") != V34_ARCHITECTURE:
        raise ValueError("glyph-motor bridge requires a V34 codec checkpoint")
    ema = payload.get("ema")
    if not isinstance(ema, Mapping) or not isinstance(ema.get("shadow"), Mapping):
        raise ValueError("V34 checkpoint lacks an EMA shadow")
    codec = ContinuousGlyphCodec(ContinuousGlyphCodecConfig())
    codec.load_state_dict(ema["shadow"], strict=True)
    codec.requires_grad_(False).eval().to(device)
    return codec, {
        "path": str(path),
        "sha256": checkpoint_hash,
        "architecture": payload.get("architecture"),
        "update": int(payload.get("update", 0)),
        "parameters": sum(parameter.numel() for parameter in codec.parameters()),
        "selection": "ema-shadow",
    }


def load_mxfont_generator(
    root: str | Path,
    *,
    device: torch.device,
    verify_hash: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    root = Path(root).expanduser().resolve()
    checkpoint_path = root / "generator.pth"
    checkpoint_hash = file_sha256(checkpoint_path)
    if verify_hash and checkpoint_hash != MXFONT_CHECKPOINT_SHA256:
        raise ValueError(f"unexpected MX-Font checkpoint SHA-256: {checkpoint_hash}")
    source_hashes = {
        relative: file_sha256(root / relative)
        for relative in MXFONT_SOURCE_SHA256
    }
    if verify_hash:
        mismatched = {
            relative: actual
            for relative, actual in source_hashes.items()
            if actual != MXFONT_SOURCE_SHA256[relative]
        }
        if mismatched:
            raise ValueError(f"unexpected MX-Font source SHA-256 values: {mismatched}")
    existing = sys.modules.get("models")
    if existing is not None:
        source = Path(getattr(existing, "__file__", "")).resolve()
        if root not in source.parents:
            raise RuntimeError("another top-level models package is already imported")
    root_text = str(root)
    sys.path.insert(0, root_text)
    try:
        module = importlib.import_module("models")
    finally:
        sys.path.remove(root_text)
    generator_class = getattr(module, "Generator")
    generator = generator_class(
        1,
        32,
        1,
        style_enc={
            "norm": "in",
            "activ": "relu",
            "pad_type": "zero",
            "skip_scale_var": False,
        },
        experts={
            "n_experts": 6,
            "norm": "in",
            "activ": "relu",
            "pad_type": "zero",
            "skip_scale_var": False,
        },
        emb_num=2,
        dec={
            "norm": "in",
            "activ": "relu",
            "pad_type": "zero",
            "out": "tanh",
        },
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping) and "generator_ema" in payload:
        payload = payload["generator_ema"]
    if not isinstance(payload, Mapping):
        raise TypeError("MX-Font checkpoint must contain a state mapping")
    generator.load_state_dict(payload, strict=True)
    generator.requires_grad_(False).eval().to(device)
    return generator, {
        "repository": MXFONT_REPOSITORY,
        "revision": MXFONT_REVISION,
        "root": str(root),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "source_sha256": source_hashes,
        "parameters": sum(parameter.numel() for parameter in generator.parameters()),
        "license": "MIT source; checkpoint provenance follows the repository",
    }


@torch.no_grad()
def mxfont_render(
    generator: torch.nn.Module,
    source_pixels: torch.Tensor,
    style_reference_pixels: torch.Tensor,
) -> torch.Tensor:
    if source_pixels.ndim != 4 or source_pixels.shape[1:] != (1, 128, 128):
        raise ValueError("MX-Font source pixels must be [B,1,128,128]")
    if (
        style_reference_pixels.ndim != 4
        or style_reference_pixels.shape[1:] != (1, 128, 128)
        or len(style_reference_pixels) < 1
    ):
        raise ValueError("MX-Font style references must be [R,1,128,128]")
    if not all(
        bool(torch.isfinite(value).all())
        and bool(((value >= 0.0) & (value <= 1.0)).all())
        for value in (source_pixels, style_reference_pixels)
    ):
        raise ValueError("MX-Font bridge pixels must be finite and in [0,1]")
    references = style_reference_pixels.unsqueeze(0).expand(
        len(source_pixels),
        -1,
        -1,
        -1,
        -1,
    )
    signed_sources = source_pixels.mul(2.0).sub(1.0).unsqueeze(1)
    signed_references = references.mul(2.0).sub(1.0)
    signed_output = generator.gen_from_style_char(
        signed_references,
        signed_sources,
    )
    output = signed_output.float().add(1.0).div(2.0).clamp(0.0, 1.0)
    if output.shape != source_pixels.shape or not bool(torch.isfinite(output).all()):
        raise ValueError("MX-Font bridge emitted an invalid raster batch")
    return output


@torch.no_grad()
def v34_project_source(
    codec: ContinuousGlyphCodec,
    source_pixels: torch.Tensor,
    *,
    latent_noise_sigma: float = 0.0,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source_pixels.ndim != 4 or source_pixels.shape[1:] != (1, 128, 128):
        raise ValueError("V34 source projection requires [B,1,128,128]")
    if latent_noise_sigma < 0.0 or seed < 0:
        raise ValueError("V34 source projection noise settings are invalid")
    coarse = F.interpolate(source_pixels, size=(32, 32), mode="area")
    latent = codec.encode(coarse)
    if latent_noise_sigma:
        generator = torch.Generator(device=latent.device).manual_seed(seed)
        latent = latent + torch.randn(
            latent.shape,
            generator=generator,
            device=latent.device,
            dtype=latent.dtype,
        ).mul(latent_noise_sigma)
    projected = codec.decode(latent).sigmoid()
    projected = F.interpolate(
        projected,
        size=(128, 128),
        mode="bilinear",
        align_corners=False,
    )
    return projected, latent


def noise_condition_name(sigma: float) -> str:
    if sigma < 0.0 or not math.isfinite(sigma):
        raise ValueError("glyph-motor noise sigma must be finite and non-negative")
    return f"v34_sigma_{sigma:.3f}".replace(".", "p")


__all__ = [
    "MXFONT_CHECKPOINT_SHA256",
    "MXFONT_REPOSITORY",
    "MXFONT_REVISION",
    "MXFONT_SOURCE_SHA256",
    "V34_CODEC_CHECKPOINT_SHA256",
    "binary_ink_f1",
    "load_mxfont_generator",
    "load_qualified_v34_codec",
    "load_unit_grayscale",
    "mxfont_render",
    "noise_condition_name",
    "render_centered_glyph",
    "unit_grayscale_to_pil",
    "v34_project_source",
]
