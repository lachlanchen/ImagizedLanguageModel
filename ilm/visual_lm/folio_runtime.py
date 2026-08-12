from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .folio import FolioRetina, folio_config_from_payload
from .folio_address import FolioAddressRetina, folio_address_config_from_payload


FOLIO_ARCHITECTURES = {
    "visual-folio-retina-v1",
    "visual-folio-interference-retina-v2",
}


def folio_encoder_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    architecture = checkpoint.get("architecture")
    if architecture == "visual-folio-retina-v1":
        model: nn.Module = FolioRetina(folio_config_from_payload(checkpoint["model_config"]))
    elif architecture == "visual-folio-interference-retina-v2":
        model = FolioAddressRetina(folio_address_config_from_payload(checkpoint["model_config"]))
    else:
        raise ValueError(f"unsupported visual folio checkpoint architecture: {architecture!r}")
    model = model.to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def load_folio_encoder(
    path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return folio_encoder_from_checkpoint(checkpoint, device), checkpoint
