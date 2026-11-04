"""A proper image-copy/raster mixture with a learned sequence stop hazard."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .canonical_glyph_language import CanonicalGlyphLanguageConfig
from .exact_raster_density_v54 import ExactRasterDensityConfig
from .hierarchical_raster_language import CausalRasterReader
from .joint_exact_raster_v55 import JointExactRasterLanguage
from .raster_beam import raster_beam_search

ARCHITECTURE = "visual-copy-raster-language-v57"


@dataclass(frozen=True)
class VisualCopyReaderConfig(CanonicalGlyphLanguageConfig):
    maximum_cells: int = 512

    def __post_init__(self):
        # Keep old experiments' configuration contract unchanged.
        CanonicalGlyphLanguageConfig(**{**asdict(self), "maximum_cells": 64})
        if self.maximum_cells != 512:
            raise ValueError("V57 requires a 512-cell maximum")


def raster_signatures(images):
    """Lossless 32-word bit packing, derived only from observed binary pixels."""
    if tuple(images.shape[-3:]) != (1, 32, 32):
        raise ValueError("image cells must end in [1,32,32]")
    if not bool(((images == 0) | (images == 1)).all()):
        raise ValueError("copy equality requires binary image cells")
    bits = images.to(torch.int64).reshape(*images.shape[:-3], 32, 32)
    weights = 1 << torch.arange(32, device=images.device, dtype=torch.int64)
    return (bits * weights).sum(-1)


def raster_equality(targets, sources):
    """Return [B,A,S] exact equality, without a symbolic identity table."""
    a, b = raster_signatures(targets), raster_signatures(sources)
    return (a[:, :, None] == b[:, None]).all(-1)


def mixture_log_probability(copy_logits, attention_logp, equal, free_nll):
    """Marginalize duplicate observations and the two normalized components."""
    present = (equal & torch.isfinite(attention_logp)).any(-1)
    masked = attention_logp.masked_fill(~equal, -torch.inf)
    # logsumexp of an all-minus-infinity row has undefined backward derivatives.
    safe = torch.where(present[:, None], masked, torch.zeros_like(masked))
    copied = torch.logsumexp(safe, dim=-1).masked_fill(~present, -torch.inf)
    return torch.logaddexp(
        F.logsigmoid(copy_logits) + copied,
        F.logsigmoid(-copy_logits) - free_nll,
    )


class VisualCopyLanguage(JointExactRasterLanguage):
    def __init__(self, reader_config, density_config, *, copy_enabled=True):
        super().__init__(
            CausalRasterReader(reader_config), density_config, reader_trainable=True
        )
        self.copy_enabled = bool(copy_enabled)
        d = reader_config.model_dim
        self.copy_query = nn.Linear(d, d, bias=False)
        self.copy_key = nn.Linear(d, d, bias=False)
        self.copy_gate = nn.Linear(d, 1)
        self.stop_head = nn.Linear(d, 1)
        nn.init.zeros_(self.copy_gate.weight)
        nn.init.zeros_(self.copy_gate.bias)
        nn.init.zeros_(self.stop_head.weight)
        nn.init.constant_(self.stop_head.bias, math.log(0.1 / 0.9))
        if not copy_enabled:
            for layer in (self.copy_query, self.copy_key, self.copy_gate):
                layer.requires_grad_(False)

    def visual_condition(self, sequence, prompt_lengths, query_positions):
        """Read only pixels; integer arguments describe packing/causal positions."""
        states = self.reader.language(sequence)
        batch = torch.arange(len(sequence), device=sequence.device)[:, None]
        if query_positions.ndim != 2 or prompt_lengths.shape != (len(sequence),):
            raise ValueError("invalid visual packing metadata")
        if not bool(
            ((query_positions >= 0) & (query_positions < sequence.shape[1])).all()
        ):
            raise ValueError("query outside visible sequence")
        if not bool(
            ((prompt_lengths > 0) & (prompt_lengths <= sequence.shape[1])).all()
        ):
            raise ValueError("invalid prompt lengths")
        hidden = states["hidden_states"][batch, query_positions].float()
        fields = states["anchor_fields"][batch, query_positions].float()
        source_count = int(prompt_lengths.max())
        keys = states["hidden_states"][:, :source_count].float()
        source_positions = torch.arange(source_count, device=sequence.device)
        visible = (source_positions[None, None] < prompt_lengths[:, None, None]) & (
            source_positions[None, None] <= query_positions[:, :, None]
        )
        score = torch.einsum(
            "bad,bsd->bas", self.copy_query(hidden), self.copy_key(keys)
        ).float() / math.sqrt(hidden.shape[-1])
        attention = score.masked_fill(~visible, -torch.inf).log_softmax(-1)
        copy_logits = self.copy_gate(hidden).squeeze(-1).float()
        if not self.copy_enabled:
            copy_logits = torch.full_like(copy_logits, -torch.inf)
        return {
            "hidden": hidden,
            "fields": fields,
            "attention_logp": attention,
            "copy_logits": copy_logits,
            "stop_logits": self.stop_head(hidden).squeeze(-1).float(),
            "source_images": sequence[:, :source_count],
            "source_visible": visible,
        }

    def event_nll(self, condition, targets, continued, stopped, *, motor=None):
        """Score valid image and stop events; padded events contribute zero."""
        if continued.shape != stopped.shape or bool((continued & stopped).any()):
            raise ValueError("continue/stop masks overlap or differ in shape")
        hidden, fields = condition["hidden"], condition["fields"]
        if targets.shape[:2] != continued.shape:
            raise ValueError("image event layout differs from masks")
        motor = torch.zeros_like(continued) if motor is None else motor
        if bool((motor & ~continued).any()):
            raise ValueError("motor conditions require an image event")
        image_logp = torch.zeros_like(condition["stop_logits"])
        if bool(continued.any()):
            h, z, y = hidden[continued], fields[continued], targets[continued]
            copying = motor[continued]
            if bool(copying.any()):
                h = h.masked_fill(copying[:, None], 0)
                exact = self.reader.field.encode_unit(y).float()
                z = torch.where(copying[:, None], exact, z)
            free_nll = self.image_nll(y, h, z)
            equal = raster_equality(targets, condition["source_images"])[continued]
            gate = condition["copy_logits"][continued].masked_fill(copying, -torch.inf)
            image_logp[continued] = mixture_log_probability(
                gate, condition["attention_logp"][continued], equal, free_nll
            )
        stop = condition["stop_logits"]
        nll = torch.zeros_like(stop)
        nll[continued] = -image_logp[continued] - F.logsigmoid(-stop[continued])
        nll[stopped] = -F.logsigmoid(stop[stopped])
        # Supplied-image motor exercises do not train the sequence stop hazard.
        nll[motor] = -image_logp[motor]
        return nll

    @torch.no_grad()
    def generate(self, prompt, *, maximum_output_cells=17, beam_width=8):
        """Hierarchical greedy decode; no reference length or answer IDs."""
        if prompt.ndim != 4 or not 1 <= len(prompt) <= 480:
            raise ValueError("native prompt must be [1..480,1,32,32]")
        prompt = (prompt >= 0.5).float()
        sequence = prompt[None]
        emitted, events = [], []
        for step in range(maximum_output_cells + 1):
            c = self.visual_condition(
                sequence,
                torch.tensor([len(prompt)], device=prompt.device),
                torch.tensor([[sequence.shape[1] - 1]], device=prompt.device),
            )
            stopping = float(c["stop_logits"][0, 0].sigmoid())
            gate = float(c["copy_logits"][0, 0].sigmoid())
            if stopping >= 0.5:
                events.append(
                    {
                        "stop_probability": stopping,
                        "copy_probability": gate,
                        "route": "stop",
                    }
                )
                return {
                    "images": torch.stack(emitted) if emitted else prompt[:0],
                    "stopped": True,
                    "capped": False,
                    "events": events,
                }
            if step == maximum_output_cells:
                break
            if self.copy_enabled and gate >= 0.5:
                equal = raster_equality(prompt[None], prompt[None])[0]
                mass = equal.float() @ c["attention_logp"][0, 0].exp()
                selected = int(mass.argmax())
                image = prompt[selected]
                event = {
                    "route": "copy",
                    "source_position": selected,
                    "marginal_copy_mass": float(mass[selected]),
                }
            else:
                result = raster_beam_search(
                    self, c["hidden"][:, 0], c["fields"][:, 0], beam_width=beam_width
                )
                image = result.images[0]
                event = {
                    "route": "free",
                    "raster_log_probability": float(result.log_probability[0]),
                }
            event.update(stop_probability=stopping, copy_probability=gate)
            events.append(event)
            emitted.append(image)
            sequence = torch.cat((sequence, image[None, None]), dim=1)
        return {
            "images": torch.stack(emitted) if emitted else prompt[:0],
            "stopped": False,
            "capped": True,
            "events": events,
        }

    def boundary_receipt(self):
        r = super().boundary_receipt()
        r.update(
            architecture=ARCHITECTURE,
            copy_enabled=self.copy_enabled,
            maximum_cells=self.reader.config.maximum_cells,
            copy_source="currently visible prompt images only",
            stop="learned transport hazard, no output glyph ID",
            decoder="hierarchical branch greedy; duplicate copy mass marginalized",
        )
        return r


def model_from_foundation(checkpoint, device, *, copy_enabled):
    rc = VisualCopyReaderConfig(**{**checkpoint["reader_config"], "maximum_cells": 512})
    model = VisualCopyLanguage(
        rc,
        ExactRasterDensityConfig(**checkpoint["model_config"]),
        copy_enabled=copy_enabled,
    )
    result = model.load_state_dict(
        checkpoint.get("ema_model", checkpoint.get("model")), strict=False
    )
    if (
        set(result.missing_keys)
        != {
            "copy_query.weight",
            "copy_key.weight",
            "copy_gate.weight",
            "copy_gate.bias",
            "stop_head.weight",
            "stop_head.bias",
        }
        or result.unexpected_keys
    ):
        raise ValueError(f"foundation mismatch: {result}")
    return model.to(device)


def model_from_checkpoint(checkpoint, device):
    if checkpoint["architecture"] != ARCHITECTURE:
        raise ValueError("not a V57 image-copy language checkpoint")
    model = VisualCopyLanguage(
        VisualCopyReaderConfig(**checkpoint["reader_config"]),
        ExactRasterDensityConfig(**checkpoint["model_config"]),
        copy_enabled=checkpoint["arm"] == "copy",
    )
    model.load_state_dict(
        checkpoint.get("ema_model", checkpoint.get("model")), strict=True
    )
    return model.to(device).requires_grad_(False).eval()
