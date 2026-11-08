"""A constant task condition assembled with live visual features.

The deployed object contains vectors, masks and positions. It accepts neither
text nor token IDs. Compilation uses the source model's tokenizer and embedding
table offline; retaining its pretrained transformer body is a separate choice.
"""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn


class CompiledVisualCondition(nn.Module):
    KEYS = {"prompt_embeds", "image_mask", "attention_mask", "position_ids", "image_grid_thw"}

    def __init__(self, tensors, *, queries):
        super().__init__()
        if set(tensors) != self.KEYS:
            raise ValueError("unexpected compiled condition fields")
        self.queries = int(queries)
        if self.queries <= 0:
            raise ValueError("positive continuous-query count required")
        for name, tensor in tensors.items():
            self.register_buffer(name, tensor.detach().clone(), persistent=True)
        b, length, hidden = self.prompt_embeds.shape
        if b != 2 or hidden <= 0 or not self.prompt_embeds.is_floating_point():
            raise ValueError("one conditional and one unconditional branch required")
        if not torch.isfinite(self.prompt_embeds).all():
            raise ValueError("nonfinite compiled task vectors")
        if self.image_mask.dtype != torch.bool or self.image_mask.shape != (b, length):
            raise ValueError("image mask shape or type differs")
        if self.attention_mask.dtype != torch.bool or self.attention_mask.shape != (b, length + self.queries):
            raise ValueError("attention mask shape or type differs")
        if self.position_ids.dtype != torch.long or self.position_ids.shape != (3, b, length + self.queries):
            raise ValueError("multimodal positions differ")
        if self.image_grid_thw.dtype != torch.long or self.image_grid_thw.shape != (2, 3):
            raise ValueError("one reference image per branch required")
        if (self.image_grid_thw <= 0).any() or not torch.equal(self.image_grid_thw[0], self.image_grid_thw[1]):
            raise ValueError("branches require the same positive reference geometry")
        if not torch.equal(self.image_mask.sum(1), self.image_mask.sum(1)[0].expand(2)) or not self.image_mask.any():
            raise ValueError("branches require the same nonzero visual feature count")
        if torch.count_nonzero(self.prompt_embeds[self.image_mask]):
            raise ValueError("compiled image slots must be zero until live features arrive")

    @classmethod
    def from_pretrained(cls, folder):
        from safetensors.torch import load_file
        folder = Path(folder)
        manifest = json.loads((folder / "manifest.json").read_text())
        payload = folder / "condition.safetensors"
        if manifest.get("format") != "ilm-fixed-visual-condition-v1":
            raise ValueError("unknown compiled visual condition format")
        if hashlib.sha256(payload.read_bytes()).hexdigest() != manifest["payload_sha256"]:
            raise ValueError("compiled condition payload changed")
        return cls(load_file(str(payload), device="cpu"), queries=manifest["queries"])

    def forward(self, query_embeds, *, image_embeds, image_grid_thw, past_key_values=None):
        if past_key_values is not None:
            raise ValueError("this fixed full-sequence condition does not support a KV cache")
        b, length, hidden = self.prompt_embeds.shape
        if query_embeds.shape != (b, self.queries, hidden):
            raise ValueError("continuous-query shape differs from compiled geometry")
        if not query_embeds.is_floating_point():
            raise ValueError("continuous-query tensor must be floating point")
        if image_embeds.shape != (int(self.image_mask.sum()), hidden):
            raise ValueError("live visual feature shape differs from compiled image slots")
        if image_embeds.device != query_embeds.device or image_embeds.dtype != query_embeds.dtype:
            raise ValueError("live image and query features must share device and dtype")
        if image_grid_thw.dtype != torch.long or not torch.equal(
            image_grid_thw.to(self.image_grid_thw.device), self.image_grid_thw
        ):
            raise ValueError("live image geometry differs from compiled positions")
        device, dtype = query_embeds.device, query_embeds.dtype
        inputs = self.prompt_embeds.to(device=device, dtype=dtype).clone()
        inputs[self.image_mask.to(device)] = image_embeds
        return dict(
            inputs_embeds=torch.cat((inputs, query_embeds), dim=1),
            attention_mask=self.attention_mask.to(device),
            position_ids=self.position_ids.to(device),
            past_key_values=None,
        )
