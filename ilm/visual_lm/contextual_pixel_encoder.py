"""Reusable image-only contextual encoder and learned source-pixel motor.

The completed experiment runner stays immutable. This runtime reproduces its
feature operations and is checked against its saved BF16 descriptors.
"""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .causal_visual_prefix import fold_writing_prompt
from .clean_image_actor import tensor_digest
from .contextual_pixel_spans import ContextualPixelSpans, units
from .grounded_image_weights import load_grounded_image_actor, file_sha
from .image_context_interface import ImageContextInterface
from .semantic_image_memory import native_budget_image


class ContextualPixelReader:
    def __init__(self, *, checkpoint, checkpoint_sha256, controls, controls_sha256,
                 ollama_root, gguf_source, device):
        if file_sha(checkpoint) != checkpoint_sha256 or file_sha(controls) != controls_sha256:
            raise ValueError("completed readout and numeric source controls required")
        self.actor, self.provenance = load_grounded_image_actor(ollama_root=ollama_root,
            gguf_source=gguf_source, device=device)
        self.actor.writer = torch.nn.Identity()
        self.actor.requires_grad_(False).eval()
        if any(bool(p.any()) for n, p in self.actor.named_parameters() if n.endswith(".up")):
            raise ValueError("calibrated frozen source required")
        self.observer = ImageContextInterface(self.actor,
            **torch.load(controls, map_location="cpu", weights_only=True)).eval()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload["completed_updates"] != 3200:
            raise ValueError("complete final pixel-span head required")
        self.head = ContextualPixelSpans().to(device).eval()
        self.head.load_state_dict(payload["model"], strict=True)
        self.device = device

    @torch.inference_mode()
    def encode(self, image):
        pixels = np.asarray(image)
        spans = units(pixels)
        physical = torch.from_numpy(pixels.copy()).float().div(127.5).sub(1)[None, None]
        folded = fold_writing_prompt(physical)
        arguments, meta = self.observer.assemble(native_budget_image(folded.pixels))
        output = self.actor.context(**arguments, use_cache=False)
        start = meta["before_positions"]
        h, w = meta["grid"][1] // 2, meta["grid"][2] // 2
        field = output.last_hidden_state[:, start:start + h * w].transpose(1, 2).reshape(1, 4096, h, w).float()
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
        visual = F.grid_sample(field, grid, align_corners=False, padding_mode="border").mean(-1).transpose(1, 2)[0]
        geometry = torch.tensor(np.stack((spans[:, 0] / pixels.shape[1], spans[:, 1] / pixels.shape[1],
            row / max(1, len(folded.source_ranges)), (spans[:, 1] - spans[:, 0]) / 32), -1), dtype=torch.float32)
        return dict(visual=visual.to(torch.bfloat16).cpu(),
            memory=output.last_hidden_state[0, -len(self.observer.after):].to(torch.bfloat16).cpu(),
            geometry=geometry, source_units=torch.from_numpy(spans.copy()), metadata=meta)

    @torch.inference_mode()
    def generate(self, image):
        features = self.encode(image)
        logp = self.head(features["visual"][None].to(self.device).float(),
            features["memory"][None].to(self.device).float(), features["geometry"][None].to(self.device),
            torch.tensor([len(features["visual"])], device=self.device))[0]
        answer, event = self.head.select(logp, image, features["source_units"].numpy())
        return answer, dict(event=event, feature_hashes={k: tensor_digest(features[k])
            for k in ("visual", "memory", "geometry", "source_units")}, metadata=features["metadata"],
            image_input_image_output=True, references_used=False, tokenizer=False, ocr=False,
            language_vocabulary=False, teacher_service=False, source_pixel_copy=True,
            free_generation_demonstrated=False, general_ilm_goal_complete=False)

    def parameter_count(self):
        return sum(p.numel() for p in self.observer.parameters()) + sum(p.numel() for p in self.head.parameters())
