"""Causal writing over physical glyph proposals and continuous pictured inputs."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import GPT2Config, GPT2Model

from ilm.visual_lm.font_invariant_reader import GlyphMetric


class CausalGlyphWriter(nn.Module):
    def __init__(self, config, projection, regression, markers, packed):
        super().__init__()
        # Only continuous inputs enter the inherited decoder.
        config.vocab_size = 1
        self.body = GPT2Model(config)
        self.body.wte = None
        self.metric = GlyphMetric()
        self.register_buffer("projection", projection)
        self.register_buffer("regression", regression)
        self.register_buffer("markers", markers)
        self.register_buffer("packed", packed)

    def encode(self, packed):
        shifts = torch.arange(7, -1, -1, device=packed.device)
        bits = ((packed[..., None] >> shifts) & 1).reshape(-1, 32, 32)
        features = F.relu(self.metric(bits) @ self.projection)
        return (features * (2 / self.projection.shape[1]) ** .5) @ self.regression

    def attach_typeface_map(self, path):
        from ilm.visual_lm.typeface_map import TypefaceMap
        self.typeface_map = TypefaceMap.restore(path, self.packed)

    def read_extra(self, packed, alphabet):
        mapper = getattr(self, "typeface_map", None)
        if mapper is None:
            return self.encode(packed)
        indices, accept = mapper.identify(packed)
        result = alphabet[indices]
        if not accept.all():
            result[~accept] = self.encode(packed[~accept])
        return result

    @torch.no_grad()
    def alphabet(self):
        # Row zero ends the response. All other logits refer to actual ink planes.
        return torch.cat([self.markers[1:2], *(self.encode(p) for p in self.packed.split(256))])

    @torch.inference_mode()
    def generate(self, prompts, alphabet, limit=48):
        if not prompts or not 1 <= limit <= self.body.config.n_positions:
            raise ValueError("Invalid batch or generation length")
        width = max(map(len, prompts))
        if width + limit > self.body.config.n_positions:
            raise ValueError("Pictured prompt exceeds available context")
        device = alphabet.device
        x = alphabet.new_zeros(len(prompts), width, alphabet.shape[-1])
        mask = torch.zeros(x.shape[:2], device=device, dtype=torch.long)
        for i, prompt in enumerate(prompts):
            x[i, -len(prompt):] = prompt
            mask[i, -len(prompt):] = 1
        pos = (mask.cumsum(1) - 1).clamp_min(0)
        cache = None
        streams = [[] for _ in prompts]
        ended = [False] * len(streams)
        for step in range(limit + 1):
            out = self.body(inputs_embeds=x, attention_mask=mask, position_ids=pos,
                            past_key_values=cache, use_cache=True)
            choice = (out.last_hidden_state[:, -1] @ alphabet.T).argmax(-1)
            for i, symbol in enumerate(choice.tolist()):
                if symbol == 0:
                    ended[i] = True
                elif not ended[i] and step < limit:
                    streams[i].append(symbol)
            if all(ended):
                break
            cache = out.past_key_values
            # alphabet is a lossless cache of encodings of the emitted bitmaps.
            x = alphabet[choice, None]
            pos = pos[:, -1:] + 1
            mask = F.pad(mask, (0, 1), value=1)
        return streams, ended

    @classmethod
    def restore(cls, folder, device="cpu"):
        folder = Path(folder)
        interface = load_file(str(folder / "interface.safetensors"))
        writer = cls(GPT2Config.from_json_file(str(folder / "config.json")), **interface)
        writer.metric.load_state_dict(load_file(str(folder / "metric.safetensors")))
        writer.body.load_state_dict(load_file(str(folder / "body.safetensors")))
        if (folder / "typeface.safetensors").is_file():
            writer.attach_typeface_map(folder / "typeface.safetensors")
        return writer.to(device).eval()


def packed_page(path):
    """Read a 32-pixel-high horizontal strip, black ink on white paper."""
    from PIL import Image
    image = np.asarray(Image.open(path).convert("L"))
    if image.shape[0] != 32 or not image.shape[1] or image.shape[1] % 32:
        raise ValueError("Use a single row of 32 by 32 glyph images")
    return torch.from_numpy(np.packbits((image < 128).reshape(32, -1, 32)
                                        .transpose(1, 0, 2).reshape(-1, 1024), axis=-1))


def write_strip(packed, path):
    from PIL import Image
    if len(packed) == 0:
        pixels = np.zeros((32, 1), np.uint8)
    else:
        pixels = np.unpackbits(np.asarray(packed, np.uint8), axis=-1).reshape(-1, 32, 32)
        pixels = pixels.transpose(1, 0, 2).reshape(32, -1)
    Image.fromarray(255 * (1 - pixels)).save(path)
