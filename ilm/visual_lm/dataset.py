from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .rendering import GlyphCorpus, RenderConfig, render_answer_page, render_prompt_page


@dataclass(frozen=True)
class VisualLanguageSample:
    prompt: Image.Image
    target: Image.Image
    metadata: dict[str, Any]


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().float().cpu().clamp(-1, 1)
    arr = ((x.permute(1, 2, 0).numpy() + 1.0) * 127.5).round().astype(np.uint8)
    return Image.fromarray(arr, "RGB")


class VisualLanguageDataset(Dataset):
    """On-the-fly image-to-image samples for ILM-V.

    Each item is a real rendered prompt page and a real rendered answer page.
    Historical glyph panels are loaded from the local SVG/bitmap glyph corpus.
    """

    def __init__(
        self,
        corpus: GlyphCorpus,
        *,
        render_config: RenderConfig,
        length: int = 1024,
        seed: int = 0,
        characters: list[str] | None = None,
    ):
        self.corpus = corpus
        self.cfg = render_config
        self.length = int(length)
        self.seed = int(seed)
        self.characters = characters or list(corpus.characters)
        if not self.characters:
            raise ValueError("VisualLanguageDataset needs at least one character.")

    def __len__(self) -> int:
        return self.length

    def render_sample(self, idx: int) -> VisualLanguageSample:
        rng = random.Random(self.seed + idx * 1009)
        char = self.characters[idx % len(self.characters)] if self.length <= len(self.characters) else rng.choice(self.characters)
        variant = rng.randrange(1_000_000)
        glyphs = self.corpus.examples_for(char, rng=rng)
        if not glyphs:
            raise ValueError(f"No glyph examples found for {char!r}")
        prompt = render_prompt_page(char, self.cfg, variant=variant)
        target = render_answer_page(char, glyphs, self.cfg, variant=variant)
        return VisualLanguageSample(
            prompt=prompt,
            target=target,
            metadata={
                "char": char,
                "codepoint": f"U+{ord(char):04X}",
                "variant": variant,
                "glyphs": [{"stage": g.stage, "label": g.label, "path": str(g.path)} for g in glyphs],
            },
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.render_sample(idx)
        return {
            "prompt": pil_to_tensor(sample.prompt),
            "target": pil_to_tensor(sample.target),
            "metadata": sample.metadata,
        }
