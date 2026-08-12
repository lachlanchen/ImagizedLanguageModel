from __future__ import annotations

import hashlib
import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from .dataset import pil_to_tensor
from .instruction_data import (
    InstructionRenderConfig,
    VisualInstructionRecord,
    render_instruction_page,
)
from .rendering import (
    GlyphCorpus,
    GlyphExample,
    RenderConfig,
    render_answer_page,
    render_prompt_page,
)


@dataclass(frozen=True)
class VisualEpisodeSpec:
    identifier: str
    source: str
    language: str
    query_variants: tuple[str, ...]
    evaluation_queries: tuple[str, ...]
    response: str | None = None
    historical_char: str | None = None
    glyphs: tuple[GlyphExample, ...] = ()
    semantic: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.query_variants:
            raise ValueError("an episode needs at least one training query image")
        if (self.response is None) == (self.historical_char is None):
            raise ValueError("an episode must be instruction or historical, not both")
        if self.historical_char is not None and not self.glyphs:
            raise ValueError("historical episodes require provenance-bearing glyph images")

    @property
    def kind(self) -> str:
        return "historical" if self.historical_char is not None else "instruction"

    def metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "identifier": self.identifier,
            "source": self.source,
            "language": self.language,
            "kind": self.kind,
        }
        if self.historical_char is not None:
            metadata.update(
                {
                    "character": self.historical_char,
                    "evidence": [
                        {"stage": glyph.stage, "label": glyph.label, "source_path": str(glyph.path)}
                        for glyph in self.glyphs
                    ],
                    "evidence_policy": "copied_attested_pixels",
                }
            )
        return metadata


def stable_episode_fraction(identifier: str) -> float:
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def instruction_episode_specs(records: Sequence[VisualInstructionRecord]) -> list[VisualEpisodeSpec]:
    return [
        VisualEpisodeSpec(
            identifier=record.identifier,
            source=record.source,
            language=record.language,
            query_variants=(record.prompt,),
            evaluation_queries=(record.prompt,),
            response=record.response,
        )
        for record in records
    ]


def _historical_query_families(character: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    training = (
        f"请依据实物字形说明“{character}”的起源和历代写法。",
        f"“{character}”字从古文字到现代字形怎样演变？请展示图证。",
        f"Explain the evidenced visual origin and evolution of the Chinese character {character}.",
        f"Show how the written form {character} changed through recorded historical stages.",
        f"漢字「{character}」の字形の由来と変遷を、実物画像に基づいて説明してください。",
    )
    held_out = (
        f"What is the origin of the Kanji {character}? Return an illustrated page with attested forms.",
        f"请问{character}的甲骨文、金文、篆书和今字之间有什么字形关系？",
        f"Where did the visible shape of {character} come from, and what did it look like earlier?",
        f"漢字「{character}」は昔どのように書かれ、その形はどう変わりましたか。",
    )
    return training, held_out


def historical_episode_specs(
    corpus: GlyphCorpus,
    characters: Sequence[str],
    *,
    teacher_records: Mapping[str, Mapping[str, Any]] | None = None,
    seed: int = 811,
) -> list[VisualEpisodeSpec]:
    teacher_records = teacher_records or {}
    episodes: list[VisualEpisodeSpec] = []
    for index, character in enumerate(characters):
        glyphs = tuple(corpus.examples_for(character, rng=random.Random(seed + index * 97)))
        if not glyphs:
            continue
        queries, evaluation_queries = _historical_query_families(character)
        episodes.append(
            VisualEpisodeSpec(
                identifier=f"historical:{character}",
                source="hanziyuan-local-evidence",
                language="zh-en-ja",
                query_variants=queries,
                evaluation_queries=evaluation_queries,
                historical_char=character,
                glyphs=glyphs,
                semantic=dict(teacher_records.get(character, {})),
            )
        )
    return episodes


def _language_for_query(text: str, fallback: str) -> str:
    if any("\u3040" <= character <= "\u30ff" for character in text):
        return "ja"
    if any("\u3400" <= character <= "\u9fff" for character in text):
        return "zh"
    return "en" if fallback == "zh-en-ja" else fallback


def render_episode_query(
    episode: VisualEpisodeSpec,
    *,
    query: str,
    image_size: int,
    variant: int,
    detailed_layout: bool = False,
    augment: bool = False,
) -> Image.Image:
    language = _language_for_query(query, episode.language)
    if detailed_layout and episode.historical_char is not None:
        semantic = dict(episode.semantic or {})
        semantic["query_en" if language == "en" else "query_zh"] = query
        image = render_prompt_page(
            episode.historical_char,
            RenderConfig(image_size=image_size),
            glyphs=episode.glyphs,
            semantic=semantic,
            language=language,
            variant=variant,
        )
    else:
        image, _ = render_instruction_page(
            query,
            role="prompt",
            language=language,
            config=InstructionRenderConfig(image_size=image_size, augment=augment),
            variant=variant,
        )
    return image


def render_episode_answer(
    episode: VisualEpisodeSpec,
    *,
    image_size: int,
    variant: int,
    augment: bool = False,
) -> Image.Image:
    if episode.historical_char is not None:
        image = render_answer_page(
            episode.historical_char,
            episode.glyphs,
            RenderConfig(image_size=image_size),
            semantic=episode.semantic,
            variant=variant,
        )
        return augment_episode_image(image, random.Random(variant)) if augment else image
    image, _ = render_instruction_page(
        episode.response or "",
        role="answer",
        language=episode.language,
        config=InstructionRenderConfig(image_size=image_size, augment=augment),
        variant=variant,
    )
    return image


def augment_episode_image(image: Image.Image, rng: random.Random) -> Image.Image:
    """Simulate typography capture damage without reading the content."""

    image = image.convert("RGB")
    if rng.random() < 0.55:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.78, 1.22))
    if rng.random() < 0.40:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.88, 1.12))
    if rng.random() < 0.38:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.15, 0.85)))
    if rng.random() < 0.30:
        offset_x = rng.uniform(-3.5, 3.5)
        offset_y = rng.uniform(-3.5, 3.5)
        image = image.transform(
            image.size,
            Image.Transform.AFFINE,
            (1.0, rng.uniform(-0.008, 0.008), offset_x, rng.uniform(-0.008, 0.008), 1.0, offset_y),
            resample=Image.Resampling.BILINEAR,
            fillcolor=(248, 248, 244),
        )
    if rng.random() < 0.32:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=rng.randint(45, 88))
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
    if rng.random() < 0.36:
        array = np.asarray(image, dtype=np.float32)
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0.0, rng.uniform(1.0, 5.0), array.shape)
        image = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), "RGB")
    return image


class VisualEpisodeDataset(Dataset):
    """Produce image-only positive views and answer images for field learning."""

    def __init__(
        self,
        episodes: Sequence[VisualEpisodeSpec],
        *,
        image_size: int = 384,
        seed: int = 0,
    ):
        if not episodes:
            raise ValueError("VisualEpisodeDataset requires at least one episode")
        self.episodes = list(episodes)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode = self.episodes[index]
        rng = random.Random(self.seed + self.epoch * 10_000_019 + index * 104_729)
        first_query = rng.choice(episode.query_variants)
        second_query = rng.choice(episode.query_variants)
        variant_a = rng.randrange(2**31)
        variant_b = rng.randrange(2**31)
        query_a = render_episode_query(
            episode,
            query=first_query,
            image_size=self.image_size,
            variant=variant_a,
            detailed_layout=episode.historical_char is not None and rng.random() < 0.45,
            augment=True,
        )
        query_b = render_episode_query(
            episode,
            query=second_query,
            image_size=self.image_size,
            variant=variant_b,
            detailed_layout=episode.historical_char is not None and rng.random() < 0.45,
            augment=True,
        )
        query_a = augment_episode_image(query_a, rng)
        query_b = augment_episode_image(query_b, rng)
        answer = render_episode_answer(
            episode,
            image_size=self.image_size,
            variant=rng.randrange(2**31),
            augment=True,
        )
        return {
            "query_a": pil_to_tensor(query_a),
            "query_b": pil_to_tensor(query_b),
            "answer": pil_to_tensor(answer),
            "metadata": episode.metadata(),
        }


def visual_episode_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty visual episode batch")
    return {
        "query_a": torch.stack([item["query_a"] for item in batch]),
        "query_b": torch.stack([item["query_b"] for item in batch]),
        "answer": torch.stack([item["answer"] for item in batch]),
        "metadata": [item["metadata"] for item in batch],
    }
