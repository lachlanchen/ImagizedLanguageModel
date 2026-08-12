from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from .folio_data import FOLIO_AVAILABLE_FONTS
from .ink_jepa_data import (
    RetinalRenderConfig,
    VisualGrammarRecord,
    extract_retinal_fovea,
    render_retinal_page,
)


def _validation_record(identifier: str, fraction: float) -> bool:
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return value < fraction


def _visible_writing(text: str) -> str:
    return "".join(character for character in text if not character.isspace())


@dataclass(frozen=True)
class SaccadeSequenceSpec:
    sequence_length: int = 48
    fovea_size: int = 32

    def __post_init__(self) -> None:
        if self.sequence_length < 4:
            raise ValueError("visual sequence must contain at least four causal transitions")
        if self.fovea_size < 16 or self.fovea_size % 8:
            raise ValueError("fovea_size must be a multiple of eight and at least 16")


def render_saccade_foveas(
    writing: str,
    *,
    render_config: RetinalRenderConfig,
    fovea_size: int,
    variant: int,
) -> torch.Tensor:
    if len(writing) > render_config.capacity:
        raise ValueError("writing sequence exceeds the retinal page capacity")
    page = render_retinal_page(writing, config=render_config, variant=variant)
    output = []
    for index in range(len(writing)):
        row, column = divmod(index, render_config.columns)
        output.append(
            extract_retinal_fovea(
                page,
                row=row,
                column=column,
                config=render_config,
                fovea_size=fovea_size,
            )
        )
    return torch.stack(output)


def render_glyph_fovea(
    character: str,
    *,
    render_config: RetinalRenderConfig,
    fovea_size: int,
    variant: int,
) -> torch.Tensor:
    if len(character) != 1 or character.isspace():
        raise ValueError("a glyph fovea requires exactly one visible character")
    return render_saccade_foveas(
        character,
        render_config=render_config,
        fovea_size=fovea_size,
        variant=variant,
    )[0]


class VisualSaccadeDataset(Dataset):
    """Render causal fixation sequences; the student receives only image tensors."""

    def __init__(
        self,
        records: Sequence[VisualGrammarRecord],
        *,
        render_config: RetinalRenderConfig,
        spec: SaccadeSequenceSpec,
        split: str,
        validation_fraction: float = 0.03,
        length: int | None = None,
        seed: int = 0,
        expose_evaluation_labels: bool = False,
    ):
        if split not in {"train", "validation", "all"}:
            raise ValueError("split must be train, validation, or all")
        if spec.sequence_length + 1 > render_config.capacity:
            raise ValueError("causal sequence does not fit on the retinal renderer")
        selected: list[tuple[VisualGrammarRecord, str]] = []
        for record in records:
            validation = _validation_record(record.identifier, validation_fraction)
            writing = _visible_writing(record.text)
            if len(writing) < spec.sequence_length + 1:
                continue
            if split == "all" or (split == "validation" and validation) or (split == "train" and not validation):
                selected.append((record, writing))
        if not selected:
            raise ValueError(f"no visual saccade records selected for split={split}")
        self.records = selected
        self.render_config = render_config
        self.spec = spec
        self.length = int(length) if length is not None else len(selected)
        self.seed = int(seed)
        self.epoch = 0
        self.expose_evaluation_labels = bool(expose_evaluation_labels)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.seed + self.epoch * 10_000_019 + index * 104_729)
        if self.length <= len(self.records):
            record, writing = self.records[index % len(self.records)]
        else:
            record, writing = rng.choice(self.records)
        start = rng.randint(0, len(writing) - self.spec.sequence_length - 1)
        segment = writing[start : start + self.spec.sequence_length + 1]
        first_variant = rng.randrange(2**31)
        second_variant = first_variant + 1
        if len(FOLIO_AVAILABLE_FONTS) > 1:
            while second_variant % len(FOLIO_AVAILABLE_FONTS) == first_variant % len(FOLIO_AVAILABLE_FONTS):
                second_variant += 1
        first = render_saccade_foveas(
            segment,
            render_config=self.render_config,
            fovea_size=self.spec.fovea_size,
            variant=first_variant,
        )
        second = render_saccade_foveas(
            segment,
            render_config=self.render_config,
            fovea_size=self.spec.fovea_size,
            variant=second_variant,
        )
        metadata: dict[str, Any] = {
            "id": record.identifier,
            "source": record.source,
            "rights": record.rights,
            "offset": start,
            "first_variant": first_variant,
            "second_variant": second_variant,
        }
        if self.expose_evaluation_labels:
            metadata["target_character"] = segment[-1]
            metadata["previous_character"] = segment[-2]
        return {
            "context": first[:-1],
            "target_ink": first[1:],
            "current_reference": second[:-1],
            "target_reference": second[1:],
            "metadata": metadata,
        }


def visual_saccade_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "context": torch.stack([item["context"] for item in batch]),
        "target_ink": torch.stack([item["target_ink"] for item in batch]),
        "current_reference": torch.stack([item["current_reference"] for item in batch]),
        "target_reference": torch.stack([item["target_reference"] for item in batch]),
        "metadata": [item["metadata"] for item in batch],
    }
