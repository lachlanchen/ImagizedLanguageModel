from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from .ink_jepa_data import (
    RetinalRenderConfig,
    VisualGrammarRecord,
    retinal_character_supported,
)
from .saccade_data import render_glyph_fovea
from .visual_binding_data import (
    CANONICAL_TARGET_VARIANT,
    LABEL_PAIRS,
    MARKER_CHARACTERS,
    OPERATIONS,
    binding_partition_receipt,
    is_han_character,
    noncanonical_variant,
    split_binding_characters,
)


PARTITION_SALT = "visual-packet-reread-stream-v24"
PAIR_HEADER = "契"
OPERATION_HEADER = "法"
QUERY_HEADER = "问"
DISTRACTOR_HEADER = "旁"
END_HEADER = "止"
PACKET_HEADERS = (
    PAIR_HEADER,
    OPERATION_HEADER,
    QUERY_HEADER,
    DISTRACTOR_HEADER,
    END_HEADER,
)
PACKET_MARKER_CHARACTERS = MARKER_CHARACTERS | frozenset(PACKET_HEADERS)
HELD_OUT_COMBINATIONS = frozenset({("异", "天地"), ("同", "左右")})
MAX_TRAIN_DISTRACTORS = 2
MAX_DISTRACTORS = 3
MIN_PACKETS = 5
MAX_PACKETS = 8
FRAMES_PER_PACKET = 3


def build_packet_character_bank(
    records: Sequence[VisualGrammarRecord],
    *,
    bank_size: int = 1_024,
) -> list[str]:
    if bank_size < 2 + MAX_DISTRACTORS + 1:
        raise ValueError("visual packet bank is too small for one episode")
    counts = Counter(
        character
        for record in records
        for character in record.text
        if is_han_character(character)
        and character not in PACKET_MARKER_CHARACTERS
        and retinal_character_supported(character)
    )
    characters = [character for character, _ in counts.most_common(bank_size)]
    if len(characters) != bank_size:
        raise ValueError(
            f"requested {bank_size} packet identities, corpus supplied "
            f"{len(characters)}"
        )
    return characters


def split_packet_characters(
    characters: Sequence[str],
    *,
    salt: str = PARTITION_SALT,
) -> dict[str, list[str]]:
    return split_binding_characters(characters, salt=salt)


def packet_partition_receipt(
    partitions: dict[str, Sequence[str]],
    *,
    salt: str = PARTITION_SALT,
) -> dict[str, Any]:
    return binding_partition_receipt(partitions, salt=salt)


def _combination_key(operation: str, label_pair: tuple[str, str]) -> tuple[str, str]:
    return operation, "".join(label_pair)


def _allowed_combinations(*, held_out: bool) -> tuple[tuple[str, tuple[str, str]], ...]:
    return tuple(
        (operation, pair)
        for operation in OPERATIONS
        for pair in LABEL_PAIRS
        if (_combination_key(operation, pair) in HELD_OUT_COMBINATIONS)
        == held_out
    )


@dataclass(frozen=True)
class VisualPacketEpisodeConfig:
    fovea_size: int = 32
    source_font_size: int = 25
    source_minimum_font_size: int = 21
    target_font_size: int = 25
    canonical_target_variant: int = CANONICAL_TARGET_VARIANT
    development_heldout_fraction: float = 0.50
    development_heldout_length_fraction: float = 0.50

    def __post_init__(self) -> None:
        if self.fovea_size != 32:
            raise ValueError("V24 fixes the fovea size to 32")
        if self.source_minimum_font_size > self.source_font_size:
            raise ValueError("source minimum font size exceeds source font size")
        for name, value in (
            ("development_heldout_fraction", self.development_heldout_fraction),
            (
                "development_heldout_length_fraction",
                self.development_heldout_length_fraction,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def source_render_config(self) -> RetinalRenderConfig:
        return RetinalRenderConfig(
            font_size=self.source_font_size,
            minimum_font_size=self.source_minimum_font_size,
            augment=True,
        )

    def target_render_config(self) -> RetinalRenderConfig:
        return RetinalRenderConfig(
            font_size=self.target_font_size,
            minimum_font_size=self.target_font_size,
            augment=False,
        )


@dataclass
class _Packet:
    kind: str
    frames: torch.Tensor
    variants: tuple[int | None, int | None, int | None]


class VisualPacketEpisodeDataset(Dataset):
    """Render V24 image packets; symbolic fields remain renderer-only metadata."""

    def __init__(
        self,
        characters: Sequence[str],
        *,
        split: str,
        length: int,
        config: VisualPacketEpisodeConfig | None = None,
        seed: int = 20260829,
    ) -> None:
        if split not in {"train", "development"}:
            raise ValueError("packet dataset split must be train or development")
        if len(characters) < 2 + MAX_DISTRACTORS + 1:
            raise ValueError("packet dataset needs at least six identities")
        if length < 1:
            raise ValueError("packet dataset length must be positive")
        self.characters = tuple(characters)
        self.split = split
        self.length = int(length)
        self.config = config or VisualPacketEpisodeConfig()
        self.seed = int(seed)
        self.epoch = 0
        self.source_config = self.config.source_render_config()
        self.target_config = self.config.target_render_config()
        self.seen_combinations = _allowed_combinations(held_out=False)
        self.heldout_combinations = _allowed_combinations(held_out=True)
        self._target_cache: dict[str, torch.Tensor] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def _blank(self) -> torch.Tensor:
        return torch.zeros(1, self.config.fovea_size, self.config.fovea_size)

    def _render_source(self, character: str, variant: int) -> torch.Tensor:
        return render_glyph_fovea(
            character,
            render_config=self.source_config,
            fovea_size=self.config.fovea_size,
            variant=variant,
        )

    def _render_target(self, character: str) -> torch.Tensor:
        cached = self._target_cache.get(character)
        if cached is None:
            cached = render_glyph_fovea(
                character,
                render_config=self.target_config,
                fovea_size=self.config.fovea_size,
                variant=self.config.canonical_target_variant,
            )
            self._target_cache[character] = cached
        return cached

    def _variant(self, rng: random.Random) -> int:
        return noncanonical_variant(rng, self.config.canonical_target_variant)

    def _packet(
        self,
        rng: random.Random,
        *,
        kind: str,
        header: str,
        content_a: str | None,
        content_b: str | None,
    ) -> _Packet:
        characters = (header, content_a, content_b)
        variants: list[int | None] = []
        frames: list[torch.Tensor] = []
        for character in characters:
            if character is None:
                variants.append(None)
                frames.append(self._blank())
            else:
                variant = self._variant(rng)
                variants.append(variant)
                frames.append(self._render_source(character, variant))
        return _Packet(kind, torch.stack(frames), tuple(variants))

    @staticmethod
    def _stack_packets(packets: Sequence[_Packet]) -> torch.Tensor:
        return torch.cat([packet.frames for packet in packets], dim=0)

    @staticmethod
    def _packet_positions(packets: Sequence[_Packet], kind: str) -> list[int]:
        return [index for index, packet in enumerate(packets) if packet.kind == kind]

    @staticmethod
    def _localization_target(packets: Sequence[_Packet]) -> torch.Tensor:
        query_index = VisualPacketEpisodeDataset._packet_positions(packets, "query")[0]
        operation_index = VisualPacketEpisodeDataset._packet_positions(
            packets, "operation"
        )[0]
        pair_indices = VisualPacketEpisodeDataset._packet_positions(packets, "pair")
        query = packets[query_index].frames[1]
        operation = packets[operation_index].frames[1]
        pair_labels = torch.stack([packets[index].frames[1] for index in pair_indices])
        pair_glyphs = torch.stack([packets[index].frames[2] for index in pair_indices])
        return torch.stack(
            (
                query,
                operation,
                pair_labels.mean(dim=0),
                pair_glyphs.mean(dim=0),
            )
        )

    def _replace_content(
        self,
        packets: Sequence[_Packet],
        *,
        packet_kind: str,
        character: str,
    ) -> list[_Packet]:
        output = [
            _Packet(packet.kind, packet.frames.clone(), packet.variants)
            for packet in packets
        ]
        index = self._packet_positions(output, packet_kind)[0]
        variant = output[index].variants[1]
        if variant is None:
            raise RuntimeError(f"{packet_kind} content unexpectedly blank")
        output[index].frames[1] = self._render_source(character, variant)
        return output

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.seed + self.epoch * 10_000_019 + index * 104_729)
        heldout_combination = (
            self.split == "development"
            and rng.random() < self.config.development_heldout_fraction
        )
        combinations = (
            self.heldout_combinations
            if heldout_combination
            else self.seen_combinations
        )
        operation, canonical_pair = rng.choice(combinations)
        labels = canonical_pair if rng.random() < 0.5 else canonical_pair[::-1]
        query_index = rng.randrange(2)
        target_index = query_index if operation == "同" else 1 - query_index
        other_operation = "异" if operation == "同" else "同"

        heldout_length = (
            self.split == "development"
            and rng.random() < self.config.development_heldout_length_fraction
        )
        distractor_count = (
            MAX_DISTRACTORS
            if heldout_length
            else rng.randrange(MAX_TRAIN_DISTRACTORS + 1)
        )
        sampled = rng.sample(self.characters, 2 + MAX_DISTRACTORS + 1)
        glyphs = tuple(sampled[:2])
        distractors = sampled[2 : 2 + distractor_count]
        spare_distractor = sampled[-1]

        packets = [
            self._packet(
                rng,
                kind="pair",
                header=PAIR_HEADER,
                content_a=label,
                content_b=glyph,
            )
            for label, glyph in zip(labels, glyphs, strict=True)
        ]
        source_glyph_images = tuple(packet.frames[2].clone() for packet in packets)
        packets.append(
            self._packet(
                rng,
                kind="operation",
                header=OPERATION_HEADER,
                content_a=operation,
                content_b=None,
            )
        )
        packets.append(
            self._packet(
                rng,
                kind="query",
                header=QUERY_HEADER,
                content_a=labels[query_index],
                content_b=None,
            )
        )
        packets.extend(
            self._packet(
                rng,
                kind="distractor",
                header=DISTRACTOR_HEADER,
                content_a=character,
                content_b=None,
            )
            for character in distractors
        )
        rng.shuffle(packets)
        packets.append(
            self._packet(
                rng,
                kind="end",
                header=END_HEADER,
                content_a=None,
                content_b=None,
            )
        )

        query_counterfactual_packets = self._replace_content(
            packets,
            packet_kind="query",
            character=labels[1 - query_index],
        )
        operation_counterfactual_packets = self._replace_content(
            packets,
            packet_kind="operation",
            character=other_operation,
        )

        permuted_body = list(packets[:-1])
        original_kinds = tuple(packet.kind for packet in permuted_body)
        rng.shuffle(permuted_body)
        if len(permuted_body) > 1 and tuple(
            packet.kind for packet in permuted_body
        ) == original_kinds:
            permuted_body = permuted_body[1:] + permuted_body[:1]
        permuted_packets = permuted_body + [packets[-1]]

        if distractor_count < MAX_DISTRACTORS:
            distractor_packet = self._packet(
                rng,
                kind="distractor",
                header=DISTRACTOR_HEADER,
                content_a=spare_distractor,
                content_b=None,
            )
            insertion = rng.randrange(len(packets))
            distractor_counterfactual_packets = (
                list(packets[:insertion])
                + [distractor_packet]
                + list(packets[insertion:])
            )
            distractor_counterfactual_kind = "insert"
        else:
            distractor_counterfactual_packets = [
                _Packet(packet.kind, packet.frames.clone(), packet.variants)
                for packet in packets
            ]
            distractor_packet_index = self._packet_positions(
                distractor_counterfactual_packets, "distractor"
            )[0]
            variant = distractor_counterfactual_packets[
                distractor_packet_index
            ].variants[1]
            if variant is None:
                raise RuntimeError("distractor content unexpectedly blank")
            distractor_counterfactual_packets[distractor_packet_index].frames[1] = (
                self._render_source(spare_distractor, variant)
            )
            distractor_counterfactual_kind = "replace"

        canonical_glyphs = tuple(self._render_target(glyph) for glyph in glyphs)
        canonical_labels = tuple(self._render_target(label) for label in labels)
        other_target_index = 1 - target_index

        def target_stream(selected_index: int) -> torch.Tensor:
            return torch.stack(
                (canonical_glyphs[selected_index], canonical_labels[selected_index])
            )

        return {
            "prompt": self._stack_packets(packets),
            "target_stream": target_stream(target_index),
            "query_counterfactual_prompt": self._stack_packets(
                query_counterfactual_packets
            ),
            "query_counterfactual_target_stream": target_stream(other_target_index),
            "operation_counterfactual_prompt": self._stack_packets(
                operation_counterfactual_packets
            ),
            "operation_counterfactual_target_stream": target_stream(
                other_target_index
            ),
            "permuted_prompt": self._stack_packets(permuted_packets),
            "permuted_target_stream": target_stream(target_index),
            "distractor_counterfactual_prompt": self._stack_packets(
                distractor_counterfactual_packets
            ),
            "distractor_counterfactual_target_stream": target_stream(target_index),
            "history_override_frame": canonical_glyphs[other_target_index],
            "history_override_target": canonical_labels[other_target_index],
            "teacher_forced_frame": canonical_glyphs[target_index],
            "localization_target": self._localization_target(packets),
            "query_counterfactual_localization_target": self._localization_target(
                query_counterfactual_packets
            ),
            "operation_counterfactual_localization_target": (
                self._localization_target(operation_counterfactual_packets)
            ),
            "oracle_reference": source_glyph_images[target_index],
            "counterfactual_oracle_reference": source_glyph_images[
                other_target_index
            ],
            "metadata": {
                "operation": operation,
                "labels": labels,
                "glyphs": glyphs,
                "query_index": query_index,
                "target_index": target_index,
                "target_character": glyphs[target_index],
                "target_label": labels[target_index],
                "counterfactual_target_character": glyphs[other_target_index],
                "counterfactual_target_label": labels[other_target_index],
                "heldout_combination": heldout_combination,
                "heldout_length": heldout_length,
                "distractor_count": distractor_count,
                "active_packets": len(packets),
                "packet_kinds": tuple(packet.kind for packet in packets),
                "permuted_packet_kinds": tuple(
                    packet.kind for packet in permuted_packets
                ),
                "distractor_counterfactual_kind": (
                    distractor_counterfactual_kind
                ),
            },
        }


PROMPT_KEYS = (
    "prompt",
    "query_counterfactual_prompt",
    "operation_counterfactual_prompt",
    "permuted_prompt",
    "distractor_counterfactual_prompt",
)


def _pad_prompts(prompts: Sequence[torch.Tensor]) -> torch.Tensor:
    maximum = max(prompt.shape[0] for prompt in prompts)
    if maximum % FRAMES_PER_PACKET:
        raise ValueError("prompt padding boundary is not packet aligned")
    if maximum > MAX_PACKETS * FRAMES_PER_PACKET:
        raise ValueError("prompt exceeds the V24 maximum length")
    output = prompts[0].new_zeros(
        len(prompts), maximum, 1, prompts[0].shape[-2], prompts[0].shape[-1]
    )
    for index, prompt in enumerate(prompts):
        if prompt.shape[0] % FRAMES_PER_PACKET:
            raise ValueError("prompt is not packet aligned")
        output[index, : prompt.shape[0]] = prompt
    return output


def visual_packet_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty visual packet batch")
    output: dict[str, Any] = {
        key: _pad_prompts([item[key] for item in batch]) for key in PROMPT_KEYS
    }
    tensor_keys = (
        "target_stream",
        "query_counterfactual_target_stream",
        "operation_counterfactual_target_stream",
        "permuted_target_stream",
        "distractor_counterfactual_target_stream",
        "history_override_frame",
        "history_override_target",
        "teacher_forced_frame",
        "localization_target",
        "query_counterfactual_localization_target",
        "operation_counterfactual_localization_target",
        "oracle_reference",
        "counterfactual_oracle_reference",
    )
    output.update(
        {
            key: torch.stack([item[key] for item in batch])
            for key in tensor_keys
        }
    )
    output["metadata"] = [item["metadata"] for item in batch]
    return output
