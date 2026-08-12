from __future__ import annotations

import torch
import pytest

from ilm.visual_lm.ink_jepa_data import VisualGrammarRecord
from ilm.visual_lm.visual_binding_data import (
    HELD_OUT_COMBINATIONS,
    LABEL_PAIRS,
    MARKER_CHARACTERS,
    OPERATIONS,
    VisualBindingEpisodeConfig,
    VisualBindingEpisodeDataset,
    binding_partition_receipt,
    build_binding_character_bank,
    split_binding_characters,
    visual_binding_collate,
)


pytestmark = pytest.mark.filterwarnings(
    "ignore:The py23 module has been deprecated:DeprecationWarning"
)


def grammar_records() -> list[VisualGrammarRecord]:
    text = "的一是在不了有和人这中大为上个国我以要他"
    return [
        VisualGrammarRecord(
            identifier="test:0",
            text=text * 4,
            language="zh",
            source="test",
            rights="test-only",
        )
    ]


def test_character_bank_excludes_task_markers() -> None:
    bank = build_binding_character_bank(grammar_records(), bank_size=16)
    assert len(bank) == 16
    assert not set(bank) & MARKER_CHARACTERS


def test_identity_partition_is_deterministic_and_disjoint() -> None:
    characters = [chr(0x4E00 + index) for index in range(1_024)]
    first = split_binding_characters(characters)
    second = split_binding_characters(characters)
    assert first == second
    assert set(first["train"]).isdisjoint(first["development"])
    assert set(first["train"]).isdisjoint(first["frozen"])
    assert set(first["development"]).isdisjoint(first["frozen"])
    assert sum(map(len, first.values())) == len(characters)
    receipt = binding_partition_receipt(first)
    assert receipt["train_identities"] == len(first["train"])
    assert receipt["frozen_images_instantiated"] is False
    assert len(receipt["frozen_identifiers_sha256"]) == 64


def test_training_dataset_never_uses_heldout_combination() -> None:
    dataset = VisualBindingEpisodeDataset(
        tuple("的一是在不了有和人这中大为上个国我以要他"),
        split="train",
        length=32,
        config=VisualBindingEpisodeConfig(),
        seed=17,
    )
    for index in range(len(dataset)):
        item = dataset[index]
        metadata = item["metadata"]
        assert metadata["heldout_combination"] is False
        key = (metadata["operation"], "".join(sorted(metadata["labels"])))
        normalized_heldout = {
            (operation, "".join(sorted(pair)))
            for operation, pair in HELD_OUT_COMBINATIONS
        }
        assert key not in normalized_heldout


def test_development_dataset_can_force_heldout_combinations() -> None:
    dataset = VisualBindingEpisodeDataset(
        tuple("的一是在不了有和人这中大为上个国我以要他"),
        split="development",
        length=12,
        config=VisualBindingEpisodeConfig(development_heldout_fraction=1.0),
        seed=23,
    )
    normalized_heldout = {
        (operation, "".join(sorted(pair)))
        for operation, pair in HELD_OUT_COMBINATIONS
    }
    for index in range(len(dataset)):
        metadata = dataset[index]["metadata"]
        key = (metadata["operation"], "".join(sorted(metadata["labels"])))
        assert metadata["heldout_combination"] is True
        assert key in normalized_heldout


def test_episode_shapes_and_counterfactual_change_only_query_frame() -> None:
    dataset = VisualBindingEpisodeDataset(
        tuple("的一是在不了有和人这中大为上个国我以要他"),
        split="train",
        length=2,
        config=VisualBindingEpisodeConfig(),
        seed=31,
    )
    item = dataset[0]
    assert item["prompt"].shape == (6, 1, 32, 32)
    assert item["target"].shape == (1, 32, 32)
    torch.testing.assert_close(
        item["prompt"][:-1],
        item["counterfactual_prompt"][:-1],
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(
        item["prompt"][-1],
        item["counterfactual_prompt"][-1],
    )
    assert not torch.equal(item["target"], item["counterfactual_target"])
    assert item["metadata"]["target_character"] != item["metadata"][
        "counterfactual_target_character"
    ]


def test_collate_keeps_symbolic_metadata_outside_tensor_inputs() -> None:
    dataset = VisualBindingEpisodeDataset(
        tuple("的一是在不了有和人这中大为上个国我以要他"),
        split="train",
        length=2,
        config=VisualBindingEpisodeConfig(),
        seed=41,
    )
    batch = visual_binding_collate([dataset[0], dataset[1]])
    assert batch["prompt"].shape == (2, 6, 1, 32, 32)
    assert batch["target"].shape == (2, 1, 32, 32)
    for key, value in batch.items():
        if key == "metadata":
            assert isinstance(value, list)
            continue
        assert isinstance(value, torch.Tensor)
        assert value.dtype.is_floating_point


def test_declared_task_grammar_is_complete() -> None:
    assert len(LABEL_PAIRS) == 4
    assert OPERATIONS == ("同", "异")
    assert HELD_OUT_COMBINATIONS == frozenset({("同", "天地"), ("异", "左右")})
