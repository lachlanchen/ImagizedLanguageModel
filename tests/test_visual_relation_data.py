from __future__ import annotations

import torch

from ilm.visual_lm.ink_jepa_data import load_visual_grammar_manifest
from ilm.visual_lm.visual_relation_data import (
    PARTITION_SALT,
    VisualRelationEpisodeDataset,
    build_relation_character_bank,
    relation_partition_receipt,
    split_relation_characters,
    visual_relation_collate,
)


EXPECTED_PARTITION = {
    "train_identities": 817,
    "development_identities": 109,
    "frozen_identities": 98,
    "development_identifiers_sha256": (
        "6e89f898a17028125a060deec8249bbf35b4d02f898f716f2f519a29cd314170"
    ),
    "frozen_identifiers_sha256": (
        "206efd6fa2a0e640368a178c61f2f82ee737260afcaed6e97226bfef1f366d0c"
    ),
}


def test_v23_partition_matches_preregistration_without_rendering() -> None:
    records = load_visual_grammar_manifest(
        "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
    )
    bank = build_relation_character_bank(records)
    partitions = split_relation_characters(bank)
    receipt = relation_partition_receipt(partitions)
    assert receipt["salt"] == PARTITION_SALT
    assert receipt["frozen_images_instantiated"] is False
    for key, expected in EXPECTED_PARTITION.items():
        assert receipt[key] == expected


def test_relation_episode_counterfactuals_change_only_declared_frames() -> None:
    dataset = VisualRelationEpisodeDataset(
        ("中", "文", "学", "习"),
        split="development",
        length=2,
        seed=97,
    )
    item = dataset[0]
    assert item["prompt"].shape == (6, 1, 32, 32)
    assert item["target"].shape == (1, 32, 32)
    torch.testing.assert_close(
        item["prompt"][:5], item["query_counterfactual_prompt"][:5]
    )
    assert not torch.equal(
        item["prompt"][5], item["query_counterfactual_prompt"][5]
    )
    torch.testing.assert_close(
        item["prompt"][:4], item["operation_counterfactual_prompt"][:4]
    )
    torch.testing.assert_close(
        item["prompt"][5], item["operation_counterfactual_prompt"][5]
    )
    assert not torch.equal(
        item["prompt"][4], item["operation_counterfactual_prompt"][4]
    )
    torch.testing.assert_close(item["prompt"][0], item["pair_swapped_prompt"][2])
    torch.testing.assert_close(item["prompt"][1], item["pair_swapped_prompt"][3])
    torch.testing.assert_close(item["prompt"][2], item["pair_swapped_prompt"][0])
    torch.testing.assert_close(item["prompt"][3], item["pair_swapped_prompt"][1])
    torch.testing.assert_close(item["target"], item["pair_swapped_target"])
    torch.testing.assert_close(
        item["query_counterfactual_target"],
        item["operation_counterfactual_target"],
    )
    assert not torch.equal(item["target"], item["query_counterfactual_target"])

    batch = visual_relation_collate([item, dataset[1]])
    assert batch["prompt"].shape == (2, 6, 1, 32, 32)
    assert batch["target"].shape == (2, 1, 32, 32)
    assert len(batch["metadata"]) == 2
