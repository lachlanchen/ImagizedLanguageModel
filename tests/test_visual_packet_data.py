from __future__ import annotations

from collections import Counter

import torch

from ilm.visual_lm.ink_jepa_data import load_visual_grammar_manifest
from ilm.visual_lm.visual_packet_data import (
    FRAMES_PER_PACKET,
    MAX_PACKETS,
    PACKET_MARKER_CHARACTERS,
    PARTITION_SALT,
    VisualPacketEpisodeDataset,
    build_packet_character_bank,
    packet_partition_receipt,
    split_packet_characters,
    visual_packet_collate,
)


EXPECTED_PARTITION = {
    "train_identities": 829,
    "development_identities": 88,
    "frozen_identities": 107,
    "development_identifiers_sha256": (
        "2b611e66778061319bb2502ad850c635b5d89e81e9eab7f9c8ef23a09514e892"
    ),
    "frozen_identifiers_sha256": (
        "d3f6d51ef6c0cb0eeeab664d89e8a2c467bc35ea7482ae55f873f4b28b85c2ab"
    ),
}


def test_v24_partition_matches_preregistration_without_rendering() -> None:
    records = load_visual_grammar_manifest(
        "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
    )
    bank = build_packet_character_bank(records)
    assert not set(bank) & PACKET_MARKER_CHARACTERS
    partitions = split_packet_characters(bank)
    receipt = packet_partition_receipt(partitions)
    assert receipt["salt"] == PARTITION_SALT
    assert receipt["frozen_images_instantiated"] is False
    for key, expected in EXPECTED_PARTITION.items():
        assert receipt[key] == expected


def _packet_multiset(prompt: torch.Tensor) -> Counter[bytes]:
    packets = prompt.reshape(-1, FRAMES_PER_PACKET, 1, 32, 32)
    return Counter(packet.numpy().tobytes() for packet in packets)


def test_v24_episode_is_a_variable_image_stream_with_causal_targets() -> None:
    dataset = VisualPacketEpisodeDataset(
        tuple("中文学习天地玄黄宇宙洪荒日月盈昃辰宿列张"),
        split="development",
        length=4,
        seed=97,
    )
    item = dataset[0]
    prompt = item["prompt"]
    metadata = item["metadata"]
    assert prompt.ndim == 4
    assert prompt.shape[1:] == (1, 32, 32)
    assert prompt.shape[0] == metadata["active_packets"] * FRAMES_PER_PACKET
    assert 5 <= metadata["active_packets"] <= MAX_PACKETS
    assert item["target_stream"].shape == (2, 1, 32, 32)
    assert item["localization_target"].shape == (4, 1, 32, 32)
    assert metadata["packet_kinds"][-1] == "end"
    assert metadata["packet_kinds"].count("pair") == 2
    assert metadata["packet_kinds"].count("query") == 1
    assert metadata["packet_kinds"].count("operation") == 1

    query_packet = metadata["packet_kinds"].index("query")
    operation_packet = metadata["packet_kinds"].index("operation")
    changed_query = torch.any(
        prompt != item["query_counterfactual_prompt"], dim=(1, 2, 3)
    ).nonzero().flatten().tolist()
    changed_operation = torch.any(
        prompt != item["operation_counterfactual_prompt"], dim=(1, 2, 3)
    ).nonzero().flatten().tolist()
    assert changed_query == [query_packet * 3 + 1]
    assert changed_operation == [operation_packet * 3 + 1]
    assert not torch.equal(
        item["target_stream"], item["query_counterfactual_target_stream"]
    )
    assert not torch.equal(
        item["target_stream"], item["operation_counterfactual_target_stream"]
    )
    torch.testing.assert_close(
        item["target_stream"], item["permuted_target_stream"]
    )
    assert _packet_multiset(prompt) == _packet_multiset(item["permuted_prompt"])
    torch.testing.assert_close(
        item["history_override_frame"],
        item["query_counterfactual_target_stream"][0],
    )
    torch.testing.assert_close(
        item["history_override_target"],
        item["query_counterfactual_target_stream"][1],
    )
    assert not torch.equal(
        item["oracle_reference"], item["counterfactual_oracle_reference"]
    )


def test_v24_collate_pads_only_with_blank_image_packets() -> None:
    dataset = VisualPacketEpisodeDataset(
        tuple("中文学习天地玄黄宇宙洪荒日月盈昃辰宿列张"),
        split="train",
        length=32,
        seed=211,
    )
    items = [dataset[index] for index in range(32)]
    by_length: dict[int, dict] = {}
    for item in items:
        by_length.setdefault(item["prompt"].shape[0], item)
    assert len(by_length) >= 2
    selected = [by_length[length] for length in sorted(by_length)[:2]]
    batch = visual_packet_collate(selected)
    maximum = max(item["prompt"].shape[0] for item in selected)
    assert batch["prompt"].shape == (2, maximum, 1, 32, 32)
    assert batch["target_stream"].shape == (2, 2, 1, 32, 32)
    assert "length" not in batch
    assert "mask" not in batch
    for index, item in enumerate(selected):
        active = item["prompt"].shape[0]
        torch.testing.assert_close(batch["prompt"][index, :active], item["prompt"])
        assert torch.count_nonzero(batch["prompt"][index, active:]) == 0
    assert len(batch["metadata"]) == 2
