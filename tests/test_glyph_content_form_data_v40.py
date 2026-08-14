from __future__ import annotations

import torch

from ilm.visual_lm.continuous_glyph_codec_data import (
    HistoricGlyphRecord,
    historic_character_partition,
)
from ilm.visual_lm.glyph_content_form_data import (
    CrossEraContentFormDataset,
    glyph_content_form_collate,
    glyph_content_form_data_boundary_receipt,
    glyph_content_form_stage_ids,
    glyph_content_form_student_batch,
)


def _records_in_one_split() -> tuple[list[HistoricGlyphRecord], str]:
    by_split: dict[str, list[str]] = {
        "train": [],
        "development": [],
        "sealed": [],
    }
    codepoint = 0x4E00
    while max(map(len, by_split.values())) < 4:
        character = chr(codepoint)
        codepoint += 1
        by_split[historic_character_partition(character)].append(character)
    split = max(by_split, key=lambda name: len(by_split[name]))
    records: list[HistoricGlyphRecord] = []
    for family_index, character in enumerate(by_split[split][:4]):
        for stage in ("oracle", "bronze", "seal"):
            records.append(
                HistoricGlyphRecord(
                    character=character,
                    stage=stage,
                    label=f"{stage}-{family_index}",
                    local_path=f"data/{stage}-{family_index}.svg",
                )
            )
    return records, split


def test_cross_era_dataset_uses_other_families_as_style_references() -> None:
    records, split = _records_in_one_split()
    pixels = torch.randint(0, 2, (len(records), 1, 32, 32), dtype=torch.uint8)
    dataset = CrossEraContentFormDataset(
        records,
        pixels,
        split=split,
        length=12,
        seed=40,
    )

    first = dataset[0]
    repeated = dataset[0]

    assert first["metadata"] == repeated["metadata"]
    assert torch.equal(first["anchor_pixels"], repeated["anchor_pixels"])
    assert first["metadata"]["anchor_stage"] != first["metadata"]["positive_stage"]
    assert first["metadata"]["anchor_style_character"] != first["metadata"]["character"]
    assert first["metadata"]["positive_style_character"] != first["metadata"]["character"]


def test_v40_collate_excludes_host_labels_from_model_inputs() -> None:
    records, split = _records_in_one_split()
    pixels = torch.randint(0, 2, (len(records), 1, 32, 32), dtype=torch.uint8)
    dataset = CrossEraContentFormDataset(
        records,
        pixels,
        split=split,
        length=8,
        seed=41,
    )
    batch = glyph_content_form_collate([dataset[0], dataset[1]])
    student = glyph_content_form_student_batch(batch)
    stage_ids, mapping = glyph_content_form_stage_ids(batch["metadata"])
    receipt = glyph_content_form_data_boundary_receipt(batch)

    assert all(value.shape == (2, 1, 32, 32) for value in student.values())
    assert stage_ids.shape == (8,)
    assert mapping
    assert receipt["metadata_excluded_from_model"] is True
    assert receipt["family_labels_are_model_inputs"] is False
    assert receipt["stage_labels_are_model_inputs"] is False

