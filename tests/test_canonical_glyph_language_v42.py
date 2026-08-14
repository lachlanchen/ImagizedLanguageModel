from __future__ import annotations

import pytest
import torch

from ilm.visual_lm.canonical_glyph_language import (
    CanonicalGlyphLanguageConfig,
    CanonicalGlyphLanguageModel,
    OrthonormalGlyphField,
    canonical_glyph_language_boundary_receipt,
    orthonormal_dct_matrix,
)
from ilm.visual_lm.canonical_glyph_language_data import (
    CanonicalGlyphLanguageDataset,
    CanonicalGlyphRenderConfig,
    canonical_glyph_collate,
    canonical_glyph_data_boundary_receipt,
    canonical_glyph_student_batch,
)
from ilm.visual_lm.canonical_glyph_language_evaluation import (
    canonical_language_boundary_is_clean,
)
from ilm.visual_lm.canonical_glyph_language_training import (
    canonical_glyph_language_loss,
    dynamic_visual_contrastive_loss,
    empirical_energy_score,
    exact_field_positive_mask,
)
from ilm.visual_lm.ink_jepa_data import VisualGrammarRecord
from ilm.visual_lm.visual_cell_data import visual_cell_partition


def _identifier_for(split: str) -> str:
    for index in range(100_000):
        identifier = f"v42-test-record-{index}"
        if visual_cell_partition(identifier) == split:
            return identifier
    raise AssertionError(f"could not find a V42 {split} identifier")


def _record(split: str = "train") -> VisualGrammarRecord:
    writing = (
        "天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往秋收冬藏"
        "闰余成岁律吕调阳云腾致雨露结为霜金生丽水玉出昆冈"
        "剑号巨阙珠称夜光果珍李柰菜重芥姜海咸河淡鳞潜羽翔"
    )
    return VisualGrammarRecord(
        identifier=_identifier_for(split),
        text=writing,
        language="zh-Hans",
        source="unit-test",
        rights="test-only",
    )


def _small_config() -> CanonicalGlyphLanguageConfig:
    return CanonicalGlyphLanguageConfig(
        model_dim=128,
        layers=2,
        heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        noise_dim=32,
        generator_layers=2,
    )


def test_orthonormal_dct_and_binary_field_round_trip_are_exact() -> None:
    basis = orthonormal_dct_matrix(32)
    identity = basis @ basis.transpose(0, 1)
    torch.testing.assert_close(identity, torch.eye(32), atol=2e-6, rtol=0.0)

    pixels = torch.zeros(3, 1, 32, 32)
    pixels[0, 0, 4:28, 14:18] = 1.0
    pixels[1, 0, 14:18, 4:28] = 1.0
    pixels[2, 0].diagonal().fill_(1.0)
    field = OrthonormalGlyphField()
    encoded = field.encode(pixels)
    unit = field.normalize(encoded)
    assert encoded.shape == (3, 1024)
    torch.testing.assert_close(
        encoded.norm(dim=-1), torch.full((3,), 32.0), atol=2e-5, rtol=0.0
    )
    assert torch.equal(field.binary(unit), pixels)


def test_dataset_removes_every_symbolic_value_from_student_batch() -> None:
    dataset = CanonicalGlyphLanguageDataset(
        [_record("train")],
        split="train",
        render_config=CanonicalGlyphRenderConfig(script_views="original"),
        seed=19,
        length=2,
        expose_evaluation_labels=True,
    )
    batch = canonical_glyph_collate([dataset[0], dataset[1]])
    assert batch["context"].shape == (2, 64, 1, 32, 32)
    assert batch["target"].shape == (2, 64, 1, 32, 32)
    assert "target_characters" in batch["metadata"][0]
    student = canonical_glyph_student_batch(batch)
    assert set(student) == {"context", "target"}
    assert all(value.is_floating_point() for value in student.values())

    receipt = canonical_glyph_data_boundary_receipt()
    assert receipt["one_canonical_font"] is True
    assert receipt["metadata_excluded_from_student"] is True
    assert all(
        receipt[key] is False
        for key in (
            "uses_strings",
            "uses_token_ids",
            "uses_unicode_ids",
            "uses_character_ids",
            "uses_ocr",
            "uses_visual_codebook",
            "uses_glyph_lookup",
            "uses_external_language_model",
            "candidate_bank_deployed",
        )
    )


def test_model_is_causal_and_has_a_clean_image_only_boundary() -> None:
    torch.manual_seed(7)
    model = CanonicalGlyphLanguageModel(_small_config()).eval()
    context = torch.rand(2, 7, 1, 32, 32)
    changed = context.clone()
    changed[:, -1] = 1.0 - changed[:, -1]
    first = model.language(context)
    second = model.language(changed)
    assert first["hidden_states"].shape == (2, 7, 128)
    assert first["anchor_fields"].shape == (2, 7, 1024)
    torch.testing.assert_close(
        first["anchor_fields"][:, :-1],
        second["anchor_fields"][:, :-1],
        atol=2e-5,
        rtol=2e-5,
    )
    assert not torch.allclose(
        first["anchor_fields"][:, -1], second["anchor_fields"][:, -1]
    )

    receipt = canonical_glyph_language_boundary_receipt(model)
    assert receipt["parameter_names_with_forbidden_fragments"] == []
    assert receipt["field_transform_is_fixed_and_invertible"] is True
    assert receipt["candidate_bank_deployed"] is False
    assert canonical_language_boundary_is_clean(model)


def test_visual_positive_mask_uses_exact_raster_geometry() -> None:
    pixels = torch.zeros(3, 1, 32, 32)
    pixels[0, 0, 4:8, 4:8] = 1.0
    pixels[1] = pixels[0]
    pixels[2] = pixels[0]
    pixels[2, 0, 4, 4] = 0.0
    field = OrthonormalGlyphField()
    targets = field.encode_unit(pixels)
    positives = exact_field_positive_mask(targets)
    assert positives.tolist() == [
        [True, True, False],
        [True, True, False],
        [False, False, True],
    ]
    loss, accuracy = dynamic_visual_contrastive_loss(
        targets,
        targets,
        scale=torch.tensor(20.0),
    )
    assert torch.isfinite(loss)
    assert accuracy == pytest.approx(1.0)


def test_energy_score_and_full_training_loss_are_finite_and_differentiable() -> None:
    torch.manual_seed(11)
    model = CanonicalGlyphLanguageModel(_small_config())
    context = torch.rand(2, 4, 1, 32, 32)
    target = torch.rand(2, 4, 1, 32, 32)
    output = model(context)
    generator = torch.Generator().manual_seed(17)
    measured = canonical_glyph_language_loss(
        model,
        output,
        target,
        generator=generator,
        maximum_contrastive_positions=8,
        maximum_energy_positions=4,
        energy_samples=2,
    )
    assert torch.isfinite(measured.loss)
    assert measured.contrastive_positions == 8
    assert measured.energy_positions == 4
    measured.loss.backward()
    assert model.anchor_head[-1].weight.grad is not None
    assert model.generator.output.weight.grad is not None
    assert torch.isfinite(model.anchor_head[-1].weight.grad).all()

    target_field = model.field.encode_unit(target[:, 0])
    exact = target_field[:, None].expand(-1, 2, -1).clone()
    shifted = -exact
    exact_score = empirical_energy_score(exact, target_field)[0]
    shifted_score = empirical_energy_score(shifted, target_field)[0]
    assert exact_score < shifted_score


def test_bank_free_generation_outputs_and_rereads_binary_images() -> None:
    torch.manual_seed(23)
    model = CanonicalGlyphLanguageModel(_small_config()).eval()
    prefix = torch.rand(1, 5, 1, 32, 32)
    generator = torch.Generator().manual_seed(29)
    sequence, trace = model.generate(
        prefix,
        new_cells=3,
        samples=2,
        generator=generator,
    )
    assert sequence.shape == (1, 8, 1, 32, 32)
    assert trace["generated_cells"].shape == (1, 3, 1, 32, 32)
    assert trace["generated_fields"].shape == (1, 3, 1024)
    assert trace["rereads_generated_pixels"].item() is True
    assert set(torch.unique(trace["generated_cells"]).tolist()).issubset({0.0, 1.0})
