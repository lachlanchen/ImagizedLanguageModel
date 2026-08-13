from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from ilm.visual_lm.conditional_visual_density_ratio import (
    ConditionalVisualDensityRatioConfig,
    ConditionalVisualDensityRatioModel,
)
from ilm.visual_lm.spatial_visual_next_field import (
    V30_GLOBAL_ROUTE,
    V30_SPATIAL_ROUTE,
    SpatialVisualNextFieldConfig,
    SpatialVisualNextFieldModel,
    model_state_sha256,
    spatial_visual_next_field_boundary_receipt,
    spatially_permute_candidate_fields,
)


def _small_config(*, route: str = V30_SPATIAL_ROUTE) -> SpatialVisualNextFieldConfig:
    return SpatialVisualNextFieldConfig(
        visual_dim=72,
        semantic_dim=72,
        model_dim=128,
        layers=2,
        heads=4,
        mlp_ratio=2.0,
        retina_base_channels=24,
        semantic_hidden_dim=96,
        field_channels=72,
        decoder_hidden_channels=72,
        score_chunk_size=3,
        route_mode=route,
    )


def _small_model(*, route: str = V30_SPATIAL_ROUTE) -> SpatialVisualNextFieldModel:
    return SpatialVisualNextFieldModel(_small_config(route=route))


def _v29_source() -> ConditionalVisualDensityRatioModel:
    return ConditionalVisualDensityRatioModel(
        ConditionalVisualDensityRatioConfig(
            visual_dim=72,
            semantic_dim=72,
            model_dim=128,
            layers=2,
            heads=4,
            mlp_ratio=2.0,
            retina_base_channels=24,
            semantic_hidden_dim=96,
            evidence_layers=1,
            evidence_heads=4,
            evidence_mlp_ratio=2.0,
            relation_hidden_dim=128,
            score_chunk_size=3,
        )
    )


def test_v30_predicts_candidate_independent_continuous_field() -> None:
    model = _small_model().eval()
    context = torch.rand(2, 7, 1, 32, 32)
    with torch.no_grad():
        field = model.predict_field(context)
    assert field.shape == (2, 16, 72)
    assert torch.isfinite(field).all()
    assert torch.allclose(field.norm(dim=-1), torch.ones(2, 16), atol=1e-5)


def test_v30_scores_shared_batched_and_paired_candidate_images() -> None:
    model = _small_model().eval()
    context = torch.rand(2, 7, 1, 32, 32)
    candidates = torch.rand(5, 1, 32, 32)
    paired_contexts = context[:, None].expand(-1, 2, -1, -1, -1, -1)
    paired_candidates = torch.rand(2, 2, 1, 32, 32)
    with torch.no_grad():
        predicted = model.predict_field(context)
        shared_fields = model.encode_route_candidates(candidates)
        shared = model.score_encoded_shared(predicted, shared_fields)
        batched = model.score_encoded_batched(
            predicted,
            shared_fields[None].expand(2, -1, -1, -1),
        )
        paired = model.score_paired_candidates(paired_contexts, paired_candidates)
    assert shared.shape == (2, 5)
    assert torch.allclose(shared, batched, atol=1e-6, rtol=1e-6)
    assert paired.shape == (2, 2, 2)


def test_v30_shared_score_is_chunk_invariant() -> None:
    model = _small_model().eval()
    context = torch.rand(2, 8, 1, 32, 32)
    candidates = torch.rand(7, 1, 32, 32)
    with torch.no_grad():
        predicted = model.predict_field(context)
        fields = model.encode_route_candidates(candidates)
        one = model.score_encoded_shared(predicted, fields, chunk_size=1)
        full = model.score_encoded_shared(predicted, fields, chunk_size=7)
    assert torch.allclose(one, full, atol=1e-6, rtol=1e-6)


def test_v30_candidate_column_permutation_is_equivariant() -> None:
    model = _small_model().eval()
    contexts = torch.rand(3, 2, 8, 1, 32, 32)
    candidates = torch.rand(3, 2, 1, 32, 32)
    with torch.no_grad():
        original = model.score_paired_candidates(contexts, candidates)
        swapped = model.score_paired_candidates(contexts, candidates.flip(1))
    assert torch.allclose(swapped, original.flip(-1), atol=1e-6, rtol=1e-6)


def test_v30_exact_suffix_has_exact_score_rows() -> None:
    model = _small_model().eval()
    suffix = torch.rand(2, 4, 1, 32, 32)
    contexts = suffix[:, None].expand(-1, 2, -1, -1, -1, -1).clone()
    candidates = torch.rand(2, 2, 1, 32, 32)
    with torch.no_grad():
        score = model.score_exact_suffix_paired(contexts, candidates)
    assert torch.equal(score[:, 0], score[:, 1])


def test_v30_spatial_permutation_changes_only_spatial_candidate_route() -> None:
    spatial = _small_model(route=V30_SPATIAL_ROUTE).eval()
    global_control = _small_model(route=V30_GLOBAL_ROUTE).eval()
    images = torch.rand(4, 1, 32, 32)
    with torch.no_grad():
        spatial_fields = spatial.encode_route_candidates(images)
        global_fields = global_control.encode_route_candidates(images)
    assert not torch.equal(
        spatial_fields,
        spatially_permute_candidate_fields(spatial_fields),
    )
    assert torch.equal(
        global_fields,
        spatially_permute_candidate_fields(global_fields),
    )


def test_v30_routes_have_identical_parameter_structure_and_initial_state() -> None:
    torch.manual_seed(117)
    spatial = _small_model(route=V30_SPATIAL_ROUTE)
    torch.manual_seed(117)
    global_control = _small_model(route=V30_GLOBAL_ROUTE)
    spatial_state = spatial.state_dict()
    global_state = global_control.state_dict()
    assert spatial_state.keys() == global_state.keys()
    assert model_state_sha256(spatial_state) == model_state_sha256(global_state)
    for name in spatial_state:
        assert torch.equal(spatial_state[name], global_state[name])


def test_v30_loads_only_v29_perception_and_context_backbone() -> None:
    source = _v29_source()
    model = _small_model()
    decoder_before = model_state_sha256(model.field_decoder.state_dict())
    receipt = model.load_v29_backbone_state(source.state_dict())
    assert receipt["discarded_candidate_critic"] is True
    assert decoder_before == model_state_sha256(model.field_decoder.state_dict())
    for name in model._BACKBONE_MODULES:
        source_state = getattr(source, name).state_dict()
        target_state = getattr(model, name).state_dict()
        assert source_state.keys() == target_state.keys()
        for key in source_state:
            assert torch.equal(source_state[key], target_state[key])
    assert all(not parameter.requires_grad for parameter in model.retina.parameters())
    assert all(
        not parameter.requires_grad
        for parameter in model.semantic_adapter.parameters()
    )


def test_v30_boundary_and_config_reject_symbolic_or_unmatched_routes() -> None:
    model = _small_model()
    with pytest.raises(TypeError, match="floating image"):
        model.predict_field(torch.ones(1, 4, 1, 32, 32, dtype=torch.long))
    with pytest.raises(ValueError, match="route"):
        replace(_small_config(), route_mode="lookup")
    receipt = spatial_visual_next_field_boundary_receipt(_small_config())
    for key in (
        "input_is_continuous_image_stream",
        "output_is_candidate_independent_continuous_field",
        "candidate_is_arbitrary_image",
        "candidate_reduction_occurs_after_local_interaction",
        "retina_is_frozen",
        "semantic_adapters_are_frozen",
    ):
        assert receipt[key] is True
    for key in (
        "uses_strings",
        "uses_token_ids",
        "uses_unicode_ids",
        "uses_character_ids",
        "uses_vocabulary_embedding",
        "uses_vocabulary_output",
        "uses_ocr",
        "uses_visual_codebook",
        "uses_glyph_lookup",
        "uses_external_language_model",
        "candidate_bank_deployed",
        "candidate_bank_in_model_state",
    ):
        assert receipt[key] is False
