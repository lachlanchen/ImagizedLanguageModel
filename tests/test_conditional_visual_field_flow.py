from __future__ import annotations

import pytest
import torch

from ilm.visual_lm.conditional_visual_field_flow import (
    V31_GLOBAL_ROUTE,
    V31_SPATIAL_ROUTE,
    ConditionalVisualFieldFlowConfig,
    ConditionalVisualFieldFlowModel,
    conditional_visual_field_flow_boundary_receipt,
    model_state_sha256,
    spatially_permute_v31_fields,
)
from ilm.visual_lm.spatial_visual_next_field import (
    SpatialVisualNextFieldConfig,
    SpatialVisualNextFieldModel,
)


def _config(*, route: str = V31_SPATIAL_ROUTE) -> ConditionalVisualFieldFlowConfig:
    return ConditionalVisualFieldFlowConfig(
        visual_dim=72,
        semantic_dim=72,
        model_dim=128,
        layers=1,
        heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        retina_base_channels=24,
        semantic_hidden_dim=96,
        field_channels=72,
        velocity_hidden_channels=72,
        velocity_blocks=2,
        velocity_dropout=0.0,
        time_embedding_dim=32,
        score_chunk_size=3,
        route_mode=route,
    )


def _model(*, route: str = V31_SPATIAL_ROUTE) -> ConditionalVisualFieldFlowModel:
    return ConditionalVisualFieldFlowModel(_config(route=route))


def _probes(
    model: ConditionalVisualFieldFlowModel,
) -> tuple[torch.Tensor, torch.Tensor]:
    return model.make_coherent_base(2), torch.tensor([0.10, 0.35])


def test_v31_coherent_base_repeats_one_global_choice() -> None:
    model = _model()
    base = model.make_coherent_base(3)
    assert base.shape == (3, 16, 72)
    assert torch.equal(base, base[:, :1].expand_as(base))
    assert torch.allclose(base[:, 0].norm(dim=-1), torch.ones(3), atol=1e-6)


def test_v31_autonomous_sampler_is_candidate_independent_and_normalized() -> None:
    model = _model().eval()
    context = torch.rand(2, 7, 1, 32, 32)
    base_vectors = torch.randn(3, 72)
    with torch.no_grad():
        sample = model.sample(context, base_vectors, steps=2)
    assert sample.shape == (2, 3, 16, 72)
    assert torch.isfinite(sample).all()
    assert torch.allclose(sample.norm(dim=-1), torch.ones(2, 3, 16), atol=1e-5)


def test_v31_shared_path_score_is_chunk_invariant() -> None:
    model = _model().eval()
    context = torch.rand(2, 7, 1, 32, 32)
    candidates = torch.rand(5, 1, 32, 32)
    probes, times = _probes(model)
    with torch.no_grad():
        condition = model.context_condition(context)
        fields = model.encode_route_candidates(candidates)
        one = model.path_score_encoded_shared(
            condition, fields, probes, times, chunk_size=1
        )
        full = model.path_score_encoded_shared(
            condition, fields, probes, times, chunk_size=5
        )
    assert one.shape == (2, 5)
    assert torch.allclose(one, full, atol=1e-6, rtol=1e-6)


def test_v31_candidate_column_permutation_is_equivariant() -> None:
    model = _model().eval()
    contexts = torch.rand(2, 2, 8, 1, 32, 32)
    candidates = torch.rand(2, 2, 1, 32, 32)
    probes, times = _probes(model)
    with torch.no_grad():
        original = model.path_score_paired_candidates(
            contexts, candidates, probes, times
        )
        swapped = model.path_score_paired_candidates(
            contexts, candidates.flip(1), probes, times
        )
    assert torch.allclose(swapped, original.flip(-1), atol=1e-6, rtol=1e-6)


def test_v31_exact_suffix_has_bitwise_equal_score_rows() -> None:
    model = _model().eval()
    suffix = torch.rand(2, 4, 1, 32, 32)
    contexts = suffix[:, None].expand(-1, 2, -1, -1, -1, -1).clone()
    candidates = torch.rand(2, 2, 1, 32, 32)
    probes, times = _probes(model)
    with torch.no_grad():
        scores = model.path_score_exact_suffix_paired(
            contexts, candidates, probes, times
        )
    assert torch.equal(scores[:, 0], scores[:, 1])


def test_v31_spatial_permutation_changes_only_spatial_route_targets() -> None:
    spatial = _model(route=V31_SPATIAL_ROUTE).eval()
    global_control = _model(route=V31_GLOBAL_ROUTE).eval()
    images = torch.rand(4, 1, 32, 32)
    with torch.no_grad():
        spatial_fields = spatial.encode_route_candidates(images)
        global_fields = global_control.encode_route_candidates(images)
    assert not torch.equal(spatial_fields, spatially_permute_v31_fields(spatial_fields))
    assert torch.equal(global_fields, spatially_permute_v31_fields(global_fields))


def test_v31_routes_have_byte_identical_initial_states() -> None:
    torch.manual_seed(311)
    spatial = _model(route=V31_SPATIAL_ROUTE)
    torch.manual_seed(311)
    global_control = _model(route=V31_GLOBAL_ROUTE)
    assert model_state_sha256(spatial.state_dict()) == model_state_sha256(
        global_control.state_dict()
    )
    for name, value in spatial.state_dict().items():
        assert torch.equal(value, global_control.state_dict()[name])


def test_v31_loads_only_v30_backbone() -> None:
    source = SpatialVisualNextFieldModel(
        SpatialVisualNextFieldConfig(
            visual_dim=72,
            semantic_dim=72,
            model_dim=128,
            layers=1,
            heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            retina_base_channels=24,
            semantic_hidden_dim=96,
            field_channels=72,
            decoder_hidden_channels=72,
            decoder_dropout=0.0,
        )
    )
    model = _model()
    velocity_before = model_state_sha256(model.velocity_decoder.state_dict())
    receipt = model.load_v30_backbone_state(source.state_dict())
    assert receipt["discarded_v30_field_decoder"] is True
    assert velocity_before == model_state_sha256(model.velocity_decoder.state_dict())
    for name in model._BACKBONE_MODULES:
        for key, value in getattr(source, name).state_dict().items():
            assert torch.equal(value, getattr(model, name).state_dict()[key])


def test_v31_boundary_rejects_symbolic_input_and_has_no_bank() -> None:
    model = _model()
    with pytest.raises(TypeError, match="floating image"):
        model.context_condition(torch.ones(1, 4, 1, 32, 32, dtype=torch.long))
    boundary = conditional_visual_field_flow_boundary_receipt(model.config)
    assert boundary["output_is_candidate_independent_continuous_distribution"]
    assert not boundary["autonomous_sampler_requires_candidates"]
    assert not boundary["candidate_bank_deployed"]
    assert not boundary["uses_token_ids"]
