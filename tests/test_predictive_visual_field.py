from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ilm.visual_lm.predictive_visual_field import (
    PredictiveVisualField,
    PredictiveVisualFieldConfig,
    hyperspherical_flow_path,
    initialize_from_retinal_flow_checkpoint,
    predictive_visual_field_loss,
    sphere_exponential_map,
)
from ilm.visual_lm.retinal_flow_lm import RetinalFlowConfig, RetinalFlowLanguageModel


def small_config(*, flow_geometry: str = "euclidean") -> PredictiveVisualFieldConfig:
    return PredictiveVisualFieldConfig(
        fovea_size=16,
        visual_dim=64,
        state_dim=128,
        state_layers=1,
        retina_base_channels=16,
        dropout=0.0,
        flow_hidden_dim=128,
        flow_blocks=2,
        time_dim=32,
        proposal_hidden_dim=128,
        proposal_blocks=1,
        condition_dropout=0.0,
        sample_temperature=0.10,
        flow_geometry=flow_geometry,
    )


def test_predictive_visual_field_has_strict_state_only_boundary() -> None:
    model = PredictiveVisualField(small_config())

    assert not hasattr(model, "writer")
    assert not hasattr(model, "energy")
    assert not hasattr(model, "embedding")
    assert all(not parameter.requires_grad for parameter in model.retina.parameters())
    assert any(parameter.requires_grad for parameter in model.dynamics.parameters())
    assert any(parameter.requires_grad for parameter in model.visual_proposal.parameters())
    assert any(parameter.requires_grad for parameter in model.state_flow.parameters())

    images = torch.rand(2, 3, 1, model.config.fovea_size, model.config.fovea_size)
    proposal = model.predict(images)["proposal_visual"]
    assert proposal.shape == (2, 3, model.config.visual_dim)
    torch.testing.assert_close(
        proposal.norm(dim=-1),
        torch.ones(2, 3),
        atol=1e-6,
        rtol=1e-6,
    )


class ConstantVelocity(nn.Module):
    def __init__(self, velocity: torch.Tensor):
        super().__init__()
        self.register_buffer("velocity", velocity)

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
        *,
        condition_present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del time, condition, condition_present
        if state.shape != self.velocity.shape:
            raise ValueError("constant velocity test shape changed")
        return self.velocity.expand_as(state)


def test_state_flow_integrates_noise_back_to_continuous_target() -> None:
    model = PredictiveVisualField(small_config())
    batch, samples, dimension = 2, 3, model.config.visual_dim
    condition = torch.randn(batch, model.config.state_dim + dimension)
    noise = torch.randn(batch, samples, dimension)
    target = torch.randn(batch, samples, dimension)
    velocity = (noise - target).reshape(batch * samples, dimension)
    model.state_flow = ConstantVelocity(velocity)

    sampled = model.sample_states(
        condition,
        samples_per_context=samples,
        steps=4,
        noise=noise,
    )

    torch.testing.assert_close(sampled, torch.nn.functional.normalize(target, dim=-1))


def test_hyperspherical_path_stays_on_manifold_and_is_reversible() -> None:
    torch.manual_seed(11)
    target = F.normalize(torch.randn(7, 64), dim=-1)
    source = F.normalize(torch.randn(7, 64), dim=-1)
    time = torch.linspace(0.1, 0.9, 7)

    point, velocity = hyperspherical_flow_path(target, source, time)
    recovered_target = sphere_exponential_map(point, -time[:, None] * velocity)
    recovered_source = sphere_exponential_map(point, (1.0 - time[:, None]) * velocity)

    torch.testing.assert_close(point.norm(dim=-1), torch.ones(7), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        (point * velocity).sum(dim=-1),
        torch.zeros(7),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(recovered_target, target, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(recovered_source, source, atol=2e-5, rtol=2e-5)


def test_hyperspherical_flow_loss_is_finite_and_differentiable() -> None:
    torch.manual_seed(29)
    model = PredictiveVisualField(small_config(flow_geometry="hypersphere")).train()
    batch, length, size = 2, 4, model.config.fovea_size
    context = torch.rand(batch, length, 1, size, size)
    target_ink = torch.rand(batch, length, 1, size, size)
    outputs = model(context, torch.rand_like(target_ink))

    loss, metrics = predictive_visual_field_loss(
        model,
        outputs,
        target_ink,
        flow_positions_per_sequence=2,
        sampled_positions_per_sequence=2,
        samples_per_context=2,
        sample_steps=2,
        sampled_endpoint_weight=1.0,
        generator=torch.Generator().manual_seed(31),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    assert 0.0 <= float(metrics["sampled_state_endpoint_geodesic"]) <= torch.pi**2
    torch.testing.assert_close(
        metrics["state_flow_objective"],
        metrics["state_flow_mse"] * model.config.visual_dim,
    )
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in model.state_flow.parameters()
    )


def test_state_flow_loss_backpropagates_through_sampled_distribution() -> None:
    torch.manual_seed(17)
    model = PredictiveVisualField(small_config()).train()
    batch, length, size = 3, 4, model.config.fovea_size
    context = torch.rand(batch, length, 1, size, size)
    target_ink = torch.rand(batch, length, 1, size, size)
    target_reference = torch.rand(batch, length, 1, size, size)
    outputs = model(context, target_reference)
    visual_anchors = outputs["target_visual"].detach().flatten(0, 1)[:, None]

    loss, metrics = predictive_visual_field_loss(
        model,
        outputs,
        target_ink,
        flow_positions_per_sequence=2,
        samples_per_context=2,
        sample_steps=2,
        visual_anchor_candidates=visual_anchors,
        visual_anchor_identity_weight=0.5,
        visual_anchor_context_weight=0.5,
        proposal_geodesic_weight=0.5,
        proposal_identity_weight=0.5,
        proposal_context_weight=0.5,
        proposal_anchor_identity_weight=0.5,
        proposal_anchor_context_weight=0.5,
        generator=torch.Generator().manual_seed(23),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in model.state_flow.parameters()
    )
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in model.dynamics.parameters()
    )
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in model.visual_proposal.parameters()
    )
    assert all(parameter.grad is None for parameter in model.retina.parameters())
    assert float(metrics["visual_anchor_active"]) == 1.0
    assert float(metrics["visual_anchor_coverage"]) == 1.0


def test_sample_density_scores_flat_and_multiview_image_states() -> None:
    model = PredictiveVisualField(small_config())
    sampled = torch.randn(4, 3, model.config.visual_dim)
    flat = torch.randn(11, model.config.visual_dim)
    multiview = torch.randn(11, 5, model.config.visual_dim)

    flat_scores = model.score_candidates(sampled, flat)
    multiview_scores = model.score_candidates(sampled, multiview)

    assert flat_scores.shape == (4, 11)
    assert multiview_scores.shape == (4, 11)
    assert torch.isfinite(flat_scores).all()
    assert torch.isfinite(multiview_scores).all()


def test_v7_initialization_loads_only_retina_and_causal_dynamics() -> None:
    source = RetinalFlowLanguageModel(
        RetinalFlowConfig(
            fovea_size=16,
            visual_dim=64,
            state_dim=128,
            state_layers=1,
            retina_base_channels=16,
            dropout=0.0,
            flow_base_channels=16,
            flow_context_dim=64,
            energy_dim=64,
            condition_dropout=0.0,
        )
    )
    target = PredictiveVisualField(small_config())
    checkpoint = {
        "architecture": "retinal-flow-language-model-v1",
        "global_step": 5_800,
        "model_config": {
            "fovea_size": 16,
            "visual_dim": 64,
            "state_dim": 128,
            "state_layers": 1,
            "retina_base_channels": 16,
        },
        "model": source.state_dict(),
    }

    receipt = initialize_from_retinal_flow_checkpoint(target, checkpoint)

    assert receipt["source_step"] == 5_800
    for expected, actual in zip(
        source.target_retina.parameters(),
        target.retina.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(expected, actual)
    for expected, actual in zip(
        source.dynamics.parameters(),
        target.dynamics.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(expected, actual)
