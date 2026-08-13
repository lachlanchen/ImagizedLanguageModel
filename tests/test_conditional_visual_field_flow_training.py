from __future__ import annotations

from dataclasses import asdict

import torch

from ilm.visual_lm.conditional_visual_field_flow import (
    ConditionalVisualFieldFlowConfig,
    ConditionalVisualFieldFlowModel,
)
from ilm.visual_lm.conditional_visual_field_flow_training import (
    conditional_visual_field_flow_training_microstep,
    flow_matching_loss,
    pair_direction_objective,
)
from ilm.visual_lm.spatial_visual_next_field import (
    V30_GLOBAL_ROUTE,
    SpatialVisualNextFieldConfig,
    SpatialVisualNextFieldModel,
)
from scripts.train_conditional_visual_field_flow_v31 import (
    _checkpoint_payload,
    build_optimizer,
    choose_device,
    load_v30_initialization,
)


def _model() -> ConditionalVisualFieldFlowModel:
    return ConditionalVisualFieldFlowModel(
        ConditionalVisualFieldFlowConfig(
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
            score_chunk_size=8,
        )
    )


def _pair_batch(batch: int = 2) -> dict[str, torch.Tensor]:
    suffix = torch.rand(batch, 4, 1, 32, 32)
    contexts = torch.cat(
        (
            torch.rand(batch, 2, 60, 1, 32, 32),
            suffix[:, None].expand(-1, 2, -1, -1, -1, -1),
        ),
        dim=2,
    )
    reference_suffix = torch.rand(batch, 4, 1, 32, 32)
    reference_contexts = torch.cat(
        (
            torch.rand(batch, 2, 60, 1, 32, 32),
            reference_suffix[:, None].expand(-1, 2, -1, -1, -1, -1),
        ),
        dim=2,
    )
    return {
        "contexts": contexts,
        "candidates": torch.rand(batch, 2, 1, 32, 32),
        "assignment": torch.tensor([[0, 1]]).expand(batch, -1).clone(),
        "reference_contexts": reference_contexts,
        "reference_candidates": torch.rand(batch, 2, 1, 32, 32),
        "reference_assignment": torch.tensor([[1, 0]]).expand(batch, -1).clone(),
    }


def test_v31_flow_matching_exact_velocity_has_zero_loss() -> None:
    model = _model().eval()
    condition = torch.rand(2, 128)
    targets = torch.nn.functional.normalize(torch.rand(2, 16, 72), dim=-1)
    base = model.make_coherent_base(2)
    times = torch.tensor([0.2, 0.7])
    expected = targets - base

    original = model.velocity
    model.velocity = lambda *_args: expected  # type: ignore[method-assign]
    try:
        loss, metrics = flow_matching_loss(
            model,
            condition,
            targets,
            times=times,
            base_fields=base,
        )
    finally:
        model.velocity = original  # type: ignore[method-assign]
    assert torch.allclose(loss, torch.zeros(()), atol=1e-7)
    assert torch.allclose(metrics["flow_endpoint_cosine"], torch.ones(()), atol=1e-6)


def test_v31_pair_objective_preserves_exact_suffix_diagnostic() -> None:
    model = _model().eval()
    batch = _pair_batch()
    probes = model.make_coherent_base(2)
    times = torch.tensor([0.10, 0.35])
    losses, metrics = pair_direction_objective(
        model,
        batch["contexts"],
        batch["candidates"],
        batch["assignment"],
        probes,
        times,
    )
    assert set(losses) == {"flow", "assignment", "positive", "order"}
    assert all(torch.isfinite(value) for value in losses.values())
    assert float(metrics["pair_suffix_row_max_error"]) == 0.0


def test_v31_complete_microstep_backpropagates_only_language_path() -> None:
    model = _model().train()
    natural = {
        "first_context": torch.rand(2, 64, 1, 32, 32),
        "second_context": torch.rand(2, 64, 1, 32, 32),
        "first_target": torch.rand(2, 1, 32, 32),
        "second_target": torch.rand(2, 1, 32, 32),
    }
    loss, metrics = conditional_visual_field_flow_training_microstep(
        model, natural, _pair_batch()
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["natural_order_gain"])
    assert any(
        parameter.grad is not None for parameter in model.velocity_decoder.parameters()
    )
    assert all(parameter.grad is None for parameter in model.retina.parameters())
    assert all(
        parameter.grad is None for parameter in model.semantic_adapter.parameters()
    )


def test_v31_initialization_loads_exact_v30_backbone(tmp_path) -> None:
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
            route_mode=V30_GLOBAL_ROUTE,
        )
    )
    path = tmp_path / "v30.pt"
    torch.save(
        {
            "architecture": "spatial-visual-next-field-v30",
            "route_mode": V30_GLOBAL_ROUTE,
            "model_config": asdict(source.config),
            "model": source.state_dict(),
            "initialization": {"retina_sha256": "retina-test"},
        },
        path,
    )
    model = _model()
    receipt = load_v30_initialization(model, path, require_expected_hash=False)
    assert receipt["discarded_v30_field_decoder"] is True
    assert receipt["source_route"] == V30_GLOBAL_ROUTE
    for name in model._BACKBONE_MODULES:
        expected = getattr(source, name).state_dict()
        measured = getattr(model, name).state_dict()
        for key in expected:
            assert torch.equal(expected[key], measured[key])


def test_v31_optimizer_and_parameter_caps_match_protocol() -> None:
    model = ConditionalVisualFieldFlowModel(ConditionalVisualFieldFlowConfig())
    optimizer = build_optimizer(
        model,
        velocity_learning_rate=3e-4,
        context_learning_rate=6e-5,
        weight_decay=0.05,
        device=torch.device("cpu"),
    )
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    assert set(groups) == {"v30_context", "conditional_velocity"}
    assert groups["v30_context"]["lr"] == 6e-5
    assert groups["conditional_velocity"]["lr"] == 3e-4
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    assert total <= 20_000_000
    assert trainable <= 18_500_000


def test_v31_explicit_cuda_device_has_an_index() -> None:
    assert choose_device("cuda") == torch.device("cuda:0")
    assert choose_device("cuda:1") == torch.device("cuda:1")


def test_v31_final_checkpoint_excludes_training_state_and_bank() -> None:
    model = _model()
    optimizer = build_optimizer(
        model,
        velocity_learning_rate=3e-4,
        context_learning_rate=6e-5,
        weight_decay=0.05,
        device=torch.device("cpu"),
    )
    payload = _checkpoint_payload(
        model,
        optimizer,
        step=1,
        final=True,
        smoke_only=True,
        exploratory=True,
        initialization={"sha256": "v30"},
        manifest_receipt={"sha256": "manifest"},
        partition={"train": 1},
        pair_receipt={"sha256": "pairs"},
        candidate_bank_receipt={
            "manifest_sha256": "candidate-manifest",
            "images_in_checkpoint": False,
            "forms_in_checkpoint": False,
            "inference_requires_bank": False,
        },
        arguments={
            "batch_size": 2,
            "gradient_accumulation": 1,
            "pair_batch_size": 2,
        },
        peak_vram_gib=0.0,
        training_metrics={"loss": 1.0},
        training_seconds=1.0,
    )
    assert payload["optimizer"] is None
    assert payload["rng_state"] is None
    assert payload["resumable"] is False
    assert payload["deployed_state_includes_training_candidate_images"] is False
    assert payload["deployed_state_includes_training_form_labels"] is False
    assert not any("bank" in name.lower() for name in payload["model"])
