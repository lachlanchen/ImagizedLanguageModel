from __future__ import annotations

import torch

from ilm.visual_lm.spatial_visual_next_field import (
    SpatialVisualNextFieldConfig,
    SpatialVisualNextFieldModel,
)
from ilm.visual_lm.spatial_visual_next_field_training import (
    aligned_field_loss,
    pair_direction_objective,
    shuffle_visual_prefix,
    spatial_visual_training_microstep,
)


def _model() -> SpatialVisualNextFieldModel:
    return SpatialVisualNextFieldModel(
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


def test_v30_prefix_shuffle_preserves_exact_suffix() -> None:
    context = torch.arange(2 * 64 * 32 * 32, dtype=torch.float32).reshape(
        2, 64, 1, 32, 32
    )
    shuffled = shuffle_visual_prefix(context)
    assert torch.equal(shuffled[:, -4:], context[:, -4:])
    assert not torch.equal(shuffled[:, :-4], context[:, :-4])
    assert torch.equal(
        shuffled[:, :-4].flatten(2).sort(dim=1).values,
        context[:, :-4].flatten(2).sort(dim=1).values,
    )


def test_v30_aligned_field_loss_is_zero_for_exact_normalized_target() -> None:
    field = torch.nn.functional.normalize(torch.rand(3, 16, 72), dim=-1)
    assert torch.allclose(aligned_field_loss(field, field), torch.zeros(()), atol=1e-6)
    reversed_loss = aligned_field_loss(field, field.flip(1))
    assert float(reversed_loss) > 0.0


def test_v30_pair_objective_uses_full_field_and_exact_suffix_diagnostic() -> None:
    model = _model().eval()
    batch = _pair_batch()
    losses, metrics = pair_direction_objective(
        model,
        batch["contexts"],
        batch["candidates"],
        batch["assignment"],
    )
    assert set(losses) == {"assignment", "field", "positive", "order"}
    assert all(torch.isfinite(value) for value in losses.values())
    assert float(metrics["pair_suffix_row_max_error"]) == 0.0


def test_v30_complete_microstep_backpropagates_without_perception_gradients() -> None:
    model = _model().train()
    natural = {
        "first_context": torch.rand(2, 64, 1, 32, 32),
        "second_context": torch.rand(2, 64, 1, 32, 32),
    }
    targets = torch.tensor([1, 3])
    bank_images = torch.rand(2, 5, 1, 32, 32)
    candidate_features = tuple(
        model.encode_route_candidates(bank_images[view])
        for view in range(2)
    )
    loss, metrics = spatial_visual_training_microstep(
        model,
        natural,
        targets,
        _pair_batch(),
        candidate_features,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["natural_order_gain"])
    assert any(
        parameter.grad is not None
        for parameter in model.field_decoder.parameters()
    )
    assert all(parameter.grad is None for parameter in model.retina.parameters())
    assert all(
        parameter.grad is None
        for parameter in model.semantic_adapter.parameters()
    )
