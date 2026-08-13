from __future__ import annotations

import torch

from ilm.visual_lm.conditional_visual_density_ratio import (
    ConditionalVisualDensityRatioConfig,
    ConditionalVisualDensityRatioModel,
)
from ilm.visual_lm.conditional_visual_density_ratio_training import (
    conditional_visual_training_microstep,
    pair_direction_objective,
    shuffle_visual_prefix,
)


def _model() -> ConditionalVisualDensityRatioModel:
    return ConditionalVisualDensityRatioModel(
        ConditionalVisualDensityRatioConfig(
            visual_dim=64,
            semantic_dim=64,
            model_dim=128,
            layers=1,
            heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            retina_base_channels=8,
            semantic_hidden_dim=96,
            evidence_layers=1,
            evidence_heads=4,
            evidence_mlp_ratio=1.0,
            evidence_dropout=0.0,
            relation_hidden_dim=64,
            score_chunk_size=8,
        )
    )


def _pair_batch(batch: int = 2) -> dict[str, torch.Tensor]:
    suffix = torch.rand(batch, 4, 1, 32, 32)
    first_prefix = torch.rand(batch, 2, 60, 1, 32, 32)
    contexts = torch.cat(
        (
            first_prefix,
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


def test_v29_prefix_shuffle_changes_prefix_and_preserves_suffix() -> None:
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


def test_v29_pair_objective_uses_exact_suffix_rows() -> None:
    model = _model().eval()
    batch = _pair_batch()
    losses, metrics = pair_direction_objective(
        model,
        batch["contexts"],
        batch["candidates"],
        batch["assignment"],
    )
    assert set(losses) == {"full", "increment", "positive", "order"}
    assert all(torch.isfinite(value) for value in losses.values())
    assert float(metrics["pair_suffix_row_max_error"]) == 0.0


def test_v29_complete_microstep_backpropagates_without_perception_gradients() -> None:
    model = _model().train()
    natural = {
        "first_context": torch.rand(2, 64, 1, 32, 32),
        "second_context": torch.rand(2, 64, 1, 32, 32),
    }
    targets = torch.tensor([1, 3])
    bank_images = torch.rand(2, 5, 1, 32, 32)
    candidate_features = tuple(
        model.encode_image_parts(bank_images[view], target=True)
        for view in range(2)
    )
    loss, metrics = conditional_visual_training_microstep(
        model,
        natural,
        targets,
        _pair_batch(),
        candidate_features,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["natural_increment_order_gain"])
    assert any(
        parameter.grad is not None
        for parameter in model.evidence_blocks.parameters()
    )
    assert all(parameter.grad is None for parameter in model.retina.parameters())
    assert all(
        parameter.grad is None
        for parameter in model.semantic_adapter.parameters()
    )
