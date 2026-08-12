from __future__ import annotations

import torch

from ilm.visual_lm.retinal_flow_lm import (
    RetinalFlowConfig,
    RetinalFlowLanguageModel,
    visual_rollout_losses,
)


def _tiny_model() -> RetinalFlowLanguageModel:
    torch.manual_seed(7)
    return RetinalFlowLanguageModel(
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


def test_sample_visual_candidates_rereads_and_reranks_images() -> None:
    model = _tiny_model().eval()
    context = torch.rand(2, 3, 1, 16, 16)
    prediction = model.predict(context)
    sampled = model.sample_visual_candidates(
        prediction["condition"][:, -1],
        context[:, -1],
        samples_per_context=3,
        steps=1,
        guidance_scale=1.0,
        generator=torch.Generator().manual_seed(11),
    )

    assert sampled["candidates"].shape == (2, 3, 1, 16, 16)
    assert sampled["candidate_visual"].shape == (2, 3, 64)
    assert sampled["energy"].shape == (2, 3)
    assert sampled["choice"].shape == (2,)
    assert sampled["selected"].shape == (2, 1, 16, 16)
    assert not sampled["selected"].requires_grad
    batch = torch.arange(2)
    assert torch.equal(sampled["choice"], sampled["energy"].argmax(dim=1))
    assert torch.equal(
        sampled["selected"],
        sampled["candidates"][batch, sampled["choice"]],
    )


def test_visual_rollout_uses_generated_feedback_and_backpropagates() -> None:
    model = _tiny_model().train()
    context = torch.rand(3, 6, 1, 16, 16)
    target_ink = torch.rand(3, 6, 1, 16, 16)
    current_reference = (0.85 * context + 0.075).clamp(0, 1)
    target_reference = (0.85 * target_ink + 0.075).clamp(0, 1)
    outputs = model(context, target_reference, current_reference)
    candidate_bank = outputs["target_visual"].float().detach().flatten(0, 1)

    losses, metrics, trace = visual_rollout_losses(
        model,
        outputs,
        context,
        target_ink,
        candidate_bank,
        rollout_batch_size=2,
        rollout_steps=2,
        rollout_candidates=2,
        rollout_sample_steps=1,
        rollout_guidance_scale=1.0,
        rollout_min_prefix=2,
        duplicate_similarity=0.90,
        endpoint_weight=0.10,
        stroke_weight=2.0,
        generator=torch.Generator().manual_seed(17),
    )
    total = losses["state"] + losses["energy"] + losses["recovery_flow"]
    total.backward()

    assert torch.isfinite(total)
    assert float(metrics["rollout_active"]) == 1.0
    assert trace["generated"].shape == (2, 2, 1, 16, 16)
    assert not trace["generated"].requires_grad
    assert 1 <= int(trace["start"]) <= 3
    assert model.online_retina.stem[1].weight.grad is not None
    assert model.dynamics.weight_ih_l0.grad is not None
    assert model.energy.condition[1].weight.grad is not None
    assert model.writer.input.weight.grad is not None
    for parameter in (
        model.online_retina.stem[1].weight,
        model.dynamics.weight_ih_l0,
        model.energy.condition[1].weight,
        model.writer.input.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
