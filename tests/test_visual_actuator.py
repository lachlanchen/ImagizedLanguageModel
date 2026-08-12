from __future__ import annotations

import torch
import torch.nn.functional as F

from ilm.visual_lm.saccade_lm import FovealRetina, VisualSaccadeConfig
from ilm.visual_lm.visual_actuator import (
    ContinuousVisualActuator,
    VisualActuatorConfig,
    evaluate_visual_actuator_batch,
    multi_positive_nce,
    visual_actuator_config_from_payload,
    visual_actuator_config_payload,
    visual_actuator_loss,
    visual_positive_mask,
)


def _tiny_actuator() -> ContinuousVisualActuator:
    torch.manual_seed(7)
    actuator = ContinuousVisualActuator(
        VisualActuatorConfig(
            fovea_size=16,
            visual_dim=64,
            style_dim=16,
            style_base_channels=8,
            flow_base_channels=16,
            flow_context_dim=64,
            condition_dropout=0.0,
        )
    )
    torch.nn.init.normal_(actuator.writer.output[-1].weight, std=0.01)
    return actuator


def _tiny_retina() -> FovealRetina:
    retina = FovealRetina(
        VisualSaccadeConfig(
            fovea_size=16,
            visual_dim=64,
            state_dim=128,
            state_layers=1,
            retina_base_channels=16,
            dropout=0.0,
            visual_hypotheses=1,
        )
    )
    return retina.eval().requires_grad_(False)


def test_visual_actuator_has_continuous_image_boundary() -> None:
    actuator = _tiny_actuator().eval()
    intended = F.normalize(torch.randn(3, 64), dim=-1)
    style = torch.rand(3, 1, 16, 16)
    state = torch.randn(3, 1, 16, 16)
    time = torch.rand(3)

    velocity = actuator(state, time, intended, style)
    generated = actuator.sample(
        intended,
        style,
        steps=2,
        initial_noise=state,
    )

    assert velocity.shape == state.shape
    assert generated.shape == state.shape
    assert float(generated.min().detach()) >= -1.0
    assert float(generated.max().detach()) <= 1.0
    assert not hasattr(actuator, "embedding")
    assert not hasattr(actuator, "classifier")
    assert not hasattr(actuator, "vocabulary")


def test_visual_actuator_loss_backpropagates_through_deployed_sampler() -> None:
    actuator = _tiny_actuator().train()
    retina = _tiny_retina()
    target = torch.rand(5, 1, 16, 16)
    semantic = (0.8 * target + 0.1).clamp(0, 1)
    style = torch.rand_like(target)

    loss, metrics, trace = visual_actuator_loss(
        actuator,
        retina,
        target,
        semantic,
        style,
        sampled_batch_size=3,
        sampled_steps=2,
        duplicate_similarity=0.99,
        generator=torch.Generator().manual_seed(11),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    assert trace["sampled_ink"].shape == (3, 1, 16, 16)
    assert trace["sampled_ink"].requires_grad
    assert actuator.style_encoder.field[0].weight.grad is not None
    assert actuator.writer.input.weight.grad is not None
    assert actuator.writer.output[-1].weight.grad is not None
    assert all(parameter.grad is None for parameter in retina.parameters())


def test_intended_visual_state_changes_pixels_with_fixed_style_and_noise() -> None:
    actuator = _tiny_actuator().eval()
    intended = F.normalize(torch.randn(4, 64), dim=-1)
    style = torch.rand(4, 1, 16, 16)
    noise = torch.randn(4, 1, 16, 16)

    correct = actuator.sample(
        intended,
        style,
        steps=2,
        initial_noise=noise,
    )
    shuffled = actuator.sample(
        intended.roll(1, dims=0),
        style,
        steps=2,
        initial_noise=noise,
    )

    assert not torch.allclose(correct, shuffled)


def test_actuator_evaluator_reports_shuffled_state_control() -> None:
    actuator = _tiny_actuator().eval()
    retina = _tiny_retina()
    target = torch.rand(4, 1, 16, 16)
    semantic = (0.8 * target + 0.1).clamp(0, 1)
    style = torch.rand_like(target)
    noise = torch.randn_like(target)

    metrics, trace = evaluate_visual_actuator_batch(
        actuator,
        retina,
        target,
        semantic,
        style,
        steps=2,
        duplicate_similarity=0.99,
        initial_noise=noise,
    )

    assert all(torch.isfinite(value) for value in metrics.values())
    assert float(metrics["condition_pixel_l1"]) > 0.0
    assert trace["correct_ink"].shape == target.shape
    assert trace["shuffled_ink"].shape == target.shape


def test_visual_positive_mask_and_multi_positive_nce_handle_duplicates() -> None:
    candidates = F.normalize(torch.randn(4, 32), dim=-1)
    candidates[1] = candidates[0]
    positive = visual_positive_mask(candidates, 0.99)
    logits = candidates @ candidates.transpose(0, 1)

    loss, top1 = multi_positive_nce(logits, positive)

    assert positive[0, 1]
    assert positive[1, 0]
    assert torch.isfinite(loss)
    assert float(top1) == 1.0


def test_visual_actuator_config_round_trip() -> None:
    config = _tiny_actuator().config
    restored = visual_actuator_config_from_payload(
        visual_actuator_config_payload(config)
    )
    assert restored == config
