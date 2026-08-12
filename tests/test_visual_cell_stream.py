from __future__ import annotations

import copy

import pytest
import torch

from ilm.visual_lm.visual_cell_stream import (
    VisualCellStreamConfig,
    VisualCellStreamModel,
    visual_cell_flow_loss,
    visual_cell_language_loss,
    visual_cell_model_boundary_receipt,
    visual_cell_model_config_from_payload,
    visual_cell_model_config_payload,
)


def tiny_config() -> VisualCellStreamConfig:
    return VisualCellStreamConfig(
        maximum_cells=8,
        visual_dim=64,
        model_dim=128,
        layers=2,
        heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        retina_base_channels=8,
        writer_base_channels=8,
        time_dim=16,
    )


def random_cells(batch: int = 2, length: int = 6) -> torch.Tensor:
    return torch.rand(batch, length, 1, 32, 32)


def test_visual_time_volume_shapes_and_image_only_boundary() -> None:
    model = VisualCellStreamModel(tiny_config()).eval()
    cells = random_cells()
    output = model.language(cells)
    assert output["context_visual"].shape == (2, 6, 64)
    assert output["context_state"].shape == (2, 6, 128)
    assert output["proposed_visual"].shape == (2, 6, 64)
    torch.testing.assert_close(
        output["proposed_visual"].norm(dim=-1),
        torch.ones(2, 6),
        rtol=1e-5,
        atol=1e-5,
    )

    receipt = visual_cell_model_boundary_receipt(model.config)
    assert receipt["visual_time_volume_axes"] == [
        "time",
        "channel",
        "height",
        "width",
    ]
    assert receipt["each_time_slice_is_a_clean_2d_cell"] is True
    assert receipt["geometric_depth_is_one"] is True
    assert receipt["rereads_generated_pixels"] is True
    forbidden = (
        "uses_strings",
        "uses_token_ids",
        "uses_unicode_ids",
        "uses_character_ids",
        "uses_vocabulary_embedding",
        "uses_ocr",
        "uses_visual_codebook",
        "uses_glyph_lookup",
        "uses_external_language_model",
        "candidate_bank_deployed",
    )
    assert all(receipt[key] is False for key in forbidden)

    with pytest.raises(TypeError, match="floating image"):
        model.language(cells.to(torch.int64))
    with pytest.raises(ValueError, match="shape"):
        model.language(torch.rand(2, 6, 32, 32))
    with pytest.raises(ValueError, match="configured context"):
        model.language(random_cells(length=9))


def test_language_field_is_strictly_causal_over_2d_slices() -> None:
    torch.manual_seed(7)
    model = VisualCellStreamModel(tiny_config()).eval()
    first = random_cells(batch=1)
    changed = first.clone()
    changed[:, 4:] = torch.rand_like(changed[:, 4:])
    first_proposal = model.language(first)["proposed_visual"]
    changed_proposal = model.language(changed)["proposed_visual"]
    torch.testing.assert_close(
        first_proposal[:, :4],
        changed_proposal[:, :4],
        rtol=0.0,
        atol=1e-6,
    )
    assert not torch.allclose(first_proposal[:, 4:], changed_proposal[:, 4:])


def test_next_visual_state_depends_on_earlier_visible_cells() -> None:
    torch.manual_seed(11)
    model = VisualCellStreamModel(tiny_config()).eval()
    first = random_cells(batch=1)
    changed = first.clone()
    changed[:, 0] = 1.0 - changed[:, 0]
    first_last = model.language(first)["proposed_visual"][:, -1]
    changed_last = model.language(changed)["proposed_visual"][:, -1]
    assert float((first_last - changed_last).abs().max().detach()) > 1e-6


def test_visual_language_loss_is_finite_and_trainable() -> None:
    torch.manual_seed(13)
    model = VisualCellStreamModel(tiny_config()).train()
    context = random_cells(batch=2, length=5)
    reference_target = random_cells(batch=2, length=5)
    output = model.forward_language(context, reference_target)
    loss, metrics = visual_cell_language_loss(
        output,
        contrastive_scale=model.contrastive_scale,
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    loss.backward()
    assert model.proposal[-1].weight.grad is not None
    assert float(model.proposal[-1].weight.grad.abs().sum()) > 0.0
    assert all(parameter.grad is None for parameter in model.online_retina.parameters())
    assert all(parameter.grad is None for parameter in model.target_retina.parameters())


def test_continuous_writer_loss_backpropagates_without_symbol_labels() -> None:
    torch.manual_seed(17)
    model = VisualCellStreamModel(tiny_config()).train()
    language = model.language(random_cells(batch=2, length=4))
    target = random_cells(batch=2, length=1)[:, 0]
    reference = random_cells(batch=2, length=1)[:, 0]
    loss, metrics = visual_cell_flow_loss(
        model,
        language["context_state"][:, -1].detach(),
        language["proposed_visual"][:, -1].detach(),
        target,
        reference,
        condition_dropout=0.0,
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    loss.backward()
    writer_gradients = [
        parameter.grad
        for parameter in model.writer.parameters()
        if parameter.grad is not None
    ]
    assert writer_gradients
    assert all(torch.isfinite(gradient).all() for gradient in writer_gradients)
    assert sum(float(gradient.abs().sum()) for gradient in writer_gradients) > 0.0


def test_generation_rereads_each_generated_pixel_slice() -> None:
    torch.manual_seed(19)
    model = VisualCellStreamModel(tiny_config()).eval()
    encoded_batch_sizes: list[int] = []

    def record_retina_batch(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        encoded_batch_sizes.append(inputs[0].shape[0])

    hook = model.online_retina.register_forward_hook(record_retina_batch)
    prefix = random_cells(batch=1, length=3)
    sequence, trace = model.generate(
        prefix,
        new_cells=2,
        candidates=2,
        flow_steps=2,
        generator=torch.Generator().manual_seed(23),
    )
    hook.remove()

    assert sequence.shape == (1, 5, 1, 32, 32)
    assert trace["generated_cells"].shape == (1, 2, 1, 32, 32)
    assert trace["reread_visual"].shape == (1, 2, 64)
    assert bool(trace["reread_generated_pixels"])
    torch.testing.assert_close(sequence[:, :3], prefix, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        sequence[:, 3:], trace["generated_cells"], rtol=0.0, atol=0.0
    )
    assert encoded_batch_sizes == [3, 4]
    assert torch.all((0.0 <= sequence[:, 3:]) & (sequence[:, 3:] <= 1.0))


def test_configuration_and_state_round_trip() -> None:
    torch.manual_seed(29)
    config = tiny_config()
    restored_config = visual_cell_model_config_from_payload(
        visual_cell_model_config_payload(config)
    )
    assert restored_config == config
    model = VisualCellStreamModel(config).eval()
    restored = copy.deepcopy(model).eval()
    restored.load_state_dict(model.state_dict())
    cells = random_cells(batch=1, length=4)
    torch.testing.assert_close(
        model.language(cells)["proposed_visual"],
        restored.language(cells)["proposed_visual"],
        rtol=0.0,
        atol=0.0,
    )
