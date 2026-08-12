from __future__ import annotations

from argparse import Namespace

import torch

from ilm.visual_lm.ink_jepa_data import VisualGrammarRecord
from ilm.visual_lm.visual_cell_data import (
    VisualCellRenderConfig,
    visual_cell_partition,
)
from ilm.visual_lm.visual_cell_stream import VisualCellStreamConfig, VisualCellStreamModel
from scripts.train_visual_cell_stream_v25 import (
    FIXED_EVIDENCE,
    FIXED_OPTIMIZATION,
    LANGUAGE_STAGE,
    WRITER_STAGE,
    _bidirectional_views,
    _effective_arguments,
    _require_fixed_evidence_arguments,
    _set_training_stage,
    load_v16_retina,
    train_stage,
)


def tiny_model() -> VisualCellStreamModel:
    return VisualCellStreamModel(
        VisualCellStreamConfig(
            maximum_cells=64,
            visual_dim=64,
            model_dim=128,
            layers=1,
            heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            retina_base_channels=8,
            writer_base_channels=8,
            time_dim=16,
        )
    )


def train_record() -> VisualGrammarRecord:
    identifier = next(
        f"visual-cell-train-{index}"
        for index in range(100_000)
        if visual_cell_partition(f"visual-cell-train-{index}") == "train"
    )
    return VisualGrammarRecord(
        identifier=identifier,
        text="天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往秋收冬藏" * 4,
        language="zh-Hans",
        source="unit-test",
        rights="test-only",
    )


def fixed_namespace(**changes: object) -> Namespace:
    values: dict[str, object] = {
        **FIXED_OPTIMIZATION,
        **FIXED_EVIDENCE,
        "smoke": False,
        "exploratory": False,
        "from_scratch_retina": False,
        "continue_writer_after_failed_language": False,
        "smoke_steps": 2,
        "out": "unused",
        "manifest": "unused",
        "retina_checkpoint": "unused",
        "resume": None,
        "device": "cpu",
        "num_workers": 0,
        "log_every": 1,
        "save_every": 1,
    }
    values.update(changes)
    return Namespace(**values)


def test_smoke_arguments_are_bounded_and_cannot_be_evidence() -> None:
    args = fixed_namespace(smoke=True, smoke_steps=3)
    effective = _effective_arguments(args)
    assert effective["language_steps"] == 3
    assert effective["writer_steps"] == 3
    assert effective["audit_windows"] == 16
    assert effective["inference_candidates"] == 1

    evidence = fixed_namespace(language_steps=2_399)
    try:
        _require_fixed_evidence_arguments(evidence)
    except ValueError as error:
        assert "language-steps" in str(error)
    else:
        raise AssertionError("changed evidence schedule was accepted")


def test_two_font_views_are_used_in_both_continuous_directions() -> None:
    batch = {
        "context": torch.zeros(2, 64, 1, 32, 32),
        "target": torch.ones(2, 64, 1, 32, 32),
        "reference_context": torch.full((2, 64, 1, 32, 32), 2.0),
        "reference_target": torch.full((2, 64, 1, 32, 32), 3.0),
    }
    views = _bidirectional_views(batch)
    assert views["context"].shape[0] == 4
    assert torch.equal(views["context"][:2], batch["context"])
    assert torch.equal(views["context"][2:], batch["reference_context"])
    assert torch.equal(views["pixel_target"][:2], batch["target"])
    assert torch.equal(views["independent_target"][:2], batch["reference_target"])
    assert torch.equal(views["independent_target"][2:], batch["target"])


def test_training_stages_have_disjoint_trainable_parameters() -> None:
    model = tiny_model()
    _set_training_stage(model, LANGUAGE_STAGE, train_retina=False)
    language_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert language_names
    assert not any(name.startswith("writer.") for name in language_names)
    assert not any(name.startswith("online_retina.") for name in language_names)

    _set_training_stage(model, WRITER_STAGE, train_retina=False)
    writer_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert writer_names
    assert all(name.startswith("writer.") for name in writer_names)
    assert language_names.isdisjoint(writer_names)


def test_retina_initialization_loads_both_online_and_target_views(tmp_path) -> None:
    source = tiny_model()
    path = tmp_path / "retina.pt"
    torch.save(
        {
            "architecture": "predictive-visual-field-state-flow-v1",
            "global_step": 9,
            "model": {
                f"retina.{name}": value
                for name, value in source.online_retina.state_dict().items()
            },
        },
        path,
    )
    target = tiny_model()
    with torch.no_grad():
        for parameter in target.online_retina.parameters():
            parameter.zero_()
    receipt = load_v16_retina(target, path, require_expected_hash=False)
    assert receipt["source_step"] == 9
    for name, value in source.online_retina.state_dict().items():
        torch.testing.assert_close(target.online_retina.state_dict()[name], value)
        torch.testing.assert_close(target.target_retina.state_dict()[name], value)


def test_both_training_stages_execute_real_gradient_updates(tmp_path) -> None:
    torch.manual_seed(41)
    model = tiny_model()
    context = {
        "smoke_only": True,
        "exploratory": True,
        "initialization": {"route": "unit-test"},
        "manifest_receipt": {"sha256": "unit-test"},
        "partition": {"frozen_images_instantiated": False},
        "render_config": VisualCellRenderConfig(),
        "arguments": {
            "batch_size": 1,
            "gradient_accumulation": 1,
            "precision": "fp32",
        },
    }
    language_before = model.proposal[-1].weight.detach().clone()
    language_metrics, _ = train_stage(
        model,
        [train_record()],
        stage=LANGUAGE_STAGE,
        device=torch.device("cpu"),
        precision="fp32",
        render_config=VisualCellRenderConfig(),
        steps=1,
        start_step=0,
        batch_size=1,
        accumulation=1,
        num_workers=0,
        base_lr=1e-4,
        warmup=1,
        minimum_lr_ratio=0.1,
        weight_decay=0.0,
        gradient_clip=1.0,
        writer_positions=1,
        dataset_seed=43,
        train_retina=False,
        output=tmp_path,
        log_every=1,
        save_every=1,
        checkpoint_context=context,
    )
    assert torch.isfinite(torch.tensor(language_metrics["loss"]))
    assert not torch.equal(language_before, model.proposal[-1].weight)

    writer_before = model.writer.output.weight.detach().clone()
    writer_metrics, _ = train_stage(
        model,
        [train_record()],
        stage=WRITER_STAGE,
        device=torch.device("cpu"),
        precision="fp32",
        render_config=VisualCellRenderConfig(),
        steps=1,
        start_step=0,
        batch_size=1,
        accumulation=1,
        num_workers=0,
        base_lr=1e-4,
        warmup=1,
        minimum_lr_ratio=0.1,
        weight_decay=0.0,
        gradient_clip=1.0,
        writer_positions=1,
        dataset_seed=47,
        train_retina=False,
        output=tmp_path,
        log_every=1,
        save_every=1,
        checkpoint_context=context,
    )
    assert torch.isfinite(torch.tensor(writer_metrics["loss"]))
    assert not torch.equal(writer_before, model.writer.output.weight)
