from __future__ import annotations

from dataclasses import asdict

import torch

from ilm.visual_lm.conditional_visual_density_ratio import (
    ConditionalVisualDensityRatioConfig,
    ConditionalVisualDensityRatioModel,
)
from ilm.visual_lm.dense_visual_future_energy import (
    DenseVisualFutureConfig,
    DenseVisualFutureModel,
)
from scripts.train_conditional_visual_density_ratio_v29 import (
    _checkpoint_payload,
    build_optimizer,
    load_v28_initialization,
)


def _v29_model() -> ConditionalVisualDensityRatioModel:
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


def _v28_source() -> DenseVisualFutureModel:
    return DenseVisualFutureModel(
        DenseVisualFutureConfig(
            visual_dim=64,
            semantic_dim=64,
            model_dim=128,
            layers=1,
            heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            retina_base_channels=8,
            semantic_hidden_dim=96,
            hypotheses=2,
        )
    )


def test_v29_initialization_loads_exact_v28_backbone(tmp_path) -> None:
    source = _v28_source()
    path = tmp_path / "v28.pt"
    torch.save(
        {
            "architecture": "dense-visual-future-energy-v28",
            "model_config": asdict(source.config),
            "model": source.state_dict(),
            "initialization": {"sha256": "retina-test"},
        },
        path,
    )
    model = _v29_model()
    receipt = load_v28_initialization(
        model, path, require_expected_hash=False
    )
    assert receipt["discarded_future_heads"] is True
    assert receipt["retina_sha256"] == "retina-test"
    for name in model._BACKBONE_MODULES:
        expected = getattr(source, name).state_dict()
        measured = getattr(model, name).state_dict()
        for key in expected:
            assert torch.equal(expected[key], measured[key])


def test_v29_optimizer_has_distinct_context_and_evidence_groups() -> None:
    model = _v29_model()
    optimizer = build_optimizer(
        model,
        evidence_learning_rate=3e-4,
        context_learning_rate=6e-5,
        weight_decay=0.05,
        device=torch.device("cpu"),
    )
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    assert set(groups) == {"v28_context", "candidate_evidence"}
    assert groups["v28_context"]["lr"] == 6e-5
    assert groups["candidate_evidence"]["lr"] == 3e-4
    assert all(not parameter.requires_grad for parameter in model.retina.parameters())


def test_v29_checkpoint_has_receipts_but_no_training_bank_tensors() -> None:
    model = _v29_model()
    optimizer = build_optimizer(
        model,
        evidence_learning_rate=3e-4,
        context_learning_rate=6e-5,
        weight_decay=0.05,
        device=torch.device("cpu"),
    )
    arguments = {
        "batch_size": 2,
        "gradient_accumulation": 1,
        "pair_batch_size": 2,
    }
    payload = _checkpoint_payload(
        model,
        optimizer,
        step=1,
        final=True,
        smoke_only=True,
        exploratory=True,
        initialization={"sha256": "v28"},
        manifest_receipt={"sha256": "manifest"},
        partition={"train": 1},
        pair_receipt={"sha256": "pairs"},
        candidate_bank_receipt={
            "manifest_sha256": "bank-manifest",
            "images_in_checkpoint": False,
            "forms_in_checkpoint": False,
        },
        arguments=arguments,
        peak_vram_gib=0.0,
        training_metrics={"loss": 1.0},
        training_seconds=1.0,
    )
    assert payload["deployed_state_includes_training_candidate_images"] is False
    assert payload["deployed_state_includes_training_form_labels"] is False
    assert "host_forms" not in payload["candidate_bank_receipt"]
    assert not any("bank" in name.lower() for name in payload["model"])


def test_fixed_v29_model_stays_inside_preregistered_parameter_caps() -> None:
    model = ConditionalVisualDensityRatioModel(
        ConditionalVisualDensityRatioConfig()
    )
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    assert trainable <= 22_000_000
    assert total <= 25_000_000
