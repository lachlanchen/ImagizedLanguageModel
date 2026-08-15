from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from ilm.visual_lm.canonical_glyph_language_data import CanonicalGlyphRenderConfig
from ilm.visual_lm.visual_future_block_language_v48 import (
    VisualFutureBlockLanguageConfigV48,
    VisualFutureBlockLanguageModelV48,
)
from scripts.eval_visual_future_block_language_v48 import (
    _fixed_evaluation_arguments,
)
from scripts.train_visual_future_block_language_v48 import (
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT,
    FIXED_OPTIMIZATION,
    _assert_production_arguments,
    _checkpoint,
    _effective_arguments,
    _global_rng_receipt,
    _restore_global_rng,
)


def _arguments(*, smoke: bool = False, exploratory: bool = False) -> Namespace:
    return Namespace(
        manifest=DEFAULT_MANIFEST,
        out=DEFAULT_OUTPUT,
        resume=None,
        device="cpu",
        precision="bf16",
        num_workers=4,
        smoke=smoke,
        exploratory=exploratory,
        **FIXED_OPTIMIZATION,
    )


def test_smoke_run_cannot_overwrite_the_production_artifact() -> None:
    effective = _effective_arguments(_arguments(smoke=True))
    assert effective["out"] == f"{DEFAULT_OUTPUT}_smoke"
    assert effective["steps"] == 2
    assert effective["batch_size"] == 2
    assert effective["gradient_accumulation"] == 1
    assert effective["maximum_positions"] == 64


def test_production_and_exploratory_arguments_are_guarded() -> None:
    production = vars(_arguments()).copy()
    _assert_production_arguments(production)
    changed = production | {"steps": 9_999}
    with pytest.raises(ValueError, match="steps"):
        _assert_production_arguments(changed)
    exploratory = vars(_arguments(exploratory=True)).copy()
    with pytest.raises(ValueError, match="distinct output"):
        _assert_production_arguments(exploratory)


def test_cpu_rng_state_round_trips_for_exact_resume() -> None:
    device = torch.device("cpu")
    torch.manual_seed(4800)
    receipt = _global_rng_receipt(device)
    expected = torch.rand(8)
    _ = torch.rand(19)
    _restore_global_rng(receipt, device=device)
    torch.testing.assert_close(torch.rand(8), expected)


def test_checkpoint_contains_every_resume_and_boundary_receipt() -> None:
    config = VisualFutureBlockLanguageConfigV48(
        model_dim=128,
        layers=1,
        heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
    )
    model = VisualFutureBlockLanguageModelV48(config)
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    arguments = _effective_arguments(_arguments(smoke=True))
    generator = torch.Generator().manual_seed(48)
    payload = _checkpoint(
        model,
        optimizer,
        scaler,
        update=1,
        arguments=arguments,
        render_config=CanonicalGlyphRenderConfig(),
        manifest_receipt={"sha256": "unit-test"},
        partition_receipt={"partition": "unit-test"},
        training_generator=generator,
        elapsed_seconds=1.0,
        peak_vram_gib=0.0,
        metrics={"loss": 1.0},
        device=torch.device("cpu"),
    )
    assert payload["update"] == 1
    assert payload["segments_consumed"] == 2
    assert set(payload["global_rng_state"]) == {"cpu"}
    assert isinstance(payload["training_generator_state"], torch.Tensor)
    assert payload["data_boundary"]["uses_token_ids"] is False
    assert payload["model_boundary"]["uses_stochastic_generator"] is False
    assert payload["protocol"]["effective_arguments"]["smoke"] is True


def test_strict_evaluator_shape_is_frozen() -> None:
    arguments = Namespace(
        precision="bf16",
        windows=2_048,
        future_windows=2_048,
        bank_size=1_024,
        pairs=512,
        closed_loop_examples=256,
    )
    assert _fixed_evaluation_arguments(arguments)
    arguments.future_windows = 2_047
    assert not _fixed_evaluation_arguments(arguments)

