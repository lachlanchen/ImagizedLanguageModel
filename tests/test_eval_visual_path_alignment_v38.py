from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from ilm.visual_lm.visual_path_alignment import (
    VisualPathAlignmentConfig,
    VisualPathAlignmentModel,
)
from ilm.visual_lm.visual_path_alignment_data import V38_DEVELOPMENT_FONT
from ilm.visual_lm.visual_path_alignment_training import VisualPathAlignmentEMA
from ilm.visual_lm.visual_semantic_distillation_data import (
    VisualSemanticDistillationRenderConfig,
)
from scripts.eval_visual_path_alignment_v38 import (
    RasterSet,
    checkpoint_tensor_boundary,
    controlled_rasters,
    infer_paths,
    load_checkpoint_model,
    render_records,
)
from ilm.visual_lm.visual_path_alignment_data import (
    load_v37_instruction_records,
    select_v38_instruction_records,
)


def _tiny_config() -> VisualPathAlignmentConfig:
    return VisualPathAlignmentConfig(
        reader_hidden_size=32,
        reader_layers=1,
        reader_heads=4,
        reader_intermediate_size=64,
        reader_dropout=0.0,
        projection_hidden_size=64,
        semantic_dim=1024,
        projection_dropout=0.0,
        answer_hidden_size=32,
        length_hidden_size=16,
    )


def test_controlled_rasters_shuffle_blank_and_crop() -> None:
    pixels = torch.arange(4 * 3 * 16 * 1024, dtype=torch.float32).reshape(
        4, 3, 16, 1024
    )
    masks = torch.ones(4, 64)
    rasters = RasterSet(("a", "b", "c", "d"), pixels, masks)

    shuffled = controlled_rasters(rasters, "shuffled")
    blank = controlled_rasters(rasters, "blank")
    cropped = controlled_rasters(rasters, "final-quarter")

    assert torch.equal(shuffled.pixels[0], pixels[-1])
    assert bool((blank.pixels == 1).all()) and not bool(blank.mask.any())
    assert bool((cropped.pixels[..., :768] == 1).all())
    assert torch.equal(cropped.pixels[..., 768:], pixels[..., 768:])
    assert not bool(cropped.mask[:, :48].any())


def test_checkpoint_boundary_rejects_detached_training_tensors() -> None:
    assert checkpoint_tensor_boundary(
        {"model": {"reader.embeddings.patch_embeddings.weight": torch.ones(2)}}
    )
    assert not checkpoint_tensor_boundary({"teacher_mean": torch.ones(2)})
    assert not checkpoint_tensor_boundary({"nearest_answer": torch.ones(2)})
    assert not checkpoint_tensor_boundary({"answer_rotation": torch.ones(2)})


def test_checkpoint_loader_applies_all_parameter_ema(tmp_path: Path) -> None:
    config = _tiny_config()
    model = VisualPathAlignmentModel(config)
    ema = VisualPathAlignmentEMA(
        model,
        tuple(name for name, _parameter in model.named_parameters()),
        decay=0.999,
    )
    first_name = next(iter(ema.shadow))
    ema.shadow[first_name].fill_(0.25)
    checkpoint = {
        "architecture": "visual-path-alignment-v38",
        "model_config": asdict(config),
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "global_update": 12,
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)

    loaded, _checkpoint, receipt = load_checkpoint_model(
        path,
        device=torch.device("cpu"),
        raw_weights=False,
    )

    assert receipt["weight_route"] == "all-parameter-ema"
    assert receipt["global_update"] == 12
    assert torch.equal(
        dict(loaded.named_parameters())[first_name],
        torch.full_like(dict(loaded.named_parameters())[first_name], 0.25),
    )


def test_render_and_one_pass_image_only_inference_align() -> None:
    all_records = load_v37_instruction_records("data/raw/alpaca_zh.json")
    records, _rejected = select_v38_instruction_records(
        all_records,
        split="development",
        render_config=VisualSemanticDistillationRenderConfig(augment=False),
    )
    rasters = render_records(
        records[:2],
        field="prompt",
        font_path=V38_DEVELOPMENT_FONT,
        render_config=VisualSemanticDistillationRenderConfig(augment=False),
    )
    model = VisualPathAlignmentModel(_tiny_config()).eval()

    output = infer_paths(
        model,
        rasters,
        device=torch.device("cpu"),
        precision="fp32",
        batch_size=2,
    )

    assert output.prompt_states.shape == (2, 1024)
    assert output.answer_states.shape == (2, 1024)
    assert output.lengths.shape == (2,)
    assert output.finite is True

