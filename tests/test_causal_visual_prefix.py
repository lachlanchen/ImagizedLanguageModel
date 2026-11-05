"""Real writing-cache information-flow checks, not a surrogate language test."""

from pathlib import Path

import pytest
import torch

from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt, prepare_teacher_forced_prefix
from ilm.visual_lm.complete_writing_data import CompleteWritingImageData


@pytest.fixture(scope="module")
def data():
    path = Path(__file__).resolve().parents[1]/"data/general_instructions/complete_lines_train_20261105"
    if not (path/"report.json").exists():
        pytest.skip("real complete-writing training cache is not installed")
    return CompleteWritingImageData(path, purpose="train",
        expected_report_sha256="0ffcfd101ea26d79c4084bfd166bd146ff5560f5fa83b7aa57c6b48558377d7e")


def test_exact_real_prompt_page_round_trip_without_ink_cuts(data):
    for index in (0, len(data)//2):
        pixels = data.prompt(index)
        folded = fold_writing_prompt(pixels)
        assert torch.equal(folded.unfold(), pixels)
        assert folded.pixels.shape[-1] <= 1024
        for start, end in folded.source_ranges[:-1]:
            assert bool((pixels[..., end-1] == 1).all())
        assert folded.pixels.untyped_storage().nbytes() == folded.pixels.numel()*4


def test_future_nan_and_source_mutation_cannot_change_visible_inputs(data):
    for index in (0, len(data)//2):
        image, boundary = data[index]
        completed = 1
        a = prepare_teacher_forced_prefix(image, prompt_blocks=boundary, completed_blocks=completed)
        changed = image.clone()
        changed[..., (boundary+completed)*64:] = torch.nan
        b = prepare_teacher_forced_prefix(changed, prompt_blocks=boundary, completed_blocks=completed)
        c = prepare_teacher_forced_prefix(image[..., :(boundary+completed)*64],
            prompt_blocks=boundary, completed_blocks=completed)
        assert torch.equal(a.prompt.pixels, b.prompt.pixels)
        assert torch.equal(a.prompt.pixels, c.prompt.pixels)
        assert all(torch.equal(x, y) for x, y in zip(a.completed_blocks, b.completed_blocks))
        assert all(torch.equal(x, y) for x, y in zip(a.completed_blocks, c.completed_blocks))
        assert len(a.completed_blocks) == completed
        before = a.completed_blocks[0].clone()
        image.fill_(1)
        assert torch.equal(before, a.completed_blocks[0])
        assert a.completed_blocks[0].untyped_storage().nbytes() == 32*64*4


def test_continuous_ink_is_rejected_instead_of_cut():
    ink = torch.full((1, 1, 32, 2048), -1, dtype=torch.float32)
    with pytest.raises(ValueError, match="no safe blank-column"):
        fold_writing_prompt(ink)
