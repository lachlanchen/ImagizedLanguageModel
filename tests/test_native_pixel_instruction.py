import pytest
import torch

from ilm.visual_lm.native_pixel_instruction import (
    answer_only_bce, complete_pairs, image_cells, native_strip, training_strip,
)


def test_native_sign_pairing_and_roundtrip():
    x = torch.randint(0, 2, (3, 1, 32, 32)).float()
    padded = complete_pairs(x)
    assert len(padded) == 4 and not padded[-1].any()
    torch.testing.assert_close(image_cells(native_strip(padded)), padded, atol=0, rtol=0)
    with pytest.raises(ValueError):
        native_strip(x)


def test_answer_loss_uses_next_block_and_excludes_prompt_labels():
    prompt = torch.ones(3, 1, 32, 32)
    answer = torch.ones(1, 1, 32, 32)
    pixels, count = training_strip(prompt, answer)
    assert count == 2 and pixels.shape[-1] == 5*64
    logits = torch.zeros_like(pixels, requires_grad=True)
    loss = answer_only_bce(logits, pixels, count)
    loss.backward()
    assert not logits.grad[..., :64].any()
    assert logits.grad[..., 64:128].abs().sum() > 0  # Last prompt state predicts answer.
    assert not logits.grad[..., -64:].any()  # No unseen future block is labelled.
    changed = pixels.clone(); changed[..., :count*64] *= -1
    torch.testing.assert_close(answer_only_bce(logits, changed, count), loss)
    # Supplying the exact next-block logits drives true answer-only NLL down.
    good = torch.zeros_like(pixels)
    good[..., (count-1)*64:-64] = 20 * pixels[..., count*64:]
    assert float(answer_only_bce(good, pixels, count)) < 1e-7
