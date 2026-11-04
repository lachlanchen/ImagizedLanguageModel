from __future__ import annotations

import torch

from ilm.visual_lm.visual_posterior_language_v51_evaluation import (
    isolated_ink_fraction,
    pixel_f1,
)


def test_pixel_f1_is_exact_for_matching_ink() -> None:
    images = torch.zeros(2, 1, 32, 32)
    images[:, :, 8:24, 15:17] = 1.0
    assert torch.equal(pixel_f1(images, images), torch.ones(2))


def test_isolated_ink_fraction_detects_speckle() -> None:
    connected = torch.zeros(1, 1, 32, 32)
    connected[:, :, 8:24, 15:17] = 1.0
    speckle = torch.zeros(1, 1, 32, 32)
    speckle[:, :, 4, 4] = 1.0
    speckle[:, :, 12, 12] = 1.0
    speckle[:, :, 24, 24] = 1.0
    assert float(isolated_ink_fraction(connected)) == 0.0
    assert float(isolated_ink_fraction(speckle)) == 1.0
