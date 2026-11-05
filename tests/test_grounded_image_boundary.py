"""Physical image boundary checks against the installed native processor."""

import numpy as np
import pytest
import torch

from ilm.visual_lm.grounded_image_actor import native_image_patches


def test_spatial_temporal_order_and_normalization_match_native_processor():
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    image = torch.linspace(-1, 1, 64*96).reshape(1, 1, 64, 96)
    packed, grid = native_image_patches(image)
    processor = Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2,
        image_mean=[.5]*3, image_std=[.5]*3, do_resize=False, do_rescale=False)
    reference = processor(images=[((image[0]+1)*.5).expand(3, -1, -1).numpy()],
        input_data_format="channels_first", return_tensors="pt")
    assert grid == tuple(reference["image_grid_thw"][0].tolist())
    assert torch.allclose(packed, reference["pixel_values"], rtol=0, atol=2e-7)
    # Independently invert the temporal/channel/spatial packing. Both repeated
    # still frames must preserve every asymmetric input pixel.
    restored = packed.reshape(1, 2, 3, 2, 2, 3, 2, 16, 16).permute(0, 6, 5, 1, 3, 7, 2, 4, 8)
    restored = restored.reshape(2, 3, 64, 96)
    assert torch.equal(restored[0], image[0].expand(3, -1, -1))
    assert torch.equal(restored[1], restored[0])


def test_packed_pixels_own_storage_and_preserve_unknown_forms():
    rng = np.random.default_rng(20266404)
    image = torch.from_numpy(rng.choice([-1., 1.], (1, 1, 32, 64)).astype(np.float32))
    packed, grid = native_image_patches(image)
    saved = packed.clone()
    image.fill_(1)
    assert torch.equal(saved, packed)
    assert grid == (1, 2, 4)
    assert torch.equal(packed.unique(), torch.tensor([-1., 1.]))


@pytest.mark.parametrize("image", [torch.zeros(1, 1, 31, 64), torch.zeros(1, 1, 32, 65),
    torch.full((1, 1, 32, 64), float("nan")), torch.ones(1, 1, 32, 64)*2])
def test_invalid_physical_images_are_rejected(image):
    with pytest.raises(ValueError):
        native_image_patches(image)
