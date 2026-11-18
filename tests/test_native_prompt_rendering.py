import numpy as np
from PIL import Image
import pytest

from scripts.infer_continuous_image_language import image_cells, text_cells
from scripts.prepare_visual_sequence_corpus import FONT, Rasterizer


@pytest.mark.skipif(not FONT.is_file(), reason='Requires the original training font')
def test_typed_prompt_matches_physical_training_page(tmp_path):
    text = '問題：請說明「言」字的演變。\n回答：'
    renderer = Rasterizer()
    packed = np.frombuffer(b''.join(renderer.tile(c) for c in text), dtype=np.uint8)
    original = np.unpackbits(packed, bitorder='big').reshape(-1, 32, 32)
    page = np.concatenate(original, axis=1)
    image = tmp_path / 'prompt.png'
    Image.fromarray(255 * (1 - page)).save(image)
    expected = image_cells(image)
    np.testing.assert_array_equal(text_cells(text), expected)
    np.testing.assert_array_equal(text_cells(text, FONT), expected)
    with pytest.raises(ValueError, match='at least one'):
        text_cells('')
