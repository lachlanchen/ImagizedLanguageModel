import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.continuous_autoregressive_language import ContinuousImageLanguage
from ilm.visual_lm.continuous_glyph_codec import ContinuousGlyphCodec, ContinuousGlyphCodecConfig


def test_complete_checkpoint_standalone_uses_actual_neural_pixels(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(13)
    model = ContinuousImageLanguage(ContinuousGlyphCodec(ContinuousGlyphCodecConfig()), components=1).eval()
    with torch.no_grad():
        model.stop.weight.zero_()
        model.stop.bias.fill_(-10)
    ink = torch.randint(0, 2, (2, 1, 32, 32), dtype=torch.uint8)
    expected, _ = model.generate(ink, maximum_new=2)
    checkpoint = tmp_path / 'full.pt'
    torch.save(dict(model=model.state_dict(), components=1, update=0), checkpoint)
    source = tmp_path / 'input.png'
    Image.fromarray((1 - ink[:, 0].numpy()).transpose(1, 0, 2).reshape(32, 64) * 255).save(source)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / 'scripts/infer_continuous_image_language.py'),
        '--checkpoint', str(checkpoint), '--image', str(source), '--output', str(tmp_path / 'out'),
        '--maximum-new', '2'], cwd=tmp_path, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(root)),
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    with np.load(tmp_path / 'out/actual_pixels.npz', allow_pickle=False) as actual:
        assert np.array_equal(actual['ink'], expected.numpy())
    assert '"output_fonts_used": false' in result.stdout
    assert '"external_teacher_or_corpus_used": false' in result.stdout
