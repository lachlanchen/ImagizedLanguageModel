import numpy as np

from ilm.visual_lm.causal_glyph_translator import DRAW, EXEMPLAR
from scripts.probe_paired_student_first_mark import pooled_choice
from scripts.train_causal_glyph_translator import prepare


def test_paired_fit_uses_original_exemplar_addresses():
    examples, (_, _, _, mode, addr, _, _) = prepare(
        [{"x": [0], "z": [2, 3]}], np.arange(4), examples=[1, 2]
    )
    assert list(examples) == [1, 2]
    assert mode[0, :2].tolist() == [EXEMPLAR, DRAW]
    assert addr[0, 0] == 1


def test_source_and_exemplar_mass_combines_before_choosing_ink():
    ids = np.array([0, 0, 1, 0, 2])
    best, certified = pooled_choice([0.11, 0.09, 0.12], [0.09, 0.24], ids, 0.21, 0.14)
    assert best == 0 and not certified
    best, certified = pooled_choice([0.14, 0.10, 0.12], [0.11, 0.22], ids, 0.21, 0.10)
    assert best == 0 and certified
    assert pooled_choice([0.1], [0.15], np.array([0, 1]), 0.65, 0.10) == (2, True)


def test_unseen_ink_can_beat_the_only_known_image():
    assert pooled_choice([0.18], [0.16], np.array([0, 0]), 0.25, 0.41) == (0, False)


def test_legacy_and_pooled_neural_entrypoints(tmp_path):
    import torch
    from safetensors.torch import save_file
    from ilm.visual_lm.block_translation import BlockTranslator
    from ilm.visual_lm.causal_glyph_translator import CausalGlyphTranslator, END
    from scripts.infer_visual_translation import generate

    packed = np.zeros((1, 128), np.uint8)
    landmarks = torch.arange(48)
    block = BlockTranslator(landmarks)
    legacy = tmp_path / "block.safetensors"
    save_file(block.state_dict(), str(legacy))
    assert generate(packed, "neural", legacy).shape[1] == 128

    model = CausalGlyphTranslator(landmarks, torch.zeros(1, 1024).bool())
    with torch.no_grad():
        model.route.weight.zero_()
        model.route.bias.fill_(-12)
        model.route.bias[END] = 12
    current = tmp_path / "reader.safetensors"
    save_file(model.state_dict(), str(current))
    for method in ("pooled", "marginal"):
        receipt = {}
        assert generate(packed, method, current, receipt=receipt).shape == (0, 128)
        assert receipt["ending"] == "learned_stop"
