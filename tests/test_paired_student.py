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
