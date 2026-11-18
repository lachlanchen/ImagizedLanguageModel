from collections import Counter

import numpy as np

from ilm.visual_lm.visual_phrase_boundary import VisualPhraseBoundary


def test_learned_removal_needs_a_matching_unconsumed_neighbour(tmp_path):
    ink = [bytes([i]) * 128 for i in range(1, 8)]
    left, centre, right, sense, phrase, unknown, second = ink
    as_rows = lambda x: np.frombuffer(b"".join(x), np.uint8).reshape(-1, 128)

    class Copy:
        def generate(self, x):
            return x.copy()

    class Senses:
        composer = Copy()
        heads = {centre: True, second: True}

        def choices(self, x, positions):
            return [sense for _ in positions]

    model = VisualPhraseBoundary(Senses())
    source = as_rows((left, centre, right))
    target = as_rows((phrase, right))
    for _ in range(4):
        model.observe(
            source,
            1,
            sense,
            phrase,
            target,
            (0, 1),
            [
                dict(left=1, right=0, outer_edit_cost=0),
                dict(left=0, right=0, outer_edit_cost=1),
            ],
        )
    path = tmp_path / "b.gz"
    model.save(path)
    frozen = VisualPhraseBoundary.restore(path, model.senses)
    assert frozen.generate(source).tobytes() == phrase + right
    assert (
        frozen.generate(as_rows([unknown, centre, right])).tobytes()
        == unknown + phrase + right
    )
    # A neighbouring active centre cannot be swallowed, regardless of its weight.
    frozen.tables[(centre, sense, 0, b"", b"")] = Counter({(None, second, phrase): 1e6})
    assert frozen.generate(as_rows([centre, second])).tobytes() == sense + sense
