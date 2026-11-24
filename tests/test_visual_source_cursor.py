import itertools
import torch
from ilm.visual_lm.source_cursor import SourceCursor


def test_forward_filter_agrees_with_enumerating_copy_and_write_paths():
    # Duplicate marks at two source locations have different continuations.
    x = [2, 3, 2, 4]
    starts = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64)
    c = torch.tensor([[0.8, 0.5, 0.9, 0.1]], dtype=torch.float64)
    b = torch.tensor([0.73], dtype=torch.float64)
    mask = torch.ones((1, 4), dtype=torch.bool)
    free = {2: 0.4, 3: 0.3, 4: 0.3}
    for observed in itertools.product((2, 3, 4), repeat=3):
        exact = {-1: 1.0}
        prev = torch.zeros_like(starts)
        joint = 1.0
        for g in observed:
            new = {}
            for old, prob in exact.items():
                keep = float(c[0, old]) if 0 <= old < len(x) - 1 else 0.0
                if keep and x[old + 1] == g:
                    new[old + 1] = new.get(old + 1, 0) + prob * keep
                reset = prob * (1 - keep)
                new[-1] = new.get(-1, 0) + reset * (1 - float(b)) * free[g]
                for j, ch in enumerate(x):
                    if ch == g:
                        new[j] = new.get(j, 0) + reset * float(b * starts[0, j])
            exact = new
            write, source = SourceCursor.paths(prev, c, starts, b, mask, True)
            assert abs(float(write + source.sum()) - 1) < 1e-12
            measured = source * (torch.tensor(x) == g)
            total = write * free[g] + measured.sum()
            joint *= float(total)
            prev = measured / total
        assert abs(joint - sum(exact.values())) < 1e-12


def test_likelihood_is_normalized_and_causal_with_repeated_tiles():
    torch.manual_seed(3)
    torch.set_num_threads(1)
    m = SourceCursor(16).eval()
    features = torch.randn(8, 256)
    ids = torch.tensor([[2, 3, 2, 7, 0]])
    for persistent in (True, False):
        # One source glyph lies outside the learned bitmap alphabet.
        probs = []
        for ch in (1, 2, 3, 4, 5, 6, 7):
            t = torch.tensor([[2, ch]])
            base = torch.tensor(
                [[0.2, 0.1 if ch in (1, 3) else (0.2 if ch in (2, 4, 5, 6) else 0.0)]]
            )
            lp = m.likelihood(features, ids, t, base, 7, persistent)
            probs.append(lp[0, 1].exp())
        assert torch.allclose(sum(probs), torch.tensor(1.0), atol=1e-5)
