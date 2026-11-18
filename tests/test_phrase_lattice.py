import itertools
import math

from rapidfuzz.distance import Levenshtein
from ilm.visual_lm.phrase_lattice import admits, best_streams, nearest_stream


class Writer:
    weight = 0.25

    def transition(self, last, first):
        return 0.1 * (last - first)


def enumerate_paths(edges, writer):
    scores = {}

    def visit(i, seq, score):
        if i == len(edges):
            scores[seq] = max(scores.get(seq, -math.inf), score)
            return
        for end, part, logp in edges[i]:
            cost = logp - 0.65
            if part:
                cost += writer.weight * max(
                    -3.0, min(3.0, writer.transition(seq[-1] if seq else -1, part[0]))
                )
            visit(end, seq + part, score + cost)

    visit(0, (), 0.0)
    return scores


def test_exact_against_all_paths_and_all_short_targets():
    edges = [
        [(1, (0,), -0.2), (1, (), -0.6), (2, (0, 1), -0.3), (2, (1,), -0.7)],
        [(2, (1,), -0.33), (2, (0,), -0.2), (2, (), -0.2), (3, (), -0.8)],
        [(3, (0,), -0.25), (3, (), -0.31), (3, (1, 2), -0.4)],
    ]
    model = Writer()
    scores = enumerate_paths(edges, model)
    for k in (1, 3, 6, 100):
        stream = best_streams(model, edges, k)
        expected = sorted(scores.items(), key=lambda p: (-p[1], p[0]))[:k]
        assert [s for s, _ in stream] == [s for s, _ in expected]
        assert all(math.isclose(v, scores[s]) for s, v in stream)
    for size in range(5):
        for target in itertools.product(range(3), repeat=size):
            dist, witness = nearest_stream(edges, target)
            assert witness in scores
            assert (
                dist
                == Levenshtein.distance(witness, target)
                == min(Levenshtein.distance(s, target) for s in scores)
            )
            assert admits(edges, target) == (dist == 0)
