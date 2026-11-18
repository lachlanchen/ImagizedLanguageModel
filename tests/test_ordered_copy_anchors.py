from itertools import combinations, product

from ilm.visual_lm.ordered_copy_anchors import forced_matches


def test_mandatory_pairs_against_every_alignment_with_repeated_ink():
    strings = [s for n in range(5) for s in product(range(2), repeat=n)]
    for x in strings:
        for y in strings:
            for n in range(min(len(x), len(y)), -1, -1):
                opt = [
                    set(zip(a, b))
                    for a in combinations(range(len(x)), n)
                    for b in combinations(range(len(y)), n)
                    if all(x[i] == y[j] for i, j in zip(a, b))
                ]
                if opt:
                    expected = sorted(set.intersection(*opt))
                    assert forced_matches(x, y) == (n, expected)
                    break
