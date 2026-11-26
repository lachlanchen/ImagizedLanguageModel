"""Check the cut in a committed draft and the identities of both arms' supervision."""

from itertools import product
import numpy as np
from rapidfuzz.distance import Levenshtein
from scripts.prepare_visual_rollins import endpoints
from scripts.train_visual_prefix import feedback


def pad(rows, width):
    a = np.zeros((len(rows), width), np.int32)
    for i, r in enumerate(rows):
        a[i, : len(r)] = r
    return a


def test_full_prefix_search_and_attained_edit_floor():
    small = [word for n in range(4) for word in product((2, 3), repeat=n)]
    refs = [word for n in range(6) for word in product((2, 3), repeat=n)]
    pairs = list(product(small, refs))
    drafts, golds = zip(*pairs)
    js, ds = endpoints(pad(drafts, 3), pad(golds, 5))
    for (p, y), j, d in zip(pairs, js, ds, strict=True):
        oracle = min(
            (Levenshtein.distance(p, y[:k]), abs(len(p) - k), k)
            for k in range(len(y) + 1)
        )
        assert (d, j) == (oracle[0], oracle[2])
        assert Levenshtein.distance(p + y[j:], y) == d


def test_only_feedback_changes_including_empty_and_premature_draft():
    for cut, draft, end in [
        (0, [], 0),
        (0, [8, 9], 0),
        (1, [8], 2),
        (2, [4, 3], 1),
        (4, [], 0),
    ]:
        gold = [2, 3, 4, 5]
        clean, teacher = feedback(gold, cut, draft, end, "reference")
        mixed, student = feedback(gold, cut, draft, end, "recovery")
        assert clean == gold + [1]
        assert mixed == gold[:cut] + draft + gold[cut + end :] + [1]
        assert (
            [x for x in teacher if x]
            == [x for x in student if x]
            == gold[:cut] + gold[cut + end :] + [1]
        )
