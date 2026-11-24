from itertools import product, combinations
from scripts.audit_visible_answer_sets import placement


def test_visible_alternatives_against_all_disjoint_occurrence_tuples():
    for length in range(1, 7):
        for chars in product("ab", repeat=length):
            page = "".join(chars)
            cache = {}
            words = ("a", "aa", "ab", "b", "ba", "bb")
            for ref in combinations(words, 3):
                choices = [
                    [(i, i + len(w)) for i in range(length) if page.startswith(w, i)]
                    for w in ref
                ]
                possible = any(
                    all(
                        a[1] <= b[0] or b[1] <= a[0] for a, b in combinations(select, 2)
                    )
                    for select in product(*choices)
                )
                assert (placement(page, ref, cache) == "placed") == possible
    assert placement("abba", ["a", "bb"], {}, budget=0) == "undecided"
