from itertools import combinations
from scripts.train_teacher_translation import balanced_indices


def test_keeps_largest_zero_sum_set():
    for values in ([], [0], [1, 3], [-2, 3, -1, 0, 3], [-1, 1, 2, -3, 1, 2, -1]):
        retained = balanced_indices(values)
        assert len(set(retained)) == len(retained)
        assert sum(values[i] for i in retained) == 0
        feasible = [len(group) for k in range(len(values)+1) for group in combinations(values,k) if sum(group)==0]
        assert len(retained) == max(feasible)
