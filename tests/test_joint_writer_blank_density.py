"""Check the probability bound against complete small trees, not Chinese scores."""
import itertools
import numpy as np

from scripts.analyze_joint_writer_blank_density import blank_path_bounds


def test_nonempty_upper_bound_covers_every_completion():
    rng = np.random.default_rng(512)
    for _ in range(20):
        tree = {}
        for length in range(4):
            for prefix in itertools.product(range(3), repeat=length):
                p = rng.dirichlet(np.ones(3))
                tree[prefix] = p
        zero_history = np.array([[np.log(tree[(0,) * j]) for j in range(4)]])
        blank, upper = blank_path_bounds(zero_history)
        for sequence in itertools.product(range(3), repeat=4):
            value = sum(np.log(tree[sequence[:j]][sequence[j]]) for j in range(4))
            if any(sequence):
                assert value <= upper[0] + 1e-12
            else:
                assert abs(value - blank[0]) < 1e-12


def test_greedy_blank_can_lose_to_a_complete_nonempty_image():
    # Greedy chooses 0 then 0: .6 * .51. Alternative 1 then 1: .4 * .99.
    lp = np.log([[[.6, .4], [.51, .49]]])
    blank, upper = blank_path_bounds(lp)
    assert np.log(.4 * .99) > blank[0]
    assert upper[0] >= np.log(.4 * .99)


def test_bound_can_certify_blank_without_enumerating_the_suffix():
    blank, upper = blank_path_bounds(np.log([[[.99, .01], [.99, .01]]]))
    assert upper[0] < blank[0]
