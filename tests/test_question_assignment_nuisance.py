"""Numerical and data-isolation checks for the training-only nuisance probe."""

import itertools

import numpy as np

from scripts.audit_question_assignment_nuisance import assignment_margin, ridge_predictions


def test_held_article_targets_cannot_affect_their_own_predictions():
    rng = np.random.default_rng(20266380)
    features = rng.normal(size=(50, 6))
    folds = np.repeat(np.arange(5), 10)
    targets = features[:, 0]*3 + rng.normal(size=50)
    baseline = ridge_predictions(features, targets, folds)
    for fold in range(5):
        changed = targets.copy()
        changed[folds == fold] += 1000
        prediction = ridge_predictions(features, changed, folds)
        # Even target normalization must exclude the held article fold.
        np.testing.assert_array_equal(prediction[folds == fold], baseline[folds == fold])
        assert np.max(np.abs(prediction[folds != fold]-baseline[folds != fold])) > 1


def test_assignment_orientation_matches_exhaustive_two_way_matching():
    for values in itertools.product((-2., 0., 3.), repeat=4):
        predicted, observed = np.array(values[:2]), np.array(values[2:])
        costs = (predicted[:, None]-observed[None, :])**2
        energies = [costs[0, p[0]]+costs[1, p[1]] for p in itertools.permutations((0, 1))]
        assert assignment_margin(predicted, observed) == energies[1]-energies[0]
        assert assignment_margin(predicted[::-1], observed) == energies[0]-energies[1]
        assert assignment_margin(np.ones(2), observed) == 0
