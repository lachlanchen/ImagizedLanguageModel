from collections import Counter

import numpy as np

from scripts.train_book_breadth import matched_pool, schedule


def test_replay_preserves_both_exact_lengths_and_domain_in_every_batch():
    lengths = np.array([[21, 8]] * 33 + [[21, 9]] * 17 + [[32, 12]], np.uint8)
    domains = np.array([0] * 16 + [1] * 17 + [0] * 18)
    mapping = matched_pool(lengths, domains)
    assert len(set(mapping)) == 2 + 3 + 3 + 1
    assert max(Counter(mapping).values()) <= 8
    for epoch in range(2):
        batches = schedule(lengths, epoch)
        assert sorted(np.concatenate(batches)) == list(range(len(lengths)))
        for ids in batches:
            assert np.array_equal(lengths[ids], lengths[mapping[ids]])
            assert np.array_equal(domains[ids], domains[mapping[ids]])
