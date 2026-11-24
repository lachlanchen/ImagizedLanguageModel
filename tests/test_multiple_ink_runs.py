from itertools import product

import numpy as np
import torch

from ilm.visual_lm.multiple_ink_runs import best_path
from scripts.evaluate_multiple_ink_runs import grade, edit
from ilm.visual_lm.paired_evidence import EvidenceWriter
from ilm.visual_lm.multiple_ink_runs import RunReader


def test_budgeted_viterbi_with_trace_and_count_reward():
    rng = np.random.default_rng(17)
    for n in range(6):
        for limit in range(4):
            scores = rng.integers(-5, 6, (n, 3), dtype=np.int64)
            counts = rng.integers(-9, 8, limit + 1, dtype=np.int64)
            tags, score = best_path(scores, counts)

            def admissible(path):
                return sum(z == 1 for z in path) <= limit and all(
                    z != 2 or (i and path[i - 1] != 0) for i, z in enumerate(path)
                )

            def utility(path):
                return (
                    sum(scores[i, z] for i, z in enumerate(path))
                    + counts[sum(z == 1 for z in path)]
                )

            assert admissible(tags) and utility(tags) == score
            assert (
                max(utility(p) for p in product(range(3), repeat=n) if admissible(p))
                == score
            )


def test_duplicate_or_extra_fact_invalidates_complete_answer():
    assert grade(["甲", "乙"], [["乙", "甲"]])["exact"]
    assert not grade(["甲", "甲", "乙"], [["甲", "乙"]])["exact"]
    assert not grade(["甲"], [["甲", "乙"]])["exact"]
    assert grade(["甲"], [["甲", "乙"], ["甲"]])["exact"]


def test_changed_ink_preserves_the_second_fact():
    r = dict(refs=[["Adam", "Bert"]], q="Who?", p="Adam met Bert.")
    i, j, text, ref = edit(r)
    assert (i, j) == (0, 4) and len(text) == len(r["p"]) and ref[0][1] == "Bert"


def test_shared_reader_is_unchanged_by_padding():
    torch.manual_seed(3)
    net = RunReader(EvidenceWriter()).eval()
    bank = torch.randn(10, 80)
    q = torch.tensor([[2, 3, 4]])
    p = torch.tensor([[5, 6, 7]])
    with torch.no_grad():
        role, count = net(bank, q, p)
        padded, c = net(bank, q, torch.nn.functional.pad(p, (0, 2)))
    torch.testing.assert_close(role, padded[:, :3], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(count, c, atol=1e-5, rtol=1e-5)
