from itertools import product
import numpy as np
import torch
from ilm.visual_lm.fact_binding import decode, pack_paths, score_paths


def test_exact_two_program_decoder_and_supplied_paths():
    rng = np.random.default_rng(229)
    for n in range(1, 6):
        paths, labels = [], []
        for row in product(range(3), repeat=n):
            if any(t == 2 and (i == 0 or row[i - 1] == 0) for i, t in enumerate(row)):
                continue
            path = []
            for i, t in enumerate(row):
                if t == 1:
                    path.append([i, i + 1])
                elif t == 2:
                    path[-1][1] = i + 1
            paths.append(path)
            labels.append(row)
        bounds = torch.from_numpy(pack_paths([paths]))
        for _ in range(4):
            roles, head, count = [
                rng.integers(-12, 13, shape) for shape in ((n, 3), (n, 2), n + 1)
            ]
            for mixture in (False, True):
                scores, _ = score_paths(
                    *(torch.from_numpy(a[None]) for a in (roles, count, head)),
                    bounds,
                    mixture,
                )
                gold = []
                for path, tags in zip(paths, labels):
                    gain = sum(head[i, t - 1] for i, t in enumerate(tags) if t)
                    gold.append(
                        sum(roles[i, t] for i, t in enumerate(tags))
                        + count[len(path)]
                        + (abs(gain) if mixture else gain)
                    )
                assert (scores.numpy()[0] + roles[:, 0].sum() == gold).all()
                chosen, best = decode(roles, count, head, mixture)
                assert best == max(gold) == gold[paths.index(chosen)]
