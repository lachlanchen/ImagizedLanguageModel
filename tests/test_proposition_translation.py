"""Physical prefix, bilingual likelihood and stopped candidate invariants."""

import math
import numpy as np
import torch
from torch.nn import functional as F
from ilm.visual_lm.bilingual_alignment import BilingualTraining
from ilm.visual_lm.reverse_translation import sequence_scores, stopped_beams, rerank
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_proposition_translation import BASE, Ink, read
from scripts.train_proposition_translation import divergence
from scripts.train_ordered_translation import pack
from scripts.write_ordered_translation import restore, stopped


def test_matched_bilingual_likelihood_and_stop():
    torch.set_num_threads(2)
    net = restore("proposition-likelihood", "cpu")
    ink = Ink(np.load(BASE / "palette.npy"))
    rows = [
        ["Tom helped Mary.", "汤姆帮助了玛丽。"],
        ["Mary helped Tom.", "玛丽帮助了汤姆。"],
    ]
    seq = [[ink.encode(r[t]) for r in rows] for t in (0, 1)]
    src = [pack(side, "cpu") for side in seq]
    tgt = [pack([v + [1] for v in side], "cpu") for side in seq]
    features = torch.tensor(observations({"tiles": ink.array()}, "ink"))
    inv = [torch.tensor(v) for v in read(BASE / "inventories.json")]
    loc = [torch.full((len(features),), -100) for _ in inv]
    for a, v in zip(loc, inv):
        a[v] = torch.arange(len(v))
    with torch.inference_mode():
        ce, aux, count = BilingualTraining(net)(
            features, src, tgt, inv, loc, torch.tensor([[0, 0], [1, 1]]), False
        )
        expected = (
            sum(
                F.cross_entropy(
                    net(features, src[1 - t], tgt[t], inv[t], t).flatten(0, 1),
                    loc[t][tgt[t]].flatten(),
                    ignore_index=-100,
                    reduction="sum",
                )
                for t in (0, 1)
            )
            / count
        )
        torch.testing.assert_close(ce, expected)
        assert aux.item() == 0 and divergence([5, 1], [5, 6, 1]) == 1
        # Raw scores include the original stopping action even for a strict-prefix alternative.
        vals = sequence_scores(net, features, seq[0], seq[1], inv[1], 1)
        for k in range(2):
            target = torch.tensor([seq[1][k] + [1]])
            lp = net(
                features, torch.tensor([seq[0][k]]), target, inv[1], 1
            ).log_softmax(-1)
            score = lp.gather(-1, loc[1][target][:, :, None]).mean()
            assert abs(vals[k] - float(score)) < 2e-5
        greedy = [stopped(v) for v in net.generate(features, src[0], inv[1], 1)]
        pick = rerank(net, features, seq[0], inv, 1, greedy)
        for i, p in enumerate(pick):
            assert p["forward"] in p["candidates"] and p["roundtrip"] in p["candidates"]
            if greedy[i][1]:
                assert greedy[i][0] in p["candidates"]
            assert all(len(v) <= 128 for v in p["candidates"])
        assert sequence_scores(net, features, [[]], [[]], inv[1], 1) == [-math.inf]
        beam = stopped_beams(
            net, net.optical(features), src[0], inv[1], 1, width=2, limit=2
        )
        assert all(len(v) <= 2 for opts in beam for v in opts)
