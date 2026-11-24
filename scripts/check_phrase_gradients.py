"""Check the full writer gradient against an independent differentiable path sum on real data."""

from collections import defaultdict
import math
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from ilm.visual_lm.phrase_region import LENGTHS
from ilm.visual_lm.marginal_phrase import MarginalPhraseWriter
from scripts.prepare_marginal_confirmation import PUB
from scripts.train_region_writer import Corpus, pack
from scripts.train_phrase_region import MODEL as LONGEST, DATA
from scripts.prepare_open_exam import dump, read, sha


def dense_reference(net, ctx, hidden, y):
    seq = y[0].tolist()
    source = ctx[2][0].tolist()
    inp = torch.cat((net.bos[None, None, :], ctx[0][y[:, :-1]]), 1)
    state, _ = net.gru(inp, hidden)
    past = torch.stack([F.pad(y, (i, 0))[:, : y.size(1)] for i in (1, 2, 3)], -1)
    sp, ptr, gate = net.distribution(ctx, state, past)
    logits = net.run_length(state)[0]
    starts = [
        [j for j in range(len(source) - n + 1) if min(source[j : j + n]) >= 2]
        for n in LENGTHS
    ]
    length = logits.masked_fill(
        torch.tensor([not v for v in starts]), -torch.inf
    ).softmax(-1)
    paths = defaultdict(list)
    paths[0].append(state.new_zeros(()))
    for i, target in enumerate(seq):
        prefix = torch.stack(paths[i]).logsumexp(0)
        if 1 <= target <= sp.size(-1):
            paths[i + 1].append(
                prefix + sp[0, i, target - 1] + F.logsigmoid(gate[0, i])
            )
        for k, n in enumerate(LENGTHS):
            if i + n > len(seq):
                continue
            aligned = [j for j in starts[k] if source[j : j + n] == seq[i : i + n]]
            if not aligned:
                continue
            probability = (
                length[i, k] * ptr[0, i, aligned].sum() / ptr[0, i, starts[k]].sum()
            )
            paths[i + n].append(prefix + F.logsigmoid(-gate[0, i]) + probability.log())
    return -torch.stack(paths[len(seq)]).logsumexp(0) / len(seq)


def main():
    torch.set_num_threads(2)
    d = Corpus("train", "cpu")
    picks = np.load(DATA / "selector/train-regions.npy")
    net = MarginalPhraseWriter().double()
    net.load_state_dict(load_file(str(LONGEST / "weights.safetensors")))
    cert = {}
    for t in ("statements", "catalogue", "restaurant"):
        i = next(
            i for i, r in enumerate(d.rows) if r[4] == t and len(d.seq[r[3][0]]) <= 90
        )
        r = d.rows[i]
        q, p, y = [
            pack([s], "cpu")
            for s in (d.seq[r[0]], d.seq[r[1][picks[i]]], np.r_[d.seq[r[3][0]], 1])
        ]

        def calc(f):
            ctx, h = net.encode(d.features.double(), d.n, q, p)
            loss = f(net, ctx, h, y).mean()
            grads = torch.autograd.grad(loss, list(net.parameters()))
            return float(loss.detach()), grads

        val, a = calc(MarginalPhraseWriter.loss)
        ref, b = calc(dense_reference)
        # distribution() normalizes the categorical scores in float32.
        diff = max(float((x - z).abs().max()) for x, z in zip(a, b))
        rel = math.sqrt(
            sum(float((x - z).square().sum()) for x, z in zip(a, b))
            / sum(float(x.square().sum()) for x in b)
        )
        assert abs(val - ref) < 1e-5 and rel < 3e-5 and diff < 2e-5, (
            t,
            val,
            ref,
            rel,
            diff,
        )
        cert[t] = dict(
            row=i,
            cells=int(y.numel()),
            loss=val,
            independent=ref,
            parameter_tensors=len(a),
            max_abs_gradient=diff,
            relative_l2_gradient=rel,
        )
    dump(PUB / "gradients.json", cert)
    print(cert)


if __name__ == "__main__":
    main()
