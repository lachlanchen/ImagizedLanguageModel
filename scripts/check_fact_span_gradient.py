"""Independently enumerate legal physical spans on three real pages and compare full gradients."""

import torch
from scripts.prepare_fact_channel import PUB, dump
from scripts.train_fact_channel import Data, load, join_regions, pack


def main():
    torch.set_num_threads(2)
    data = Data("reader", "cpu")
    model = load("reader", "cpu").double()
    probe = []
    for domain, repeat in (("squad", False), ("exam", False), ("squad", True)):
        i = next(
            i
            for i, r in enumerate(data.rows)
            if r[-2] == domain
            and 40 < sum(len(data.seq[b]) for b in r[1]) < 220
            and (not repeat or len(r[2]) > 1)
            and i not in probe
        )
        probe.append(i)
    measurements = []
    for i in probe:
        r = data.rows[i]
        pg = join_regions(data.pages([i])[0])
        p = pack([pg], "cpu")
        q = pack([data.seq[r[0]]], "cpu")
        captured = []
        hook = model.boundaries.register_forward_hook(
            lambda mod, args, out: captured.append(out)
        )
        try:
            scores = model(data.features.double(), q, p)
        finally:
            hook.remove()
        k = scores.shape[-1]
        aliases = torch.tensor([[s * k + n - 1 for s, n in r[2]]])
        value = model.loss(scores, aliases).sum()
        raw = captured[0][0].float()
        starts = []
        ends = []
        right = []
        accepted = {tuple(pg[s : s + n].tolist()) for s, n in r[2]}
        for s in range(len(pg)):
            for e in range(s, min(s + 256, len(pg))):
                if pg[e] == 0:
                    break
                starts.append(s)
                ends.append(e)
                if tuple(pg[s : e + 1].tolist()) in accepted:
                    right.append(len(starts) - 1)
        ss, ee = map(torch.tensor, (starts, ends))
        v = raw[ss, 0] + raw[ee, 1] + model.length_bias[ee - ss]
        independent = v.logsumexp(0) - v[right].logsumexp(0)
        assert len(right) == len(r[2])
        grads = torch.autograd.grad(value, tuple(model.parameters()), retain_graph=True)
        ref = torch.autograd.grad(independent, tuple(model.parameters()))
        diff = torch.cat([(a - b).flatten() for a, b in zip(grads, ref)])
        rel = float(diff.norm() / torch.cat([a.flatten() for a in ref]).norm())
        assert abs(float((value - independent).detach())) < 2e-6 and rel < 2e-5
        measurements.append(
            dict(
                row=i,
                domain=r[-2],
                page_cells=len(pg),
                candidates=len(starts),
                accepted=len(right),
                loss=float(value.detach()),
                enumerated=float(independent.detach()),
                tensors=len(grads),
                relative_l2=rel,
                max_abs=float(diff.abs().max()),
            )
        )
    dump(PUB / "real-gradients.json", measurements)
    print(measurements)


if __name__ == "__main__":
    main()
