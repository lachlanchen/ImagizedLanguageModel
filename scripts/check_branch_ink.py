"""Validate the differentiable prefix ratio against the actual sequential optical writer."""

import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from scripts.train_branch_ink import branch_log_prob
from scripts.train_open_exam import corpus
from scripts.write_visual_discourse import Writer
from scripts.write_recurrent_evidence import load, generate
from scripts.prepare_branch_ink import PUB, WORK, dump
from scripts.train_paired_evidence import gpu_owner


def main():
    rows, tiles, n = corpus("train")
    cuts = np.load(WORK / "cuts.npy")
    indices = [int(np.flatnonzero(cuts == v)[0]) for v in np.unique(cuts) if v >= 0]
    indices.extend(
        i
        for i in range(len(rows))
        if cuts[i] >= 0 and rows[i]["y"][cuts[i]] == 1 and i not in indices
    )
    assert any(rows[i]["y"][cuts[i]] == 1 for i in indices)
    writer = Writer(device="cuda")
    base = writer.model.requires_grad_(False)
    net = load("dynamic", "cuda").train()
    x = torch.from_numpy(observations({"tiles": tiles}, "ink")).cuda()
    bk = base.encode(x).detach()
    measured = branch_log_prob(
        net, base, x, bk, n, [rows[i] for i in indices], cuts[indices]
    )
    trials = []
    for j, i in enumerate(indices):
        r = rows[i]
        prefix = r["y"][: cuts[i] + 1]
        seq = generate(
            base,
            net,
            x,
            n,
            [r["qi"]],
            [r["pi"]],
            [r["pi"] + r["qi"]],
            len(prefix),
            forced=torch.tensor([prefix], device=x.device),
        )[0, -1].item()
        trials.append(
            dict(
                training_index=i,
                cut=int(cuts[i]),
                stop=prefix[-1] == 1,
                actual=seq,
                ratio=measured[j].item(),
                error=abs(seq - measured[j].item()),
            )
        )
    (-measured.mean()).backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    norm = sum(g.float().square().sum().item() for g in grads) ** 0.5
    err = max(r["error"] for r in trials)
    passed = err < 0.06 and np.isfinite(norm) and norm > 0
    dump(
        PUB / "preflight.json",
        dict(passed=bool(passed), max_error=err, gradient_norm=norm, trials=trials),
    )
    if not passed:
        raise ValueError(f"Mismatch {err}, gradient norm {norm}")
    print(
        len(trials), "actual questions, maximum log error", err, "gradient norm", norm
    )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    with gpu_owner():
        main()
