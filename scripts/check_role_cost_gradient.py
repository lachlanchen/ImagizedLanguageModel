"""Check the role objective's exact logit derivative on every eligible development glyph."""

import math
from pathlib import Path

import numpy as np
import torch

from ilm.visual_lm.role_attention import window_roles
from scripts.train_role_grounding import MODELS, PUB, RUN, library, material, pack, read, sha, dump
from scripts.write_ordered_translation import restore


def main():
    torch.set_num_threads(3)
    net = restore("role-grounded", "cpu")
    net.requires_grad_(False)
    features, inv, _ = material("cpu")
    pairs = library("dev")
    tagfile = RUN / "dev.npy"
    if sha(tagfile) != read(PUB / "alignment.json")["files"][tagfile.name]:
        raise ValueError("Annotation mismatch")
    tags = torch.from_numpy(np.load(tagfile))
    count, max_abs, max_fd = 0, 0., 0.
    rng = torch.Generator().manual_seed(618319)
    sums = torch.zeros(3, dtype=torch.float64)
    for t in (0, 1):
        for start in range(0,len(pairs),12):
            batch = pairs[start:start+12]
            x = pack([r["cells"][1-t] for r in batch], "cpu")
            y = pack([r["cells"][t]+[1] for r in batch], "cpu")
            roles = tags[start:start+len(batch)]
            target = torch.zeros_like(y)
            width = min(y.size(1), roles.size(2))
            target[:, :width] = roles[:, t, :width]
            frac = window_roles(x, roles[:, 1-t, :x.size(1)], net.stride).double()
            membership = frac.transpose(1,2).gather(1, target.unsqueeze(-1).expand(-1,-1,frac.size(1)))
            eligible = target.ne(0) & membership.amax(-1).gt(0)
            saved = []

            def capture(module, args):
                query,(keys,_),valid = args
                q = module.heads(module.q(query)).double()
                saved.append(((q @ keys.double().transpose(-1,-2)) / math.sqrt(q.size(-1)), valid))

            h = net.dec[-1].cross.register_forward_pre_hook(capture)
            try:
                with torch.no_grad():
                    net(features, x, y, inv[t], t)
            finally:
                h.remove()
            logits, valid = saved[0]
            logits.requires_grad_()
            def compute(z):
                pi = z.masked_fill(~valid, -torch.inf).softmax(-1)
                head = (pi * membership[:,None]).sum(-1)
                mass = head.mean(1).masked_fill(~eligible,1.)
                return -mass.log(), pi, head, mass

            loss, pi, mean, mass = compute(logits)
            grad, = torch.autograd.grad(loss.sum(), logits)
            with torch.no_grad():
                predicted = pi*(mean.unsqueeze(-1)-membership[:,None]) / (pi.size(1)*mass[:,None,:,None])
                predicted *= eligible[:,None,:,None]
                delta = torch.randn(logits.shape, generator=rng, dtype=torch.float64)
                step = 1e-5
                finite = (compute(logits+step*delta)[0]-compute(logits-step*delta)[0])/(2*step)
                exact = (predicted*delta).sum((1,3))
                max_abs = max(max_abs,(grad-predicted).abs().max().item())
                max_fd = max(max_fd,(finite-exact).abs().max().item())
                sums += torch.stack((loss.sum(), mean.mean(1)[eligible].sum(), eligible.sum()))
                count += eligible.sum().item()
    if count != 8249 or max_abs > 1e-8 or max_fd > 1e-7:
        raise ValueError((count, max_abs, max_fd))
    result = dict(model=sha(MODELS/"grounded/weights.safetensors"),program=sha(Path(__file__)),
                  positions=count, mean_mass=sums[1].item()/count, role_nll=sums[0].item()/count,
                  max_autograd_delta=max_abs,max_directional_difference=max_fd, difference_step=step)
    dump(PUB/"gradient.json", result)
    print(result)


if __name__ == "__main__":
    main()
