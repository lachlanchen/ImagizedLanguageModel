"""Check complete-distribution bounds and analytic softmax gradients for glyph deficits."""
from itertools import product
from pathlib import Path
import torch

from ilm.visual_lm.target_glyph_teaching import deficit_loss
from scripts.reserve_target_glyph_book_panel import PUB, ROOT, dump, sha


def verify():
    torch.manual_seed(918)
    logits = torch.randn((192,17), dtype=torch.float64, requires_grad=True)
    teacher = torch.randn_like(logits).softmax(-1)
    floors = teacher*torch.rand_like(teacher)
    codes = torch.arange(17).expand(192,-1)
    logq = logits.log_softmax(-1)
    loss = deficit_loss(logq, codes, floors)
    kl = (teacher*(teacher.log()-logq)).sum(-1)
    if (loss < -1.e-14).any() or (loss>kl+1.e-13).any():
        raise ValueError("Violation of complete KL bound")
    actual, = torch.autograd.grad(loss.sum(),logits)
    q = logq.detach().exp()
    d = (floors-q).clamp_min(0)
    analytic = d.sum(-1,keepdim=True)*q-d
    if not torch.allclose(actual,analytic,rtol=1e-11,atol=1e-12):
        raise ValueError("Wrong deficit gradient")
    feasible = deficit_loss(logq.detach(),codes,q*.99)
    if feasible.count_nonzero(): raise ValueError("Feasible completion penalized")
    # Independent exhaustive finite check of the Lean least-unavoidable bound.
    simplices = [x for x in product(range(7),repeat=3) if sum(x)==6]
    sub = [x for x in product(range(7),repeat=3) if sum(x)<=6]
    cases = 0
    for f in sub:
        consistent = [p for p in simplices if all(a>=b for a,b in zip(p,f))]
        for s in simplices:
            known = sum(max(0,l-q) for l,q in zip(f,s))
            optimum = min(sum(max(0,a-b) for a,b in zip(p,s)) for p in consistent)
            if optimum!=known: raise ValueError("Inconsistent minimum variation")
            cases+=1
    return dict(kl_cases=len(logits),max_gradient_error=float((analytic-actual).abs().max()),
                zero_feasible_loss=feasible.max().item(),finite_completion_cases=cases,
                program=sha(Path(__file__)),proof=sha(ROOT/"formal/span_list/OpticalMassCompletion.lean"))


if __name__=="__main__":
    dump(PUB/"loss_checks.json", verify())
