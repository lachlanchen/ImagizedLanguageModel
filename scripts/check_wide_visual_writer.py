"""Check wider logits, autonomous ink, and actual learning on the full released alphabet."""

from time import monotonic
import torch
from scripts.reserve_visual_prefix import BASE, PUB, ROOT, read, dump, sha
from scripts.prepare_visual_rollins import training_rows
from scripts.train_ordered_translation import pack
from scripts.train_paired_ink import conditional, material
from scripts.write_ordered_translation import restore, stopped
from ilm.visual_lm.wide_translation import widen


def check():
    torch.set_num_threads(3)
    torch.manual_seed(791)
    cells = training_rows()
    f, inv, local = material("cpu")
    small = restore("book-rehearsal", "cpu")
    large = widen(small).eval()
    start = monotonic()
    sampled = list(range(0, len(cells), max(1, len(cells) // 32)))[:32]
    worst = 0.0
    emitted = 0
    for t in (0, 1):
        parts = [[side[side != 0].tolist() for side in row] for row in cells[sampled]]
        x = pack([p[1 - t] for p in parts], "cpu")
        y = pack([p[t] + [1] for p in parts], "cpu")
        with torch.no_grad():
            a, b = (m(f, x, y, inv[t], t) for m in (small, large))
            error = float((a - b).abs().max())
            if error > 0.002:
                raise ValueError(("logits", error))
            worst = max(worst, error)
            p, q = (m.generate(f, x, inv[t], t) for m in (small, large))
            for left, right in zip(p, q, strict=True):
                if stopped(left) != stopped(right):
                    raise ValueError(
                        "An autonomous source output changed at initialization"
                    )
                emitted += 1
    # A real Adam step must distinguish cloned optical rows and allow the loss to decrease.
    large.train()
    opt = torch.optim.AdamW(large.parameters(), lr=0.00002)
    objective = (
        lambda: -conditional(large, f, x, y, inv[t], local[t], t).sum()
        / y.count_nonzero()
    )
    loss = objective()
    loss.backward()
    grad = large.optical[0].weight.grad
    delta = float((grad[:160] - grad[160:320]).abs().max())
    torch.nn.utils.clip_grad_norm_(large.parameters(), 1)
    opt.step()
    with torch.no_grad():
        after = float(objective())
    if not (delta > 0 and after < float(loss.detach())):
        raise ValueError("Widened learning failed")
    dump(
        PUB / "wide-preflight.json",
        dict(
            inputs=emitted,
            independent_png_streams=False,
            actual_stopped_streams_identical=emitted,
            max_logit_error=worst,
            first_layer_replica_gradient_difference=delta,
            batch_nll_before=float(loss.detach()),
            batch_nll_after=after,
            narrow_parameters=sum(p.numel() for p in small.parameters()),
            wide_parameters=sum(p.numel() for p in large.parameters()),
            seconds=monotonic() - start,
            device="CPU",
            retained_updates=0,
            source=sha(ROOT / "ilm/visual_lm/wide_translation.py"),
        ),
    )


if __name__ == "__main__":
    check()
